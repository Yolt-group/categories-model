import argparse
import datetime as dt
import os
import time
from pathlib import Path
from typing import Dict, AnyStr

import pandas as pd
import yaml
from datascience_model_commons.deploy.config.domain import (
    YDSProjectConfig,
    YDSEnvironment,
)
from datascience_model_commons.deploy.config.load import load_config_while_in_job
from datascience_model_commons.deploy.config.schema import YDSProjectConfigSchema
from datascience_model_commons.utils import get_logger

from categories_model.config.domain import ModelType
from categories_model.config.utils import get_domain_config
from categories_model.training.model import CategoryClassifier
from categories_model.training.utils import read_data, copy_model_and_metadata

logger = get_logger()


def train(
    project_config: YDSProjectConfig,
    script_config: Dict[AnyStr, AnyStr],
    preprocess_path: AnyStr,
    tries=1,
    max_retry=2,
):
    domain_config = get_domain_config(project_config.model_name)

    """Training categories model"""
    # store configuration in training metadata
    domain_config_as_dict = vars(domain_config)
    logger.info(f"Training config: \n{domain_config_as_dict}")
    training_metadata = {"train_config": vars(domain_config)}

    # add categories to the metadata
    training_metadata["categories"] = domain_config.CATEGORIES

    # instantiate new estimator
    model = CategoryClassifier(
        domain_config=domain_config,
        log_device_placement=script_config.get("log_device_placement", False),
    )
    logger.info("CategoryClassifier initialized")

    # read preprocessed datasets
    (df, df_production_sample) = (
        read_data(file_path=table_path)
        for table_path in [
            f"{preprocess_path}/{domain_config.TRAINING_DATA_FILE}",
            f"{preprocess_path}/{domain_config.PRODUCTION_DATA_FILE}",
        ]
    )

    # extract samples parameters
    n_validation_samples = script_config.get(
        "n_validation_samples", domain_config.N_VALIDATION_SAMPLES
    )

    n_test_samples = script_config.get("n_test_samples", domain_config.N_TEST_SAMPLES)

    # For each domain split of datasets and combine
    train_data, validation_data, test_data = (
        pd.concat(df, axis=0)
        for df in zip(
            *(
                # split off dataset for each domain
                model.split_training_data_frame(
                    df=df[df["domain"] == domain],
                    n_validation_samples=n_validation_samples,
                    n_test_samples=n_test_samples,
                )
                for domain in domain_config.LIST_OF_DOMAINS
            )
        )
    )

    training_metadata["n_training_samples"] = n_training_samples = len(train_data)
    training_metadata["n_validation_samples"] = n_validation_samples = len(
        validation_data
    )
    training_metadata["n_test_samples"] = n_test_samples = len(test_data)
    training_metadata["n_production_samples"] = n_production_samples = len(
        df_production_sample
    )
    logger.info(
        "Data split in train, validation and test: \n"
        f"\t Training samples: {n_training_samples:,} \n "
        f"\t Validation samples: {n_validation_samples:,} \n"
        f"\t Test samples : {n_test_samples:,}"
        f"\t Production samples : {n_production_samples:,}"
    )

    # fit
    start_time = time.time()
    batch_size = script_config.get("batch_size")
    model.fit(df_train=train_data, df_validation=validation_data, batch_size=batch_size)
    fit_time = time.time() - start_time
    training_metadata["fit_time"] = time.strftime("%H:%M:%S", time.gmtime(fit_time))
    logger.info("Model fitted")

    # calculate some metadata based on some recent production data
    # 1. store output label distribution
    predictions_sample = pd.Series(model.predict(df_production_sample))
    training_metadata["output_label_distribution"] = predictions_sample.value_counts(
        normalize=True
    ).to_dict()
    logger.info("Output label distribution computed")

    # 2. store input length histogram
    training_metadata[
        "input_description_length_distribution"
    ] = model.calculate_description_length_distribution(df=df_production_sample)
    logger.info("Input description length distribution computed")

    # evaluate on test set
    t = time.time()
    training_metadata["metrics"] = metrics = model.evaluate(df=test_data)
    training_metadata["evaluate_time"] = time.time() - t
    logger.info("Model evaluated on test set")

    # Temporarily save model
    test_path = Path("/tmp") / Path(project_config.model_name)
    copy_model_and_metadata(
        domain_config=domain_config,
        docker_output_path=test_path,
        model=model,
        training_metadata=training_metadata,
    )
    # Load model to test custom model code
    if domain_config.MODEL_TYPE == ModelType.SME_CATEGORIES_MODEL:
        from categories_model.predict.sme import Model
    elif domain_config.MODEL_TYPE == ModelType.RETAIL_CATEGORIES_MODEL:
        from categories_model.predict.retail import Model
    else:
        raise Exception(f"Unsupported modeltype: {domain_config.MODEL_TYPE.value}")
    production_model = Model(test_path)

    # check cucumber tests & model performance
    cucumber_test_sample = model.create_cucumber_test_sample(
        domain_config.CUCUMBER_TEST_SAMPLES
    )
    cucumber_test_sample["predicted_category"] = production_model.predict(
        df=cucumber_test_sample
    )["category"]

    cucumber_sample_count = len(cucumber_test_sample)

    training_metadata["cucumber_tests_result"] = {
        domain_config.TARGET_COLUMN: list(
            cucumber_test_sample[domain_config.TARGET_COLUMN]
        ),
        "predicted_category": list(cucumber_test_sample["predicted_category"]),
        "failed_samples": [
            f"Sample #{i+1} (from range {1}-{cucumber_sample_count}): predicted {predicted}, but should be {target}"
            for (i, (predicted, target)) in enumerate(
                cucumber_test_sample[
                    ["predicted_category", domain_config.TARGET_COLUMN]
                ].to_numpy()
            )
            if predicted != target
        ],
    }
    training_metadata["is_performant"] = is_performant = model.check_performance(
        metrics=metrics, cucumber_tests_df=cucumber_test_sample
    )

    # serialize training log
    training_metadata_yml = yaml.dump(training_metadata)
    logger.info(f"Training metadata: \n{training_metadata_yml}")

    env = project_config.env
    # Todo get docker output path from defaults
    docker_output_path = Path(script_config.get("docker_output_path", "/opt/ml/model"))
    if env != YDSEnvironment.PRD:
        logger.info(
            f"Environment is {env} and model is not performant, but we ignore it!"
        )
        copy_model_and_metadata(
            domain_config=domain_config,
            docker_output_path=docker_output_path,
            model=model,
            training_metadata=training_metadata,
        )
    elif is_performant:
        logger.info(f"After #{tries} of tries the model performance meets expectations")
        copy_model_and_metadata(
            domain_config=domain_config,
            docker_output_path=docker_output_path,
            model=model,
            training_metadata=training_metadata,
        )
    elif tries <= max_retry:
        logger.warning(
            f"Tried {tries} times and model performance is below expectations and we try again"
        )
        train(
            project_config=project_config,
            script_config=script_config,
            preprocess_path=preprocess_path,
            tries=tries + 1,
        )
    else:
        logger.warning("Model performance is below expectations so we stop processing")
        copy_model_and_metadata(
            domain_config=domain_config,
            docker_output_path=docker_output_path,
            model=model,
            training_metadata=training_metadata,
        )
        # only raise exception if not performant on master
        if project_config.git_branch == "master":
            raise Exception(f"After #{tries} the model wasn't performant")


if __name__ == "__main__":
    logger.warning("STARTING JOB")
    parser = argparse.ArgumentParser()
    # --- Don't change the following block unless you're familiar with how YDS works! ---
    # Positional args that are provided when starting the job via YDS.
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--config", type=str, default=os.environ["SM_CHANNEL_CONFIG"])
    parser.add_argument(
        "--preprocessing_output",
        type=str,
        default=os.environ["SM_CHANNEL_PREPROCESSING_OUTPUT"],
    )
    parser.add_argument("--env", type=str)
    args, _ = parser.parse_known_args()
    logger.info(f"Going to load config from {args.config}")
    logger.info(f"Preprocessing output located in {args.preprocessing_output}")
    logger.info(
        f"Preprocessing output files {list(Path(args.preprocessing_output).glob('*'))}"
    )

    # The args.config argument is a training input channel which means we only get the folder
    # name and not the file name. So, we have to manually add the filename here.
    project_config = load_config_while_in_job(Path(args.config) / "yds.yaml")
    script_config = project_config.training.script_config
    # --- Feel free to change from here on out :) ---

    logger.info(f"Loaded config: {project_config}")
    name = "Training"

    execution_date = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    project_config_as_dict = YDSProjectConfigSchema.instance_as_dict(project_config)
    max_train_retry = script_config.get("max_train_retry", 2)

    logger.info(f"Training project config: \n{project_config_as_dict}")
    logger.info(f"Max train retry: \n{max_train_retry}")

    train(
        project_config=project_config,
        script_config=script_config,
        preprocess_path=args.preprocessing_output,
        max_retry=max_train_retry,
    )
