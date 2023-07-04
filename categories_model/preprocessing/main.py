import argparse
import datetime as dt
import logging
import time
from pathlib import Path
from typing import AnyStr, Dict, List, Optional

import pandas as pd
import yaml
from datascience_model_commons.deploy.config.domain import (
    YDSProjectConfig,
    YDSEnvironment,
)
from datascience_model_commons.deploy.config.load import load_config_while_in_job
from datascience_model_commons.deploy.config.schema import YDSProjectConfigSchema
from datascience_model_commons.general import upload_metadata
from datascience_model_commons.spark import (
    upload_parquet,
    get_spark_session,
    read_data,
)
from datascience_model_commons.utils import get_logger
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame as SparkDataFrame
import pyspark.sql.functions as f

from categories_model.config.domain import ModelType
from categories_model.config.utils import get_domain_config
from categories_model.preprocessing.create_training_data import (
    create_synthetic_feedback_df,
    filter_yoltapp_sme_transactions,
)
from categories_model.preprocessing.data import (
    list_folders_on_s3,
    read_data_and_select_columns,
    create_training_data_frame,
)
from categories_model.preprocessing.rules import CATEGORY_RULES

logger = get_logger()


def generate_training_data(
    *,
    users: SparkDataFrame,
    accounts: SparkDataFrame,
    transactions: SparkDataFrame,
    category_rules: Dict[AnyStr, Dict],
    list_of_clients=Optional[List[AnyStr]],
) -> SparkDataFrame:
    # Store logging level and reset fo WARN
    # This prevents an overload of logging while writing to file
    orig_level = logger.getEffectiveLevel()
    logger.setLevel("WARN")

    synthetic_feedback_df = create_synthetic_feedback_df(
        users=users,
        accounts=accounts,
        transactions=transactions,
        category_rules=category_rules,
        list_of_clients=list_of_clients,
    )

    logger.warning("Done")

    # Restore logging level
    logger.setLevel(orig_level)

    return synthetic_feedback_df


def preprocess(
    spark: SparkSession,
    project_config: YDSProjectConfig,
):
    """processing data for categories model"""
    execution_date = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    project_config_as_dict = YDSProjectConfigSchema.instance_as_dict(project_config)
    logger.info(f"Preprocessing project config: \n{project_config_as_dict}")

    script_config = project_config.preprocessing.script_config
    domain_config = get_domain_config(project_config.model_name)
    domain_config_as_dict = vars(domain_config)

    # store configuration in metadata
    project_config_as_dict = YDSProjectConfigSchema.instance_as_dict(project_config)
    preprocessing_metadata = {
        "execution_date": execution_date,
        "config": domain_config_as_dict,
        "project_config": project_config_as_dict,
    }

    start_datetime = dt.datetime.now()
    # data samples size configuration
    # to get a recent production data sample for monitoring, take last 30 days as default
    default_sample_start_date = (start_datetime - dt.timedelta(days=30)).strftime(
        "%Y-%m-%d"
    )
    sample_start_date = script_config.get(
        "sample_start_date", default_sample_start_date
    )
    sample_end_date = script_config.get(
        "sample_end_date", start_datetime.strftime("%Y-%m-%d")
    )

    n_model_samples_per_country = int(
        script_config.get(
            "n_model_samples_per_country", domain_config.N_MODEL_SAMPLES_PER_COUNTRY
        )
    )
    n_production_samples = int(
        script_config.get("n_production_samples", domain_config.N_PRODUCTION_SAMPLES)
    )

    logger.info(f"Sample from {sample_start_date} to {sample_end_date}")
    logger.info(
        f"Using {n_model_samples_per_country} samples per country and {n_production_samples} production samples"
    )

    logger.info("Loading data: \n")
    data_file_paths = script_config.get("data_file_paths")
    domain_table_dict = {
        domain: {
            key: read_data_and_select_columns(
                table=table,
                spark=spark,
                domain_config=domain_config,
                data_file_paths=data_file_paths,
            )
            for key, table in domain_config.TABLE_DOMAIN_MAPPING[domain].items()
        }
        for domain in domain_config.LIST_OF_DOMAINS
    }

    feedback_sources_dict = {
        domain: {
            table: read_data_and_select_columns(
                table=table,
                spark=spark,
                domain_config=domain_config,
                data_file_paths=data_file_paths,
            )
            for table in domain_config.FEEDBACK_TABLES[domain]
        }
        for domain in domain_config.LIST_OF_DOMAINS
    }

    # filter transactions for training after defined training date
    start_training_date = domain_config.START_TRAINING_DATE
    training_date_range = f.col("date") >= f.to_date(f.lit(start_training_date))

    # create training data
    start_time = time.time()

    # store datasets in parquet files
    docker_output_path = Path(
        script_config.get("docker_output_path", "/opt/ml/processing/output")
    )

    for domain in domain_config.LIST_OF_DOMAINS:
        logger.info(f"Filtering transactions for {domain}")

        tables_dict = domain_table_dict[domain]
        tables_dict["transactions"] = tables_dict["transactions"].where(
            training_date_range
        )

        # Leave out inspection set if specified
        if "inspection_set" in script_config:
            inspection_set_path = script_config.get("inspection_set")
            logging.warning(f"Leaving out inspection set {inspection_set_path}")
            # read inspected samples that should be excluded from training
            df_excluded_trx = read_data(
                file_path=inspection_set_path,
                spark=spark,
            )
            count_before = tables_dict["transactions"].count()
            tables_dict["transactions"] = tables_dict["transactions"].join(
                df_excluded_trx,
                on=["user_id", "account_id", "transaction_id"],
                how="leftanti",
            )
            count_after = tables_dict["transactions"].count()
            logging.warning(
                f"{count_before-count_after} transactions have been left out from {domain} as these are in the inspection set"
            )

    if domain_config.MODEL_TYPE == ModelType.SME_CATEGORIES_MODEL:
        synthetic_feedback_path = Path(
            script_config.get("docker_output_path")
        ) / script_config.get("synthetic_feedback_prefix")

        logger.info("Starting training data generation")
        for domain in domain_config.LIST_OF_DOMAINS:
            logger.info(f"Generating training data for {domain}")

            tables_dict = domain_table_dict[domain]

            test_users = tables_dict["test_users"]
            users = tables_dict["users"]
            accounts = tables_dict["accounts"]
            transactions = tables_dict["transactions"]

            # Exclude any test users
            if test_users:
                users = users.join(test_users, "user_id", "leftanti")

            # Filter for SME in yoltapp prd
            if domain == "yoltapp" and project_config.env == YDSEnvironment.PRD:
                users, accounts, transactions = filter_yoltapp_sme_transactions(
                    users=users, accounts=accounts, transactions=transactions
                )

            synthetic_feedback_df = generate_training_data(
                users=users,
                accounts=accounts,
                transactions=transactions,
                category_rules=CATEGORY_RULES.get(domain),
                list_of_clients=domain_config.LIST_OF_CLIENTS.get(domain),
            ).repartition(1)
            synthetic_feedback_df.cache().count()

            destination = f"file://{synthetic_feedback_path}_{domain}"
            synthetic_feedback_df.write.format("parquet").mode("overwrite").save(
                destination
            )
            feedback_sources_dict[domain][
                f"synthetic_feedback_{domain}"
            ] = synthetic_feedback_df

        logger.info("Training data generation completed")

    # add tables under "historical" feedback path to the metadata
    if project_config.model_bucket != "local":
        preprocessing_metadata["feedback_paths_subfolders"] = {
            table: list_folders_on_s3(path=data_file_paths[table])
            for domain in domain_config.LIST_OF_DOMAINS
            for table in domain_config.FEEDBACK_TABLES[domain]
        }

    training_df_dict = {
        domain: create_training_data_frame(
            **domain_table_dict[domain],
            feedback_sources=feedback_sources_dict[domain],
            n_model_samples_per_country=n_model_samples_per_country,
            n_production_samples=n_production_samples,
            sample_start_date=sample_start_date,
            sample_end_date=sample_end_date,
            domain_config=domain_config,
            domain=domain,
        )
        for domain in domain_config.LIST_OF_DOMAINS
    }

    # Merge domains in one dataframe:
    training_df = pd.concat(
        (
            training_domain_df.assign(domain=domain)
            for domain, (training_domain_df, _) in training_df_dict.items()
        )
    )
    production_sample_df = pd.concat(
        (
            production_sample_domain_df.assign(domain=domain)
            for domain, (_, production_sample_domain_df) in training_df_dict.items()
        )
    )

    preprocessing_time = time.time() - start_time
    preprocessing_metadata["preprocessing_time"] = time.strftime(
        "%H:%M:%S", time.gmtime(preprocessing_time)
    )

    for domain, (train_df, sample_df) in training_df_dict.items():
        preprocessing_metadata[
            f"training_data_size_per_country_{domain}"
        ] = training_data_size = (
            train_df["country_code"].value_counts().sort_index().to_dict()
        )
        preprocessing_metadata[
            f"production_data_size_per_country_{domain}"
        ] = production_data_size = (
            sample_df["country_code"].value_counts().sort_index().to_dict()
        )

        logger.info(
            f"Training data created; Number of transactions used per country ({domain}): {training_data_size}"
        )
        logger.info(
            f"Production data created; Number of transactions used per country ({domain}): {production_data_size}"
        )

    # serialize training log
    preprocessing_metadata_yml = yaml.dump(preprocessing_metadata)
    logger.info(f"Preprocessing metadata: \n{preprocessing_metadata_yml}")

    upload_parquet(
        df=training_df,
        path=Path(docker_output_path),
        file_name=domain_config.TRAINING_DATA_FILE,
    )

    upload_parquet(
        df=production_sample_df,
        path=Path(docker_output_path),
        file_name=domain_config.PRODUCTION_DATA_FILE,
    )

    # store preprocessing metadata
    upload_metadata(
        metadata=preprocessing_metadata,
        path=Path(docker_output_path),
        file_name=domain_config.PREPROCESSING_METADATA_FILE,
    )


if __name__ == "__main__":
    logger.error("STARTING JOB")
    parser = argparse.ArgumentParser()
    # Positional args that are provided when starting the job
    parser.add_argument("env", type=str)
    parser.add_argument("yds_config_path", type=str)
    args, _ = parser.parse_known_args()
    project_config = load_config_while_in_job(Path(args.yds_config_path))

    app_name = f"{project_config.model_name}_preprocessing"
    spark = get_spark_session(app_name, log_level="WARN")
    logger.info(f"Spark settings: {spark.sparkContext.getConf().toDebugString()}")
    logger.info("getExecutorMemoryStatus (spark.memory.offHeap.size)")
    logger.info(spark.sparkContext._jsc.sc().getExecutorMemoryStatus())
    logger.info("java heap (spark.driver.memory)")
    r = spark.sparkContext._jvm.java.lang.Runtime.getRuntime()
    logger.info(f"total: {r.totalMemory()}")
    logger.info(f"max: {r.maxMemory()}")
    logger.info(f"free: {r.freeMemory()}")

    preprocess(
        spark=spark,
        project_config=project_config,
    )
