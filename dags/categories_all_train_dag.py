# flake8: noqa
import os
from airflow.decorators import dag, task
from airflow.models.variable import Variable
from airflow.operators.dummy import DummyOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from datetime import datetime

default_args = {"provide_context": True, "start_date": datetime(2021, 7, 1)}
# Because otherwise we're "blackholing" traffic on PRD and we get
# connection timeouts
if os.environ["ENV"] == "management-prd":
    nexus_host = "nexus.yolt.io:443"
    nexus_address = f"https://{nexus_host}"
    extra_flags = ["--trusted-host", nexus_host]
else:
    nexus_address = "https://nexus.yolt.io:443"
    extra_flags = []

virtualenv_requirements = [
    "--extra-index-url",
    f"{nexus_address}/repository/pypi-hosted/simple",
    *extra_flags,
    "datascience_model_commons==0.3.11.3",
]

categories_model_config = "./dags/categories_model_yds.yaml"
sme_categories_model_config = "./dags/sme_categories_model_yds.yaml"


@dag(
    default_args=default_args,
    schedule_interval="0 12 * * 0",  # run every Sunday at 12:00 UTC
    tags=["datascience"],
    catchup=False,
)
def categories_all_train():
    def generate_preprocessing(config_file_location: str, task_id: str):
        @task.virtualenv(
            task_id=task_id,
            use_dill=True,
            system_site_packages=True,
            requirements=virtualenv_requirements,
        )
        def preprocessing(_config_file_location):
            # All imports being used within this function scope should be done
            # inside this function. Everything in this scope will run in a
            # separate virtualenv isolated from this DAG file.
            from datascience_model_commons.airflow import (
                airflow_run_spark_preprocessing,
            )

            airflow_run_spark_preprocessing(_config_file_location)

        return preprocessing(config_file_location)

    def generate_training(config_file_location: str, task_id: str):
        @task.virtualenv(
            task_id=task_id,
            use_dill=True,
            system_site_packages=True,
            requirements=virtualenv_requirements,
            multiple_outputs=True,  # because we're returning a Dict[str, str]
        )
        def training(_config_file_location):
            # All imports being used within this function scope should be done
            # inside this function. Everything in this scope will run in a
            # separate virtualenv isolated from this DAG file.
            from datetime import datetime
            from datascience_model_commons.airflow import (
                airflow_run_tensorflow_training_job,
            )

            training_start = datetime.now()
            estimator = airflow_run_tensorflow_training_job(_config_file_location)

            # This is the S3 path to the trained model
            return {
                "model_artifact_uri": estimator.model_data,
                "training_run_start": training_start.strftime("%Y-%m-%d-%H-%M"),
            }

        return training(config_file_location)

    def generate_copy_trained_model(
        _trained_model_details: dict, config_file_location: str, task_id: str
    ):
        @task.virtualenv(
            task_id=task_id,
            use_dill=True,
            system_site_packages=True,
            requirements=virtualenv_requirements,
        )
        def copy_trained_model(_trained_model_details, _config_file_location=None):
            # All imports being used within this function scope should be done
            # inside this function. Everything in this scope will run in a
            # separate virtualenv isolated from this DAG file.
            from datascience_model_commons.deploy.config.load import (
                load_config_while_in_job,
            )

            from datascience_model_commons.airflow import invoke_copy_lambda
            from pathlib import Path
            import logging

            logging.info(
                f"Going to copy trained model based on details: {_trained_model_details}"
            )
            project_config = load_config_while_in_job(Path(_config_file_location))

            # This is a full S3 uri like s3://bucket/prefix/model.tar.gz
            # so we need to split
            model_artifact_uri = (
                _trained_model_details["model_artifact_uri"]
                .replace("s3://", "")
                .split("/")
            )
            destination_bucket = f"yolt-dp-{project_config.env.value}-exchange-yoltapp"
            destination_prefix = f"artifacts/datascience/{project_config.model_name}/{project_config.git_branch}/{_trained_model_details['training_run_start']}"  # noqa
            destination_filename = model_artifact_uri[-1]
            invoke_copy_lambda(
                source_bucket=model_artifact_uri[0],
                source_key="/".join(model_artifact_uri[1:]),
                dst_bucket=destination_bucket,
                # This is formatted this way because of backwards compatibility.
                # Ideally, we would indicate the model artifact via a {branch, deploy_id, training_start}
                # identifier.
                dst_prefix=destination_prefix,  # noqa
                new_key=destination_filename,
            )

            return (
                f"s3://{destination_bucket}/{destination_prefix}/{destination_filename}"
            )

        return copy_trained_model(_trained_model_details, config_file_location)

    @task.virtualenv(
        use_dill=True,
        system_site_packages=True,
        requirements=virtualenv_requirements,
    )
    def send_success_notification():
        from datascience_model_commons.airflow import (
            send_dag_finished_to_slack_mle_team,
        )

        send_dag_finished_to_slack_mle_team()

    categories_preprocessing = generate_preprocessing(
        categories_model_config, "categories_preprocessing"
    )
    categories_training = generate_training(
        categories_model_config, "categories_training"
    )
    categories_preprocessing >> categories_training
    categories_copy_trained = generate_copy_trained_model(
        categories_training, categories_model_config, "categories_copy"
    )

    sme_categories_preprocessing = generate_preprocessing(
        sme_categories_model_config, "sme_categories_preprocessing"
    )
    sme_categories_training = generate_training(
        sme_categories_model_config, "sme_categories_training"
    )
    sme_categories_preprocessing >> sme_categories_training
    sme_categories_copy_trained = generate_copy_trained_model(
        sme_categories_training, sme_categories_model_config, "sme_categories_copy"
    )

    env = os.environ["ENV"]
    task_name = "trigger_build_all_categories_serving"
    if env == "management-dta":
        (
            [
                categories_copy_trained,
                sme_categories_copy_trained,
            ]
            >> DummyOperator(task_id=task_name)
            >> send_success_notification()
        )
    elif env == "management-prd":
        gitlab_token = Variable.get("gitlab-categories")
        payload = {
            "token": gitlab_token,
            "ref": "master",
            "variables[CATEGORIES_MODEL_URI]": categories_copy_trained,
            "variables[SME_CATEGORIES_MODEL_URI]": sme_categories_copy_trained,
        }

        (
            SimpleHttpOperator(
                task_id=task_name,
                http_conn_id="gitlab",
                endpoint="api/v4/projects/555/trigger/pipeline",
                method="POST",
                data=payload,
                log_response=True,
                retries=25,
            )
            >> send_success_notification()
        )


categories_all_train_dag = categories_all_train()
