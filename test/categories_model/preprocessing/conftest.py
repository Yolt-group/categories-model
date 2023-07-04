from pathlib import Path
from typing import AnyStr, Dict

import pytest
from datascience_model_commons.deploy.config.domain import (
    YDSPreprocessingConfig,
)
from datascience_model_commons.spark import get_spark_session
from pyspark.sql import SparkSession

from categories_model.preprocessing.data import (
    read_data_and_select_columns,
    create_training_data_frame,
)


@pytest.fixture(scope="session")
def spark_session(request) -> SparkSession:
    spark = get_spark_session("categories_test")
    request.addfinalizer(lambda: spark.stop())

    return spark


@pytest.fixture(scope="module")
def df(project_config, spark_session, domain_config, data_file_paths, script_config):
    (transactions, accounts, users, test_users,) = (
        read_data_and_select_columns(
            table=table,
            spark=spark_session,
            domain_config=domain_config,
            data_file_paths=data_file_paths,
        )
        for table in [
            "transactions",
            "accounts",
            "users",
            "test_users",
        ]
    )

    # create dictionary of used feedback source PySpark DataFrames
    feedback_sources = {
        table: read_data_and_select_columns(
            table=table,
            spark=spark_session,
            domain_config=domain_config,
            data_file_paths=data_file_paths,
        )
        for table in domain_config.FEEDBACK_TABLES["yoltapp"]
    }

    df, _ = create_training_data_frame(
        transactions=transactions,
        accounts=accounts,
        users=users,
        test_users=test_users,
        feedback_sources=feedback_sources,
        n_model_samples_per_country=script_config.get(
            "n_model_samples_per_country", domain_config.N_MODEL_SAMPLES_PER_COUNTRY
        ),
        n_production_samples=script_config.get(
            "n_production_samples", domain_config.N_PRODUCTION_SAMPLES
        ),
        sample_start_date=script_config.get("sample_start_date"),
        sample_end_date=script_config.get("sample_end_date"),
        domain_config=domain_config,
        domain=project_config.domain.value,
    )

    return df


@pytest.fixture(scope="module")
def user_tables(spark_session, domain_config, data_file_paths):
    (users, test_users,) = (
        read_data_and_select_columns(
            table=table,
            spark=spark_session,
            domain_config=domain_config,
            data_file_paths=data_file_paths,
        )
        for table in [
            "users",
            "test_users",
        ]
    )
    return users, test_users


@pytest.fixture(scope="module")
def data_file_paths(script_config) -> AnyStr:
    return script_config.get("data_file_paths")


@pytest.fixture(scope="module")
def docker_output_path(script_config) -> Path:
    return Path(script_config.get("docker_output_path"))


@pytest.fixture(scope="module")
def script_config(preprocessing_config) -> Dict[AnyStr, AnyStr]:
    return preprocessing_config.script_config


@pytest.fixture(scope="module")
def preprocessing_config(project_config) -> YDSPreprocessingConfig:
    return project_config.preprocessing
