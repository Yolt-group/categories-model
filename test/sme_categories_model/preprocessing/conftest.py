from typing import Dict, AnyStr

import pytest
from pyspark.sql import SparkSession
import pyspark.sql as pysql
from categories_model.preprocessing.create_training_data import (
    clean_column,
    FEEDBACK_COLUMNS,
)
import pandas as pd
from datascience_model_commons.deploy.config.domain import (
    YDSDomain,
    YDSPreprocessingConfig,
)
from datascience_model_commons.spark import get_spark_session

from categories_model.preprocessing.create_training_data import (
    create_synthetic_feedback_df,
)
from categories_model.preprocessing.data import (
    read_data_and_select_columns,
    create_training_data_frame,
)
from categories_model.preprocessing.rules import CATEGORY_RULES


@pytest.fixture(scope="session")
def spark_session(request) -> SparkSession:
    spark = get_spark_session("categories_test")
    request.addfinalizer(lambda: spark.stop())

    return spark


@pytest.fixture(scope="session")
def training_data_test_rule():
    category_rule = {
        "transaction_type": "debit",
        "description": {
            "+": ["etentje"],
            "-": ["weglaten"],
        },
        "counterparty": {
            "+": ["zakelijke.*spaarrekening"],
            "-": [],
        },
    }
    return category_rule


@pytest.fixture(scope="session")
def cleaned_df(spark_session, test_df) -> pysql.DataFrame:
    test_df = clean_column(test_df, "description")
    test_df = clean_column(test_df, "bank_counterparty_name")
    return test_df


@pytest.fixture(scope="session")
def test_df(spark_session) -> pysql.DataFrame:
    df = pd.DataFrame(
        [
            {
                "description": "storting privé",
                "bank_counterparty_name": "",
                "transaction_type": "debit",
                "expected": "storting prive",
                "should_match": False,
            },
            {
                "description": "etentje",
                "bank_counterparty_name": "",
                "transaction_type": "credit",
                "expected": "etentje",
                "should_match": False,
            },
            {
                "description": "234235sdf",
                "bank_counterparty_name": "",
                "transaction_type": "debit",
                "expected": "digits sdf",
                "should_match": False,
            },
            {
                "description": "xxx-555-zzz",
                "bank_counterparty_name": "zakelijke oranje spaarrekening",
                "transaction_type": "debit",
                "expected": "repeating chars digits repeating chars",
                "should_match": True,
            },
            {
                "description": "weglaten",
                "bank_counterparty_name": "zakelijke oranje spaarrekening",
                "transaction_type": "debit",
                "expected": "weglaten",
                "should_match": False,
            },
        ]
    )
    df[FEEDBACK_COLUMNS] = "none"
    spark_session.conf.set("spark.sql.execution.arrow.enabled", "true")
    return spark_session.createDataFrame(df)


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
            "transactions_app",
            "accounts_app",
            "users_app",
            "test_users_app",
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

    # tag training data
    feedback_sources["synthetic_feedback"] = create_synthetic_feedback_df(
        users=users,
        accounts=accounts,
        transactions=transactions,
        category_rules=CATEGORY_RULES["yoltapp"],
        list_of_clients=domain_config.LIST_OF_CLIENTS["yoltapp"],
    )

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
        domain=YDSDomain.YoltApp.value,
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
            "users_app",
            "test_users_app",
        ]
    )
    return users, test_users


@pytest.fixture(scope="module")
def data_file_paths(script_config) -> AnyStr:
    return script_config.get("data_file_paths")


@pytest.fixture(scope="module")
def docker_output_path(script_config) -> AnyStr:
    return script_config.get("docker_output_path")


@pytest.fixture(scope="module")
def script_config(preprocessing_config) -> Dict[AnyStr, AnyStr]:
    return preprocessing_config.script_config


@pytest.fixture(scope="module")
def preprocessing_config(project_config) -> YDSPreprocessingConfig:
    return project_config.preprocessing
