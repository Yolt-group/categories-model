import boto3
import datetime
import functools
import logging
import numpy as np
import pandas as pd
import pyspark
import pyspark.sql.functions as f
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import Window
from typing import Tuple, AnyStr, Union, Dict
from urllib.parse import urlparse

from categories_model.config.domain import DomainConfig
from datascience_model_commons.spark import read_data


def list_folders_on_s3(*, path: AnyStr) -> list:
    """
    Function that creates a list with folders under given path on s3

    :param path: s3a path where to list folders
    :return: a list of folders in given path on s3
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects")

    # split path into bucket and prefix
    # - since s3 paths are URIs, we can use urlparse to extract bucket and prefix
    #    urlparse transform path into: <scheme>://<netloc><path>
    url_path = urlparse(path)
    bucket, prefix = (url_path.netloc, url_path.path)

    # extract prefix parent since the path is directly to '*' or '*.csv'
    # - note that the prefix includes slash at the beginning therefore [1:] is used
    # - note that the prefix directory has to end up with slash
    prefix_directory = f"{Path(prefix[1:]).parent}/"

    # append folders to the list
    folders_list = []
    for result in paginator.paginate(
        Bucket=bucket, Prefix=prefix_directory, Delimiter="/"
    ):
        for prefix in result.get("CommonPrefixes", []):
            full_prefix_path = Path(prefix.get("Prefix"))
            folder_name = full_prefix_path.name
            folders_list.append(folder_name)

    return folders_list


def read_data_and_select_columns(
    table: AnyStr,
    spark: SparkSession,
    domain_config: DomainConfig,
    data_file_paths: Dict[AnyStr, AnyStr],
) -> Union[pyspark.sql.DataFrame, None]:
    """
    Function that reads data and selects the relevant columns

    :param table: name of the table that should be read
    :param spark: spark session
    :param domain_config: categories_model configuration
    :param data_file_paths: dictionary with data paths
    :return: pyspark table with relevant columns
    """
    # when the table path is not specified return None type
    if table not in data_file_paths:
        return None

    # extract table path from categories configuration
    file_path = data_file_paths[table]

    # extract columns & aliases it exists from selected table
    columns = domain_config.TABLE_COLUMNS[table]["columns"]
    aliases = domain_config.TABLE_COLUMNS[table].get("aliases", False)

    # read data
    df = read_data(file_path=file_path, spark=spark).select(columns)

    # if columns need to be renamed, do so
    if aliases:
        for column_name, alias in aliases.items():
            df = df.withColumnRenamed(column_name, alias)

    logging.info(f"{table}: {file_path}")

    return df


def create_training_data_frame(
    *,
    transactions: pyspark.sql.DataFrame,
    accounts: pyspark.sql.DataFrame,
    users: pyspark.sql.DataFrame,
    test_users: pyspark.sql.DataFrame,
    feedback_sources: dict,
    n_model_samples_per_country: int,
    n_production_samples: int,
    sample_start_date: datetime,
    sample_end_date: datetime,
    domain_config: DomainConfig,
    domain: AnyStr,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a data frame for model training

    :param transactions: PySpark DataFrame referring to the raw transactions table
    :param accounts: PySpark DataFrame referring to the raw account table
    :param users: PySpark DataFrame referring to the raw users table
    :param test_users: PySpark DataFrame referring to the interim table including all test users
    :param feedback_sources: Dictionary consisting of PySpark DataFrames of different feedback sources
    :param n_model_samples_per_country: number of samples per country included in final filtered data frame
    :param n_production_samples: number of samples included in production sample data frame
    :param sample_start_date: the first date of the transactions in the production sample in a Tuple
    :param sample_end_date: the last date of the transactions in the production sample in a Tuple
    :return: tuple containing filtered Pandas DataFrame and sample mimicking production data
    """
    # extract users sample
    df_users_sample = extract_users_sample(
        users=users,
        test_users=test_users,
        domain_config=domain_config,
        domain=domain,
    )

    # extract relevant accounts
    df_relevant_accounts = extract_relevant_accounts(
        accounts=accounts,
        domain_config=domain_config,
    )

    # extract target category for recent feedback date
    df_last_feedback = extract_recent_feedback_category(
        feedback_sources=feedback_sources,
        domain_config=domain_config,
    )

    # extract final users & accounts sample
    #   use broadcast joins since we merge large table - feedback union - with users and accounts
    #   broadcast send a copy of a table to all the executor nodes
    df_base = df_last_feedback.join(df_users_sample, on="user_id", how="inner").join(
        df_relevant_accounts, on=["user_id", "account_id"], how="inner"
    )

    # join base table with transactions in order to extract features for training the model
    unique_transaction_identifier = [
        "user_id",
        "account_id",
        "transaction_id",
        "date",
        "pending",
    ]

    df = df_base.join(
        transactions, on=unique_transaction_identifier, how="inner"
    ).cache()

    # extract production sample
    df_production_sample = extract_production_sample(
        df_base=df,
        sample_start_date=sample_start_date,
        sample_end_date=sample_end_date,
        n_production_samples=n_production_samples,
        domain_config=domain_config,
    )

    # extract final training sample
    df_training_sample = extract_training_sample(
        df_base=df,
        start_training_date=domain_config.START_TRAINING_DATE,
        n_model_samples_per_country=n_model_samples_per_country,
        domain_config=domain_config,
    )

    return df_training_sample, df_production_sample


def extract_users_sample(
    *,
    users: pyspark.sql.DataFrame,
    test_users: pyspark.sql.DataFrame,
    domain_config: DomainConfig,
    domain: AnyStr,
) -> pyspark.sql.DataFrame:
    """
    Extract users sample based on predefined countries and excluding test users

    :param users: PySpark DataFrame referring to the raw users table
    :param test_users: PySpark DataFrame referring to the interim table including all test users
    :param domain_config: Config file that contains model-env specific variables
    :param domain: Domain of the model, yts or yoltapp
    :return: final users sample
    """
    # define rule for including country specific users
    # only for NL we want to include test users in our sample
    countries_ex_nl = list(set(domain_config.COUNTRIES) - {"NL"})

    country_specific_users = (
        f.col("country_code").isin(countries_ex_nl) & (f.col("test_user") == 0)
    ) | (f.col("country_code") == "NL")

    selected_clients = f.col("client_id").isin(domain_config.LIST_OF_CLIENTS[domain])

    df_users_sample = (
        (
            # exclude test users
            users.join(
                test_users.withColumn("test_user", f.lit(1)),
                on="user_id",
                how="left",
            ).fillna(0, subset="test_user")
            if test_users is not None
            else users.withColumn("test_user", f.lit(0))
        )
        .where(country_specific_users)
        .where(selected_clients)
    )

    return df_users_sample


def extract_relevant_accounts(
    *,
    accounts: pyspark.sql.DataFrame,
    domain_config: DomainConfig,
) -> pyspark.sql.DataFrame:
    """
    Extract accounts sample excluding deleted accounts and test accounts

    :param accounts: PySpark DataFrame referring to the raw account table
    :param domain_config: Config file that contains model-env specific variables
    :return: final accounts sample
    """
    # define rule for including non deleted accounts
    non_deleted_accounts = f.col("deleted").isNull()
    test_accounts = f.col("site_id") == domain_config.TEST_SITE_ID

    df_relevant_accounts = accounts.where(non_deleted_accounts).where(~test_accounts)

    return df_relevant_accounts


def extract_recent_feedback_category(
    *,
    feedback_sources: dict,
    domain_config: DomainConfig,
) -> pyspark.sql.DataFrame:
    """
    Extract target category for recent feedback date

    :param feedback_sources: dictionary consisting of PySpark DataFrames referring to different feedback sources
    :param domain_config: Config file that contains model-env specific variables
    :return: final table with the most recent feedback category assigned to the transaction
    """
    # define feedback columns so that we make sure that the order of the columns in each table is the same
    feedback_columns = [
        "user_id",
        "account_id",
        "transaction_id",
        "date",
        "pending",
        "feedback_time",
        "category",
    ]

    # append all feedback tables
    feedback_combined = functools.reduce(
        lambda a, b: a.unionAll(b),
        [table.select(feedback_columns) for table in feedback_sources.values()],
    ).dropDuplicates()

    # extract the events where the time format is not corrupted - can be transformed to timestamp
    correct_feedback_time_format = f.col("feedback_time").isNotNull()

    feedback_combined = feedback_combined.withColumn(
        "feedback_time", f.to_timestamp(f.col("feedback_time"))
    ).where(correct_feedback_time_format)

    # select last feedback time for given transaction
    unique_transaction_identifier = [
        "user_id",
        "account_id",
        "transaction_id",
        "date",
        "pending",
    ]

    window_over_transaction = Window.partitionBy(unique_transaction_identifier).orderBy(
        f.desc("feedback_time")
    )
    most_recent_feedback = f.col("row_nr") == 1

    df_last_feedback = (
        feedback_combined.withColumn(
            "row_nr", f.row_number().over(window_over_transaction)
        )
        .where(most_recent_feedback)
        .withColumnRenamed("category", domain_config.TARGET_COLUMN)
    )

    return df_last_feedback


def extract_production_sample(
    *,
    df_base: pyspark.sql.DataFrame,
    sample_start_date: datetime,
    sample_end_date: datetime,
    n_production_samples: int,
    domain_config: DomainConfig,
) -> pyspark.sql.DataFrame:
    """
    Extract a sample of data to mimic future production data

    :param df_base: PySpark DataFrame referring to final transaction base used in training
    :param sample_start_date: the first date of the transactions in the production sample in a Tuple
    :param sample_end_date: the last date of the transactions in the production sample in a Tuple
    :param n_production_samples: number of samples included in production sample data frame
    :param domain_config: Config file that contains model-env specific variables
    :return: production data sample
    """
    # define final column set
    relevant_columns = (
        domain_config.PREPROCESSING_COLUMNS
        + [domain_config.TARGET_COLUMN]
        + ["country_code"]
    )

    # define production date range
    production_date_range = (f.col("date") >= sample_start_date) & (
        f.col("date") <= sample_end_date
    )

    df_production = df_base.where(production_date_range).select(relevant_columns)
    df_production_sample = sample_pyspark_df_to_pandas_df(
        df=df_production, n_samples=n_production_samples
    )

    # add target column transformed to integer
    df_production_sample = map_category_to_int(
        df=df_production_sample,
        domain_config=domain_config,
    )

    return df_production_sample


def extract_training_sample(
    *,
    df_base: pyspark.sql.DataFrame,
    start_training_date: datetime,
    n_model_samples_per_country: int,
    domain_config: DomainConfig,
) -> pyspark.sql.DataFrame:
    """
    Extract a sample of data used for model training

    :param df_base: PySpark DataFrame referring to final transaction base used in training
    :param start_training_date: the first date of transaction in training sample
    :param n_model_samples_per_country: number of samples included in training sample per country
    :param domain_config: Config file that contains model-env specific variables
    :return: production data sample
    """
    # define final column set
    relevant_columns = (
        domain_config.PREPROCESSING_COLUMNS
        + [domain_config.TARGET_COLUMN]
        + ["country_code"]
    )

    # filter transactions for training after defined training date
    training_date_range = f.col("date") >= f.to_date(f.lit(start_training_date))

    df_filtered = df_base.where(training_date_range).select(relevant_columns).cache()

    # sample data per country
    df_per_country = {}

    for country in domain_config.COUNTRIES:
        country_specific_tx = f.col("country_code") == country
        df_per_country[country] = sample_pyspark_df_to_pandas_df(
            df=df_filtered.where(country_specific_tx),
            n_samples=n_model_samples_per_country,
        )

    # union all training samples
    df_training_sample = pd.concat(df_per_country.values())

    # add target column transformed to integer
    df_training_sample = map_category_to_int(
        df=df_training_sample, domain_config=domain_config
    )

    return df_training_sample


def sample_pyspark_df_to_pandas_df(
    *, df: pyspark.sql.DataFrame, n_samples: int
) -> pd.DataFrame:
    """
    Extract the sample of pyspark dataframe and save it as pandas df

    :param df: PySpark DataFrame
    :param n_samples: number of samples to extract
    :return: Data sample
    """

    df_rows = df.count()
    if df_rows > 0:
        # return sample
        df_sample_ratio = np.min(
            [n_samples / df_rows, 1.0]
        )  # make sure the ratio is not higher than 1 - it may happen that for some countries we have smaller sample
        df_sample = df.sample(
            withReplacement=False, fraction=df_sample_ratio, seed=123
        ).toPandas()
    else:
        # nothing to sample: return empty DataFrame
        df_sample = pd.DataFrame(columns=df.columns)

    return df_sample


def map_category_to_int(df: pd.DataFrame, domain_config: DomainConfig) -> pd.DataFrame:
    """
    Define mapping of training labels to integers

    :param df: pandas DataFrame referring to final training set
    :param domain_config: Config file that contains model-env specific variables
    :return: pandas DataFrame with additional column - map of labels to integers
    """
    # define mapping of categories to int
    category_to_int = dict(
        zip(domain_config.CATEGORIES, map(int, range(len(domain_config.CATEGORIES))))
    )
    df[domain_config.TARGET_COLUMN_INT] = df[domain_config.TARGET_COLUMN].map(
        category_to_int
    )

    return df
