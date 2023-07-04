import functools
import unicodedata
from typing import Dict, AnyStr, List, Optional, Tuple

from pyspark.sql import functions as f
from pyspark import sql as pysql
import numpy as np
from pyspark.sql.functions import udf
from datetime import datetime

FEEDBACK_COLUMNS = ["user_id", "account_id", "transaction_id", "date", "pending"]


def create_base_tagging_df(
    *,
    users: pysql.DataFrame,
    accounts: pysql.DataFrame,
    transactions: pysql.DataFrame,
    list_of_clients: Optional[List[AnyStr]],
) -> pysql.DataFrame:
    users_from_selected_clients = users.where(
        f.col("client_id").isin(list_of_clients)
    ).select(["user_id", "country_code", "client_id"])
    non_deleted_accounts = accounts.where(f.col("deleted").isNull()).select(
        ["user_id", "account_id"]
    )

    transaction_columns = [
        "user_id",
        "account_id",
        "transaction_id",
        "date",
        "amount",
        "pending",
        "transaction_type",
        "bank_counterparty_name",
        "description",
    ]

    users_accounts = users_from_selected_clients.join(
        non_deleted_accounts, on="user_id", how="inner"
    )

    return (
        transactions.select(transaction_columns)
        .join(
            users_accounts.hint("broadcast"),
            on=["user_id", "account_id"],
            how="inner",
        )
        .withColumn("year_month", f.col("date"))
    )


remove_diacritics = udf(
    lambda s: (
        ""
        if s is None
        else "".join(
            (
                c
                for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn"
            )
        )
    )
)


def clean_column(df: pysql.DataFrame, column_name: AnyStr) -> pysql.DataFrame:
    column = f.col(column_name)
    return (
        df.withColumn(column_name, remove_diacritics(column))
        .withColumn(column_name, f.lower(column))
        .withColumn(column_name, f.regexp_replace(column, r"([0-9])+", " digits "))
        .withColumn(column_name, f.regexp_replace(column, r"[^a-z ]", " "))
        .withColumn(
            column_name,
            f.regexp_replace(column, r"\b([a-z])\1{2,}\b", " repeating chars "),
        )
        .withColumn(column_name, f.regexp_replace(column, r"\ +", " "))
        .withColumn(column_name, f.trim(column))
    )


def match_column(*, column_name: str, word_list: list) -> pysql.Column:
    no_match = f.lit(False)
    word_list_regex = "|".join(word_list)
    return f.col(column_name).rlike(word_list_regex) if len(word_list) > 0 else no_match


def match_category_rule(
    df: pysql.DataFrame, category_rule: Dict[AnyStr, Dict]
) -> pysql.DataFrame:
    match_list = category_rule["description"]["+"]
    exclude_list = category_rule["description"]["-"]
    match_counterparty = category_rule["counterparty"]["+"]
    exclude_counterparty = category_rule["counterparty"]["-"]
    transaction_type = category_rule["transaction_type"]
    iai = f.lit(False)
    if "amount_based" in category_rule.keys():
        for amount_rule in category_rule["amount_based"]:
            amount_min_threshold = amount_rule.get("min", -np.Infinity)
            amount_max_threshold = amount_rule.get("max", np.Infinity)
            amount_description_match_list = amount_rule["description"]
            amount_counterparty_match_list = amount_rule["counterparty"]
            iam = (f.col("amount") >= f.lit(amount_min_threshold)) & (
                f.col("amount") <= f.lit(amount_max_threshold)
            )
            idai = match_column(
                column_name="description", word_list=amount_description_match_list
            )
            icai = match_column(
                column_name="bank_counterparty_name",
                word_list=amount_counterparty_match_list,
            )
            iai |= iam & (idai | icai)

    # Create boolean columns to filter matched data
    idi = match_column(column_name="description", word_list=match_list)
    ide = match_column(column_name="description", word_list=exclude_list)

    # Create boolean columns to filter matched data
    ici = match_column(
        column_name="bank_counterparty_name", word_list=match_counterparty
    )
    ice = match_column(
        column_name="bank_counterparty_name", word_list=exclude_counterparty
    )

    # Create rule for checking transaction_type
    trx_type = f.col("transaction_type") == f.lit(transaction_type)
    result_df = df.where(trx_type & (idi | ici | iai) & ~(ide | ice)).select(
        FEEDBACK_COLUMNS
    )
    return result_df


def filter_yoltapp_sme_transactions(
    users: pysql.DataFrame, accounts: pysql.DataFrame, transactions: pysql.DataFrame
) -> Tuple[pysql.DataFrame, pysql.DataFrame, pysql.DataFrame]:
    filtered_transactions = (
        transactions.join(
            users.where(f.col("country_code") == "GB")
            .select("user_id")
            .hint("broadcast"),
            "user_id",
            "inner",
        )
        .where(f.col("category") != "Internal")
        .withColumn("description", f.lower(f.col("description")))
    )
    filtered_accounts = (
        filtered_transactions.where(f.col("transaction_type") == "debit")
        .where(
            f.col("description").rlike(r"hmrc.*\bvat\b|\bxero\b|quickbooks|zoho.?books")
        )
        .where(f.col("amount") > 30)
        .select("account_id")
        .distinct()
    )

    transactions = filtered_transactions.join(
        filtered_accounts.hint("broadcast"), "account_id", "inner"
    )
    accounts = accounts.join(filtered_accounts, "account_id", "inner")
    users = users.join(accounts.select("user_id").distinct(), "user_id", "inner")

    return users, accounts, transactions


def create_synthetic_feedback_df(
    *,
    users: pysql.DataFrame,
    accounts: pysql.DataFrame,
    transactions: pysql.DataFrame,
    category_rules: Dict[AnyStr, Dict],
    list_of_clients: Optional[List[AnyStr]],
    **kwargs,
) -> pysql.DataFrame:
    base_df = create_base_tagging_df(
        users=users,
        accounts=accounts,
        transactions=transactions,
        list_of_clients=list_of_clients,
    )
    # clean columns
    base_df = clean_column(base_df, "bank_counterparty_name")
    base_df = clean_column(base_df, "description")
    base_df.cache()

    # append all tagged transactions
    result_df = functools.reduce(
        lambda a, b: a.unionAll(b),
        [
            match_category_rule(base_df, category_rules)
            .withColumn("category", f.lit(category))
            .withColumn(
                "feedback_time", f.lit(datetime.now().strftime("%Y-%m-%d %H:%M"))
            )
            for category, category_rules in category_rules.items()
        ],
    )

    return result_df
