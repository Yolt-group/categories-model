from typing import AnyStr, Dict

import pytest
import pandas as pd

from datascience_model_commons.deploy.config.domain import (
    YDSTrainingConfig,
)


@pytest.fixture(scope="module")
def tax_df(script_config, domain_config):
    df = (
        pd.DataFrame(
            columns=[
                "description",
                "amount",
                "transaction_type",
                "internal_transaction",
                "country_code",
                "counter_account_type",
                domain_config.TARGET_COLUMN,
                "bank_counterparty_name",
                "bank_counterparty_iban",
                "target_country_code",
                "counterparty_name",
            ]
        )
        .append(
            [
                {
                    "description": "hmrc corporation t 4106400364a00102a BBP",
                    "amount": 250.0,
                    "transaction_type": "debit",
                    "internal_transaction": "",
                    "country_code": "GB",
                    "target_country_code": "GB",
                    domain_config.TARGET_COLUMN: "Corporate Income Tax",
                }
            ],
            ignore_index=True,
        )
        .append(
            [
                {
                    "description": "hmrc corporation t 4106400364a00102a BBP",
                    "amount": 250.0,
                    "transaction_type": "debit",
                    "internal_transaction": "",
                    "bank_counterparty_iban": "GB92345874235987",
                    "target_country_code": "GB",
                    domain_config.TARGET_COLUMN: "Corporate Income Tax",
                }
            ],
            ignore_index=True,
        )
        .append(
            [
                {
                    "description": "dvla vehicle",
                    "amount": 250.0,
                    "transaction_type": "debit",
                    "internal_transaction": "",
                    "country_code": "GB",
                    "target_country_code": "GB",
                    domain_config.TARGET_COLUMN: "Unspecified Tax",
                }
            ],
            ignore_index=True,
        )
        .append(
            [
                {
                    "description": "dvla vehicle",
                    "amount": 250.0,
                    "transaction_type": "debit",
                    "internal_transaction": "",
                    "country_code": "",
                    "target_country_code": "",
                    domain_config.TARGET_COLUMN: "Food and Drinks",  # This sample should not fire UK tax rule has no CURRENT_ACCOUNT type
                }
            ],
            ignore_index=True,
        )
        .append(
            [
                {
                    "description": "dvla vehicle",
                    "amount": 250.0,
                    "transaction_type": "debit",
                    "internal_transaction": "",
                    "counter_account_type": "CURRENT_ACCOUNT",
                    "target_country_code": "GB",
                    domain_config.TARGET_COLUMN: "Unspecified Tax",
                    # This sample should fire UK tax rule as it has CURRENT_ACCOUNT type
                }
            ],
            ignore_index=True,
        )
        .append(
            [
                {
                    "description": "SEPA Overboeking, IBAN: NL86INGB0002445588, BIC: INGBNL2A, Naam: Belastingdienst Apeldoorn, Betalingskenm.: 8107471771101210",
                    "amount": 249.0,
                    "transaction_type": "debit",
                    "account_type": "CURRENT_ACCOUNT",
                    "country_code": "NL",
                    "bank_counterparty_name": "Belastingsdienst",
                    "bank_counterparty_iban": "NL86INGB0002445588",
                    "target_country_code": "NL",
                    domain_config.TARGET_COLUMN: "Sales Tax",
                }
            ],
            ignore_index=True,
        )
        .append(
            [
                {
                    "description": "TERUGGAAF NR. 851494523V760112 VPB.2017 (JACHTTRUST )",
                    "amount": 2772.0,
                    "transaction_type": "credit",
                    "account_type": "CURRENT_ACCOUNT",
                    "country_code": "NL",
                    "bank_counterparty_name": "Belastingsdienst",
                    "bank_counterparty_iban": "NL86INGB0002445588",
                    "target_country_code": "NL",
                    domain_config.TARGET_COLUMN: "Tax Returns",
                }
            ],
            ignore_index=True,
        )
        .append(
            [
                {
                    "description": "TERUGGAAF NR. 851494523V760112 VPB.2017 (JACHTTRUST )",
                    "amount": 2772.0,
                    "transaction_type": "credit",
                    "counter_account_type": "CURRENT_ACCOUNT",
                    "bank_counterparty_name": "Belastingsdienst",
                    "bank_counterparty_iban": "NL86INGB0002445588",
                    "target_country_code": "NL",
                    domain_config.TARGET_COLUMN: "Tax Returns",
                }
            ],
            ignore_index=True,
        )
        .append(
            [
                {
                    "description": domain_config.TEST_DESCRIPTION_PREFIX,
                    "amount": 2772.0,
                    "transaction_type": "credit",
                    "account_type": "CURRENT_ACCOUNT",
                    "country_code": "FR",
                    "bank_counterparty_name": "Other Income",
                    "bank_counterparty_iban": "NL86INGB0002445588",
                    "target_country_code": "FR",
                    domain_config.TARGET_COLUMN: "Other Income",
                }
            ],
            ignore_index=True,
        )
        .append(
            [
                {
                    "description": domain_config.TEST_DESCRIPTION_PREFIX,
                    "amount": 2772.0,
                    "transaction_type": "credit",
                    "account_type": "CURRENT_ACCOUNT",
                    "bank_counterparty_name": "Other Income",
                    "bank_counterparty_iban": "NL86INGB0002445588",
                    "target_country_code": "NL",
                    domain_config.TARGET_COLUMN: "Other Income",
                }
            ],
            ignore_index=True,
        )
        .append(
            [
                {
                    "description": "test the counterparty_name",
                    "amount": 123.0,
                    "transaction_type": "debit",
                    "account_type": "CURRENT_ACCOUNT",
                    "bank_counterparty_name": "Shell",
                    "counterparty_name": "Shell",
                    "bank_counterparty_iban": "NL86INGB0001234567",
                    "target_country_code": "NL",
                    domain_config.TARGET_COLUMN: "Vehicles and Driving Expenses",
                }
            ],
            ignore_index=True,
        )
        .fillna("")
    )
    df.reset_index(inplace=True)
    return df


@pytest.fixture(scope="module")
def docker_output_path(script_config) -> AnyStr:
    return script_config.get("docker_output_path")


@pytest.fixture(scope="module")
def script_config(training_config) -> Dict[AnyStr, AnyStr]:
    return training_config.script_config


@pytest.fixture(scope="module")
def training_config(project_config) -> YDSTrainingConfig:
    return project_config.training
