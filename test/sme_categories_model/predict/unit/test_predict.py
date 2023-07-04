from categories_model.predict.sme import Model
from categories_model.predict.arms_of_kamaji import category_counterparty_map
import pandas as pd

from categories_model.config.sme import SME


def test_extract_country_code(tax_df):
    result = Model.extract_country_code(tax_df)

    pd.testing.assert_series_equal(
        result.country_code, result.target_country_code, check_names=False
    )


def test_validate_kamajis_arms():
    list_of_all_counterparties = []

    for category, counterparties in category_counterparty_map.items():
        list_of_all_counterparties += counterparties

    assert_check = (
        "Verify counterparties map to 1 unique category, and not appear twice."
    )
    assert len(list_of_all_counterparties) == len(
        set(list_of_all_counterparties)
    ), assert_check


def test_category_validity():
    SME_Config = SME()

    assert_check = "Verify all counterparties map to valid categories"

    for category, counterparties in category_counterparty_map.items():
        assert category in SME_Config.CATEGORIES, assert_check
