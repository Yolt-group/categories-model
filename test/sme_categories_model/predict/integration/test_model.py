import pytest
from pandas.testing import assert_series_equal

from categories_model.predict.sme import Model


@pytest.mark.third
def test_model(docker_output_path, tax_df):
    model = Model(docker_output_path)

    tax_preds = model.predict(df=tax_df)
    tax_preds.reset_index(inplace=True)

    assert_series_equal(tax_preds.category, tax_df.target_category, check_names=False)
