import pytest
import pandas as pd
from categories_model.predict.retail import Model


@pytest.mark.third
def test_model(project_config, domain_config, docker_output_path):
    model = Model(docker_output_path)

    # define input features for examples
    df = pd.DataFrame(domain_config.CUCUMBER_TEST_SAMPLES).fillna("")
    model.predict(df=df)

    assert True
