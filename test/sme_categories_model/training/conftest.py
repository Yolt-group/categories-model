from typing import AnyStr, Dict

import pytest
from datascience_model_commons.deploy.config.domain import (
    YDSTrainingConfig,
)

from categories_model.training.model import CategoryClassifier
from categories_model.training.utils import read_data


@pytest.fixture(scope="module")
def df():
    df = read_data(file_path="/tmp/categories_model_test/preprocessed_training_data")
    return df


@pytest.fixture(scope="module")
def model_instance(domain_config) -> CategoryClassifier:
    return CategoryClassifier(domain_config=domain_config)


@pytest.fixture(scope="module")
def docker_output_path(script_config) -> AnyStr:
    return script_config.get("docker_output_path")


@pytest.fixture(scope="module")
def script_config(training_config) -> Dict[AnyStr, AnyStr]:
    return training_config.script_config


@pytest.fixture(scope="module")
def training_config(project_config) -> YDSTrainingConfig:
    return project_config.training
