from typing import AnyStr, Dict

import pytest
from datascience_model_commons.deploy.config.domain import (
    YDSTrainingConfig,
)


@pytest.fixture(scope="module")
def docker_output_path(script_config) -> AnyStr:
    return script_config.get("docker_output_path")


@pytest.fixture(scope="module")
def script_config(training_config) -> Dict[AnyStr, AnyStr]:
    return training_config.script_config


@pytest.fixture(scope="module")
def training_config(project_config) -> YDSTrainingConfig:
    return project_config.training
