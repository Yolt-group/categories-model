import os
from pathlib import Path

import pytest

from categories_model.preprocessing.main import preprocess


@pytest.mark.first
def test_preprocess(project_config, domain_config, spark_session, docker_output_path):
    preprocess(
        spark=spark_session,
        project_config=project_config,
    )

    assert os.path.exists(Path(docker_output_path) / domain_config.TRAINING_DATA_FILE)
    assert os.path.exists(Path(docker_output_path) / domain_config.PRODUCTION_DATA_FILE)
    assert os.path.exists(
        Path(docker_output_path) / domain_config.PREPROCESSING_METADATA_FILE
    )
