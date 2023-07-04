import pytest

from categories_model.training.main import train


class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


@pytest.mark.second
def test_train(project_config, script_config, docker_output_path):
    train(
        project_config=project_config,
        script_config=script_config,
        preprocess_path=docker_output_path,
    )

    assert True
