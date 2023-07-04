from pathlib import Path

import pytest
from datascience_model_commons.deploy.config.domain import (
    YDSProjectConfig,
    YDSDomain,
    YDSEnvironment,
    DeployingUser,
    YDSPreprocessingConfig,
    PreprocessingType,
    YDSTrainingConfig,
    TrainingType,
)

import categories_model
from categories_model.config.domain import DomainConfig
from categories_model.config.utils import get_domain_config


@pytest.fixture(scope="module")
def domain_config(project_config) -> DomainConfig:
    return get_domain_config(project_config.model_name)


@pytest.fixture(scope="module")
def project_root() -> Path:
    return Path(categories_model.__file__).parent.parent


@pytest.fixture(scope="module")
def project_config(project_root) -> YDSProjectConfig:
    return YDSProjectConfig(
        model_name="categories-model",
        domain=YDSDomain.YoltApp,
        model_bucket="local",
        aws_iam_role_name="local",
        env=YDSEnvironment.DTA,
        deploy_id="local",
        deploying_user=DeployingUser(first_name="test", last_name="user"),
        git_branch="",
        git_commit_short="",
        package_dir="categories_model",
        preprocessing=YDSPreprocessingConfig(
            processing_type=PreprocessingType.SPARK,
            entrypoint="main.py",
            sagemaker_processor_kwargs={},
            script_config={
                "sample_start_date": "2018-02-01",
                "sample_end_date": "2018-03-01",
                "n_model_samples_per_country": 3000,
                "n_production_samples": 1000,
                "n_validation_samples": 1000,
                "n_test_samples": 1000,
                "data_file_paths": {
                    "transactions": f"{project_root}/test/resources/transactions.csv",
                    "accounts": f"{project_root}/test/resources/account.csv",
                    "users": f"{project_root}/test/resources/user.csv",
                    "test_users": f"{project_root}/test/resources/test_users.csv",
                    "user_single_feedback_created": f"{project_root}/test/resources/data_science_events/user_single_feedback_created.json",
                    "user_multiple_feedback_created": f"{project_root}/test/resources/data_science_events/user_multiple_feedback_created.json",
                    "user_multiple_feedback_applied": f"{project_root}/test/resources/data_science_events/user_multiple_feedback_applied.json",
                    "historical_feedback": f"{project_root}/test/resources/historical_feedback/*.csv",
                },
                "docker_output_path": "/tmp/categories_model_test",
            },
        ),
        training=YDSTrainingConfig(
            training_type=TrainingType.TENSORFLOW,
            entrypoint="main.py",
            sagemaker_processor_kwargs={},
            script_config={
                "docker_output_path": "/tmp/categories_model_test",
                "batch_size": 128,
            },
        ),
    )
