from pathlib import Path

import pytest
from datascience_model_commons.deploy.config.domain import (
    YDSProjectConfig,
    YDSDomain,
    YDSEnvironment,
    YDSPreprocessingConfig,
    PreprocessingType,
    YDSTrainingConfig,
    TrainingType,
    DeployingUser,
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
        model_name="sme-categories-model",
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
                "n_model_samples_per_country": 2000,
                "n_production_samples": 1000,
                "n_validation_samples": 250,
                "n_test_samples": 250,
                "data_file_paths": {
                    "users_app": f"{project_root}/test/resources/user.csv",
                    "test_users_app": f"{project_root}/test/resources/test_users.csv",
                    "transactions_app": f"{project_root}/test/resources/transactions.csv",
                    "accounts_app": f"{project_root}/test/resources/account.csv",
                    "users_yts": f"{project_root}/test/resources/user_yts.csv",
                    "transactions_yts": f"{project_root}/test/resources/transactions.csv",
                    "accounts_yts": f"{project_root}/test/resources/account.csv",
                },
                "docker_output_path": "/tmp/sme_categories_model_test",
                "synthetic_feedback_prefix": "synthetic_feedback",
            },
        ),
        training=YDSTrainingConfig(
            training_type=TrainingType.TENSORFLOW,
            entrypoint="main.py",
            sagemaker_processor_kwargs={},
            script_config={
                "docker_output_path": "/tmp/sme_categories_model_test",
                "batch_size": 128,
            },
        ),
    )
