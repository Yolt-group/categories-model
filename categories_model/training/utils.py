import shutil
from pathlib import Path
from typing import AnyStr
from datascience_model_commons.general import upload_metadata, upload_artifact
from categories_model import predict
from categories_model.config.domain import DomainConfig, ModelType
import pandas as pd

from categories_model.predict import sme as sme_model
from categories_model.predict import retail as retail_model


def read_data(*, file_path: AnyStr) -> pd.DataFrame:
    """Read parquet/csv data"""
    if Path(file_path).suffix == ".csv":
        return pd.read_csv(file_path, header=True)
    else:
        return pd.read_parquet(file_path)


# always save model in docker output path as "model.tar.gz" artifact (automatically created by AWS)
def copy_model_and_metadata(
    domain_config: DomainConfig, docker_output_path: Path, model, training_metadata
):
    upload_artifact(
        model=model,
        path=docker_output_path,
        file_name=domain_config.MODEL_ARTIFACT_FILE,
    )

    # Copy right model type to model.py
    if domain_config.MODEL_TYPE == ModelType.SME_CATEGORIES_MODEL:
        shutil.copyfile(
            sme_model.__file__,
            docker_output_path / Path("model.py"),
        )
    elif domain_config.MODEL_TYPE == ModelType.RETAIL_CATEGORIES_MODEL:
        shutil.copyfile(
            retail_model.__file__,
            docker_output_path / Path("model.py"),
        )

    # Copy all predict code and dependencies
    predict_root = Path(predict.__file__).parent
    for origin in predict_root.glob("*"):
        if origin.is_file():
            destination = docker_output_path / origin.relative_to(predict_root)
            shutil.copyfile(origin, destination)

    # store metadata
    training_metadata.update(model.metadata)
    upload_metadata(
        metadata=training_metadata,
        path=docker_output_path,
        file_name=domain_config.MODEL_METADATA_FILE,
    )
