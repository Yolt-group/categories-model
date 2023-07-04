from datascience_model_commons.model import BaseModel, ModelException
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_text  # noqa: F401
import yaml

DEFAULT_VALUES_DICT = {tf.string: (tf.string, ""), tf.float32: (tf.float32, 0.0)}


# Enable loading of metadata without model source code
class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
    def ignore_unknown(self, node):
        return None


class Model(BaseModel):
    def __init__(self, model_path):
        self.load_model(model_path=model_path)

    def load_model(self, model_path):
        def find_artifact(root):
            import os

            # Traverse root to find model signature files (.pb file extension)
            for root, dirs, files in os.walk(root):
                for file in files:
                    if file.endswith(".pb"):
                        return root

            raise ModelException("No .pb model artifact found!")

        try:
            artifact_location = find_artifact(root=model_path)
            model = tf.saved_model.load(artifact_location)
        except Exception as e:
            raise ModelException(f"Unable to load model {model_path}") from e

        metadata_path = f"{model_path}/training_metadata.yaml"
        try:
            with open(metadata_path, "r") as stream:
                SafeLoaderIgnoreUnknown.add_constructor(
                    None, SafeLoaderIgnoreUnknown.ignore_unknown
                )
                root = yaml.load(stream, Loader=SafeLoaderIgnoreUnknown)
                categories = root["categories"]
        except IOError as ioerr:
            raise ModelException(
                f'Error reading yaml config file: "{metadata_path}"'
            ) from ioerr
        except yaml.YAMLError as yerr:
            raise ModelException(f'Unable to parse "{metadata_path}"') from yerr

        self.model = model
        self.categories = categories
        self.predict_function = self.model.signatures["serving_default"]
        self.input_tensors = [
            tensor
            for tensor in self.predict_function.inputs
            if "unknown" not in tensor.name
        ]
        self.input_description = {
            tensor.name.split(":")[0]: DEFAULT_VALUES_DICT.get(tensor.dtype)
            for tensor in self.input_tensors
        }

    def df_to_tensor_dict(self, df):
        N = df.shape[0]
        return {
            key: np.atleast_2d(
                df[key].values.astype(dtype=dtype.as_numpy_dtype)
                if key in df
                else np.repeat(default_value, N)
            ).T
            for key, (dtype, default_value) in self.input_description.items()
        }

    def scores_to_category(self, scores):
        return np.choose(scores.argmax(axis=1), self.categories)

    @staticmethod
    def apply_business_rules(input_df: pd.DataFrame) -> pd.DataFrame:
        """Apply business rules here, category column should be overwritten where business rules apply"""
        return input_df[["category", "category_source"]]

    def predict(self, *, df: pd.DataFrame):
        if df.empty:  # Nothing to predict so return empty dataframe
            categories = pd.Series(dtype=object)
            source = pd.Series(dtype=object)
        else:
            predictions = self.predict_function(**self.df_to_tensor_dict(df))
            scores = predictions["postprocessed_model"].numpy()
            categories = self.scores_to_category(scores=scores)
            source = np.atleast_1d(predictions["postprocessed_model_1"].numpy()).astype(
                str
            )

        return self.apply_business_rules(
            input_df=df.assign(
                category=categories,
                category_source=source,
            )
        )
