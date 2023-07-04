from pathlib import Path

from datascience_model_commons.model import BaseModel, ModelException
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_text  # noqa: F401
import yaml

from .arms_of_kamaji import series_map

DEFAULT_VALUES_DICT = {tf.string: (tf.string, ""), tf.float32: (tf.float32, 0.0)}


# Enable loading of metadata without model source code
class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
    def ignore_unknown(self, node):
        return None


class Model(BaseModel):
    def __init__(self, model_path):
        from datascience_model_commons.transactions.business_rules.dutch_tax_rules import (
            apply_tax_rules as apply_nl_tax_rules,
        )
        from datascience_model_commons.transactions.business_rules.uk_tax_rules import (
            apply_tax_rules as apply_uk_tax_rules,
        )

        model_directory = Path(__file__).parent
        with open(model_directory / "tax_config.yaml", "r") as f:
            self.tax_config_dict = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)
            self.tax_mapping_dict = self.tax_config_dict.get("mapping")
            self.tax_label_column = self.tax_config_dict.get("tax_label_column")

        # Dummy function (no operation) just returns the input_df
        def apply_nop(input_df: pd.DataFrame):
            return input_df

        self.tax_function_dict = {
            "NL": apply_nl_tax_rules,
            "GB": apply_uk_tax_rules,
            "": apply_nop,
        }

        # Default mapping is combination of UK and NL mapping
        tax_all_mapping = {}
        for local_mapping in self.tax_mapping_dict.values():
            for transaction_type, mapping in local_mapping.items():
                if transaction_type not in tax_all_mapping:
                    tax_all_mapping[transaction_type] = {}
                tax_all_mapping[transaction_type].update(mapping)
        self.tax_mapping_dict[""] = tax_all_mapping

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
    def extract_country_code(df: pd.DataFrame):
        # Make sure country code field is available
        if "country_code" not in df:
            df = df.assign(country_code="")
        else:
            df["country_code"].fillna("", inplace=True)

        # Select transactions that are missing country code
        has_no_country_code = df["country_code"] == ""
        # and where counterparty_iban is available
        if "bank_counterparty_iban" not in df:
            df = df.assign(counterparty_iban="")
        has_iban = df["bank_counterparty_iban"].fillna("") != ""
        selected_records = has_no_country_code & has_iban

        # Take first country code letters from counterparty_iban
        derived_country_code = df.loc[
            selected_records, "bank_counterparty_iban"
        ].str.slice(start=0, stop=2)
        # and assign to country code
        df.loc[selected_records, "country_code"] = derived_country_code

        # ToDo: Make sure country code becomes available from serving
        # assign country_code=GB when account type is CURRENT_ACCOUNT
        # and country_code could not be derived from bank_counterparty_iban
        has_no_country_code = df["country_code"] == ""
        is_current_account = df["counter_account_type"] == "CURRENT_ACCOUNT"
        selected_records = has_no_country_code & is_current_account
        df.loc[selected_records, "country_code"] = "GB"

        return df

    def apply_tax_rules(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        input = Rules column
        output = SME categories
        """
        for country_code, local_tax_mapping in self.tax_mapping_dict.items():
            subset_df = input_df.loc[input_df["country_code"] == country_code]

            # Nothing to do continue to next country code
            if len(subset_df) == 0:
                continue

            tax_mapping_function = self.tax_function_dict.get(country_code)
            subset_df = tax_mapping_function(subset_df)
            # if subset_df is empty no tax mapping should take place
            if self.tax_label_column in subset_df:
                for transaction_type, mapping in local_tax_mapping.items():
                    to_be_mapped_df = subset_df.loc[
                        (subset_df["transaction_type"] == transaction_type)
                        & (
                            subset_df[self.tax_label_column].notna()
                        )  # only when the rules apply
                    ]
                    # loc only changes the relevant transactions and columns in input_df
                    input_df.loc[to_be_mapped_df.index, "category"] = to_be_mapped_df[
                        self.tax_label_column
                    ].map(
                        mapping
                    )  # this applies the tax re-mapping
                    input_df.loc[
                        to_be_mapped_df.index, "category_source"
                    ] = "ModelPrediction"
        return input_df

    def apply_kamaji_the_counterparty_mapper(
        self, input_df: pd.DataFrame
    ) -> pd.DataFrame:

        if "counterparty_name" in input_df:
            kamaji_answer = series_map(input_df.counterparty_name)
            return kamaji_answer

        else:
            nones = pd.Series(None, input_df.index, dtype=str)
            return nones

    def apply_business_rules(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Apply business rules here, category column should be overwritten where business rules apply"""
        input_df = self.extract_country_code(df=input_df)

        result_df = self.apply_tax_rules(input_df=input_df)

        # Make sure interest paid after tax .* deducted is categorised as Other Income as it is not Loans
        result_df.loc[
            result_df.description.str.lower().str.match(
                "interest paid after tax .* deducted"
            ),
            ["category", "category_source"],
        ] = ("Other Income", "ModelPrediction")

        return result_df[["category", "category_source"]]

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

        stage_two = self.apply_business_rules(
            input_df=df.assign(
                category=categories,
                category_source=source,
            )
        )

        kamajis_answer = self.apply_kamaji_the_counterparty_mapper(df)
        stage_two["category"][~kamajis_answer.isna()] = kamajis_answer[
            ~kamajis_answer.isna()
        ]

        return stage_two
