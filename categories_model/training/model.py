import uuid

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_text as text
from sklearn.metrics import classification_report, label_ranking_average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.data import Dataset as TfDataset
from typing import AnyStr, Dict, Tuple, List, Union, Optional

from categories_model.config.domain import DomainConfig, ModelType
from datascience_model_commons.utils import get_logger

logger = get_logger()


class CategoryClassifier:
    def __init__(
        self,
        domain_config: DomainConfig,
        version_number: AnyStr = "",
        log_device_placement: bool = False,
    ):
        self.model = None
        self.postprocessed_model = None
        self.metrics = dict()
        self.metadata = dict()
        self.version_number = version_number
        self.amount_scaling_params = dict()
        if not self.version_number.isdigit():
            self.version_number = "1"

        self.domain_config = domain_config

        # enable to log device placement (GPU/CPU)
        tf.debugging.set_log_device_placement(log_device_placement)

    def fit(
        self,
        df_train: pd.DataFrame,
        df_validation: pd.DataFrame,
        batch_size: Optional[int] = None,
    ):
        """Fit the model on a training data set and evaluate performance on a hold-out set"""

        if not batch_size:
            batch_size = self.domain_config.BATCH_SIZE

        # exclude given categories from train & validation dataset
        df_train_filtered = df_train[
            ~df_train[self.domain_config.TARGET_COLUMN].isin(
                self.domain_config.EXCLUDED_TRAINING_LABELS
            )
        ]
        df_validation_filtered = df_validation[
            ~df_validation[self.domain_config.TARGET_COLUMN].isin(
                self.domain_config.EXCLUDED_TRAINING_LABELS
            )
        ]

        # generate preprocessing parameters
        self.amount_scaling_params = {
            "min": np.float32(df_train["amount"].min()),
            "max": np.float32(df_train["amount"].max()),
        }

        # check missing labels on train and validation sets
        self.check_labels(df=df_train_filtered, df_name="Training set")
        self.check_labels(df=df_validation_filtered, df_name="Validation set")

        # convert pandas to preprocessed tf dataset
        train_dataset, validation_dataset = (
            self.raw_to_tf_preprocessed_dataset(
                df=df,
                batch_size=batch_size,
                include_postprocessing=False,
                shuffle_and_repeat=True,
            )
            for df in [df_train_filtered, df_validation_filtered]
        )

        # build model
        self._build_model()

        # define callbacks for reducing learning rate and early stopping
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=2,
            min_lr=0.0001,
            verbose=0,
            min_delta=0.001,
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, min_delta=0.001, verbose=0
        )

        # define steps per each epoch
        steps_per_epoch = len(df_train_filtered) // batch_size
        validation_steps = len(df_validation_filtered) // batch_size

        # train model
        self.model.fit(
            train_dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=self.domain_config.MAX_EPOCHS,
            verbose=2,
            validation_data=validation_dataset,
            validation_steps=validation_steps,
            callbacks=[reduce_lr, early_stopping],
        )

        return self

    def save(self, *, path: AnyStr):
        input_amount = tf.keras.layers.Input(
            shape=(1,),
            dtype=tf.dtypes.float32,
            name="amount",
        )

        input_description = tf.keras.layers.Input(
            shape=(1,), dtype=tf.dtypes.string, name="description"
        )

        input_bank_counterparty_name = tf.keras.layers.Input(
            shape=(1,), dtype=tf.dtypes.string, name="bank_counterparty_name"
        )

        serving_inputs = {
            "description": input_description,
            "bank_counterparty_name": input_bank_counterparty_name,
            "amount": input_amount,
            **self.build_postprocessing_inputs(
                self.domain_config.POSTPROCESSING_COLUMNS
            ),
        }

        preprocessing_layer = tf.keras.layers.Lambda(
            function=self.serving_fn,
            name="preprocessing_fn",
        )(serving_inputs)
        serving_output = self.postprocessed_model(preprocessing_layer)

        serving_model = tf.keras.Model(
            inputs=serving_inputs, outputs=serving_output, name="serving_model"
        )
        serving_model.compile(loss=self.dummy_loss)

        """Save tf model to tar gz file"""
        serving_model.save(filepath=f"{str(path)}/{self.version_number}")

    def predict_similarity(self, df: pd.DataFrame) -> np.array:
        """Return predicted similarities for each category using the fitted model"""
        test_dataset = self.raw_to_tf_preprocessed_dataset(
            df=df,
            batch_size=len(df),
            shuffle_and_repeat=False,
        )
        similarity_score, _, _, _ = self.postprocessed_model.predict(
            test_dataset, steps=1
        )
        return similarity_score

    def predict(self, df: pd.DataFrame) -> np.array:
        """Return predicted final category using the fitted model"""
        y_score = self.predict_similarity(df)
        return np.array(self.domain_config.CATEGORIES)[np.argmax(y_score, axis=1)]

    def evaluate(self, df: pd.DataFrame) -> Dict:
        """Evaluate performance of fitted model on a validation data set"""

        # count rows and check if all classes are present in validation data
        self.metadata["n_test_samples"] = len(df)
        self.check_labels(df=df, df_name="Test set")

        # get predictions for test set; note that both scores and final category are needed for model metrics
        y_score = self.predict_similarity(df)
        y_pred = self.predict(df)
        logger.debug("Predictions output")

        self.metadata["metrics"] = metrics = self.compute_model_metrics(
            df=df, y_score=y_score, y_pred=y_pred
        )
        logger.debug("Metrics computed")

        return metrics

    @property
    def model_type(self) -> ModelType:
        return self.domain_config.MODEL_TYPE

    def _build_model(self):
        """Define model architecture"""

        # preprocessed input description - generate embeddings
        input_description = tf.keras.layers.Input(
            shape=(self.domain_config.SEQUENCE_LENGTH,),
            dtype=tf.dtypes.int32,
            name="preprocessed_description",
        )
        x = input_description
        x = tf.keras.layers.Embedding(
            input_dim=self.domain_config.VOCABULARY_SIZE,
            input_length=self.domain_config.SEQUENCE_LENGTH,
            output_dim=self.domain_config.EMBEDDING_SIZE,
            mask_zero=True,
        )(x)
        description_embeddings = ZeroMaskedAverage(name="description_embedding")(x)
        description_embeddings.trainable = False

        # preprocessed numeric features
        input_numeric_features = tf.keras.layers.Input(
            shape=(self.domain_config.N_NUMERIC_FEATURES,),
            dtype=tf.dtypes.float32,
            name="preprocessed_numeric_features",
        )

        # combine description and numeric features into transaction embedding
        transaction_features = tf.keras.layers.Concatenate()(
            [description_embeddings, input_numeric_features]
        )
        x = tf.keras.layers.Dense(
            units=self.domain_config.EMBEDDING_SIZE, activation="linear"
        )(transaction_features)
        x = tf.keras.layers.Lambda(
            lambda v: tf.nn.l2_normalize(v, axis=1), name="transaction_embeddings"
        )(x)
        transaction_embeddings = x

        # extract similarities between transactions & labels embeddings
        similarities = LabelEmbeddingSimilarity(
            name="similarities", domain_config=self.domain_config
        )(transaction_embeddings)

        # model definition
        self.model = tf.keras.Model(
            inputs=[
                input_description,
                input_numeric_features,
            ],
            outputs=similarities,
        )

        # define optimizer for loss function
        optimizer = tf.keras.optimizers.Adam(lr=self.domain_config.LEARNING_RATE)

        # compile model with custom triplet loss
        self.model.compile(optimizer=optimizer, loss=self.triplet_loss)

        # auxiliary inputs for business rules
        postprocessing_inputs = {
            **self.build_postprocessing_inputs(
                self.domain_config.POSTPROCESSING_COLUMNS
            ),
            "cleaned_description": tf.keras.layers.Input(
                shape=(1,), dtype=tf.dtypes.string, name="cleaned_description"
            ),
            "test_category": tf.keras.layers.Input(
                shape=(1,), dtype=tf.dtypes.string, name="test_category"
            ),
        }

        # use business rules transformer to overwrite similarities
        #   since we use model.output as input to postprocessing layer, it needs to be defined in the same function
        if self.model_type == ModelType.RETAIL_CATEGORIES_MODEL:
            from categories_model.training.retail import PostProcessingLayer
        elif self.model_type == ModelType.SME_CATEGORIES_MODEL:
            from categories_model.training.sme import PostProcessingLayer
        else:
            raise ValueError(f"Unknown model: {self.model_type}")

        postprocessed_similarities = PostProcessingLayer(
            name="postprocessed_similarities"
        )(
            {
                **postprocessing_inputs,
                "scores": self.model.output,
            }
        )

        self.postprocessed_model = tf.keras.Model(
            inputs={
                "preprocessed_description": self.model.input[0],
                "preprocessed_numeric_features": self.model.input[1],
                **postprocessing_inputs,
            },
            outputs=(*postprocessed_similarities, transaction_embeddings, similarities),
            name="postprocessed_model",
        )

        for output in self.postprocessed_model.outputs:
            output.trainable = False
        self.postprocessed_model.trainable = False
        self.postprocessed_model.compile(loss=self.dummy_loss)

        return self

    def check_labels(self, df: pd.DataFrame, df_name: AnyStr = ""):
        """Check whether all labels are present in the data frame"""
        labels_present = df[self.domain_config.TARGET_COLUMN].unique()
        label_difference = set(self.domain_config.CATEGORIES).difference(
            set(labels_present)
        )
        n_missing = len(label_difference)

        if n_missing > 0:
            logger.warning(
                f"{df_name}: {n_missing} missing categories: {label_difference}"
            )

    # build_postprocessing_inputs
    @staticmethod
    def build_postprocessing_inputs(columns: list) -> dict:
        return {
            column_name: tf.keras.layers.Input(
                shape=(1,), dtype=tf.dtypes.string, name=column_name
            )
            for column_name in columns
        }

    def split_training_data_frame(
        self,
        df: pd.DataFrame,
        n_validation_samples: Union[int, float],
        n_test_samples: Union[int, float],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split Pandas DataFrame in three stratified sets

        :param df: PySpark DataFrame to split
        :param n_validation_samples: number of rows in the validation frame
        :param n_test_samples: number of rows in the test frame
        :return: tuple of (train, validation, test) pandas datasets
        """

        # split dataset into train & test
        split_train_test = StratifiedShuffleSplit(
            n_splits=1, test_size=n_test_samples, random_state=42
        )

        train_index, test_index = next(
            split_train_test.split(df, df[self.domain_config.TARGET_COLUMN_INT])
        )
        df_train_val = df.iloc[train_index]
        df_test = df.iloc[test_index]

        # split train dataset into train & validation
        split_train_val = StratifiedShuffleSplit(
            n_splits=1, test_size=n_validation_samples, random_state=42
        )

        train_index, val_index = next(
            split_train_val.split(
                df_train_val, df_train_val[self.domain_config.TARGET_COLUMN_INT]
            )
        )
        df_train = df_train_val.iloc[train_index]
        df_val = df_train_val.iloc[val_index]

        return df_train, df_val, df_test

    @tf.function
    def triplet_loss(self, labels, similarities):
        """
        Triplet loss function

        :param labels: target column labels
        :param similarities: dot product of text and label embeddings
        :return: loss value
        """
        # extract positive and all negative examples
        # - positive example is for target category; negative is for all other categories
        labels_indices = tf.cast(labels, tf.int32)
        pos = tf.gather(similarities, labels_indices, batch_dims=1)
        negatives_mask = tf.squeeze(
            1
            - tf.one_hot(
                labels_indices, depth=self.domain_config.N_TRAINING_LABELS, axis=1
            ),
            axis=2,
        )
        neg = tf.reshape(
            tf.boolean_mask(similarities, negatives_mask),
            (tf.shape(negatives_mask)[0], self.domain_config.N_TRAINING_LABELS - 1),
        )

        # select the negative example that has the smallest similarity
        smallest_neg_similarity_tiled = tf.multiply(
            tf.ones_like(neg), tf.reduce_min(neg, axis=1, keepdims=True)
        )

        # if a similarity for negative example is lower than for positive example -> take neg;
        # else if negative example has higher similarity than positive example -> take the one with the smallest similarity
        neg_similarity_smaller_than_pos = tf.where(
            tf.less(neg, pos), neg, smallest_neg_similarity_tiled
        )

        # as final negative example, select the 'closest' negative example to the positive example
        semi_hard_neg = tf.reduce_max(
            neg_similarity_smaller_than_pos, axis=1, keepdims=True
        )

        # minimize the distance between the positive example and 'closest' negative example + margin
        loss = tf.reduce_mean(
            tf.maximum(0.0, -(pos - semi_hard_neg) + self.domain_config.MARGIN)
        )

        return loss

    def raw_to_tf_preprocessed_dataset(
        self,
        df: pd.DataFrame,
        batch_size: int,
        include_postprocessing=True,
        shuffle_and_repeat: bool = True,
    ) -> TfDataset:
        """
        Convert numpy array to tf preprocessed dataset generator
        Recommended order of transformations (source: https://cs230.stanford.edu/blog/datapipeline):
            - create the dataset
            - shuffle
            - repeat
            - preprocess
            - batch
            - prefetch

        :param df: pandas dataframe with feature columns and target column
        :param batch_size: batch size
        :param include_postprocessing: include postprocessing inputs in preprocessing (set to False for training)
        :param shuffle_and_repeat: whether we should shuffle and repeat the dataset
        :return: tuple of (train, validation, test) tensorflow datasets
        """
        # force all postprocessing columns to exist in that data frame by merging with an empty frame
        # replace missing values since Tensor does not accept None values and convert amount to float since spark generates
        #     Decimal()
        df_preprocessed = df.fillna(
            value=self.domain_config.MISSING_VALUES_REPLACEMENT
        ).astype({"amount": np.float32})

        if include_postprocessing:
            # determine missing postprocessing columns
            missing_postprocessing_columns = set(
                self.domain_config.POSTPROCESSING_COLUMNS
            ).difference(df.columns)

            # create defaults dict for missing postprocessing columns
            postprocessing_defaults = {
                key: value
                for key, value in self.domain_config.MISSING_VALUES_REPLACEMENT.items()
                if key in missing_postprocessing_columns
            }
        else:
            postprocessing_defaults = {}

        # create df for missing postprocessing columns with default values from
        # MISSING_VALUES_REPLACEMENT
        df_postprocessing_defaults = pd.DataFrame(
            data=postprocessing_defaults,
            index=df.index,
            dtype=np.str,
        )

        # append missing postprocessing columns
        df_dict = dict(pd.concat([df_preprocessed, df_postprocessing_defaults], axis=1))

        # convert df dictionary to tensorflow dataset
        dataset = tf.data.Dataset.from_tensor_slices(df_dict)

        if shuffle_and_repeat:
            # shuffle rows with a buffer size equal to the length of the dataset; this ensures good shuffling
            #    for more details check: https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle/48096625#48096625
            shuffle_buffer_size = len(df)
            dataset = dataset.shuffle(shuffle_buffer_size)

            # repeat dataset elements
            #    since we do shuffle and then repeat we make sure that we always see every element in the dataset at each epoch
            dataset = dataset.repeat()

        # preprocess dataset
        #    use num_parallel_calls to parallelize
        dataset = dataset.map(
            lambda ds: (
                self.preprocessing_fn(
                    inputs=ds,
                    include_postprocessing=include_postprocessing,
                ),
                ds[self.domain_config.TARGET_COLUMN_INT],
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        # split dataset into batches
        #    use drop_remainder to not end up with the last batch having small number of rows
        dataset = dataset.batch(batch_size, drop_remainder=True)

        # pre-fetching the data
        #    it will always have <buffer_size> batch ready to be loaded
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    def compute_model_metrics(
        self, df: pd.DataFrame, y_score: np.array, y_pred: np.array
    ) -> Dict:
        """Compute model metrics: classification & ranking & coverage"""

        # generate classification metrics
        metrics = classification_report(
            y_true=df[self.domain_config.TARGET_COLUMN].values,
            y_pred=y_pred,
            labels=self.domain_config.CATEGORIES,
            output_dict=True,
        )

        # generate ranking metrics
        mean_reciprocal_rank, recall_at_rank = self.ranking_metrics(
            y_true_int=df[self.domain_config.TARGET_COLUMN_INT].values, y_score=y_score
        )
        metrics["mean reciprocal rank"] = float(mean_reciprocal_rank)

        # generate average recall at given rank
        metrics["mean recall"] = dict()

        for n in range(1, (self.domain_config.MAX_RECALL_RANK + 1)):
            if len(recall_at_rank) > 0:
                mean_recall_at_n = recall_at_rank[:, (n - 1)].mean()
            else:
                mean_recall_at_n = 0
            metrics["mean recall"][f"rank={n}"] = float(mean_recall_at_n)

        # generate coverage metric - % of tx classified to non excluded categories
        n_excluded = np.sum(
            [
                metrics[cat]["support"]
                for cat in self.domain_config.EXCLUDED_TRAINING_LABELS
            ]
        )
        n_total = metrics["weighted avg"]["support"]
        metrics["coverage"] = 1.0 - (n_excluded / n_total) if n_total != 0.0 else 1.0

        return metrics

    def ranking_metrics(
        self, y_true_int: np.array, y_score: np.array
    ) -> Tuple[float, List]:
        """Compute ranking metrics: mean reciprocal rank & recall at each rank until MAX_RANK define in the settings"""

        # transform true value to the array of values with 1. in the index of given category
        #   i.e. y_true=3 is changed to y_true=[0.,0.,0.,1.,...] with the length of number of categories
        y_true_int_transformed = np.zeros_like(y_score)
        y_true_int_transformed[np.arange(len(y_score)), y_true_int] = 1.0

        # label_ranking_average_precision_score is equal to mean reciprocal rank since there is exactly one relevant
        #   label per given user and partner
        mean_reciprocal_rank = label_ranking_average_precision_score(
            y_true=y_true_int_transformed,
            y_score=y_score,
        )

        # sort arguments to extract top n recommendations; minus score is used for descending order
        # i.e. for given list of scores y_score = [0.6, 0.5, 0.7], the output would be y_rank_ind = [1, 2, 0]
        y_rank_ind = np.argsort(-y_score, axis=1)

        # sort y_true based on score at rank n
        n_relevant_at_rank = np.take_along_axis(
            y_true_int_transformed, y_rank_ind, axis=1
        ).cumsum(axis=1)

        # calculate recall for each rank with a limit of max recall rank
        recall_at_rank = (
            n_relevant_at_rank[:, : (self.domain_config.MAX_RECALL_RANK + 1)] / 1.0
        )

        return mean_reciprocal_rank, recall_at_rank

    def create_cucumber_test_sample(self, test_samples) -> pd.DataFrame:
        """
        Generate test sample for cucumber tests; passing all test is required to make model performant

        :return: pandas dataframe with input columns for the model and target category
        """
        cucumber_test_sample = pd.DataFrame.from_records(test_samples).fillna(
            self.domain_config.MISSING_VALUES_REPLACEMENT
        )
        cucumber_test_sample = cucumber_test_sample.assign(
            user_id="test", account_id="test"
        )
        cucumber_test_sample["transaction_id"] = cucumber_test_sample.apply(
            lambda _: uuid.uuid4(), axis=1
        )
        return cucumber_test_sample

    def check_performance(self, metrics: Dict, cucumber_tests_df: pd.DataFrame) -> bool:
        """
        Verify if critical performance metrics are above some preset threshold and whether cucumber tests pass.

        :param metrics: dictionary containing model performance metrics which is the output of model.evaluate
            therefore we use predefined keys below
        :param cucumber_tests_df: cucumber test dataframe with predictions and expected category
        :return: boolean indicating if standards are met
        """

        # check whether cucumber tests passed
        cucumber_tests_passed = all(
            cucumber_tests_df[self.domain_config.TARGET_COLUMN]
            == cucumber_tests_df["predicted_category"]
        )

        if not cucumber_tests_passed:
            logger.warning("Cucumber tests failing")

        # extract weighted average metrics and compare with thresholds
        weighted_average_metrics = metrics["weighted avg"]
        metrics_above_thresholds = all(
            weighted_average_metrics[key]
            >= (self.domain_config.WEIGHTED_METRICS_MIN_THRESHOLDS[key])
            for key in self.domain_config.WEIGHTED_METRICS_MIN_THRESHOLDS
        )

        if not metrics_above_thresholds:
            logger.warning("Weighted recall, precision or f1-score below threshold")

        # compare coverage (% of tx classified to non General category) with the threshold
        coverage_above_threshold = (
            metrics["coverage"] >= self.domain_config.COVERAGE_MIN_THRESHOLD
        )

        if not coverage_above_threshold:
            logger.warning("Coverage below threshold")

        return (
            cucumber_tests_passed & metrics_above_thresholds & coverage_above_threshold
        )

    @staticmethod
    def calculate_description_length_distribution(df: pd.DataFrame) -> Dict:
        """
        Compute histogram of number of characters in input description

        :param df: data frame containing input data
        :return: dictionary containing histogram {bucket_right_edge: probability}
        """

        n_samples = len(df)
        buckets = list(range(0, 260, 10))
        input_description_length_counts = np.histogram(
            df["description"].str.len(), bins=buckets
        )[0]

        input_description_length_distribution = dict(
            zip(buckets[1:], (input_description_length_counts / n_samples).tolist())
        )

        return input_description_length_distribution

    @tf.function
    def preprocessing_fn(
        self,
        inputs: TfDataset,
        include_postprocessing: bool = True,
    ) -> TfDataset:
        """
        Preprocess tensorflow dataset

        :param inputs: tensorflow dataset with input columns
        :param include_postprocessing: include postprocessing inputs in preprocessing (set to False for training)
        :return: preprocessed tensorflow dataset
        """
        sequence_length = self.domain_config.SEQUENCE_LENGTH
        vocabulary_size = self.domain_config.VOCABULARY_SIZE
        amount_scaling_params = self.amount_scaling_params
        preprocessed_numeric_columns = self.domain_config.PREPROCESSED_NUMERIC_COLUMNS
        n_numeric_features = self.domain_config.N_NUMERIC_FEATURES
        postprocessing_columns = self.domain_config.POSTPROCESSING_COLUMNS
        test_description_prefix = self.domain_config.TEST_DESCRIPTION_PREFIX

        # initialize output dictionary
        outputs = dict()

        # preprocess description

        # concatenate decription and bank_counter_party_name for sme categories model
        concatenated_description = (
            tf.strings.join(
                (inputs["description"], inputs["bank_counterparty_name"]), separator=" "
            )
            if self.domain_config.MODEL_TYPE == ModelType.SME_CATEGORIES_MODEL
            else inputs["description"]
        )

        # Remove diacritics
        # NFD will convert to Canonical Decomposition form (accents separated from letters)
        #   (see https://unicode.org/reports/tr15/)
        # Regex \p{M} stands for Mark (M)ark and will match accents that should be removed
        #   (see https://javascript.info/regexp-unicode)
        description_without_accents = tf.strings.regex_replace(
            text.normalize_utf8(
                concatenated_description,
                "NFD",  # Normalize characters to Canonical Decomposition form
            ),
            pattern=r"\p{M}+",  # Remove accents
            rewrite="",
        )
        cleaned_description = tf.strings.strip(
            tf.strings.lower(
                tf.strings.regex_replace(
                    description_without_accents, pattern=r"[^a-zA-Z]+", rewrite=" "
                )
            )
        )

        reshaped_clean_description = tf.squeeze(
            tf.reshape(cleaned_description, (-1, 1)), axis=1
        )
        # divide sequence length by 2 when splitting the description since by creating 1-ngrams and 2-grams we will double
        #    the sequence length
        tokens = tf.strings.split(
            reshaped_clean_description,
            maxsplit=(sequence_length / 2 - 1),
        )
        ngrams = tf.strings.ngrams(data=tokens, ngram_width=(1, 2), separator=" ")
        token_indices = tf.strings.to_hash_bucket_fast(
            input=ngrams, num_buckets=vocabulary_size
        ).to_tensor(shape=[None, sequence_length], default_value=0)
        token_indices_padded = tf.squeeze(input=token_indices)
        token_indices_padded.set_shape(sequence_length)
        outputs["preprocessed_description"] = token_indices_padded

        # preprocess numeric features
        numeric_features = dict()

        # scale amount
        numeric_features["scaled_amount"] = (
            inputs["amount"] - amount_scaling_params["min"]
        ) / (amount_scaling_params["max"] - amount_scaling_params["min"])

        # debit transaction flag
        numeric_features["is_debit_transaction"] = tf.cast(
            tf.equal(inputs["transaction_type"], "debit"), tf.dtypes.float32
        )

        # internal transaction flag
        numeric_features["is_internal_transaction"] = tf.cast(
            tf.not_equal(inputs["internal_transaction"], ""), tf.dtypes.float32
        )

        # combine all numeric features
        # NOTE: the order is important! we extract the features based on the index in postprocessing layer
        preprocessed_numeric_features = tf.transpose(
            tf.squeeze(
                tf.stack(
                    [numeric_features[col] for col in preprocessed_numeric_columns],
                    axis=0,
                )
            )
        )
        preprocessed_numeric_features.set_shape(n_numeric_features)
        outputs["preprocessed_numeric_features"] = preprocessed_numeric_features

        if include_postprocessing:
            # pass on features for postprocessing
            for column_name in postprocessing_columns:
                outputs[column_name] = inputs[column_name]
            outputs["cleaned_description"] = cleaned_description
            match_pattern = f"{test_description_prefix}([a-zA-Z ]+)"
            is_test_description = tf.strings.regex_full_match(
                input=concatenated_description, pattern=match_pattern
            )
            outputs["test_category"] = tf.where(
                condition=is_test_description,
                x=tf.strings.strip(
                    tf.strings.regex_replace(
                        input=concatenated_description,  # Test mode on concatenated description
                        pattern=match_pattern,  # Match prefix
                        rewrite=r"\1",
                    )
                ),
                y=tf.constant(value="", dtype=tf.dtypes.string),
            )

        return outputs

    @tf.function
    def serving_fn(
        self,
        inputs: TfDataset,
    ):
        features = self.preprocessing_fn(inputs=inputs)
        sequence_length = self.domain_config.SEQUENCE_LENGTH
        n_numeric_features = self.domain_config.N_NUMERIC_FEATURES
        postprocessing_columns = self.domain_config.POSTPROCESSING_COLUMNS

        # reshape is required for estimator to be exported correctly with known shapes
        features["preprocessed_description"] = tf.reshape(
            features["preprocessed_description"], (-1, sequence_length)
        )
        features["preprocessed_numeric_features"] = tf.reshape(
            features["preprocessed_numeric_features"],
            (-1, n_numeric_features),
        )

        # postprocessing features
        for column_name in postprocessing_columns:
            features[column_name] = tf.reshape(features[column_name], (-1, 1))
        return features

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, 1), dtype=tf.dtypes.int64, name="labels"),
            tf.TensorSpec(
                shape=(None, None), dtype=tf.dtypes.float32, name="similarities"
            ),
        ]
    )
    def dummy_loss(self, _, y_pred):
        """
        Dummy loss function which returns 0; its required for the compilation of postprocessed model

        :param y_pred: predictions
        :return: loss value
        """
        # note that when using tf.constant([0.]) there is an error that Variable has `None` for gradient
        #  therefore multiplication of y_pred with 0. is used
        loss = y_pred * 0.0

        return loss


class ZeroMaskedAverage(tf.keras.layers.Layer):
    """
    This layer is called after an Embedding layer.
    It zeros out all of the masked-out embeddings and returns the average of word embeddings.
    """

    def __init__(self, **kwargs):
        self.support_mask = True
        super().__init__(**kwargs)
        self.trainable = False

    def build(self, input_shape):
        self.output_dim = input_shape[1]
        self.repeat_dim = input_shape[2]

    def call(self, inputs, mask=None, **kwargs):
        # create a mask
        mask = tf.cast(mask, "float32")
        mask = tf.keras.backend.repeat(mask, self.repeat_dim)
        mask = tf.keras.backend.permute_dimensions(mask, (0, 2, 1))

        # number of that rows are not all zeros
        number_of_non_zeros_elements = tf.reduce_sum(
            tf.cast(mask, tf.float32), axis=1, keepdims=False
        )

        # the mean of a zero-length vector is undefined, but for a quick and dirty fix we extract the max
        number_of_non_zeros_elements = tf.maximum(number_of_non_zeros_elements, 1.0)

        # extract the mean from word embeddings to create a description embedding
        average_embedding = (
            tf.reduce_sum(inputs * mask, axis=1, keepdims=False)
            / number_of_non_zeros_elements
        )

        return average_embedding


class LabelEmbeddingSimilarity(tf.keras.layers.Layer):
    """
    Layer used to generate similarity between label embeddings and transaction embeddings
    """

    def __init__(self, domain_config: DomainConfig, **kwargs):
        super().__init__(**kwargs)
        self.n_training_labels = domain_config.N_TRAINING_LABELS
        self.embedding_size = domain_config.EMBEDDING_SIZE
        self.n_training_labels = domain_config.N_TRAINING_LABELS

    def build(self, input_shape):
        self.label_embeddings = self.add_weight(
            "label_embeddings",
            shape=[
                self.n_training_labels,
                self.embedding_size,
            ],
            initializer="uniform",
            trainable=True,
        )

        super(LabelEmbeddingSimilarity, self).build(input_shape)

    def call(self, inputs, **kwargs):
        label_embeddings = tf.nn.l2_normalize(self.label_embeddings, axis=1)
        similarities = tf.matmul(inputs, label_embeddings, transpose_b=True)

        return similarities

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.n_training_labels
