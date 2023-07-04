import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from categories_model.config.domain import ModelType

from categories_model.training.model import (
    ZeroMaskedAverage,
    LabelEmbeddingSimilarity,
)


def test_split_training_data_frame(df, model_instance):
    # split training data in train and test
    train, val, test = model_instance.split_training_data_frame(
        df=df, n_validation_samples=250, n_test_samples=250
    )

    # check whether datasets are not empty
    # note that rows in train + val + test won't be equal to df since additional filters on train and test
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0

    # for test dataset we can check the exact number of rows (not possible in case of train and val due to
    #   additional filters
    assert len(test) == 250


def test_calculate_description_length_distribution(df, model_instance):
    input_description_length_distribution = (
        model_instance.calculate_description_length_distribution(df)
    )

    # check last element (first element might not be stable due to some randomization inside the model)
    assert input_description_length_distribution[250] == 0.0

    # check if histogram is properly normalized
    np.testing.assert_almost_equal(
        sum(input_description_length_distribution.values()), 1, decimal=1
    )


def test_preprocessing_fn(model_instance):
    """Test the data preprocessing"""

    # define pd input example to generate preprocessing params
    df = pd.DataFrame.from_dict(
        {
            "description": ["Avro Energy Energy 5"],
            "amount": [4000.0],
            "account_type": ["CURRENT_ACCOUNT"],
            "internal_transaction": ["12341id"],
            "transaction_type": ["debit"],
            "bank_counterparty_name": ["Avro Energy"],
            "bank_specific__paylabels": [""],
            "bank_specific__mcc": [""],
            "counter_account_type": ["current account"],
        }
    )

    # define tf dataset to test preprocessing function
    inputs = {
        "description": tf.constant(df["description"], shape=[]),
        "amount": tf.constant(np.float32(df["amount"]), shape=[]),
        "account_type": tf.constant(df["account_type"], shape=[]),
        "internal_transaction": tf.constant(df["internal_transaction"], shape=[]),
        "transaction_type": tf.constant(df["transaction_type"], shape=[]),
        "bank_counterparty_name": tf.constant(df["bank_counterparty_name"], shape=[]),
        "bank_specific__paylabels": tf.constant(
            df["bank_specific__paylabels"], shape=[]
        ),
        "bank_specific__mcc": tf.constant(df["bank_specific__mcc"], shape=[]),
        "counter_account_type": tf.constant(df["counter_account_type"], shape=[]),
    }

    # FIXME: move tensorflow session to fixtures using tf.test

    model_instance.amount_scaling_params = {
        "min": tf.constant(df["amount"].min(), dtype=tf.dtypes.float32),
        "max": tf.constant(df["amount"].max(), dtype=tf.dtypes.float32),
    }
    outputs = model_instance.preprocessing_fn(
        inputs=inputs,
    )

    domain_config = model_instance.domain_config

    # check the output shape of preprocessed text input
    assert outputs["preprocessed_description"].shape == (domain_config.SEQUENCE_LENGTH,)

    # check the output shape of preprocessed numeric input
    assert outputs["preprocessed_numeric_features"].shape == (
        domain_config.N_NUMERIC_FEATURES,
    )

    # check whether five elements in preprocessed text are non zero - since we create 1-gram and 2-grams on words
    assert np.count_nonzero(outputs["preprocessed_description"]) == 9

    # check whether number of unique elements in preprocessed text is equal to the number of expected ngrams
    assert len(np.unique(outputs["preprocessed_description"])) == 6


@pytest.mark.first
def test_description_embedding_masking(model_instance):
    """Test masking logic while generating sentence embedding"""

    # define the same architecture logic as in model.py
    input_text = tf.keras.layers.Input(shape=(4,), dtype=tf.int32)
    x = input_text
    x = tf.keras.layers.Embedding(
        input_dim=50, input_length=4, output_dim=8, mask_zero=True
    )(x)
    x = ZeroMaskedAverage()(x)
    embedding = x

    # train dummy model
    model = tf.keras.Model(inputs=input_text, outputs=embedding)
    model.compile(optimizer="adam", loss=model_instance.dummy_loss)
    model.fit(np.array([[32, 22, 0, 0]]), np.array([0]), epochs=0, verbose=0)

    # generate predictions
    prediction_for_zeros = model.predict(np.array([[0, 0, 0, 0]]))
    prediction_for_known_word = model.predict(np.array([[32, 0, 0, 0]]))
    prediction_for_known_word_occurred_twice = model.predict(np.array([[32, 32, 0, 0]]))

    # embedding for padded index should be filled in with zeros
    assert (prediction_for_zeros == 0).all()

    # embedding for a sentence with known word should not be zeros
    assert (prediction_for_known_word != 0).any()

    # embedding for a sentence with two same words should be the same as with one word - since we take the mean
    assert (prediction_for_known_word == prediction_for_known_word_occurred_twice).all()


@pytest.mark.first
def test_label_embedding_similarity(domain_config):
    """Test custom layer: whether the shape is as expected"""
    # define random inputs
    np.random.seed(0)
    transaction_embeddings = np.random.rand(1, domain_config.EMBEDDING_SIZE)

    # define model architecture
    input_transaction_embeddings = tf.keras.layers.Input(
        shape=(domain_config.EMBEDDING_SIZE,)
    )
    similarities = LabelEmbeddingSimilarity(domain_config=domain_config)(
        input_transaction_embeddings
    )
    model = tf.keras.Model(inputs=input_transaction_embeddings, outputs=similarities)

    predicted_similarities = model.predict(transaction_embeddings)

    # check if the output shape is as expected
    assert predicted_similarities.shape == (1, domain_config.N_TRAINING_LABELS)


@pytest.mark.first
def test_compute_model_metrics(model_instance):
    domain_config = model_instance.domain_config

    """Test whether the classification & ranking metrics are calculated correctly"""
    # create test data with target columns used to compute metrics
    df = pd.DataFrame().assign(
        **{
            domain_config.TARGET_COLUMN: domain_config.CATEGORIES,
            domain_config.TARGET_COLUMN_INT: np.arange(0, domain_config.N_CATEGORIES),
        }
    )

    # add dummy predictions
    y_score = np.tile(
        np.linspace(start=0, stop=1, num=domain_config.N_CATEGORIES), (len(df), 1)
    )
    y_pred = np.array(domain_config.CATEGORIES)[np.argmax(y_score, axis=1)]
    metrics = model_instance.compute_model_metrics(
        df=df, y_score=y_score, y_pred=y_pred
    )

    # check some of the metrics
    if model_instance.model_type == ModelType.RETAIL_CATEGORIES_MODEL:
        assert np.round(metrics["weighted avg"]["precision"], 3) == 0.001
        assert np.round(metrics["weighted avg"]["recall"], 3) == 0.032
        assert np.round(metrics["weighted avg"]["f1-score"], 3) == 0.002
        assert metrics["weighted avg"]["support"] == 31
        assert np.round(metrics["mean reciprocal rank"], 3) == 0.130
        assert np.round(metrics["mean recall"]["rank=1"], 3) == 0.032
    elif model_instance.model_type == ModelType.SME_CATEGORIES_MODEL:
        assert np.round(metrics["weighted avg"]["precision"], 3) == 0.002
        assert np.round(metrics["weighted avg"]["recall"], 3) == 0.042
        assert np.round(metrics["weighted avg"]["f1-score"], 3) == 0.003
        assert metrics["weighted avg"]["support"] == 24
        assert np.round(metrics["mean reciprocal rank"], 3) == 0.157
        assert np.round(metrics["mean recall"]["rank=1"], 3) == 0.042
    else:
        assert False


@pytest.mark.first
def test_check_performance(model_instance):
    domain_config = model_instance.domain_config

    is_performant = model_instance.check_performance(
        metrics={
            "weighted avg": {
                "support": 1000,
                "recall": 0.3,
                "precision": 0.3,
                "f1-score": 0.3,
            },
            "coverage": 0.6,
        },
        cucumber_tests_df=pd.DataFrame(
            {domain_config.TARGET_COLUMN: ["Bills"], "predicted_category": ["Bills"]}
        ),
    )
    assert ~is_performant

    is_performant = model_instance.check_performance(
        metrics={
            "weighted avg": {
                "support": 1000,
                "recall": 0.8,
                "precision": 0.8,
                "f1-score": 0.8,
            },
            "coverage": 0.93,
        },
        cucumber_tests_df=pd.DataFrame(
            {domain_config.TARGET_COLUMN: ["Bills"], "predicted_category": ["Housing"]}
        ),
    )
    assert ~is_performant

    is_performant = model_instance.check_performance(
        metrics={
            "weighted avg": {
                "support": 1000,
                "recall": 0.8,
                "precision": 0.8,
                "f1-score": 0.8,
            },
            "coverage": 0.93,
        },
        cucumber_tests_df=pd.DataFrame(
            {domain_config.TARGET_COLUMN: ["Bills"], "predicted_category": ["Bills"]}
        ),
    )
    assert is_performant
