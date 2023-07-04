import pytest
import numpy as np
import tensorflow as tf
import pandas as pd
from categories_model.config.retail import Retail

from categories_model.training.retail import (
    PostProcessingLayer,
    apply_rule_to_scores,
    add_uncategorized_column,
    apply_single_feature_business_rules,
    apply_internal_savings_business_rules,
    apply_remaining_business_rules,
    apply_fallback_rules,
    load_mcc_table,
)


@pytest.mark.first
def test__postprocessing_layer(domain_config):
    """Test the results after model postprocessing"""

    # extract indexes for categories that we modify in postprocessing layer
    category_indices = {
        category: index for index, category in enumerate(domain_config.CATEGORIES)
    }

    # define input features for examples
    df = pd.DataFrame(
        [
            {
                "internal_transaction": "",
                "transaction_type": "debit",
                "account_type": "CURRENT_ACCOUNT",
                "bank_specific__paylabels": "[!RO]",
                "bank_specific__mcc": "",
                "counter_account_type": "CURRENT_ACCOUNT",
                "category_source_target": "ModelPrediction",
                "cleaned_description": "test",
            },
            # row nr 1: internal transaction - set max similarity for Savings above threshold
            {
                "internal_transaction": "id_1412x",
                "transaction_type": "credit",
                "account_type": "CURRENT_ACCOUNT",
                "bank_specific__paylabels": "",
                "bank_specific__mcc": "",
                "counter_account_type": "CURRENT_ACCOUNT",
                "category_source_target": "ModelPrediction",
                "cleaned_description": "test",
            },
            # row nr 2: internal transaction - set max similarity for Savings below threshold
            {
                "internal_transaction": "id_192",
                "transaction_type": "credit",
                "account_type": "CURRENT_ACCOUNT",
                "bank_specific__paylabels": "",
                "bank_specific__mcc": "",
                "counter_account_type": "CURRENT_ACCOUNT",
                "category_source_target": "ModelPrediction",
                "cleaned_description": "test",
            },
            # row nr 3: assign all similarities below threshold
            {
                "internal_transaction": "",
                "transaction_type": "debit",
                "account_type": "CURRENT_ACCOUNT",
                "bank_specific__paylabels": "",
                "bank_specific__mcc": "",
                "counter_account_type": "CURRENT_ACCOUNT",
                "category_source_target": "ModelFallback",
                "cleaned_description": "test",
            },
            # row nr 4: internal transaction with pay label mapping to Savings
            {
                "internal_transaction": "id_9876",
                "transaction_type": "credit",
                "account_type": "CURRENT_ACCOUNT",
                "bank_specific__paylabels": "[!ES]",
                "bank_specific__mcc": "",
                "counter_account_type": "CURRENT_ACCOUNT",
                "category_source_target": "ModelPrediction",
                "cleaned_description": "test",
            },
            # row nr 5: internal transaction with pay label mapping to Income
            {
                "internal_transaction": "id_3456",
                "transaction_type": "credit",
                "account_type": "CURRENT_ACCOUNT",
                "bank_specific__paylabels": "[!RW]",
                "bank_specific__mcc": "",
                "counter_account_type": "CURRENT_ACCOUNT",
                "category_source_target": "ModelPrediction",
                "cleaned_description": "test",
            },
            # row nr 6: transaction should map to Travel via MCC fallback
            {
                "internal_transaction": "",
                "transaction_type": "debit",
                "account_type": "CURRENT_ACCOUNT",
                "bank_specific__paylabels": "",
                "bank_specific__mcc": "3000",
                "counter_account_type": "CURRENT_ACCOUNT",
                "category_source_target": "ModelFallback",
                "cleaned_description": "test",
            },
            # row nr 7: transaction should map to Income
            {
                "internal_transaction": "",
                "transaction_type": "credit",
                "account_type": "CURRENT_ACCOUNT",
                "bank_specific__paylabels": "",
                "bank_specific__mcc": "3736",
                "counter_account_type": "CURRENT_ACCOUNT",
                "category_source_target": "ModelPrediction",
                "cleaned_description": "test",
            },
            # row nr 8: transaction should map to Internal
            {
                "internal_transaction": "",
                "transaction_type": "debit",
                "account_type": "SAVINGS_ACCOUNT",
                "bank_specific__paylabels": "",
                "bank_specific__mcc": "",
                "counter_account_type": "CURRENT_ACCOUNT",
                "category_source_target": "ModelPrediction",
                "cleaned_description": "test",
            },
            # row nr 9: transaction should map to General
            {
                "internal_transaction": "",
                "transaction_type": "debit",
                "account_type": "CURRENT_ACCOUNT",
                "bank_specific__paylabels": "",
                "bank_specific__mcc": "",
                "counter_account_type": "CURRENT_ACCOUNT",
                "category_source_target": "ModelFallback",
                "cleaned_description": "test",
            },
            # row nr 10: savings transaction - set max similarity for Savings below threshold
            # - counter_account_type equals SAVINGS_ACCOUNT
            {
                "internal_transaction": "id_1412x",
                "transaction_type": "credit",
                "account_type": "CURRENT_ACCOUNT",
                "bank_specific__paylabels": "",
                "bank_specific__mcc": "",
                "counter_account_type": "SAVINGS_ACCOUNT",
                "category_source_target": "ModelPrediction",
                "cleaned_description": "test",
            },
        ]
    )

    input_shape = len(df)

    # define random similarities below threshold
    similarities = np.random.uniform(
        low=0,
        high=domain_config.GENERAL_SIMILARITY_THRESHOLD,
        size=(input_shape, domain_config.N_TRAINING_LABELS),
    )

    # row nr 1: internal transaction - set max similarity for Savings above threshold
    similarities[1, category_indices["Savings"]] = 1.0

    # row nr 2: internal transaction - set max similarity for Savings below threshold
    similarities[2, category_indices["Savings"]] = (
        domain_config.GENERAL_SIMILARITY_THRESHOLD - 0.1
    )

    # row nr 6: internal transaction - set max similarity for Savings below threshold
    similarities[6, :] = -1.0

    similarities[7, :] = -1.0

    similarities[9, :] = -1.0

    # row nr 10: internal transaction - set max similarity for Savings below threshold
    similarities[10, category_indices["Savings"]] = (
        domain_config.GENERAL_SIMILARITY_THRESHOLD - 0.1
    )

    # directly input similarities to postprocessing model
    input_similarities = tf.keras.layers.Input(shape=(domain_config.N_TRAINING_LABELS,))

    # axillary inputs for business rules
    input_internal_transaction = tf.keras.layers.Input(
        shape=(1,), dtype=tf.string, name="internal_transaction"
    )
    input_transaction_type = tf.keras.layers.Input(
        shape=(1,), dtype=tf.string, name="transaction_type"
    )
    input_account_type = tf.keras.layers.Input(
        shape=(1,), dtype=tf.string, name="account_type"
    )
    input_bank_specific__paylabels = tf.keras.layers.Input(
        shape=(1,), dtype=tf.string, name="bank_specific__paylabels"
    )
    input_bank_specific__mcc = tf.keras.layers.Input(
        shape=(1,), dtype=tf.string, name="bank_specific__mcc"
    )
    input_counter_account_type = tf.keras.layers.Input(
        shape=(1,), dtype=tf.string, name="counter_account_type"
    )
    input_cleaned_description = tf.keras.layers.Input(
        shape=(1,), dtype=tf.string, name="cleaned_description"
    )

    postprocessed_similarities = PostProcessingLayer(name="postprocessed_similarities")(
        dict(
            internal_transaction=input_internal_transaction,
            transaction_type=input_transaction_type,
            account_type=input_account_type,
            bank_specific__paylabels=input_bank_specific__paylabels,
            bank_specific__mcc=input_bank_specific__mcc,
            counter_account_type=input_counter_account_type,
            cleaned_description=input_cleaned_description,
            scores=input_similarities,
        )
    )

    postprocessed_model = tf.keras.Model(
        inputs=[
            input_internal_transaction,
            input_transaction_type,
            input_account_type,
            input_bank_specific__paylabels,
            input_bank_specific__mcc,
            input_counter_account_type,
            input_cleaned_description,
            input_similarities,
        ],
        outputs=postprocessed_similarities,
    )

    predicted_scores, category_source = postprocessed_model.predict(
        [
            df["internal_transaction"],
            df["transaction_type"],
            df["account_type"],
            df["bank_specific__paylabels"],
            df["bank_specific__mcc"],
            df["counter_account_type"],
            df["cleaned_description"],
            similarities,
        ]
    )

    category_source_target = df["category_source_target"].apply(str.encode).values

    np.testing.assert_array_equal(category_source, category_source_target)

    # check the shape of predictions
    assert predicted_scores.shape == (input_shape, domain_config.N_CATEGORIES)

    # pay labels: check whether predictions for Internal is 1.0 when "[!RO]" is provided in pay labels
    # debit transaction: check whether predictions for Income equals to -1
    assert predicted_scores[0, category_indices["Internal"]] == 1.0
    assert predicted_scores[0, category_indices["Income"]] == -1.0
    assert predicted_scores[0, category_indices["General"]] == -1.0

    # internal transaction: check whether predictions for Internal equals to 1 / -1
    # row nr 1: where Savings is the best predicted category above threshold
    # but counter_account_type is CURRENT_ACCOUNT, Internal should be 1 and Savings -1
    assert predicted_scores[1, category_indices["Savings"]] == 1.0
    assert predicted_scores[1, category_indices["Internal"]] != 1.0
    assert predicted_scores[1, category_indices["General"]] == -1.0

    # row nr 2: where Savings is the best predicted category below threshold, Internal should be 1
    assert predicted_scores[2, category_indices["Internal"]] == 1.0
    assert predicted_scores[2, category_indices["Savings"]] != 1.0
    assert predicted_scores[2, category_indices["General"]] == -1.0

    # row nr 3: where there is no internal transaction id, Internal should be -1, General should be 1
    assert predicted_scores[3, category_indices["Internal"]] == -1.0
    assert predicted_scores[3, category_indices["General"]] == 1.0

    # where Savings is the category dictated by the pay label [!ES], Internal should be -1 and Savings should be 1
    assert predicted_scores[4, category_indices["Savings"]] == 1.0
    assert predicted_scores[4, category_indices["General"]] == -1.0

    # where Income is the category dictated by the pay label [!RW], Internal should be -1
    assert predicted_scores[5, category_indices["Internal"]] != 1.0
    assert predicted_scores[5, category_indices["Income"]] == 1.0
    assert predicted_scores[5, category_indices["General"]] == -1.0

    # when none of the rules apply and none above threshold apply mcc fallback
    # this should map mcc code 3000 to Travel
    assert predicted_scores[6, category_indices["Travel"]] == 1.0

    # this should map to income as it is a credit transaction
    assert predicted_scores[7, category_indices["Income"]] == 1.0

    # account_type=SAVINGS_ACCOUNT should map to internal
    assert predicted_scores[8, category_indices["Internal"]] == 1.0

    # when no fallback available the transaction should map to general
    assert predicted_scores[9, category_indices["General"]] == 1.0

    # row nr 10: where Savings is the best predicted category below threshold,
    # but counter_account_type equals SAVINGS_ACCOUNT Savings should be 1
    assert predicted_scores[10, category_indices["Savings"]] == 1.0
    assert predicted_scores[10, category_indices["General"]] == -1.0

    # there should be at most one prediction per row with value 1.0
    assert ((predicted_scores == 1.0).sum(axis=1) <= 1).all()


class PostProcessingTest(tf.test.TestCase):
    """Test postprocessing layer components"""

    def init(self, **kwargs):
        super().__init__(**kwargs)

    @pytest.mark.first
    def test__add_uncategorized_column(self):

        scores = tf.constant([[0.9, 0.8], [0.5, 0.6]])
        uncategorized_zero_scores = tf.zeros((2, 1))
        expected_scores = tf.concat([scores, uncategorized_zero_scores], axis=-1)

        scores = add_uncategorized_column(scores=scores)

        self.assertAllEqual(scores, expected_scores)

    @pytest.mark.first
    def test__apply_rule_to_scores(self):

        mask = tf.constant([[True], [True]], dtype=tf.dtypes.bool)
        scores = tf.constant([[0.9, 0.8, 0.0], [0.5, 0.6, 0.0]])
        # replace scores to -1. based on the indices
        expected_scores = tf.tensor_scatter_nd_update(scores, [[1, 2]], [-1.0])
        expected_mask = tf.constant([[True], [False]], dtype=tf.dtypes.bool)

        mask, scores = apply_rule_to_scores(
            category_mask=tf.constant([[False, False, True], [False, False, True]]),
            condition_mask=tf.constant([[False], [True]]),
            category_value=(tf.ones_like(scores) * -1.0),
            scores=scores,
            mask=mask,
        )

        self.assertAllEqual(scores, expected_scores)
        self.assertAllEqual(mask, expected_mask)

    @pytest.mark.first
    def test__apply_single_feature_business_rules(self):

        scores = tf.constant(
            [
                [0.9, 0.8, 0.0, 0.0],
                [0.5, 0.6, 0.0, 0.0],
                [0.4, 0.5, 0.0, 0.0],
                [0.1, 0.2, 0.3, 0.4],
            ]
        )
        # replace scores to -1./-1 based on the indices for business rules categories
        expected_scores = tf.tensor_scatter_nd_update(
            scores,
            [[0, 1], [0, 0], [2, 0]],
            [-1.0, 1.0, 1.0],
        )

        _, scores = apply_single_feature_business_rules(
            inputs={
                "transaction_type": tf.constant(["debit", "credit", "debit", "debit"]),
                "internal_transaction": tf.constant(["", "id_43543", "", ""]),
                "account_type": tf.constant(["", "", "SAVINGS_ACCOUNT", ""]),
                "bank_specific__paylabels": tf.constant(
                    ["[!RO],[!ES]", "", "", "[!RO],[!ES]"]
                ),
            },
            mask=tf.constant([[True], [True], [True], [False]], dtype=tf.dtypes.bool),
            scores=scores,
            category_masks={
                "General": tf.constant([False, False, False, True]),
                "Savings": tf.constant([False, False, True, False]),
                "Income": tf.constant([False, True, False, False]),
                "Internal": tf.constant([True, False, False, False]),
            },
            category_values={
                False: (tf.ones_like(scores) * -1.0),
                True: tf.ones_like(scores),
            },
        )

        self.assertAllEqual(scores, expected_scores)

    @pytest.mark.first
    def test__apply_internal_savings_business_rules(self):

        scores = tf.constant(
            [
                [0.9, 0.8, 0],
                [0.5, 0.6, 0],
                [0.2, 0.1, 0],
                [-0.1, 0.2, 0.5],
                [0.1, 0.8, 0.2],
                [0.1, 0.8, 0.2],
            ]
        )
        # replace scores to -1./1. based on the indices of internal category; the logic for internal transactions:
        #   - if Savings has the highest score, then Internal should be -1
        #   - if Savings doesn't have the highest score, then Internal should be 1 no matter if the score is above
        #      or below threshold
        expected_scores = tf.tensor_scatter_nd_update(
            scores,
            [[0, 1], [1, 1], [2, 1], [4, 0], [5, 0]],
            [-1.0, 1.0, 1.0, 1.0, 1.0],
        )

        _, scores = apply_internal_savings_business_rules(
            inputs={
                "internal_transaction": tf.constant(
                    ["id_1021", "id_098", "id_123", "id_345", "id_3411", ""]
                ),
                "account_type": tf.constant(
                    [
                        "CURRENT_ACCOUNT",
                        "CURRENT_ACCOUNT",
                        "CURRENT_ACCOUNT",
                        "CURRENT_ACCOUNT",
                        "CURRENT_ACCOUNT",
                        "CURRENT_ACCOUNT",
                    ]
                ),
                "counter_account_type": tf.constant(
                    [
                        "CURRENT_ACCOUNT",
                        "CURRENT_ACCOUNT",
                        "CURRENT_ACCOUNT",
                        "CURRENT_ACCOUNT",
                        "SAVINGS_ACCOUNT",
                        "SAVINGS_ACCOUNT",
                    ]
                ),
            },
            mask=tf.constant(
                [[True], [True], [True], [False], [True], [True]], dtype=tf.dtypes.bool
            ),
            scores=scores,
            all_trues=tf.cast(tf.ones_like(scores), tf.bool),
            category_masks={
                "Savings": tf.constant([True, False, False]),
                "Internal": tf.constant([False, True, False]),
            },
            category_values={
                False: (tf.ones_like(scores) * -1.0),
                True: tf.ones_like(scores),
            },
            category_indices={"Savings": 0, "Internal": 1, "General": 2},
        )

        self.assertAllEqual(scores, expected_scores)

    @pytest.mark.first
    def test__remaining_business_rules(self):

        scores = tf.constant(
            [
                [0.7, 0.6, 0.0, 0.0, 0.0],
                [0.7, 0.6, 0.8, 0.0, 0.0],
                [0.7, 0.6, 0.0, 0.0, 0.8],
                [0.7, 0.6, 0.0, 0.0, 0.8],
                [0.7, 0.6, 0.0, 0.8, 0.0],
                [0.7, 0.6, 0.0, 0.8, 0.0],
            ]
        )
        # replace scores to -1./1. based on the indices of general category
        expected_scores = tf.tensor_scatter_nd_update(
            scores, [[0, 0], [1, 2], [2, 2], [3, 1]], [1.0, 1.0, 1.0, 1.0]
        )

        _, scores = apply_remaining_business_rules(
            mask=tf.constant(
                [[True], [True], [True], [True], [True], [False]],
                dtype=tf.dtypes.bool,
            ),
            inputs={
                "account_type": tf.constant(
                    [
                        "CURRENT_ACCOUNT",
                        "CURRENT_ACCOUNT",
                        "CURRENT_ACCOUNT",
                        "CURRENT_ACCOUNT",
                        "CURRENT_ACCOUNT",
                        "CURRENT_ACCOUNT",
                    ],
                    dtype=tf.dtypes.string,
                ),
                "internal_transaction": tf.constant(
                    ["", "", "id_324", "", "", ""], dtype=tf.dtypes.string
                ),
                "transaction_type": tf.constant(
                    ["debit", "debit", "credit", "credit", "debit", "credit"],
                    dtype=tf.dtypes.string,
                ),
            },
            scores=scores,
            category_masks={
                "Savings": tf.constant([True, False, False, False, False]),
                "Income": tf.constant([False, True, False, False, False]),
                "Internal": tf.constant([False, False, True, False, False]),
                "Leisure": tf.constant([False, False, False, True, False]),
                "General": tf.constant([False, False, False, False, True]),
            },
            category_values={
                False: (tf.ones_like(scores) * -1.0),
                True: tf.ones_like(scores),
            },
            category_indices={
                "Savings": 0,
                "Income": 1,
                "Internal": 2,
                "Leisure": 3,
                "General": 4,
            },
        )

        self.assertAllEqual(scores, expected_scores)

    @pytest.mark.first
    def test__apply_fallback_rules(self):
        scores = tf.constant(
            [
                [0.9, 0.8, 0.0, 0.0, 0.0],
                [0.2, 0.6, 0.0, 0.0, 0.0],
                [0.2, 0.2, 0.1, 0.3, 0.0],
                [0.2, 0.2, 0.2, 0.3, 0.0],
            ]
        )
        # replace scores to -1./1. based on the indices of general category
        expected_scores = tf.tensor_scatter_nd_update(
            scores, [[0, 4], [1, 4], [2, 2], [3, 3]], [-1.0, 1.0, 1.0, 1.0]
        )
        expected_source = tf.constant(
            [
                "ModelPrediction",
                "ModelPrediction",
                "ModelFallback",
                "ModelFallback",
            ],
            dtype=tf.dtypes.string,
        )

        mcc_category_indices = tf.transpose(
            tf.constant([[2, 3, 2, 3]], dtype=tf.dtypes.int32)
        )
        _, scores, source = apply_fallback_rules(
            mask=tf.constant([[True], [True], [True], [True]], dtype=tf.dtypes.bool),
            scores=scores,
            inputs={
                "cleaned_description": tf.constant(
                    [
                        "Test",
                        "",
                        "Test",
                        "Test",
                    ],
                    dtype=tf.dtypes.string,
                ),
                "transaction_type": tf.constant(["debit", "debit", "debit", "debit"]),
                "account_type": tf.constant(
                    [
                        "CURRENT_ACCOUNT",
                        "CURRENT_ACCOUNT",
                        "CURRENT_ACCOUNT",
                        "CURRENT_ACCOUNT",
                    ]
                ),
                "counter_account_type": tf.constant(
                    [
                        "CURRENT_ACCOUNT",
                        "CURRENT_ACCOUNT",
                        "CURRENT_ACCOUNT",
                        "CURRENT_ACCOUNT",
                    ]
                ),
            },
            mcc_category_indices=mcc_category_indices,
            category_masks={
                "Savings": tf.constant([True, False, False, False, False]),
                "Income": tf.constant([False, True, False, False, False]),
                "Travel": tf.constant([False, False, True, False, False]),
                "Leisure": tf.constant([False, False, False, True, False]),
                "General": tf.constant([False, False, False, False, True]),
            },
            category_values={
                False: (tf.ones_like(scores) * -1.0),
                True: tf.ones_like(scores),
            },
            debit_transactions=tf.constant(
                [[True], [True], [True], [True]],
                dtype=tf.dtypes.bool,
            ),
        )

        self.assertAllEqual(scores, expected_scores)
        self.assertAllEqual(source, expected_source)

    @pytest.mark.first
    def test_mcc_lookup(self):
        mcc_values = tf.constant(["3000", "3736", "sdf"], dtype=tf.dtypes.string)
        target_categories = ["Travel", "Leisure"]
        unknown_lookup_value = -1
        target_indices = tf.constant(
            [Retail().CATEGORY_INDICES[category] for category in target_categories]
            + [unknown_lookup_value],
            dtype=tf.dtypes.int32,
        )
        mcc_table = load_mcc_table()
        mcc_indices = mcc_table.lookup(mcc_values)

        self.assertAllEqual(mcc_indices, target_indices)
