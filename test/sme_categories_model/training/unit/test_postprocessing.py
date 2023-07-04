import pytest
import numpy as np
import tensorflow as tf
import pandas as pd

from categories_model.training.sme import (
    PostProcessingLayer,
    apply_rule_to_scores,
    add_uncategorized_column,
    apply_business_rules,
    apply_fallback_rules,
)

from categories_model.config.sme import SME as DomainConfig


@pytest.mark.first
def test__postprocessing_layer(domain_config):
    """Test the results after model postprocessing"""

    # extract indexes for categories that we modify in postprocessing layer
    category_indices = domain_config.CATEGORY_INDICES
    debit_indices = list(domain_config.OUTGOING_CATEGORIES_INDICES.values())
    credit_indices = list(domain_config.INCOMING_CATEGORIES_INDICES.values())

    # define input features for examples
    df = pd.DataFrame(
        [
            # row nr 0: no description debit should become Miscellaneous expenses
            {
                "cleaned_description": "",
                "internal_transaction": "",
                "transaction_type": "debit",
                "account_type": "CURRENT_ACCOUNT",
                "counter_account_type": "",
                "category_source_target": "ModelPrediction",
                "test_category": "",
            },
            # row nr 1: no description credit should become Other income
            {
                "cleaned_description": "",
                "internal_transaction": "",
                "transaction_type": "credit",
                "account_type": "CURRENT_ACCOUNT",
                "counter_account_type": "",
                "category_source_target": "ModelPrediction",
                "test_category": "",
            },
            # row nr 2: debit transaction to SAVINGS_ACCOUNT should be Savings Deposits
            {
                "cleaned_description": "",
                "internal_transaction": "",
                "transaction_type": "debit",
                "account_type": "CURRENT_ACCOUNT",
                "counter_account_type": "SAVINGS_ACCOUNT",
                "category_source_target": "ModelPrediction",
                "test_category": "",
            },
            # row nr 3: debit transaction to SAVINGS_ACCOUNT should be Savings Withdrawal
            {
                "cleaned_description": "",
                "internal_transaction": "",
                "transaction_type": "credit",
                "account_type": "CURRENT_ACCOUNT",
                "counter_account_type": "SAVINGS_ACCOUNT",
                "category_source_target": "ModelPrediction",
                "test_category": "",
            },
            # row nr 4: this transaction should map to General Expenses
            {
                "cleaned_description": "Test",
                "internal_transaction": "",
                "transaction_type": "debit",
                "account_type": "CURRENT_ACCOUNT",
                "counter_account_type": "",
                "category_source_target": "ModelPrediction",
                "test_category": "",
            },
            # row nr 5: this transaction should map to Revenue
            {
                "cleaned_description": "Test",
                "internal_transaction": "",
                "transaction_type": "credit",
                "account_type": "CURRENT_ACCOUNT",
                "counter_account_type": "",
                "category_source_target": "ModelPrediction",
                "test_category": "",
            },
            # row nr 6: transaction should map to Miscellaneous expenses and ModelFallback
            {
                "cleaned_description": "Test",
                "internal_transaction": "",
                "transaction_type": "debit",
                "account_type": "CURRENT_ACCOUNT",
                "counter_account_type": "",
                "category_source_target": "ModelFallback",
                "test_category": "",
            },
            # row nr 7: transaction should map to Other Income and ModelFallback
            {
                "cleaned_description": "Test",
                "internal_transaction": "",
                "transaction_type": "credit",
                "account_type": "CURRENT_ACCOUNT",
                "counter_account_type": "",
                "category_source_target": "ModelFallback",
                "test_category": "",
            },
            # row nr 8: transaction should map to Other Income and ModelFallback
            {
                "cleaned_description": "Test",
                "internal_transaction": "",
                "transaction_type": "debit",
                "account_type": "CURRENT_ACCOUNT",
                "counter_account_type": "",
                "category_source_target": "ModelPrediction",
                "test_category": "Interest and Repayments",
            },
        ]
    )

    df["bank_specific__paylabels"] = ""
    df["bank_specific__mcc"] = ""
    input_shape = len(df)

    # define random similarities below threshold
    similarities = np.random.uniform(
        low=0,
        high=domain_config.GENERAL_SIMILARITY_THRESHOLD,
        size=(input_shape, domain_config.N_TRAINING_LABELS),
    )

    similarities[0:4, :] = np.random.uniform(
        low=domain_config.GENERAL_SIMILARITY_THRESHOLD,
        high=1.0,
        size=(4, domain_config.N_TRAINING_LABELS),
    )

    expected_scores = np.hstack(
        (
            similarities,
            -np.ones((input_shape, len(domain_config.EXCLUDED_TRAINING_LABELS))),
        )
    )
    # Similarities for debit transactions should have -1.0 on credit categories
    for r in df[df.transaction_type == "debit"].index:
        expected_scores[r, credit_indices] = -1.0
    # Similarities for credit transactions should have -1.0 on debit categories
    for r in df[df.transaction_type == "credit"].index:
        expected_scores[r, debit_indices] = -1.0

    # row nr 0: no description debit should become Miscellaneous expenses
    expected_scores[0, category_indices["Other Expenses"]] = 1.0
    # row nr 1: no description credit should become Other income
    expected_scores[1, category_indices["Other Income"]] = 1.0
    # row nr 2: debit transaction to SAVINGS_ACCOUNT should be Savings Deposits
    expected_scores[2, category_indices["Corporate Savings Deposits"]] = 1.0
    # row nr 3: debit transaction to SAVINGS_ACCOUNT should be Savings Withdrawal
    expected_scores[3, category_indices["Equity Withdrawal"]] = 1.0
    # row nr 4: this transaction should map to General Expenses
    similarities[4, category_indices["Other Operating Costs"]] = 1.0
    expected_scores[4, category_indices["Other Operating Costs"]] = 1.0
    # row nr 5: this transaction should map to Revenue
    similarities[5, category_indices["Revenue"]] = 1.0
    expected_scores[5, category_indices["Revenue"]] = 1.0
    # row nr 6: transaction should map to Miscellaneous expenses and ModelFallback
    expected_scores[6, category_indices["Other Expenses"]] = 1.0
    # row nr 7: transaction should map to Other Income and ModelFallback
    expected_scores[7, category_indices["Other Income"]] = 1.0
    # row nr 8: transaction should map to Interest and Repayments
    expected_scores[8, category_indices["Interest and Repayments"]] = 1.0

    # directly input similarities to postprocessing model
    input_similarities = tf.keras.layers.Input(shape=(domain_config.N_TRAINING_LABELS,))

    # axillary inputs for business rules
    input_cleaned_description = tf.keras.layers.Input(
        shape=(1,), dtype=tf.string, name="cleaned_description"
    )
    input_test_category = tf.keras.layers.Input(
        shape=(1,), dtype=tf.string, name="test_category"
    )
    input_internal_transaction = tf.keras.layers.Input(
        shape=(1,), dtype=tf.string, name="internal_transaction"
    )
    input_bank_specific__paylabels = tf.keras.layers.Input(
        shape=(1,), dtype=tf.string, name="bank_specific__paylabels"
    )
    input_bank_specific__mcc = tf.keras.layers.Input(
        shape=(1,), dtype=tf.string, name="bank_specific__mcc"
    )
    input_transaction_type = tf.keras.layers.Input(
        shape=(1,), dtype=tf.string, name="transaction_type"
    )
    input_account_type = tf.keras.layers.Input(
        shape=(1,), dtype=tf.string, name="account_type"
    )
    input_counter_account_type = tf.keras.layers.Input(
        shape=(1,), dtype=tf.string, name="counter_account_type"
    )

    postprocessed_similarities = PostProcessingLayer(name="postprocessed_similarities")(
        dict(
            cleaned_description=input_cleaned_description,
            test_category=input_test_category,
            internal_transaction=input_internal_transaction,
            transaction_type=input_transaction_type,
            account_type=input_account_type,
            counter_account_type=input_counter_account_type,
            scores=input_similarities,
            bank_specific__paylabels=input_bank_specific__paylabels,
            bank_specific__mcc=input_bank_specific__mcc,
        )
    )

    postprocessed_model = tf.keras.Model(
        inputs=[
            input_cleaned_description,
            input_test_category,
            input_internal_transaction,
            input_transaction_type,
            input_account_type,
            input_counter_account_type,
            input_bank_specific__paylabels,
            input_bank_specific__mcc,
            input_similarities,
        ],
        outputs=postprocessed_similarities,
    )

    predicted_scores, category_source = postprocessed_model.predict(
        [
            df["cleaned_description"],
            df["test_category"],
            df["internal_transaction"],
            df["transaction_type"],
            df["account_type"],
            df["counter_account_type"],
            df["bank_specific__paylabels"],
            df["bank_specific__mcc"],
            similarities,
        ]
    )

    category_source_target = df["category_source_target"].apply(str.encode).values

    np.testing.assert_array_equal(category_source, category_source_target)

    # check the shape of predictions
    assert predicted_scores.shape == (input_shape, domain_config.N_CATEGORIES)

    # check that the outcome matches the expected outcome
    np.allclose(expected_scores, predicted_scores)

    # there should be at most one prediction per row with value 1.0
    assert ((predicted_scores == 1.0).sum(axis=1) <= 1).all()


class PostProcessingTest(tf.test.TestCase):
    """Test postprocessing layer components"""

    @pytest.mark.first
    def test__add_uncategorized_column(self):
        scores = tf.constant([[0.9, 0.8], [0.5, 0.6]])
        n_excluded_labels = DomainConfig().N_EXCLUDED_TRAINING_LABELS
        uncategorized_zero_scores = tf.zeros((2, n_excluded_labels))
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
    def test__apply_internal_savings_business_rules(self):
        scores = tf.constant(
            [
                [-0.1, 0.2, 0.5, 0.5],
                [0.1, 0.8, 0.2, 0.5],
                [0.1, 0.8, 0.2, 0.5],
            ]
        )
        # replace scores to -1./1. based on the indices of internal category; the logic for internal transactions:
        #   - if Savings has the highest score, then Internal should be -1
        #   - if Savings doesn't have the highest score, then Internal should be 1 no matter if the score is above
        #      or below threshold
        expected_scores = tf.tensor_scatter_nd_update(
            scores,
            [[1, 0], [2, 1]],
            [1.0, 1.0],
        )

        _, scores = apply_business_rules(
            inputs={
                "transaction_type": tf.constant(["credit", "debit", "credit"]),
                "account_type": tf.constant(
                    [
                        "CURRENT_ACCOUNT",
                        "CURRENT_ACCOUNT",
                        "CURRENT_ACCOUNT",
                    ]
                ),
                "counter_account_type": tf.constant(
                    [
                        "CURRENT_ACCOUNT",
                        "SAVINGS_ACCOUNT",
                        "SAVINGS_ACCOUNT",
                    ]
                ),
            },
            mask=tf.constant([[False], [True], [True]], dtype=tf.dtypes.bool),
            scores=scores,
            category_masks={
                "Corporate Savings Deposits": tf.constant([True, False, False, False]),
                "Equity Financing": tf.constant([False, True, False, False]),
                "Other Expenses": tf.constant([False, False, True, False]),
                "Other Income": tf.constant([False, False, False, True]),
            },
            category_values={
                False: (tf.ones_like(scores) * -1.0),
                True: tf.ones_like(scores),
            },
            debit_transactions=tf.constant(
                [[False], [True], [False]], dtype=tf.dtypes.bool
            ),
            credit_transactions=tf.constant(
                [[True], [False], [True]], dtype=tf.dtypes.bool
            ),
        )

        self.assertAllEqual(scores, expected_scores)

    @pytest.mark.first
    def test__apply_fallback_rules(self):
        scores = tf.constant(
            [
                [0.9, 0.8, 0.0, 0.0, 0.0],
                [0.2, 0.81, 0.0, 0.0, 0.0],
                [0.2, 0.2, 0.1, 0.3, 0.0],
                [0.2, 0.2, 0.2, 0.3, 0.0],
                [0.2, 0.2, 0.1, 0.3, 0.0],
                [0.2, 0.2, 0.2, 0.3, 0.0],
            ]
        )

        # replace scores to -1./1. based on the indices of general category
        category_indices = {"Other Expenses": 3, "Other Income": 4}
        expected_scores_numpy = scores.numpy()
        expected_scores_numpy[0, category_indices["Other Expenses"]] = -1.0
        expected_scores_numpy[0, category_indices["Other Income"]] = -1.0
        expected_scores_numpy[1, category_indices["Other Expenses"]] = -1.0
        expected_scores_numpy[1, category_indices["Other Income"]] = -1.0
        expected_scores_numpy[2, category_indices["Other Expenses"]] = 1.0
        expected_scores_numpy[3, category_indices["Other Income"]] = 1.0
        expected_scores_numpy[4, category_indices["Other Expenses"]] = 1.0
        expected_scores_numpy[5, category_indices["Other Income"]] = 1.0
        expected_scores = tf.constant(expected_scores_numpy, dtype=tf.dtypes.float32)

        expected_source = tf.constant(
            [
                "ModelPrediction",
                "ModelPrediction",
                "ModelFallback",
                "ModelFallback",
                "ModelPrediction",
                "ModelPrediction",
            ],
            dtype=tf.dtypes.string,
        )

        _, scores, source = apply_fallback_rules(
            mask=tf.constant(
                [[True], [True], [True], [True], [True], [True]], dtype=tf.dtypes.bool
            ),
            inputs={
                "cleaned_description": tf.constant(
                    ["Test", "Test", "Test", "Test", "", ""], dtype=tf.dtypes.string
                ),
                "transaction_type": tf.constant(
                    ["debit", "credit", "debit", "credit", "debit", "credit"]
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
                        "SAVINGS_ACCOUNT",
                        "SAVINGS_ACCOUNT",
                        "CURRENT_ACCOUNT",
                        "CURRENT_ACCOUNT",
                        "CURRENT_ACCOUNT",
                        "CURRENT_ACCOUNT",
                    ]
                ),
            },
            scores=scores,
            category_masks={
                "Travel Expenses": tf.constant([True, False, False, False, False]),
                "Equity Withdrawal": tf.constant([False, True, False, False, False]),
                "Corporate Savings Deposits": tf.constant(
                    [False, False, True, False, False]
                ),
                "Other Expenses": tf.constant([False, False, False, True, False]),
                "Other Income": tf.constant([False, False, False, False, True]),
            },
            category_values={
                False: (tf.ones_like(scores) * -1.0),
                True: tf.ones_like(scores),
            },
            debit_transactions=tf.constant(
                [[False], [True], [True], [False], [True], [False]],
                dtype=tf.dtypes.bool,
            ),
            credit_transactions=tf.constant(
                [[True], [False], [False], [True], [False], [True]],
                dtype=tf.dtypes.bool,
            ),
        )

        self.assertAllEqual(scores, expected_scores)
        self.assertAllEqual(source, expected_source)
