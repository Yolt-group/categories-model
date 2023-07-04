import tensorflow as tf
from typing import Dict, Tuple

from categories_model.config.sme import SME

DOMAIN_CONFIG = SME()


class PostProcessingLayer(tf.keras.layers.Layer):
    """
    Layer which applies postprocessing logic and modifies model scores accordingly
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        keys_tensor = tf.constant(DOMAIN_CONFIG.CATEGORIES, dtype=tf.dtypes.string)
        vals_tensor = tf.constant(
            range(DOMAIN_CONFIG.N_CATEGORIES), dtype=tf.dtypes.int32
        )
        self.category_to_int = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
            default_value=-1,
        )

    def call(self, inputs, **kwargs):
        """
        Apply business rules IMPORTANT NOTE: in theory the business logic shouldn't conflict, but in practice it may
        happen since: - we apply rule one by one, so it might happen that assigning 1. based on the first rule may
        influence the other rule results where we check the threshold - there is additional business logic for
        Savings and Internal categories specified in retail.py which may also conflict with the results of business
        rules definition result: in any conflicting case, the model will return the first category with the highest
        score :return: a tuple containing: postprocessed scores and category source
        """
        scores = inputs["scores"]

        category_indices = self.category_to_int.lookup(
            tf.reshape(inputs["test_category"], (-1, 1))
        )

        # transform postprocessing inputs
        for column_name in DOMAIN_CONFIG.POSTPROCESSING_COLUMNS:
            inputs[column_name] = tf.cast(inputs[column_name], tf.dtypes.string)

        # add column for General category
        scores = add_uncategorized_column(scores)

        # create some shared tensors
        all_ones = tf.clip_by_value(scores, 1.0, 1.0)
        all_minus_ones = tf.clip_by_value(scores, -1.0, -1.0)
        all_trues = tf.cast(all_ones, tf.bool)
        debit_transactions = tf.equal(
            tf.reshape(inputs["transaction_type"], (-1, 1)), "debit"
        )
        credit_transactions = tf.equal(
            tf.reshape(inputs["transaction_type"], (-1, 1)), "credit"
        )
        category_values = {False: all_minus_ones, True: all_ones}
        category_masks = {
            category: all_trues & tf.equal(DOMAIN_CONFIG.CATEGORIES, category)
            for category in DOMAIN_CONFIG.CATEGORIES
        }
        from functools import reduce

        category_masks["debit"] = reduce(
            tf.logical_or,
            (
                category_masks[category]
                for category in DOMAIN_CONFIG.OUTGOING_CATEGORIES
            ),
        )
        category_masks["credit"] = reduce(
            tf.logical_or,
            (
                category_masks[category]
                for category in DOMAIN_CONFIG.INCOMING_CATEGORIES
            ),
        )

        # credit transactions cannot have debit categories
        _, scores = apply_rule_to_scores(
            category_mask=category_masks["debit"],
            condition_mask=credit_transactions,
            category_value=category_values[False],
            scores=scores,
        )

        # debit transactions cannot have credit categories
        _, scores = apply_rule_to_scores(
            category_mask=category_masks["credit"],
            condition_mask=debit_transactions,
            category_value=category_values[False],
            scores=scores,
        )

        # create mask for keeping track of transactions where already a rule has applied
        mask = tf.reduce_any(all_trues, axis=1, keepdims=True)

        mask, scores = apply_business_rules(
            mask=mask,
            inputs=inputs,
            scores=scores,
            category_values=category_values,
            category_masks=category_masks,
            debit_transactions=debit_transactions,
            credit_transactions=credit_transactions,
        )

        mask, scores, source = apply_fallback_rules(
            mask=mask,
            inputs=inputs,
            scores=scores,
            category_values=category_values,
            category_masks=category_masks,
            debit_transactions=debit_transactions,
            credit_transactions=credit_transactions,
        )

        # predict desired outcome for test desriptions
        scores, source = apply_test_rules(
            source=source,
            category_indices=category_indices,
            scores=scores,
            all_trues=all_trues,
        )

        return scores, source

    def compute_output_shape(self, input_shape):
        return {
            "postprocessed_model": (
                input_shape["account_type"][0],
                DOMAIN_CONFIG.N_CATEGORIES,
            ),
            "postprocessed_model_1": (input_shape["account_type"][0]),
        }


@tf.function
def add_uncategorized_column(scores: tf.Tensor) -> tf.Tensor:
    """
    Add placeholder for uncategorized columns

    :param scores: the output of the model
    :return: scores with additional columns
    """
    n_excluded_labels = DOMAIN_CONFIG.N_EXCLUDED_TRAINING_LABELS
    # the transformation scores[:, None, 0] is used instead of scores[:, 0] to keep the dimension of scores
    return tf.concat(
        [
            scores,
            *([tf.clip_by_value(scores[:, None, 0], 0.0, 0.0)] * n_excluded_labels),
        ],
        axis=-1,
    )


@tf.function
def apply_test_rules(
    *,
    source: tf.Tensor,
    category_indices: tf.Tensor,
    scores: tf.Tensor,
    all_trues: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    condition = tf.math.greater(category_indices, -1)
    test_categories = tf.squeeze(
        tf.one_hot(
            indices=category_indices,
            depth=scores.shape[1],
        ),
    )

    # Replace similarities for test description with one hot encoded target category
    mask, scores = apply_rule_to_scores(
        category_mask=all_trues,
        condition_mask=condition,
        category_value=test_categories,
        scores=scores,
    )

    # Prediction source for test description is ModelPrediction
    is_test_description = tf.math.not_equal(category_indices, -1)
    source = tf.where(
        condition=tf.squeeze(is_test_description),
        x=tf.constant("ModelPrediction", dtype=tf.dtypes.string),
        y=source,
        name="category_source",
    )

    return scores, source


@tf.function
def apply_business_rules(
    *,
    mask: tf.Tensor,
    inputs: Dict,
    scores: tf.Tensor,
    category_values: Dict,
    category_masks: Dict,
    debit_transactions: tf.Tensor,
    credit_transactions: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Incoming transactions between current and savings account should be Savings Withdrawal
    Outgoing transactions between current and savings account should be Savings Deposit
    :param mask: mask with
        - True values for transactions where no business rule applied yet
        - False values for transactions that already have a category assigned due to a business rule
    :param inputs: dictionary with postprocessing input features
    :param scores: model scores
    :param category_values: dictionary with values assigned to True/False category
    :param category_masks: masks for each category
    :param debit_transactions: a tensor with True for each debit transaction
    :param credit_transactions: a tensor with True for each credit transaction
    :return: tuple containing: - modified mask for the categories where business rules have been applied
                               - modified scores for the categories included in business rules
    """

    # if the account_type equals "CURRENT_ACCOUNT" and counter_account_type equals "SAVINGS_ACCOUNT"
    # then category should be Savings Deposit for debit transactions
    current_savings_account_trx = tf.logical_and(
        tf.equal(tf.reshape(inputs["account_type"], (-1, 1)), "CURRENT_ACCOUNT"),
        tf.equal(
            tf.reshape(inputs["counter_account_type"], (-1, 1)), "SAVINGS_ACCOUNT"
        ),
    )
    condition = tf.logical_and(
        current_savings_account_trx,
        debit_transactions,
    )
    mask, scores = apply_rule_to_scores(
        category_mask=category_masks["Corporate Savings Deposits"],
        condition_mask=condition,
        category_value=category_values[True],
        scores=scores,
        mask=mask,
    )

    # if the account_type equals "CURRENT_ACCOUNT" and counter_account_type equals "SAVINGS_ACCOUNT"
    # then category should be Savings Withdrawal for credit transactions
    condition = tf.logical_and(
        current_savings_account_trx,
        credit_transactions,
    )
    mask, scores = apply_rule_to_scores(
        category_mask=category_masks["Equity Financing"],
        condition_mask=condition,
        category_value=category_values[True],
        scores=scores,
        mask=mask,
    )

    return mask, scores


@tf.function
def apply_fallback_rules(
    *,
    mask: tf.Tensor,
    inputs: Dict,
    scores: tf.Tensor,
    category_values: Dict,
    category_masks: Dict,
    debit_transactions: tf.Tensor,
    credit_transactions: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Modify scores based on the fallback rules
    :param mask: mask with
        - True values for transactions where no business rule applied yet
        - False values for transactions that already have a category assigned due to a business rule
    :param inputs: dictionary with postprocessing input features
    :param scores: model scores
    :param category_values: dictionary with values assigned to True/False category
    :param category_masks: masks for each category
    :param debit_transactions: a tensor with True for each debit transaction
    :param credit_transactions: a tensor with True for each credit transaction
    :return: tuple containing: - modified mask for the categories where business rules have been applied
                               - modified scores for the categories included in business rules
                               - category source tensor with prediction source (ModelPrediction or ModelFallback)
    """

    # -------empty description without business rule applied should be Other Income or Miscellaneous expenses--
    empty_description = tf.equal(tf.reshape(inputs["cleaned_description"], (-1, 1)), "")

    # if transaction type equals "debit" and cleaned description is empty
    # then category should be Miscellaneous expenses
    condition = tf.logical_and(
        empty_description,
        debit_transactions,
    )
    mask, scores = apply_rule_to_scores(
        category_mask=category_masks["Other Expenses"],
        condition_mask=condition,
        category_value=category_values[True],
        scores=scores,
        mask=mask,
    )
    # if transaction type equals "credit" and cleaned description is empty
    # then category should be Other Income
    condition = tf.logical_and(
        empty_description,
        credit_transactions,
    )
    mask, scores = apply_rule_to_scores(
        category_mask=category_masks["Other Income"],
        condition_mask=condition,
        category_value=category_values[True],
        scores=scores,
        mask=mask,
    )

    # -------apply threshold_for_uncategorized-------
    # set to general score to -1.0 if at least one category is above threshold
    has_max_score_at_least_threshold = tf.greater_equal(
        tf.reduce_max(scores, axis=1, keepdims=True),
        DOMAIN_CONFIG.GENERAL_SIMILARITY_THRESHOLD,
    )

    # do not update mask when rejecting categories
    condition = tf.logical_and(
        has_max_score_at_least_threshold,
        tf.logical_not(empty_description),
    )
    _, scores = apply_rule_to_scores(
        category_mask=tf.logical_or(
            category_masks["Other Expenses"],
            category_masks["Other Income"],
        ),
        condition_mask=condition,
        category_value=category_values[False],
        scores=scores,
        # No condition mask for this rule
    )
    # -------end apply threshold_for_uncategorized-------

    # -------Fallback rules-------
    # remaining transactions where:
    # -no business rule applied
    # will get Miscellaneous expenses category when debit transaction
    source = tf.where(
        condition=tf.squeeze(has_max_score_at_least_threshold),
        x=tf.constant("ModelPrediction", dtype=tf.dtypes.string),
        y=tf.constant("ModelFallback", dtype=tf.dtypes.string),
        name="category_source",
    )

    condition = tf.logical_and(
        debit_transactions, tf.logical_not(has_max_score_at_least_threshold)
    )
    mask, scores = apply_rule_to_scores(
        category_mask=category_masks["Other Expenses"],
        condition_mask=condition,
        category_value=category_values[True],
        scores=scores,
        mask=mask,
    )

    # will get Other Income category when credit transaction
    condition = tf.logical_and(
        credit_transactions, tf.logical_not(has_max_score_at_least_threshold)
    )
    mask, scores = apply_rule_to_scores(
        category_mask=category_masks["Other Income"],
        condition_mask=condition,
        category_value=category_values[True],
        scores=scores,
        mask=mask,
    )
    # -------End Fallback rules-------

    return mask, scores, source


@tf.function
def apply_rule_to_scores(
    *,
    category_mask: tf.Tensor,
    condition_mask: tf.Tensor,
    category_value: tf.Tensor,
    scores: tf.Tensor,
    mask: tf.Tensor = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Overwrite scores for given category based on provided condition

    :param category_mask: mask for given category
    :param condition_mask: mask for given condition
    :param category_value: value to be assigned for given category
    :param scores: model scores
    :param mask: mask for transactions where no rule has been applied yet default: None
    :return: Tuple[mask, modified scores]
    """
    if mask is not None:
        condition_mask = tf.logical_and(mask, condition_mask)
        mask = tf.logical_and(mask, tf.logical_not(condition_mask))

    scores = tf.where(
        tf.logical_and(category_mask, condition_mask),
        category_value,
        scores,
    )
    return mask, scores
