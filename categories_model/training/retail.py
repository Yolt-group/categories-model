import tensorflow as tf
import pandas as pd
from typing import Dict, Tuple

from categories_model.config.retail import Retail

DOMAIN_CONFIG = Retail()


class PostProcessingLayer(tf.keras.layers.Layer):
    """
    Layer which applies postprocessing logic and modifies model scores accordingly
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.MCC_LOOKUP_TABLE = load_mcc_table()

    def call(self, inputs, **kwargs):
        """
        Apply business rules
        IMPORTANT NOTE: in theory the business logic shouldn't conflict, but in practice it may happen since:
         - we apply rule one by one, so it might happen that assigning 1. based on the first rule may influence the other rule
           results where we check the threshold
         - there is additional business logic for Savings and Internal categories specified in retail.py which may
            also conflict with the results of business rules definition
         result: in any conflicting case, the model will return the first category with the highest score
        :return: a tuple containing: postprocessed scores and category source
        """
        mcc_category_indices = self.MCC_LOOKUP_TABLE.lookup(
            tf.reshape(inputs["bank_specific__mcc"], (-1, 1))
        )
        scores = inputs["scores"]

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
        category_values = {False: all_minus_ones, True: all_ones}
        category_masks = {
            category: all_trues & tf.equal(DOMAIN_CONFIG.CATEGORIES, category)
            for category in DOMAIN_CONFIG.CATEGORIES
        }

        # create mask for keeping track of transactions where already a rule has applied
        mask = tf.reduce_any(all_trues, axis=1, keepdims=True)

        mask, scores = apply_single_feature_business_rules(
            mask=mask,
            inputs=inputs,
            scores=scores,
            category_values=category_values,
            category_masks=category_masks,
        )

        mask, scores = apply_internal_savings_business_rules(
            mask=mask,
            inputs=inputs,
            scores=scores,
            all_trues=all_trues,
            category_values=category_values,
            category_masks=category_masks,
            category_indices=DOMAIN_CONFIG.CATEGORY_INDICES,
        )

        mask, scores = apply_remaining_business_rules(
            mask=mask,
            inputs=inputs,
            scores=scores,
            category_values=category_values,
            category_masks=category_masks,
            category_indices=DOMAIN_CONFIG.CATEGORY_INDICES,
        )

        mask, scores, source = apply_fallback_rules(
            mask=mask,
            scores=scores,
            inputs=inputs,
            mcc_category_indices=mcc_category_indices,
            category_values=category_values,
            category_masks=category_masks,
            debit_transactions=debit_transactions,
        )

        return scores, source

    def compute_output_shape(self, input_shape):
        return [
            ([input_shape["account_type"][0], DOMAIN_CONFIG.N_CATEGORIES]),
            ([input_shape["account_type"][0]]),
        ]


@tf.function
def add_uncategorized_column(scores: tf.Tensor) -> tf.Tensor:
    """
    Add placeholder for uncategorized column

    :param scores: the output of the model
    :return: scores with additional column
    """
    # the transformation scores[:, None, 0] is used instead of scores[:, 0] to keep the dimension of scores
    return tf.concat([scores, tf.clip_by_value(scores[:, None, 0], 0.0, 0.0)], axis=-1)


@tf.function
def apply_single_feature_business_rules(
    *,
    mask: tf.Tensor,
    inputs: Dict,
    scores: tf.Tensor,
    category_values: Dict,
    category_masks: Dict,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Modify scores based on the business rules - each rule depends on single input feature
    :param mask: mask with
        - True values for transactions where no business rule applied yet
        - False values for transactions that already have a category assigned due to a business rule
    :param inputs: dictionary with postprocessing input features
    :param scores: model scores
    :param category_values: dictionary with values assigned to True/False category
    :param category_masks: masks for each category
    :return: tuple containing: - modified mask for the categories where business rules have been applied
                               - modified scores for the categories included in business rules
    """

    # transactions on SAVINGS account type are considered Internal
    condition = tf.equal(tf.reshape(inputs["account_type"], (-1, 1)), "SAVINGS_ACCOUNT")
    mask, scores = apply_rule_to_scores(
        category_mask=category_masks["Internal"],
        condition_mask=condition,
        category_value=category_values[True],
        scores=scores,
        mask=mask,
    )

    # transactions on CURRENT_ACCOUNT with internal_transaction=="" cannot be Internal
    condition = tf.logical_and(
        tf.equal(tf.reshape(inputs["internal_transaction"], (-1, 1)), ""),
        tf.equal(tf.reshape(inputs["account_type"], (-1, 1)), "CURRENT_ACCOUNT"),
    )
    # do not update mask when rejecting categories
    _, scores = apply_rule_to_scores(
        category_mask=category_masks["Internal"],
        condition_mask=condition,
        category_value=category_values[False],
        scores=scores,
        mask=mask,
    )

    # debit transactions cannot be Income
    condition = tf.equal(tf.reshape(inputs["transaction_type"], (-1, 1)), "debit")
    # do not update mask when rejecting categories
    _, scores = apply_rule_to_scores(
        category_mask=category_masks["Income"],
        condition_mask=condition,
        category_value=category_values[False],
        scores=scores,
        mask=mask,
    )

    # ------Paylabel rules-------
    for pay_label, target_category in DOMAIN_CONFIG.PAY_LABEL_CATEGORY_RULES.items():
        condition = tf.strings.regex_full_match(
            tf.reshape(inputs["bank_specific__paylabels"], (-1, 1)), pay_label
        )
        mask, scores = apply_rule_to_scores(
            category_mask=category_masks[target_category],
            condition_mask=condition,
            category_value=category_values[True],
            scores=scores,
            mask=mask,
        )
    # ------End Paylabel rules-------

    return mask, scores


@tf.function
def apply_internal_savings_business_rules(
    *,
    mask: tf.Tensor,
    inputs: Dict,
    scores: tf.Tensor,
    all_trues: tf.Tensor,
    category_values: Dict,
    category_masks: Dict,
    category_indices: Dict,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Modify scores for Internal category
    - if the transaction is internal and best predicted category above threshold is not Savings, assign 1 for Internal
    - if the transaction is internal and best predicted category above threshold is Savings, assign -1 for Internal
    - note: for savings accounts the rule will not be used as above threshold for Savings due to business rule

    :param mask: mask with
        - True values for transactions where no business rule applied yet
        - False values for transactions that already have a category assigned due to a business rule
    :param inputs: dictionary with postprocessing input features
    :param scores: model scores
    :param all_trues: scores mask filled with True values
    :param category_values: dictionary with values assigned to True/False category
    :param category_masks: masks for each category
    :param category_indices: Dictionary that maps category string values to the corresponding list index number
    :return: tuple containing: - modified mask for the categories where business rules have been applied
                               - modified scores for the categories included in business rules
    """

    # if the account_type equals "CURRENT_ACCOUNT" and counter_account_type equals "SAVINGS_ACCOUNT"
    # then category should be Savings regardless whether it is a debit or credit transaction
    condition = tf.logical_and(
        tf.equal(tf.reshape(inputs["account_type"], (-1, 1)), "CURRENT_ACCOUNT"),
        tf.equal(
            tf.reshape(inputs["counter_account_type"], (-1, 1)), "SAVINGS_ACCOUNT"
        ),
    )
    mask, scores = apply_rule_to_scores(
        category_mask=category_masks["Savings"],
        condition_mask=condition,
        category_value=category_values[True],
        scores=scores,
        mask=mask,
    )

    # create a mask with True if Savings is the best predicted category
    savings_best_predicted_category = tf.reshape(
        tf.equal(tf.argmax(scores, axis=1), category_indices["Savings"]),
        (-1, 1),
    )

    has_max_score_below_threshold = tf.less(
        tf.reduce_max(scores, axis=1, keepdims=True),
        DOMAIN_CONFIG.GENERAL_SIMILARITY_THRESHOLD,
    )

    savings_predicted_mask = (
        all_trues
        # all False or all True for each row in batch where max score is NOT below_threshold
        & ~has_max_score_below_threshold
        # all False or all True for each row in batch where Savings is predicted
        & savings_best_predicted_category
    )

    # all False or all True for each row in batch where internal transaction field is not empty
    internal_transaction = all_trues & tf.not_equal(
        tf.reshape(inputs["internal_transaction"], (-1, 1)), ""
    )

    # if the transaction is internal and best predicted category above threshold is Savings, assign -1 for Internal
    condition = tf.logical_and(internal_transaction, savings_predicted_mask)
    # do not update mask when rejecting categories
    _, scores = apply_rule_to_scores(
        category_mask=category_masks["Internal"],
        condition_mask=condition,
        category_value=category_values[False],
        scores=scores,
        mask=mask,
    )

    # if the transaction is internal and best predicted category above threshold is not Savings, assign 1 for Internal
    condition = tf.logical_and(internal_transaction, ~savings_predicted_mask)
    mask, scores = apply_rule_to_scores(
        category_mask=category_masks["Internal"],
        condition_mask=condition,
        category_value=category_values[True],
        scores=scores,
        mask=mask,
    )

    return mask, scores


@tf.function
def apply_remaining_business_rules(
    *,
    mask: tf.Tensor,
    inputs: Dict,
    scores: tf.Tensor,
    category_values: Dict,
    category_masks: Dict,
    category_indices: Dict,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    There rules were originally in the scala code

    :param mask: mask with
        - True values for transactions where no business rule applied yet
        - False values for transactions that already have a category assigned due to a business rule
    :param inputs: dictionary with postprocessing input features
    :param scores: model scores
    :param category_values: dictionary with values assigned to True/False category
    :param category_masks: masks for each category
    :param category_indices: Dictionary that maps category string values to the corresponding list index number
    :return: tuple containing: - modified mask for the categories where business rules have been applied
                               - modified scores for the categories included in business rules
    """
    # ------Keep savings and internal if above similarity threshold--------
    has_max_score_below_threshold = tf.less(
        tf.reduce_max(scores, axis=1, keepdims=True),
        DOMAIN_CONFIG.GENERAL_SIMILARITY_THRESHOLD,
    )
    condition = tf.reshape(
        tf.equal(tf.argmax(scores, axis=1), category_indices["Savings"]), (-1, 1)
    )
    condition = tf.logical_and(condition, ~has_max_score_below_threshold)
    mask, scores = apply_rule_to_scores(
        category_mask=category_masks["Savings"],
        condition_mask=condition,
        category_value=category_values[True],
        scores=scores,
        mask=mask,
    )

    condition = tf.reshape(
        tf.equal(tf.argmax(scores, axis=1), category_indices["Internal"]), (-1, 1)
    )
    condition = tf.logical_and(condition, ~has_max_score_below_threshold)
    mask, scores = apply_rule_to_scores(
        category_mask=category_masks["Internal"],
        condition_mask=condition,
        category_value=category_values[True],
        scores=scores,
        mask=mask,
    )
    # ------End keep savings and internal if above similarity threshold--------

    # ----------------------------------------------
    # Remainder originates from scala business rules
    # ----------------------------------------------

    condition = tf.not_equal(tf.reshape(inputs["internal_transaction"], (-1, 1)), "")
    mask, scores = apply_rule_to_scores(
        category_mask=category_masks["Internal"],
        condition_mask=condition,
        category_value=category_values[True],
        scores=scores,
        mask=mask,
    )

    condition = tf.equal(tf.reshape(inputs["transaction_type"], (-1, 1)), "credit")
    mask, scores = apply_rule_to_scores(
        category_mask=category_masks["Income"],
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
    mcc_category_indices: tf.Tensor,
    category_values: Dict,
    category_masks: Dict,
    debit_transactions: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Modify scores based on the fallback rules
    :param mask: mask with
        - True values for transactions where no business rule applied yet
        - False values for transactions that already have a category assigned due to a business rule
    :param mcc_category_indices: tensor with -the mcc category index value for successful mcc lookup
                                             -value equals -1 when mcc lookup fails
    :param scores: model scores
    :param category_values: dictionary with values assigned to True/False category
    :param category_masks: masks for each category
    :return: tuple containing: - modified mask for the categories where business rules have been applied
                               - modified scores for the categories included in business rules
                               - category source tensor with prediction source (ModelPrediction or ModelFallback)
    """

    # -------Apply fallback rule for empty description--
    empty_description = tf.equal(tf.reshape(inputs["cleaned_description"], (-1, 1)), "")

    # if transaction type equals "debit" and cleaned description is empty
    # then category should be General
    condition = tf.logical_and(
        empty_description,
        debit_transactions,
    )
    mask, scores = apply_rule_to_scores(
        category_mask=category_masks["General"],
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
    source = tf.where(
        condition=tf.squeeze(has_max_score_at_least_threshold),
        x=tf.constant("ModelPrediction", dtype=tf.dtypes.string),
        y=tf.constant("ModelFallback", dtype=tf.dtypes.string),
        name="category_source",
    )
    # do not update mask when rejecting categories
    condition = tf.logical_and(
        has_max_score_at_least_threshold,
        tf.logical_not(empty_description),
    )

    _, scores = apply_rule_to_scores(
        category_mask=category_masks["General"],
        condition_mask=condition,
        category_value=category_values[False],
        scores=scores,
        # No condition mask for this rule
    )
    mask = tf.logical_and(mask, tf.logical_not(has_max_score_at_least_threshold))
    # -------end apply threshold_for_uncategorized-------

    # -------Fallback rules-------

    condition = tf.math.greater(mcc_category_indices, -1)
    mcc_categories = tf.squeeze(
        tf.one_hot(
            indices=mcc_category_indices,
            depth=scores.shape[1],
        ),
    )
    mask, scores = apply_rule_to_scores(
        category_mask=tf.equal(mcc_categories, 1),
        condition_mask=condition,
        category_value=category_values[True],
        scores=scores,
        mask=mask,
    )

    # remaining transactions where:
    # -no business rule applied
    # -mcc codes did not resolve
    # will get General category
    mask, scores = apply_rule_to_scores(
        category_mask=category_masks["General"],
        condition_mask=mask,
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


# load mcc fallback code file and convert to Tensorflow lookup table
def load_mcc_table() -> tf.lookup.StaticHashTable:
    mcc_table = pd.read_csv(
        DOMAIN_CONFIG.MCC_FALLBACK_RULE_FILE,
        sep=";",
        names=["code", "category"],
        dtype=str,
    )
    mcc_table = pd.merge(
        mcc_table,
        pd.DataFrame(enumerate(DOMAIN_CONFIG.CATEGORIES), columns=["id", "category"]),
    )
    keys_tensor = tf.constant(mcc_table.code, dtype=tf.dtypes.string)
    vals_tensor = tf.constant(mcc_table.id, dtype=tf.dtypes.int32)
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
        default_value=-1,
    )
