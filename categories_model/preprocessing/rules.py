import re
import itertools
import yaml

from categories_model.preprocessing.yoltapp_rules import (
    CATEGORY_RULES as YOLTAPP_CATEGORY_RULES,
)
from categories_model.preprocessing.yts_rules import (
    CATEGORY_RULES as YTS_CATEGORY_RULES,
)


CATEGORY_RULES = dict(yts=YTS_CATEGORY_RULES, yoltapp=YOLTAPP_CATEGORY_RULES)


def _merge_description_counterparty_lists(rule):
    def strip_regex(items):
        return set(map(lambda s: re.sub(r"\\b|[\^\$]", "", s), items))

    description_positive = rule["description"]["+"]
    counterparty_positive = rule["counterparty"]["+"]
    description_negative = rule["description"]["-"]
    counterparty_negative = rule["counterparty"]["-"]

    return (
        strip_regex(description_positive).union(strip_regex(counterparty_positive)),
        strip_regex(description_negative).union(strip_regex(counterparty_negative)),
    )


def _split_debit_credit_keys(rule):
    for transaction_type in ("debit", "credit"):
        yield {k for k, v in rule.items() if v["transaction_type"] == transaction_type}


def positive_negative_conflicts(rules):
    for k, v in rules.items():
        pos, neg = _merge_description_counterparty_lists(v)
        i = pos.intersection(neg)
        if bool(i):
            yield f"description + and - list overlap: {k}, {yaml.dump(list(i))}"


def inter_domain_rule_conflicts(rules):
    # Split in debit and credit subsets
    for keys in _split_debit_credit_keys(rules):
        # For each subset check for conflicts
        for k1, k2 in itertools.combinations(keys, 2):
            v1 = rules[k1]
            v2 = rules[k2]
            v1_pos, v1_neg = _merge_description_counterparty_lists(v1)
            v2_pos, v2_neg = _merge_description_counterparty_lists(v2)
            i = v1_pos.intersection(v2_pos)
            if bool(i):
                yield f"({k1}+, {k2}+): overlap: {yaml.dump(list(i))}"


def cross_domain_rule_conflicts(rules_a, rules_b):
    # Split rules into debit and credit subset pairs
    # eg. (debit_keys_a, debit_keys_b), (credit_keys_a, credit_keys_b)
    subset_key_tuples = zip(
        _split_debit_credit_keys(rules_a), _split_debit_credit_keys(rules_b)
    )

    # Iterate over debit-rule pairs and credit-rule pairs
    for keys_tuples in subset_key_tuples:
        # Check for overlap in each category combination
        for k1, k2 in itertools.product(*keys_tuples):
            # Only check different categories
            if k1 != k2:
                v1 = rules_a[k1]
                v2 = rules_b[k2]
                v1_pos, _ = _merge_description_counterparty_lists(v1)
                v2_pos, _ = _merge_description_counterparty_lists(v2)
                i = v1_pos.intersection(v2_pos)
                if bool(i):
                    yield f"({k1}+, {k2}+): overlap: {yaml.dump(list(i))}"
