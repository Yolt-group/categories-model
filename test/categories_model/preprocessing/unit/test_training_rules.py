import itertools

from categories_model.preprocessing.rules import (
    inter_domain_rule_conflicts,
    positive_negative_conflicts,
    cross_domain_rule_conflicts,
)

from categories_model.preprocessing.yoltapp_rules import CATEGORY_RULES as YOLTAPP_RULES
from categories_model.preprocessing.yts_rules import CATEGORY_RULES as YTS_RULES

ALL_RULES = (YTS_RULES, YOLTAPP_RULES)


def test_positive_negative_conflicts():
    for rule in ALL_RULES:
        found_conflicts = positive_negative_conflicts(rule)
        print(*found_conflicts, sep="\n")
        assert not any(found_conflicts)


def test_internal_rule_conflicts():
    for rule in ALL_RULES:
        found_conflicts = inter_domain_rule_conflicts(rule)
        print(*found_conflicts, sep="\n")
        assert not any(found_conflicts)


def test_cross_domain_conflicts():
    # Iterate over all rule set pairs (currently YTS and YOLTAPP)
    for rules_tuple in itertools.combinations(ALL_RULES, 2):
        found_conflicts = cross_domain_rule_conflicts(*rules_tuple)
        print(*found_conflicts, sep="\n")
        assert not any(found_conflicts)
