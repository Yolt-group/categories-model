from .mapping_dictionary import category_counterparty_map
import pandas as pd

counterparty_category_map = {}

for category, counterparties in category_counterparty_map.items():
    for counterparty in counterparties:
        counterparty_category_map[counterparty] = category


def mapper(counterparty: str):
    kamajis_answer = counterparty_category_map.get(counterparty, None)
    return kamajis_answer


def series_map(counterparty_names: pd.Series):
    kamajis_answers = counterparty_names.map(counterparty_category_map)
    return kamajis_answers
