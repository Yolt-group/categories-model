from typing import AnyStr, Optional

from categories_model.config.domain import ModelType, DomainConfig
from categories_model.config.sme import SME
from categories_model.config.retail import Retail


DOMAIN_DICT = {
    ModelType.RETAIL_CATEGORIES_MODEL.value: Retail(),
    ModelType.SME_CATEGORIES_MODEL.value: SME(),
}


def get_domain_config(model_name: AnyStr) -> Optional[DomainConfig]:
    return DOMAIN_DICT.get(model_name)
