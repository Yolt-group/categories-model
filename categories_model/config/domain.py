from dataclasses import dataclass
from enum import Enum
from typing import List, AnyStr, Dict, Union


class ModelType(Enum):
    RETAIL_CATEGORIES_MODEL = "categories-model"
    SME_CATEGORIES_MODEL = "sme-categories-model"


@dataclass
class DomainConfig:
    # Generated using uuid.uuid4()
    TEST_DESCRIPTION_PREFIX: AnyStr = (
        "test_description:5cb92fb6-93c5-452f-aa27-53a485b8f370:"
    )
    TRAINING_LABELS: List[AnyStr] = None
    OUTGOING_TRAINING_LABELS: List[AnyStr] = None
    INCOMING_TRAINING_LABELS: List[AnyStr] = None
    EXCLUDED_OUTGOING_TRAINING_LABELS: List[AnyStr] = None
    EXCLUDED_INCOMING_TRAINING_LABELS: List[AnyStr] = None
    OUTGOING_CATEGORIES: List[AnyStr] = None
    INCOMING_CATEGORIES: List[AnyStr] = None
    OUTGOING_CATEGORIES_INDICES: Dict[AnyStr, int] = None
    INCOMING_CATEGORIES_INDICES: Dict[AnyStr, int] = None
    N_TRAINING_LABELS: int = None
    EXCLUDED_TRAINING_LABELS: List[AnyStr] = None
    N_EXCLUDED_TRAINING_LABELS: int = None
    CATEGORIES: List[AnyStr] = None
    N_CATEGORIES: int = None
    CATEGORY_INDICES: Dict[AnyStr, int] = None
    PREPROCESSED_NUMERIC_COLUMNS: List[AnyStr] = None
    N_NUMERIC_FEATURES: int = None
    PREPROCESSING_COLUMNS: List[AnyStr] = None
    POSTPROCESSING_COLUMNS: List[AnyStr] = None
    FEEDBACK_TABLES: Dict[AnyStr, List[AnyStr]] = None
    INPUT_COLUMNS: List[AnyStr] = None
    TARGET_COLUMN: AnyStr = None
    TARGET_COLUMN_INT: int = None
    TRAINING_DATA_FILE: AnyStr = None
    PRODUCTION_DATA_FILE: AnyStr = None
    TEST_SITE_ID: AnyStr = None
    PREPROCESSING_METADATA_FILE: AnyStr = None
    MODEL_ARTIFACT_FILE: AnyStr = None
    MODEL_METADATA_FILE: AnyStr = None
    USER_FEEDBACK_EVENTS_COLUMNS: Dict[AnyStr, Dict] = None
    TABLE_COLUMNS: Dict[AnyStr, Dict] = None
    COUNTRIES: List[AnyStr] = None
    START_TRAINING_DATE: AnyStr = None
    N_MODEL_SAMPLES_PER_COUNTRY: int = None
    N_PRODUCTION_SAMPLES: Union[int, float] = None
    N_VALIDATION_SAMPLES: Union[int, float] = None
    N_TEST_SAMPLES: Union[int, float] = None
    BATCH_SIZE: int = None
    SEQUENCE_LENGTH = None
    VOCABULARY_SIZE: int = None
    EMBEDDING_SIZE: int = None
    MAX_EPOCHS: int = None
    LEARNING_RATE: float = None
    MARGIN: float = None
    GENERAL_SIMILARITY_THRESHOLD: float = None
    MAX_RECALL_RANK: int = None
    WEIGHTED_METRICS_MIN_THRESHOLDS: Dict[AnyStr, float] = None
    COVERAGE_MIN_THRESHOLD: float = None
    PAY_LABEL_CATEGORY_MAPPING: Dict = None
    PAY_LABEL_CATEGORY_RULES: Dict = None
    MISSING_VALUES_REPLACEMENT: Dict = None
    MCC_FALLBACK_RULE_FILE: AnyStr = None
    LIST_OF_CLIENTS: Dict[AnyStr, List[AnyStr]] = None
    LIST_OF_DOMAINS: List[AnyStr] = None
    TABLE_DOMAIN_MAPPING: Dict[AnyStr, Dict[AnyStr, AnyStr]] = None
    CUCUMBER_TEST_SAMPLES: List[Dict] = None
    MODEL_TYPE: ModelType = None

    # Required to make config hashable ans serializable (yaml dump)
    def __hash__(self):
        return hash(repr(self))
