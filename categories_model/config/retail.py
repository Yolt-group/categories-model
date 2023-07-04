import re
from abc import ABCMeta
from dataclasses import dataclass
from pathlib import Path

from categories_model.config.domain import DomainConfig, ModelType


@dataclass
class Retail(DomainConfig, metaclass=ABCMeta):
    # Required to make config hashable and serializable (yaml dump)
    def __hash__(self):
        return hash(repr(self))

    def __init__(self):
        super().__init__()
        self.MODEL_TYPE = ModelType.RETAIL_CATEGORIES_MODEL

        # ---------------------------
        # Target label settings
        # ---------------------------
        self.TRAINING_LABELS = [
            "Housing",
            "Personal care",
            "Groceries",
            "Eating out",
            "Shopping",
            "Travel",
            "Transport",
            "Bills",
            "Transfers",
            "Cash",
            "Leisure",
            "Internal",
            "Income",
            "Charity",
            "Coffee",
            "Drinks",
            "Education",
            "Expenses",
            "Investments",
            "Lunch",
            "Gifts",
            "Kids",
            "Takeaway",
            "Petrol",
            "Rent",
            "Mortgage",
            "Utilities",
            "Vehicle",
            "Pets",
            "Savings",
        ]
        self.N_TRAINING_LABELS = len(self.TRAINING_LABELS)

        # NOTE: due to mapping categories to integers, excluded labels must be assigned to the highest number
        self.EXCLUDED_TRAINING_LABELS = ["General"]
        self.N_EXCLUDED_TRAINING_LABELS = len(self.EXCLUDED_TRAINING_LABELS)
        self.CATEGORIES = self.TRAINING_LABELS + self.EXCLUDED_TRAINING_LABELS
        self.N_CATEGORIES = len(self.CATEGORIES)

        self.CATEGORY_INDICES = {
            category: index for index, category in enumerate(self.CATEGORIES)
        }

        # ---------------------------
        # Model input settings
        # ---------------------------
        self.PREPROCESSED_NUMERIC_COLUMNS = [
            "scaled_amount",
            "is_debit_transaction",
            "is_internal_transaction",
        ]
        self.N_NUMERIC_FEATURES = len(self.PREPROCESSED_NUMERIC_COLUMNS)
        self.PREPROCESSING_COLUMNS = [
            "description",
            "amount",
            "transaction_type",
            "internal_transaction",
        ]
        self.POSTPROCESSING_COLUMNS = [
            "internal_transaction",
            "bank_specific__paylabels",
            "bank_specific__mcc",
            "transaction_type",
            "account_type",
            "counter_account_type",
        ]

        self.FEEDBACK_TABLES = {
            "yoltapp": [
                "user_single_feedback_created",
                "user_multiple_feedback_created",
                "user_multiple_feedback_applied",
                "historical_feedback",
            ],
        }

        self.INPUT_COLUMNS = list(
            set(self.PREPROCESSING_COLUMNS + self.POSTPROCESSING_COLUMNS)
        )
        self.TARGET_COLUMN = "target_category"
        self.TARGET_COLUMN_INT = self.TARGET_COLUMN + "_int"

        # ---------------------------
        # App configuration
        # ---------------------------
        self.TRAINING_DATA_FILE = "preprocessed_training_data"
        self.PRODUCTION_DATA_FILE = "preprocessed_production_data"
        self.TEST_SITE_ID = "e278a008-bf45-4d19-bb5d-b36ff755be58"
        self.PREPROCESSING_METADATA_FILE = "preprocessing_metadata.yaml"
        self.MODEL_ARTIFACT_FILE = "categories-model"
        self.MODEL_METADATA_FILE = "training_metadata.yaml"

        # ---------------------------
        # Preprocessing settings
        # ---------------------------

        # ---------------------------
        # Input data settings
        # ---------------------------
        #   columns contains column names, required for all tables
        #   if some column names should be renamed, these are stored in aliases
        self.USER_FEEDBACK_EVENTS_COLUMNS = {
            "columns": [
                "id__userId",
                "id__accountId",
                "id__transactionId",
                "id__localDate",
                "id__pendingType",
                "kafka__timestamp",
                "fact__category",
            ],
            "aliases": {
                "id__userId": "user_id",
                "id__accountId": "account_id",
                "id__transactionId": "transaction_id",
                "id__localDate": "date",
                "id__pendingType": "pending",
                "kafka__timestamp": "feedback_time",
                "fact__category": "category",
            },
        }
        self.TABLE_COLUMNS = {
            "users": {
                "columns": ["id", "country_code", "client_id"],
                "aliases": {"id": "user_id"},
            },
            "test_users": {"columns": ["user_id"]},
            "accounts": {
                "columns": ["id", "user_id", "deleted", "type"],
                "aliases": {"id": "account_id", "type": "account_type"},
            },
            "transactions": {
                "columns": [
                    "user_id",
                    "account_id",
                    "transaction_id",
                    "date",
                    "pending",
                    "description",
                    "internal_transaction",
                    "transaction_type",
                    "amount",
                    "bank_specific",
                ]
            },
            "historical_feedback": {
                "columns": [
                    "user_id",
                    "account_id",
                    "transaction_id",
                    "pending",
                    "date",
                    "feedback_time",
                    "category",
                ]
            },
            "user_single_feedback_created": self.USER_FEEDBACK_EVENTS_COLUMNS,
            "user_multiple_feedback_created": self.USER_FEEDBACK_EVENTS_COLUMNS,
            "user_multiple_feedback_applied": self.USER_FEEDBACK_EVENTS_COLUMNS,
        }

        self.TABLE_DOMAIN_MAPPING = {
            "yoltapp": {
                "users": "users",
                "test_users": "test_users",
                "accounts": "accounts",
                "transactions": "transactions",
            }
        }

        # ---------------------------
        # Training data filters
        # ---------------------------
        self.COUNTRIES = ["GB", "FR", "IT", "NL"]
        self.START_TRAINING_DATE = (
            "2018-01-01"  # NOTE: start training date may change over time
        )
        self.N_MODEL_SAMPLES_PER_COUNTRY = 3000000
        self.N_PRODUCTION_SAMPLES = 100000

        # ---------------------------
        # Training settings
        # ---------------------------

        # Data samples size configuration
        self.N_VALIDATION_SAMPLES = 0.1
        self.N_TEST_SAMPLES = 0.1

        # Model parameters
        self.BATCH_SIZE = 512
        self.SEQUENCE_LENGTH = 60
        self.VOCABULARY_SIZE = 600000
        self.EMBEDDING_SIZE = 32

        self.MAX_EPOCHS = 40
        self.LEARNING_RATE = 0.01
        self.MARGIN = 0.1

        self.GENERAL_SIMILARITY_THRESHOLD = 0.45

        # Performance metrics configuration
        self.MAX_RECALL_RANK = 3
        self.WEIGHTED_METRICS_MIN_THRESHOLDS = {
            "precision": 0.60,
            "recall": 0.60,
            "f1-score": 0.60,
        }
        self.COVERAGE_MIN_THRESHOLD = 0.80

        # ---------------------------
        # Postprocessing business rules
        # ---------------------------
        self.PAY_LABEL_CATEGORY_MAPPING = {
            "[!RO]": "Internal",
            "[!TP]": "Internal",
            "[!DC]": "General",
            "[!DF]": "General",
            "[!DW]": "General",
            "[!DA]": "General",
            "[!DO]": "General",
            "[!DT]": "Internal",
            "[!XG]": "General",
            "[!RW]": "Income",
            "[!RF]": "Income",
            "[!EM]": "Internal",
            "[!ES]": "Savings",
        }
        # note that each key is concatenated with ".*" since pay labels field can have multiple labels and we use regex match
        #   to find particular value in a string
        self.PAY_LABEL_CATEGORY_RULES = {
            f".*{re.escape(key)}.*": value
            for key, value in self.PAY_LABEL_CATEGORY_MAPPING.items()
        }

        self.MISSING_VALUES_REPLACEMENT = {
            "description": "",
            "transaction_type": "",
            "account_type": "CURRENT_ACCOUNT",
            "amount": 0.0,
            "internal_transaction": "",
            "bank_specific__paylabels": "",
            "bank_specific__mcc": "",
            "counter_account_type": "CURRENT_ACCOUNT",
            "bank_counterparty_name": "",
        }

        self.MCC_FALLBACK_RULE_FILE = Path(__file__).parent.parent / Path(
            "training/mcc_codes.csv"
        )

        self.LIST_OF_CLIENTS = {"yoltapp": ["297ecda4-fd60-4999-8575-b25ad23b249c"]}

        self.LIST_OF_DOMAINS = ["yoltapp"]

        self.CUCUMBER_TEST_SAMPLES = [
            {
                "description": "EE & T-MOBILE Debit ON 03 APR BCC",
                "amount": 50.0,
                "transaction_type": "debit",
                self.TARGET_COLUMN: "Bills",
            },
            {
                "description": "VODAFONE LTD",
                "amount": 100.0,
                "transaction_type": "debit",
                self.TARGET_COLUMN: "Bills",
            },
            {
                "description": "Monthly Account Fee",
                "amount": 10.0,
                "transaction_type": "debit",
                self.TARGET_COLUMN: "Bills",
            },
            {
                "description": "ikea",
                "amount": 3000.0,
                "transaction_type": "debit",
                self.TARGET_COLUMN: "Housing",
            },
            {
                "description": "AMAZON UK MARKETPLACE",
                "amount": 300.0,
                "transaction_type": "debit",
                self.TARGET_COLUMN: "Shopping",
            },
            {
                "description": "SAINSBURY'S SUPERMARKET LONDON",
                "amount": 230.0,
                "transaction_type": "debit",
                self.TARGET_COLUMN: "Groceries",
            },
            {
                "description": "ATM Withdrawal LINK - ATM Withdrawal LINK",
                "amount": 2000.0,
                "transaction_type": "debit",
                self.TARGET_COLUMN: "Cash",
            },
            {
                "description": "?????",
                "amount": 1000.0,
                "transaction_type": "debit",
                self.TARGET_COLUMN: "General",
            },
            {
                "description": "canbeanything",
                "amount": 10.0,
                "transaction_type": "debit",
                "account_type": "SAVINGS_ACCOUNT",
                self.TARGET_COLUMN: "Internal",
            },
            {
                "description": "canbeanythingagain",
                "amount": 10.0,
                "transaction_type": "credit",
                "account_type": "SAVINGS_ACCOUNT",
                self.TARGET_COLUMN: "Internal",
            },
            {
                "description": "Round up",
                "bank_specific__paylabels": "[!RO]",
                "amount": 0.01,
                "transaction_type": "credit",
                self.TARGET_COLUMN: "Internal",
            },
            {
                "description": "Money Jar Savings",
                "bank_specific__paylabels": "[!ES]",
                "amount": 100,
                "transaction_type": "credit",
                self.TARGET_COLUMN: "Savings",
            },
            {
                "description": "Dispute missed top-up",
                "bank_specific__paylabels": "[!DT]",
                "amount": 5,
                "transaction_type": "credit",
                self.TARGET_COLUMN: "Internal",
            },
            {
                "description": "Reward",
                "bank_specific__paylabels": "[!RW]",
                "amount": 50,
                "transaction_type": "credit",
                "internal_transaction": "id123",
                self.TARGET_COLUMN: "Income",
            },
            {
                "description": "Referrals",
                "bank_specific__paylabels": "[!RF]",
                "amount": 50,
                "transaction_type": "credit",
                self.TARGET_COLUMN: "Income",
            },
            {
                "description": "Empty the money jar towards main account",
                "bank_specific__paylabels": "[!EM]",
                "amount": 50,
                "transaction_type": "credit",
                "internal_transaction": "id123",
                self.TARGET_COLUMN: "Internal",
            },
            {
                "description": "Empty the money jar towards savings",
                "bank_specific__paylabels": "[!ES]",
                "amount": 50,
                "transaction_type": "credit",
                "internal_transaction": "id123",
                self.TARGET_COLUMN: "Savings",
            },
            {
                "description": "Transfer to OLEKSII",
                "amount": 123.0,
                "transaction_type": "debit",
                self.TARGET_COLUMN: "Transfers",
            },
            {
                "description": "Avro Energy",
                "amount": 100.02,
                "transaction_type": "debit",
                self.TARGET_COLUMN: "Utilities",
            },
        ]
