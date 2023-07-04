from abc import ABCMeta
from dataclasses import dataclass
from categories_model.config.domain import DomainConfig, ModelType


@dataclass
class SME(DomainConfig, metaclass=ABCMeta):
    # Required to make config hashable ans serializable (yaml dump)
    def __hash__(self):
        return hash(repr(self))

    def __init__(self):
        super().__init__()

        self.MODEL_TYPE = ModelType.SME_CATEGORIES_MODEL

        # ---------------------------
        # Target label settings
        # ---------------------------
        self.OUTGOING_TRAINING_LABELS = [
            "Interest and Repayments",
            "Investments",
            "Food and Drinks",
            "Vehicles and Driving Expenses",
            "Rent and Facilities",
            "Travel Expenses",
            "Marketing and Promotion",
            "Other Operating Costs",
            "Utilities",
            "Collection Costs",
            "Pension Payments",
            "Salaries",
            "Corporate Savings Deposits",
            "Equity Withdrawal",
            "Unspecified Tax",
        ]

        self.INCOMING_TRAINING_LABELS = [
            "Tax Returns",
            "Equity Financing",
            "Loans",
            "Revenue",
        ]
        self.TRAINING_LABELS = (
            self.OUTGOING_TRAINING_LABELS + self.INCOMING_TRAINING_LABELS
        )
        self.N_TRAINING_LABELS = len(self.TRAINING_LABELS)

        # NOTE: due to mapping categories to integers, excluded labels must be assigned to the highest number
        self.EXCLUDED_OUTGOING_TRAINING_LABELS = [
            "Sales Tax",
            "Payroll Tax",
            "Corporate Income Tax",
            "Other Expenses",
        ]
        self.EXCLUDED_INCOMING_TRAINING_LABELS = ["Other Income"]
        self.EXCLUDED_TRAINING_LABELS = (
            self.EXCLUDED_OUTGOING_TRAINING_LABELS
            + self.EXCLUDED_INCOMING_TRAINING_LABELS
        )
        self.N_EXCLUDED_TRAINING_LABELS = len(self.EXCLUDED_TRAINING_LABELS)
        self.OUTGOING_CATEGORIES = (
            self.OUTGOING_TRAINING_LABELS + self.EXCLUDED_OUTGOING_TRAINING_LABELS
        )
        self.INCOMING_CATEGORIES = (
            self.INCOMING_TRAINING_LABELS + self.EXCLUDED_INCOMING_TRAINING_LABELS
        )
        self.CATEGORIES = self.TRAINING_LABELS + self.EXCLUDED_TRAINING_LABELS
        self.N_CATEGORIES = len(self.CATEGORIES)

        self.CATEGORY_INDICES = {
            category: index for index, category in enumerate(self.CATEGORIES)
        }

        self.OUTGOING_CATEGORIES_INDICES = {
            category: index
            for category, index in self.CATEGORY_INDICES.items()
            if category in self.OUTGOING_CATEGORIES
        }

        self.INCOMING_CATEGORIES_INDICES = {
            category: index
            for category, index in self.CATEGORY_INDICES.items()
            if category in self.INCOMING_CATEGORIES
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
            "bank_counterparty_name",
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

        # These are empty as SME training data is generated in preprocessing
        self.FEEDBACK_TABLES = {
            "yts": [],
            "yoltapp": [],
        }

        self.INPUT_COLUMNS = list(
            set(self.PREPROCESSING_COLUMNS + self.POSTPROCESSING_COLUMNS)
        )
        self.TARGET_COLUMN = "target_category"
        self.TARGET_COLUMN_INT = self.TARGET_COLUMN + "_int"

        # ---------------------------
        # App configuration
        # ---------------------------
        self.TRAINING_DATA_FILE = "preprocessed_training_data_yoltapp_yts"
        self.PRODUCTION_DATA_FILE = "preprocessed_production_data_yoltapp_yts"
        self.TEST_SITE_ID = "e278a008-bf45-4d19-bb5d-b36ff755be58"
        self.PREPROCESSING_METADATA_FILE = "preprocessing_metadata.yaml"
        self.MODEL_ARTIFACT_FILE = "categories-model-yoltapp-yts"
        self.MODEL_METADATA_FILE = "training_metadata.yaml"

        # ---------------------------
        # Preprocessing settings
        # ---------------------------

        # ---------------------------
        # Input data settings
        # ---------------------------
        #   columns contains column names, required for all tables
        #   if some column names should be renamed, these are stored in aliases
        self.FEEDBACK_COLUMNS = [
            "user_id",
            "account_id",
            "transaction_id",
            "pending",
            "date",
            "feedback_time",
            "category",
        ]
        self._USERS_COLUMNS = {
            "columns": ["id", "country_code", "client_id"],
            "aliases": {"id": "user_id"},
        }

        self._ACCOUNTS_COLUMNS = {
            "columns": ["id", "user_id", "deleted", "type"],
            "aliases": {"id": "account_id", "type": "account_type"},
        }
        self._TRANSACTIONS_COLUMMS = {
            "columns": [
                "user_id",
                "account_id",
                "transaction_id",
                "date",
                "category",
                "pending",
                "description",
                "internal_transaction",
                "transaction_type",
                "amount",
                "bank_counterparty_name",
            ]
        }

        self.TABLE_COLUMNS = {
            "users_app": self._USERS_COLUMNS,
            "accounts_app": self._ACCOUNTS_COLUMNS,
            "transactions_app": self._TRANSACTIONS_COLUMMS,
            "test_users_app": {"columns": ["user_id"]},
            "users_yts": self._USERS_COLUMNS,
            "accounts_yts": self._ACCOUNTS_COLUMNS,
            "transactions_yts": self._TRANSACTIONS_COLUMMS,
            "synthetic_feedback_yts": {"columns": self.FEEDBACK_COLUMNS},
            "synthetic_feedback_yoltapp": {"columns": self.FEEDBACK_COLUMNS},
        }

        self.TABLE_DOMAIN_MAPPING = {
            "yoltapp": {
                "users": "users_app",
                "test_users": "test_users_app",
                "accounts": "accounts_app",
                "transactions": "transactions_app",
            },
            "yts": {
                "users": "users_yts",
                "test_users": None,
                "accounts": "accounts_yts",
                "transactions": "transactions_yts",
            },
        }

        # ---------------------------
        # Training data filters
        # ---------------------------
        self.COUNTRIES = ["GB", "NL"]
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

        self.GENERAL_SIMILARITY_THRESHOLD = 0.65

        # Performance metrics configuration
        self.MAX_RECALL_RANK = 3
        self.WEIGHTED_METRICS_MIN_THRESHOLDS = {
            "precision": 0.60,
            "recall": 0.60,
            "f1-score": 0.60,
        }
        self.COVERAGE_MIN_THRESHOLD = 0.80

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
            "bank_counterparty_iban": "",
        }

        self.LIST_OF_CLIENTS = {
            "yoltapp": ["297ecda4-fd60-4999-8575-b25ad23b249c"],
            "yts": ["3e3aae2f-e632-4b78-bdf8-2bf5e5ded17e"],
        }

        self.LIST_OF_DOMAINS = [
            "yts",
            "yoltapp",
        ]

        self.CUCUMBER_TEST_SAMPLES = [
            # Outgoing Categories:
            {
                "description": "storting prive",
                "amount": 50.0,
                "transaction_type": "debit",
                "account_type": "CURRENT_ACCOUNT",
                self.TARGET_COLUMN: "Equity Withdrawal",
            },
            {
                "description": "naar zakelijke spaarrekening",
                "amount": 5000.0,
                "transaction_type": "debit",
                "account_type": "CURRENT_ACCOUNT",
                self.TARGET_COLUMN: "Corporate Savings Deposits",
            },
            {
                "description": "annual dividend payment",
                "amount": 50.0,
                "transaction_type": "debit",
                "account_type": "CURRENT_ACCOUNT",
                self.TARGET_COLUMN: "Equity Withdrawal",
            },
            {
                "description": "Bank Kosten",
                "amount": 2.0,
                "transaction_type": "debit",
                "account_type": "CURRENT_ACCOUNT",
                self.TARGET_COLUMN: "Other Operating Costs",
            },
            {
                "description": "lening rente",
                "amount": 2.0,
                "transaction_type": "debit",
                "account_type": "CURRENT_ACCOUNT",
                self.TARGET_COLUMN: "Interest and Repayments",
            },
            {
                "description": "parcel shipment",
                "amount": 2.0,
                "transaction_type": "debit",
                "account_type": "CURRENT_ACCOUNT",
                self.TARGET_COLUMN: "Other Operating Costs",
            },
            {
                "description": "zakelijk belasting",
                "bank_counterparty_name": "belastingdienst",
                "amount": 200.0,
                "transaction_type": "debit",
                "account_type": "CURRENT_ACCOUNT",
                self.TARGET_COLUMN: "Unspecified Tax",
            },
            {
                "description": "tankstation",
                "amount": 20.0,
                "transaction_type": "debit",
                "account_type": "CURRENT_ACCOUNT",
                self.TARGET_COLUMN: "Vehicles and Driving Expenses",
            },
            {
                "description": "zakelijk reiskosten",
                "amount": 20.0,
                "transaction_type": "debit",
                "account_type": "CURRENT_ACCOUNT",
                self.TARGET_COLUMN: "Travel Expenses",
            },
            {
                "description": "electric en energie kosten maand",
                "amount": 80.0,
                "transaction_type": "debit",
                "account_type": "CURRENT_ACCOUNT",
                self.TARGET_COLUMN: "Utilities",
            },
            {
                "description": "middageten restaurant",
                "amount": 20.0,
                "transaction_type": "debit",
                "account_type": "CURRENT_ACCOUNT",
                self.TARGET_COLUMN: "Food and Drinks",
            },
            {
                "description": "gerechtsdeurwaarders",
                "amount": 20.0,
                "transaction_type": "debit",
                "account_type": "CURRENT_ACCOUNT",
                self.TARGET_COLUMN: "Collection Costs",
            },
            {
                "description": "salaris",
                "amount": 200.0,
                "transaction_type": "debit",
                "account_type": "CURRENT_ACCOUNT",
                self.TARGET_COLUMN: "Salaries",
            },
            {
                "description": "pensioen bijdrage",
                "amount": 200.0,
                "transaction_type": "debit",
                "account_type": "CURRENT_ACCOUNT",
                self.TARGET_COLUMN: "Pension Payments",
            },
            {
                "description": "google ads",
                "amount": 200.0,
                "transaction_type": "debit",
                "account_type": "CURRENT_ACCOUNT",
                self.TARGET_COLUMN: "Marketing and Promotion",
            },
            # Incoming categories
            {
                "description": "etentje",
                "amount": 2000.0,
                "transaction_type": "credit",
                "account_type": "CURRENT_ACCOUNT",
                self.TARGET_COLUMN: "Other Income",
            },
            {
                "description": "interest paid after tax 0.0 deducted",
                "amount": 10.0,
                "transaction_type": "credit",
                "account_type": "CURRENT_ACCOUNT",
                self.TARGET_COLUMN: "Other Income",
            },
            {
                "description": "sparen storting",
                "amount": 200.0,
                "transaction_type": "credit",
                "account_type": "CURRENT_ACCOUNT",
                self.TARGET_COLUMN: "Equity Financing",
            },
            {
                "description": "prive storting",
                "amount": 200.0,
                "transaction_type": "credit",
                "account_type": "CURRENT_ACCOUNT",
                self.TARGET_COLUMN: "Equity Financing",
            },
            {
                "description": "invoice",
                "amount": 200.0,
                "transaction_type": "credit",
                "account_type": "CURRENT_ACCOUNT",
                self.TARGET_COLUMN: "Revenue",
            },
            {
                "description": "teruggaaf",
                "bank_counterparty_name": "belastingdienst",
                "amount": 200.0,
                "transaction_type": "credit",
                "account_type": "CURRENT_ACCOUNT",
                "bank_counterparty_iban": "NL58INGB0649306597",
                self.TARGET_COLUMN: "Tax Returns",
            },
            # Test business rules
            # Credit with empty description
            {
                "description": "",
                "amount": 20.0,
                "transaction_type": "credit",
                "account_type": "CURRENT_ACCOUNT",
                self.TARGET_COLUMN: "Other Income",
            },
            # Debit with empty description
            {
                "description": "",
                "amount": 20.0,
                "transaction_type": "debit",
                "account_type": "CURRENT_ACCOUNT",
                self.TARGET_COLUMN: "Other Expenses",
            },
            # Equity Financing
            {
                "description": "",
                "amount": 20.0,
                "transaction_type": "credit",
                "account_type": "CURRENT_ACCOUNT",
                "counter_account_type": "SAVINGS_ACCOUNT",
                self.TARGET_COLUMN: "Equity Financing",
            },
            # Debit with empty description
            {
                "description": "",
                "amount": 20.0,
                "transaction_type": "debit",
                "account_type": "CURRENT_ACCOUNT",
                "counter_account_type": "SAVINGS_ACCOUNT",
                self.TARGET_COLUMN: "Corporate Savings Deposits",
            },
            # Combination of description and bank_counterparty_name should be concatenated and trigger test mode.
            # The test mode should output given the category "Payroll Tax"
            {
                "description": DomainConfig().TEST_DESCRIPTION_PREFIX,
                "bank_counterparty_name": "Payroll Tax",
                "amount": 20.0,
                "transaction_type": "debit",
                "account_type": "CURRENT_ACCOUNT",
                "country_code": "NL",
                "counter_account_type": "SAVINGS_ACCOUNT",
                self.TARGET_COLUMN: "Payroll Tax",
            },
            # Test mode should output given category
            {
                "description": f"{DomainConfig().TEST_DESCRIPTION_PREFIX}Other Income",
                "amount": 20.0,
                "transaction_type": "credit",
                "account_type": "CURRENT_ACCOUNT",
                "country_code": "NL",
                "counter_account_type": "SAVINGS_ACCOUNT",
                self.TARGET_COLUMN: "Other Income",
            },
        ]
