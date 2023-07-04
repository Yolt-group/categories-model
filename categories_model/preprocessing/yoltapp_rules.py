from categories_model.preprocessing.domain_data import YoltApp

YOLTAPP_DOMAIN = YoltApp()

CATEGORY_RULES = {
    "Equity Withdrawal": {
        "transaction_type": "debit",
        "description": {
            "+": [
                "dividend",
                "divident",
                "dividends",
                r"private.*transfer",
                r"transfer.*private",
                r"owner.*deposit",
            ],
            "-": [
                r"\bwages\b",
                r"\bwage\b",
                "salary",
                "salaries",
            ],
        },
        "counterparty": {
            "+": [],
            "-": [],
        },
    },
    "Unspecified Tax": {
        "transaction_type": "debit",
        "description": {
            "+": [
                # goverment tax
                "self assess",
                "hmrc",
                # vat
                r"\bvat\b",
                "vehicle tax",
            ],
            "-": [],
        },
        "counterparty": {
            "+": [],
            "-": [],
        },
    },
    "Vehicles and Driving Expenses": {
        "transaction_type": "debit",
        "description": {
            "+": [
                # general
                "carwash",
                "vehicle",
                r"\bparkin",  # replacing 'parking' keyword
                r"\bmotors\b",
                r"\bmotor\b",
                "rent a car",
                r"\bpump\b",
                "car park",
                "fuel",  # this includes also when paying for employee expenses
                # car brands
                r"\btesla\b",
                r"\bbmw\b",
                r"\bvolkswagen\b",
                r"\bmercedes\b",
                # fuel
                "shell",
                r"\besso\b",
                "bp gerrards cross",
                "texaco",
                r"\bgulf\b",
                "asda f stn",
                r"tesco pfs",
                r"\blease\b",  # replacing 'lease' --> produces unrelated results
                "autolease",
                # merchants
                "drivetech",
                "enterprise rent",
                "leaseplan",
                r"\bscooter\b",
                r"\bgarage\b",
                # car-related services
                "dvla",  # Driver and Vehicle Licensing Agency
            ],
            "-": [
                "energy",
                "grotto",
                "food",
                r"\bspa\b",
                r"\bfunds\b",
                "shell energy",
                r"\btax\b",
            ],
        },
        "counterparty": {
            "+": [],
            "-": [],
        },
    },
    "Travel Expenses": {
        "transaction_type": "debit",
        "description": {
            "+": [
                # general keywords
                "hotel",
                "travel",
                r"\bflight\b",
                r"\bairways\b",
                r"\bairlines\b",
                "ticket machine",
                # travel insurance:
                "coverwise co uk",
                # uk transportation
                "tfl gov uk",
                r"\bgwr\b",
                r"\btrains\b",
                # taxi etc. services
                r"\btaxi\b",
                r"\buber\b",
                r"\bbuses\b",
                r"\bchiltern\b",
                "cabapp",
                "cmt uk",
                "cityfleet",
                r"\btrainline\b",
                # airport-flying services
                "teechu sells",
                # flight tickets
                "jetabroad",
                r"\bopodo\b",
                # airlines
                r"\btui\b",
                "ryanair",
                "transavia",
                r"\biberia\b",
                r"\bklm\b",
                # hotels etc.
                "hampton inn",
                "hyatt regency",
                "airbnb",
                r"\bbooking com\b",
                "hilton",
                "doubletree",
                r"\bibis\b",
                "marriot",
            ],
            "-": [
                # transportation-related keywords
                r"\beats\b",
                *YOLTAPP_DOMAIN.ATM,
                r"\bpools\b",
                "chiltern street",
                "sports",
                "valley",
                "hardware",
                "district",
                "nursery",
                r"\batm\b",
                # travel related keywords
                "student",
                "club",
                r"\bdeli\b",
                r"\bdoor\b",
                r"\bdoors\b",
                "primrose",
                r"\brusty\b",
                "the garage london",
            ],
        },
        "counterparty": {
            "+": [],
            "-": [],
        },
    },
    "Utilities": {
        "transaction_type": "debit",
        "description": {
            "+": [
                # general terms
                r"\butilities\b",
                "energy",
                r"\bgas\b",
                "electricity",  # instead of 'electric' from nl terms
                # r"\bwater\b",  # !!! water is too general and commented out for now
                # merchants
                "outfox",
                r"\bpure planet\b",
                r"\bnpower\b",
                "scottish power",
                r"\be on\b",  # tricky provider name! revisit...
                # mobile/internet
                # general terms
                "broadband",
                "telecom",
                # merchants
                "talktalk",
                "talkmobile",
                "t mobile",
                "vodafone",
                "plusnet",
                "hyperoptic",
                r"\bnow tv\b",
                "sky digital",
                "sky mobile",
                "bt group plc",
                "virgin media",
                r"\blebara\b",
                "tesco mobile",
                "giffgaff",
                "three co gb",
                r"\bthree billpay\b",
                "shell energy",
                "zen internet",
                "kcom group plc",
                "tescointernational",
                r"\bh digits g digits\b",
                r"\bh digits g\b",
            ],
            "-": [
                r"\bstreet\b",
                "uber",
                "spotify",
                r"\bbar\b",
                "garage",
                "bookshop",
                "sounds",
                "parking",
                "pure collection",
                "hsnf",
                "brasserie",
                "jeans",
                "shop",
                "futurehk",
                "edmondos",
                "airbnb",
                "purelife",
                "wellness",
                "charity",
                "coaching",
                r"\bcycle\b",
                r"\bbusiness\b",
                "dental",
                r"\bice\b",
                "accountancy",
                r"\bbricks\b",
                r"\bbar\b",
                r"\bhotel\b",
                r"\bgrill\b",
                r"\bticket\b",
                r"\bfish\b",
                "cafe",
                "office",
                r"\bmoon\b",
                "comedy",
                r"\bwitch\b",
                r"\bflour\b",
                r"\bcultu\b",
                r"\bpoet\b",
                "water park",
                "delivery",
                r"\bforest\b",
                "wellbeing",
                r"\bnext\b",
                r"\bbooze\b",
                # merchants excluded because of \bwater\b' term
                r"\bpennon\b",
                "holland barrett",
                "wildwoods",
                "just add water",
                "fed by water",
                "blue water",
                "world of water",
                "winwater",
                "sainsbury",
                "osmio",
                "oliver bonas",
                "open water",
                "rail water",
                "water rats",
                "water house",
                "castle water",
                "spotless water",
                "marks and spencer",
                r"\bshell\b",
                "high water",
                r"\bthe water\b",
            ],
        },
        "counterparty": {
            "+": [],
            "-": [],
        },
    },
    "Food and Drinks": {
        "transaction_type": "debit",
        "description": {
            "+": [
                # from nl terms
                *YOLTAPP_DOMAIN.FOOD_GENERIC,
                *YOLTAPP_DOMAIN.SUPERMARKETS,
                *YOLTAPP_DOMAIN.RESTAURANTS,
                # food delivery
                "deliveroo",
                "uber eats",
                # counterparties
                "nespresso",
                "starbucks",
            ],
            "-": [
                *YOLTAPP_DOMAIN.ATM,
                "travel",
                "moneysupermarket",
                "petrol",
                "fuel",
                "f stn",
                "photo",
                "tesco pfs",
                r"\bpump\b",
                "tesco mobile",
                "tescointernational",
                "shell",
                # create and include here list of gas stations
            ],
        },
        "counterparty": {
            "+": [],
            "-": [],
        },
    },
    "Rent and Facilities": {
        "transaction_type": "debit",
        "description": {
            "+": [
                r"\brent\b",
            ],
            "-": [
                "rent a car",
                "enterprise rent",
                r"\bcar\b",
            ],
        },
        "counterparty": {
            "+": [],
            "-": [],
        },
    },
    "Collection Costs": {
        "transaction_type": "debit",
        "description": {
            "+": [
                r"\bdebt\b",  # Todo is this term really related to collection costs?
                r"\bcollect\b",
                # counterparties
                "fredrickson",
            ],
            "-": [
                r"\bcar\b",
                r"\btax\b",
                r"\bcosta\b",
            ],
        },
        "counterparty": {
            "+": [],
            "-": [],
        },
    },
    "Other Operating Costs": {
        "transaction_type": "debit",
        "description": {
            "+": [
                # generic terms
                r"\bmail\b",
                "post office",
                "software",
                "stationer",
                "insurance",
                "healthcare",
                "accounting",
                "financial",
                r"\blaw\b",
                "lawyer",
                "legal",
                r"\bevents\b",
                "security",
                "packaging",
                # counterparties
                *YOLTAPP_DOMAIN.POSTAL_SERVICES,
                *YOLTAPP_DOMAIN.SOFTWARE_SUPPLIERS,
                *YOLTAPP_DOMAIN.HARDWARE_SUPPLIERS,
                *YOLTAPP_DOMAIN.LEGAL_SERVICES,
                *YOLTAPP_DOMAIN.FINANCIAL_SERVICES,
                *YOLTAPP_DOMAIN.EVENTS,
                # insurance
                "unum ltd",
                "bupa",
                # recruiting
                "reed co uk",
                # foreign exchange/bank fees
                "foreign",
                "fgn csh fee",
                "fgn pur fee",
                "non stg trans fee",
                "non stg purch fee",
                "non sterling",
                "visa rate",
                # overdraft fee
                "overdraft",
                "daily od fee",
                # other bank fees
                "club lloyds fee",
                "debit commission",  # check if correct category
                # immigration/visas fees
                "ukvi",
                # govermental company-related services
                "compani eshouse gov",
                "companieshouse",
                "companies house",
                # city_council services/fees
                "council",
                "city cou",
                "rb ken chelsea",
                "lbhf",
                "repeating chars islington gov",
                "plumbing",
                "plumber",
                "fork lift",
                "forklift",
                # raw materials
                "angel springs",
                "blinds digits go ltd",
                "selecta uk",
            ],
            "-": [
                "top ups",
                "tyre",
                *YOLTAPP_DOMAIN.ATM,
                r"\bgoogle (ad(w)?s|play)\b",
                "helppay",
                r"\bpay\b",
                "financial times",
                r"\bbar\b",
                r"\bpub\b",
                "dividend",
                r"private.*transfer",
                r"transfer.*private",
                r"owner.*deposit",
                "alarms",
                "mortgage",
                r"\bmtg\b",
                r"\bmrg\b",
                *YOLTAPP_DOMAIN.FOOD_GENERIC,
                r"\bshop\b",
            ],
        },
        "counterparty": {
            "+": [],
            "-": [],
        },
    },
    "Investments": {
        "transaction_type": "debit",
        "description": {
            "+": [
                "investment",
                # merchants
                "brokerbility",
                "octopus choice",
            ],
            "-": [
                "physical",
            ],
        },
        "counterparty": {
            "+": [],
            "-": [],
        },
        # add amount to criteria
    },
    "Corporate Savings Deposits": {
        "transaction_type": "debit",
        "description": {
            "+": [
                "corporate saving",
                r"business.*saving",
            ],
            "-": [
                "mortgage",
                r"\bmtg\b",
                r"\bmrg\b",
                "tax",
                r"\bwages\b",
                r"\bwage\b",
                "salary",
                "salaries",
                r"\bincome\b",
                "interest",
            ],
        },
        "counterparty": {
            "+": [],
            "-": [],
        },
    },
    "Interest and Repayments": {
        "transaction_type": "debit",
        "description": {
            "+": [
                # general terms
                r"\brepayment\b",
                r"\binterest\b",  # --->validate transactions (if they make sense)
                # mortgage
                r"\bmortgage\b",
                r"\bmtg\b",
                r"\bmrg\b",
                # loans
                r"\bloan\b",
                # loan counterparties
                "ratesetter",
                "boostcapital",
                "amigoloan",
                "bambooloan",
                "concern",
                "redemption",
            ],
            "-": [
                "worldwide",
                "charity",
                "roaster",
                "roalondon",
                r"\bbar\b",
                "redemption london",
            ],
        },
        "counterparty": {
            "+": [],
            "-": [],
        },
    },
    "Salaries": {
        "transaction_type": "debit",
        "description": {
            "+": [
                r"\bwage\b",
                r"\bwages\b",
                "salary",
                "salaries",
                r"\bincome\b",
            ],
            "-": [],
        },
        "counterparty": {
            "+": [],
            "-": [],
        },
    },
    "Pension Payments": {
        "transaction_type": "debit",
        "description": {
            "+": [
                r"\bpension\b",
                r"\bpensions\b",
                # merchants
                "youinvest",
                "legal general",
                "legal and general",
            ],
            "-": [
                "smart pension",
                r"\bwage\b",
                r"\bwages\b",
            ],
        },
        "counterparty": {
            "+": [],
            "-": [],
        },
    },
    "Marketing and Promotion": {
        "transaction_type": "debit",
        "description": {
            "+": [
                "promotion",
                "marketing",
                "campaign",
                r"\bcreative\b",
                r"\bgoogle ad(w)?s\b",
                "linkedin",
                "photoshop",
                "content",
                "sponsor",
                "tiktok",
                "indeed",
                # counterparties
                "httpscanva co",
                r"\bcanva\b",
                "awin ltd",
                "mailchimp",
                "anicca digital",
                "colewood",
                "vistapr",
                "moo com",
                "webcom enable",
                "lengow",
                "xsellco",
            ],
            "-": [
                r"\bcastle\b",
                "wiltshire",
                "curtains",
                "factory",
                "market",
                "cloud",
                "adobe",
                "dancers",
            ],
        },
        "counterparty": {
            "+": [],
            "-": [],
        },
    },
    "Other Expenses": {
        "transaction_type": "debit",
        "description": {
            "+": [
                # generic
                "furniture",
                "therapy",
                r"\bchemist\b",
                "nutrition",
                "fitness",
                "dry clean",
                r"\btheatre\b",
                "sports",
                r"\bdentist\b",
                r"\bdental\b",
                r"\bhospital\b",
                r"\bclub\b",  # term missclassifies restaurants in this category intead of food&drinks
                r"\bbeauty\b",
                "leisure",
                "skincare",
                "printing",
                *YOLTAPP_DOMAIN.LEISURE,
                *YOLTAPP_DOMAIN.PERSONAL_CARE,
                # stores
                r"\bretail\b",
                r"\bwatches\b",
                *YOLTAPP_DOMAIN.STORES,
                *YOLTAPP_DOMAIN.FURNITURE,
                *YOLTAPP_DOMAIN.WHOLESALERS,
                # subscriptions
                "google play",
                "youtube",
                "financial time",
                "the spectator",
                "rakuten",
                "spotify",
                "netflix",
                "itunes",
                # online shopping platforms
                "etsy",
                "trusted shops",
                r"\bebay\b",
                "ebayuk",
                # atm withdrawals
                *YOLTAPP_DOMAIN.ATM,
                # payments or transfer via Online Banking or Telephone Banking
                # a unique CALL REF.NO.XXXX is assigned to all the transactions made during that session.
                "call ref no digits",
                # printing and cards
                "scribbler",
                "paperchase",
                "galore",
                # other counterparties
                "trackmy time",
                "filmtronics",
                "timpson ltd",
                "thefreshcollection",
                *YOLTAPP_DOMAIN.CHARITY,
                *YOLTAPP_DOMAIN.GAMBLING,
            ],
            "-": [
                r"\btransport",
                "hillindon",
                "lloyds",
                *YOLTAPP_DOMAIN.FOOD_GENERIC,
                *YOLTAPP_DOMAIN.SUPERMARKETS,
            ],
        },
        "counterparty": {
            "+": [],
            "-": [],
        },
    },
    "Equity Financing": {
        "transaction_type": "credit",
        "description": {
            "+": [
                "dividend",
                "divident",
                "dividends",
                "shares",
                "save",
                "saving",
                r"private.*transfer",
                r"transfer.*private",
            ],
            "-": [
                r"\bwages\b",
                r"\bwage\b",
                "salary",
                "salaries",
                r"\bincome\b",
            ],
        },
        "counterparty": {
            "+": [],
            "-": [],
        },
    },
    "Tax Returns": {
        "transaction_type": "credit",
        "description": {
            "+": [
                # generic keywords
                "tax return",
                # income tax
                "hmrc",
            ],
            "-": [
                "interest",
                r"\bloan\b",
                "dividend",
                r"private.*transfer",
                r"transfer.*private",
                r"owner.*deposit",
                r"\bwages\b",
                r"\bwage\b",
                "salary",
                "salaries",
            ],
        },
        "counterparty": {
            "+": [],
            "-": [],
        },
    },
    "Loans": {
        "transaction_type": "credit",
        "description": {
            "+": [
                # general terms
                r"\bloan\b",
                # add loan counterparties from list
                # loan counterparties
                "ratesetter",
                "bambooloan",
            ],
            "-": [
                "mortgage",
                "interest paid after tax digits digits deducted",
            ],
        },
        "counterparty": {
            "+": [],
            "-": [],
        },
    },
    "Revenue": {
        "transaction_type": "credit",
        "description": {
            "+": [
                # general terms
                "revenue",
                "payment",
                "profit",
                "invoice",
                r"\bearn",  # Todo not sure if this is revenue (Other Income maybe?)
                r"\bsale\b",
                r"\bsales\b",
                r"\bshopify\b",
                r"\bpayout\b",
                "settlement",
            ],
            "-": [
                "refund",
                r"\btax\b",
                "amazon",
                "interest",
                "earnest",
                r"\btransfer\b",
                r"\breturn\b",
                r"\bloan\b",
                "dividend",
                r"private.*transfer",
                r"transfer.*private",
                r"owner.*deposit",
            ],
        },
        "counterparty": {
            "+": [],
            "-": [],
        },
    },
    "Other Income": {
        "transaction_type": "credit",
        "description": {
            "+": [
                # generic keywords
                r"\binterest\b",
                r"\btax\b",
                # Crypto income:
                "kraken tx",
                "coinify",
                *YOLTAPP_DOMAIN.GAMBLING,
                r"\bwage\b",
                r"\bwages\b",
                "salary",
                "salaries",
                r"\bincome\b",
                r"\bfund(s)?\b",
                "interest paid after tax digits digits deducted",
            ],
            "-": [
                "hmrc",
                "tax_return",
                # make sure all tax return keywords
                # are included here
            ],
        },
        "counterparty": {
            "+": [],
            "-": [],
        },
    },
}
