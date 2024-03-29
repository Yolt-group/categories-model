from categories_model.preprocessing.domain_data import YTS, Generic

YTS_DOMAIN = YTS()
GENERIC_DOMAIN = Generic()

CATEGORY_RULES = {
    ###########################################################################
    #                  Categories for Outgoing Transactions                   #
    ##########################################################################
    "Equity Withdrawal": {
        "transaction_type": "debit",
        "description": {
            "+": [
                r"prive.*(opname|ontrekking|overboeking)",
                r"prive.*outgoing transfer",
                r"prive.*storting",
                r"storting.*prive",
                "zakgeld",
                "dividend",
                "divident",
                "sparen bv",
            ],
            "-": [
                "belasting",
                "hypotheek",
                "loon",
                "salaris",
                "rente",
                "huur",
                "jortt",
                "lening",
                "reiskosten",
                "sparen(?! bv)",
                "spaar",
                "kosten",
                r"ziggo",
                r"\bkpn\b",
                "amazon",
                "drink",
                "huur",
                "tank",
                "zakelijk",
                "loon",
                "credit",
                "card",
                "verrekening",
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
            "+": [r"\bnotari(s(sen)?|aat).+(aankoop|aanschaf)\b"],
            "-": [],
        },
        "counterparty": {
            "+": [
                r"\b(derden|kwaliteits).+notari(s(sen)?|aat)\b",
                r"\bnotari(s(sen)?|aat).+(derden|kwaliteits)\b",
            ],
            "-": [],
        },
        "amount_based": [
            {
                "min": 15000,
                "description": ["notari(s(sen)?|aat)"],
                "counterparty": ["notari(s(sen)?|aat)"],
            },
        ],
    },
    "Corporate Savings Deposits": {
        "transaction_type": "debit",
        "description": {
            "+": [
                r"\bzakelijke.+spaarrekening\b",
            ],
            "-": [
                "belasting",
                "hypotheek",
                "salaris",
                "jortt",
                "onderhoud",
                "rente",
                "straat",
                r"\bjumbo\b",
                "vakantie",
                "prive",
            ],
        },
        "counterparty": {
            "+": [r"\bzakelijke.+spaarrekening\b"],
            "-": [],
        },
    },
    "Interest and Repayments": {
        "transaction_type": "debit",
        "description": {
            "+": [
                "belang",
                "rente",
                "aflossing",
                "rentevastlening",
                r"\blening",
                "interest",
                "hypotheek",
                r"\bterugbetaling.+lening]\b",
            ],
            "-": [
                "salaris",
                "belasting",
                "adverten",
                r"\bpinterest\b",
                *GENERIC_DOMAIN.CAR_BRANDS,
                "verzeker",
            ],
        },
        "counterparty": {
            "+": [],
            "-": [*YTS_DOMAIN.REAL_ESTATE_COUNTERPARTIES],
        },
    },
    "Unspecified Tax": {
        "transaction_type": "debit",
        "description": {
            "+": ["belasting", "\b.*(gemeente|belasting).*\b{2,}"],
            "-": ["salaris", "huur", "park", "waternet"],
        },
        "counterparty": {
            "+": [*YTS_DOMAIN.TAX_COUNTERPARTIES],
            "-": [*YTS_DOMAIN.FUEL],
        },
    },
    "Vehicles and Driving Expenses": {
        "transaction_type": "debit",
        "description": {
            "+": [
                # General Keywords
                "huurauto",
                "autoverhuur",
                "autobedrijf",
                "autoparts",
                "autos",
                "lease",
                "motor",
                "scooter",
                "hundai",
                r"\bfiets\b",
                "transport",
                r"bo.?rent",
                # Car Brands
                *GENERIC_DOMAIN.CAR_BRANDS,
                r"\bavis\b",
                # Auto parts
                "vereende",
                "autoonderdelen",
                # Parking
                "parkmobile",
                "parkeren",
                r"\bpark\b",
                r"\bparkee\b",
                "yellowbrick",
                "parkbee",
                "garagepark",
                "parking",
                r"\bdirk park fil\b",
                # Fuel
                "brandstof",
                "tankstation",
                r"\btank\b",
                "supertank",
                "diesel",
                *YTS_DOMAIN.FUEL,
                *YTS_DOMAIN.CAR_RELATED,
                # car insurance
                r"\banwb\b",
                r"auto.*verzeker",
                r"\bbenevia\b",
                # fuel cards
                r"\btravelcard\b",
            ],
            "-": [
                *YTS_DOMAIN.POSTAL_COMPANIES,
                "salaris",
                "belasting",
                "praktijk",
                "sanoma",
                r"\bbooking com\b",
                "microsoft",
                "google",
                r"\btransport(vrz|verzekering)\b",
                r"\bbar\b",
                r"\btransport eigen vervoer\b",
                r"\bhs pas\b",
            ],
        },
        "counterparty": {
            "+": [
                *YTS_DOMAIN.VEHICLE_LEASE_COUNTERPARTIES,
                *YTS_DOMAIN.FUEL_COUNTERPARTIES,
            ],
            "-": [
                *YTS_DOMAIN.FOOD_COUNTERPARTIES,
                *YTS_DOMAIN.GROCERY_COUNTERPARTIES,
                *YTS_DOMAIN.TAX_COUNTERPARTIES,
            ],
        },
    },
    "Travel Expenses": {
        "transaction_type": "debit",
        "description": {
            "+": [
                # airlines
                *YTS_DOMAIN.AIRLINES,
                "flight",
                # hotel
                "hotel",
                "marriot",
                "doubletree",
                "hilton",
                "wyndham",
                r"best.?west",
                "hyatt",
                r"\bibis\b",
                # general
                "reiskost",
                "travel",
                "ticket",
                "chipkaart",
                "overnacht",
                # booking
                "booking com",
                "airbnb",
                "agoda",
                # train
                "trein",
            ],
            "-": [
                "foodticket",
                "restaurant",
                "belasting",
                *YTS_DOMAIN.POSTAL_COMPANIES,
                r"\btravelcard\b",
                "brandstof",
                r"\bfood\b",
                r"\bborrel\b",
            ],
        },
        "counterparty": {
            "+": [*YTS_DOMAIN.TRAVEL_COUNTERPARTIES],
            "-": [
                *YTS_DOMAIN.VEHICLE_LEASE_COUNTERPARTIES,
                *YTS_DOMAIN.FUEL_COUNTERPARTIES,
                "foodtruckbooking com",
            ],
        },
    },
    "Utilities": {
        "transaction_type": "debit",
        "description": {
            "+": [
                # Heating
                "verwarming",
                # Electricity
                "elektriciteit",
                "energie",
                "vattenfall",
                "greenchoice",
                r"\beneco\b",
                r"\bessent\b",
                "solar",
                "boiler",
                "energy",
                "electric",
                # Water
                "waterbehandeling",
                "waterleiding",
                "waterschap",
                "waternet",
                # Internet
                r"ziggo",
                r"\bkpn\b",
                "internet",
                r"\btele\b",
                # Mobile
                "vodafone",
                "mobielefactuur",
                "mobile facturen",
                "mobiel",
                "youfone",
                "simpel",
                "simyo",
                "datamobile",
                "t mobile",
                "telecom",
                "telefoon",
            ],
            "-": [
                "belasting",
                "maandtariferingsnota",
                "huur",
                *GENERIC_DOMAIN.CAR_BRANDS,
                *YTS_DOMAIN.CAR_RELATED,
                *YTS_DOMAIN.E_COMMERCE,
                *YTS_DOMAIN.FUEL,
                "avis",
                "restaurant",
                "salaris",
                "internetbankieren",
                r"\bcar\b",
                "boekhoud",
                "google",
                "verzeker",
                r"\bcoffee\b",
                "travel",
                "actua assurantien",
                "financieel adv",
                "assurantiekantoor",
                r"\badobe\b",
                r"\bmicrosoft\b",
                r"\binternet security\b",
                r"\bharmony service center\b",
                r"\bmcdonalds\b",
                *YTS_DOMAIN.SUPERMARKETS,
            ],
        },
        "counterparty": {
            "+": [
                *YTS_DOMAIN.UTILITY_COUNTERPARTIES,
                *YTS_DOMAIN.TELECOM_COUNTERPARTIES,
            ],
            "-": [*YTS_DOMAIN.INSURANCE_COUNTERPARTIES],
        },
    },
    "Food and Drinks": {
        "transaction_type": "debit",
        "description": {
            "+": [
                # Groceries
                "hellofresh",
                "picnic",
                "albert heijn",
                "aldi",
                "jumbo",
                "lidl",
                "hoogvliet",
                "gall gall",
                "marqt",
                r"\bdirk\b",
                r"\bcoop\b",
                r"\bspar\b",
                r"\bvomar\b"
                # Takeaway
                "uber eats",
                "thuisbezorgd",
                "deliveroo",
                # General Keywords
                *YTS_DOMAIN.FOOD_GENERIC,
                "boodschappen",
                "vivino",
                "restaurant",
                "breakfast",
                "ontbijt",
                "dinner",
                "lunch",
                r"\bcafe\b",
                "diner",
                # Specific Places
                "eatpoint",
                "eatcorner",
                "eataly",
                "eat fresh",
                "wijngaarden",
                "mcdonald",
                "nespresso",
                "amazing oriental",
                "dominos",
                r"burger.?king",
                r"papa.?john",
                r"\bkfc\b",
                "loetje",
            ],
            "-": [
                "salaris",
                "belasting",
                "action",
                *YTS_DOMAIN.FUEL,
                *GENERIC_DOMAIN.CAR_BRANDS,
                r"\bdirk park fil\b",
                "gemeente",
                r"\bkpn\b",
                "premie",
                "carwash",
                r"\badvocaten\b",
                "verzekering",
                "toolstation",
                "fiets",
                r"\bauto",
                r"\bvereende\b",
                r"\bavia\b",
                "spaarrekening",
                r"\bpark dak hoogvliet\b",
                r"\bhuur\b",
                r"\bbraas partners\b",
                "overnachting",
                "foodtruckbestellen be",
            ],
        },
        "counterparty": {
            "+": [*YTS_DOMAIN.GROCERY_COUNTERPARTIES, *YTS_DOMAIN.FOOD_COUNTERPARTIES],
            "-": [
                *YTS_DOMAIN.WHOLESALE_COUNTERPARTIES,
                *YTS_DOMAIN.CHARITY_COUNTERPARTIES,
                *YTS_DOMAIN.FUEL_COUNTERPARTIES,
            ],
        },
    },
    "Rent and Facilities": {
        "transaction_type": "debit",
        "description": {
            "+": ["huisvest", "vastgoed"],
            "-": [
                r"(huur)?auto((ver)?huur)?\b",
                "brandstof",
                "benzine",
                "salaris",
                "jortt",
                "scooter",
                "hundai",
                "motor",
                r"bo.?rent",
                *GENERIC_DOMAIN.CAR_BRANDS,
            ],
        },
        "counterparty": {
            "+": [*YTS_DOMAIN.REAL_ESTATE_COUNTERPARTIES],
            "-": [
                *YTS_DOMAIN.FOOD_COUNTERPARTIES,
                *YTS_DOMAIN.INSURANCE_COUNTERPARTIES,
            ],
        },
        "amount_based": [
            {
                "min": 500,
                "description": [r"\b(ver)huur\b", r"\brent\b"],
                "counterparty": [],
            },
        ],
    },
    "Collection Costs": {
        "transaction_type": "debit",
        "description": {
            "+": [
                "nova incasso",
                r"incasso.?partners",
                "eb incasso",
                "bos incasso",
                "nationaal incasso",
                "bleeker incasso",
                "deurwaarder",
            ],
            "-": [
                "salaris",
                "jortt",
                "belasting",
                "pensioen",
                "pension",
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
            "+": [  # Bank cost related
                "bank kost",
                "bankkost",
                "bank kosten",
                "maandtariferingsnota",
                "kosten zakelijk betalingsverkeer",
                "oranjepakket",
                "degiro",
                r"invest.?kosten",
                "internetbankieren nota",
                "rabo directpakket",
                r"knab.*subs",
                "pakketkosten subscription fee",
                "kosten gebruik betaalrekening",
                "knab boekhoudkoppeling"
                # Software
                "adobe",
                "microsoft",
                "hostnet",
                "cloud platform",
                "happybytes",
                "trustly group",
                "neostrada",
                "cloud",
                "shiftbase",
                "evidos",
                "strato",
                "norton",
                "security",
                "codestream",
                "google",
                # post
                *YTS_DOMAIN.POSTAL_COMPANIES,
                r"\bpost\b",
                "parcel",
                "shipment",
                r"\bcargo\b",
                "logistic",
                # Financial services
                "moneybird",
                "moneymonk",
                r"\bjortt bv\b",
                "skillsource",
                "buitengewoon",
                "adviesbureau",
                "reeleezee",
                "boekhoud",
                "boschland",
                "kruisposten",
                "administratiekosten",
                # insurance & financial services
                "polis",
                "financial",
                "fbto",
                "chubb",
                "allianz",
                "verzeker",
                "boekhoud",
                # Notary
                "notaris",
                # freelance
                "fiverr",
                "eventbrite",
                "freelance"
                # automation
                "automatisering",
                *YTS_DOMAIN.SECURITY_COMPANIES,
                # legal.
                "advocaat",
                "advocaten",
                "legaal",
                # raw materials
                "grondstof",
                "material",
                "bouwmaat",
                r"inkoop.?kost",
                "inventaris",
                "goederen",
                "hornbach",
                "toolstation",
                "onderdelen",
                r"\bkabel",
                "business goods",
                "purchase of goods",
                "hanos",
                "makro",
                "sligro",
                "groothandel",
                "wholesale",
                "import"
                # office related
                r"kantoor.?artikelen",
                r"kantoor.?benodigdheden",
                r"office,?supp",
                r"office.?centre",
                "ikea",
                "manutan",
                "printer",
                "tuin",
                "vandijk",
                "vendingwork",
                "inrichting",
                "hulshoff",
                r"\binkt nl\b",
                # maintenance, repairs and cleaning
                "reparatie",
                "hertstel",
                "maintenance",
                "onderhoud",
                "upkeep",
                "repair",
                "schoonmaak",
                "cleaning",
                "verbouw(en)?",
                *YTS_DOMAIN.CLEANING_COMPANIES,
            ],
            "-": [
                "belasting",
                "salaris",
                "belasting",
                "auto",
                "scooter",
                "motor",
                r"\bcar",
                "health",
                "zorg.*verz",
                # "knab",
                "zorgkoste",
                "brandstof",
                *GENERIC_DOMAIN.CAR_BRANDS,
                *YTS_DOMAIN.AIRLINES,
                *YTS_DOMAIN.STORES,
                *YTS_DOMAIN.E_COMMERCE,
                *YTS_DOMAIN.FUEL,
                "autoonderdelen",
                "fiets",
                "vereende",
                r"\bbenevia\b",
                "spaar",
                r"\banwb\b",
                "motorverzekering",
                r"\air tickets\b",
                r"\breiskosten\b",
                r"\bmcdonalds\b",
                r"\bairbnb\b",
                r"\btravelcard\b",
                "huur",
                r"\blego\b",
                "ticket",
                # "knab",
                "vastgoed",
                "deurwaarder",
                "advies",
                "belang",
                "rente",
                "aflossing",
                "rentevastlening",
                r"\blening",
                "interest",
                "hypotheek",
                r"\bterugbetaling.+lening]\b",
                r"\bgoogle (ad(w)?s|play)\b",
            ],
        },
        "counterparty": {
            "+": [
                r"\bknab\b",
                *YTS_DOMAIN.INSURANCE_COUNTERPARTIES,
                *YTS_DOMAIN.POSTAL_COUNTERPARTIES,
                r"^kamer van koophandel",
                *YTS_DOMAIN.WHOLESALE_COUNTERPARTIES,
                *YTS_DOMAIN.CLEANING_COMPANIES,
            ],
            "-": [
                *YTS_DOMAIN.UTILITY_COUNTERPARTIES,
                *YTS_DOMAIN.VEHICLE_LEASE_COUNTERPARTIES,
                *YTS_DOMAIN.FUEL_COUNTERPARTIES,
                *YTS_DOMAIN.FOOD_COUNTERPARTIES,
            ],
        },
        "amount_based": [
            {
                "max": 500,
                "description": [r"\b(ver)huur\b", r"\brent\b"],
                "counterparty": [],
            },
        ],
    },
    "Salaries": {
        "transaction_type": "debit",
        "description": {
            "+": [r"\b(netto)?loon\b", "salaris", "personnel", "verloning", "payroll"],
            "-": ["reiskosten", "lunch", "belasting", "jortt"],
        },
        "counterparty": {
            "+": [],
            "-": [*YTS_DOMAIN.FREELANCE_BROKERS],
        },
    },
    "Pension Payments": {
        "transaction_type": "debit",
        "description": {
            "+": [
                r"(\b(stichting|.*pensio(en(en)?|neerden).*|.*fonds*.|voorziening)\b){2,}|pensioen"
            ],
            "-": [],
        },
        "counterparty": {
            "+": [
                r"(\b(stichting|.*pensio(en(en)?|neerden).*|.*fonds*.|voorziening)\b){2,}|pensioen"
            ],
            "-": [],
        },
    },
    "Marketing and Promotion": {
        "transaction_type": "debit",
        "description": {
            "+": [
                "drukwerk",
                "campagne",
                "facebook",
                r"\bgoogle (ad(w)?s)\b",
                "instagram",
                "advertentie",
                r"\bads\b",
                "campaign",
                "tiktok",
                r"\bsnap",
                "promotie",
                "reklame",
                r"media\b",
                "adwords",
                "content",
                "creative",
                "marketing",
                "drukzo",
                r"\bcanva\b",
                "creatief",
                "design",
                "fotosjop",
                "linkedin",
                "vistaprint",
                "boomerangprocurement",
                "sponsor",
            ],
            "-": [
                "salaris",
                "belasting",
                "jortt",
                "park",
                "creative cloud",
                "media.?markt",
                r"\bgoogle cloud\b",
                "spaarrekening",
                "storage",
                "adobe",
                r"\bplatform\b",
                r"\bstorage\b",
                r"\bsuite\b",
            ],
        },
        "counterparty": {
            "+": [
                *YTS_DOMAIN.ADVERTISING_COUNTERPARTIES,
            ],
            "-": [
                *YTS_DOMAIN.FREELANCE_BROKERS,
                *YTS_DOMAIN.REAL_ESTATE_COUNTERPARTIES,
                *YTS_DOMAIN.VEHICLE_LEASE_COUNTERPARTIES,
            ],
        },
    },
    "Other Expenses": {
        "transaction_type": "debit",
        "description": {
            "+": [
                # stores
                *YTS_DOMAIN.STORES,
                # ecommerce
                *YTS_DOMAIN.E_COMMERCE,
                # mass retailers
                "hema",
                "kruidvat",
                "action",
                "holland barrett",
                "etos",
                "trekpleister",
                "blokker",
                # health
                "dutch health center",
                "therapeut",
                "zilveren kruis",
                "huisart",
                "dental",
                "medisch",
                "pharma",
                "gezondheid",
                "eigenrisico",
                # sport
                "decathlon",
                r"\bsport\b",
                # cinema
                "pathe",
                "cinema",
                # subscriptions
                "disney",
                "spotify",
                "hulu",
                "netflix",
                "itunes",
                "volkskrant",
                "netfl",
                # charity
                "rode kruis",
                "gofundme",
                r"\bgift\b",
            ],
            "-": [
                *GENERIC_DOMAIN.CAR_BRANDS,
                "salaris",
                r"\beten\b",
                "boodschappen",
                "drinks",
                "drank",
                "restaurant",
                "food",
                r"\beat\b",
                "dinner",
                "lunch",
                "breakfast",
                "ontbijt",
                "supermarkt",
                "pizza",
                "coffee",
                "borrel",
                "slagerij",
                "bakkerij",
                r"\bcafe\b",
                "bakker",
                "bagels",
                "print",
                r"\bloon\b",
                "verzeker",
                r"\bq park\b",
            ],
        },
        "counterparty": {
            "+": [
                *YTS_DOMAIN.E_COMMERCE_COUNTERPARTIES,
                r"^action$",
                r"^kruidvat$",
            ],
            "-": [*YTS_DOMAIN.ADVERTISING_COUNTERPARTIES],
        },
    },
    ###########################################################################
    #                  Categories for Incoming Transactions                   #
    ##########################################################################
    "Equity Financing": {
        "transaction_type": "credit",
        "description": {
            "+": [
                "aandelen",
                "kapitaal",
                "dividend",
                "spaar",
                "sparen",
                r"prive.*storting",
                r"storting.*prive",
                r"prive.*incoming transfer",
            ],
            "-": [
                "airbnb",
                "retour",
                "hypotheek",
                "loan",
                "langlopende",
                "schulden",
                "interest",
                "lening",
                "hypotheek",
                "loan",
                "omzet",
                "verkoop",
                "sales",
                "revenue",
                "earn",
                "profit",
                "shopify",
                "payments",
                "payout",
                "salaris",
                "collecte",
                "huurgelden",
                "bruto",
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
            "+": [],
            "-": ["huurtoeslag"],
        },
        "counterparty": {
            "+": ["belastingdienst"],
            "-": [],
        },
    },
    "Loans": {
        "transaction_type": "credit",
        "description": {
            "+": [
                "langlopende",
                "schulden",
                "lening",
                "hypotheek",
                "loan",
            ],
            "-": [
                "spaar",
                "sparen",
                "^rente interest$",
                "^interest payment$",
                "terugbetaling",
                "aflossing",
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
                "omzet",
                "verkoop",
                "sales",
                "revenue",
                "earn",
                "profit",
                "shopify",
                "payments",
                "payout",
                "collecte",
                "huurgelden",
                "bruto",
                "afrekening",
                "invoice",
                r"\bfact(uur)?\b",
            ],
            "-": [
                "belasting",
                "refund",
                "teruggaaf",
                "spa(ar|ren)",
                "aandelen",
                "kapitaal",
                "lening",
                "terugbetaling",
                "aandelen",
                "kapitaal",
                "dividend",
                r"prive.*storting",
                r"storting.*prive",
                r"prive.*incoming transfer",
            ],
        },
        "counterparty": {
            "+": [
                "consorfrut polska sp z o o",
                "fruit expert poland sp zo o",
                r"\bb( )?v\b",
            ],
            "-": [],
        },
    },
    "Other Income": {
        "transaction_type": "credit",
        "description": {
            "+": [
                "interest",
                "^rente interest$",
                "^interest payment$",
                "terugbetaling",
                "aflossing",
                "salaris",
                r"\bfund(s)?\b",
                # Crypto income:
                "kraken tx",
            ],
            "-": [],
        },
        "counterparty": {
            "+": [],
            "-": [],
        },
    },
}
