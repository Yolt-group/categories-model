from dataclasses import dataclass


def exact_match(items):
    return [f"\\b{item}\\b" for item in items]


"""general remark: counterparties are added in description match list; Because counterparty is not available in yoltapp data, we look for the counterparty in transaction description to identify the sme category instead """


@dataclass
class Generic:
    CAR_BRANDS = exact_match(
        [
            "bmw",
            "peugeot",
            "mercedes( benz)?",
            "citroen",
            "toyota",
            "ford",
            "volkswagen",
            "peugeot",
            "ducati",
            "volvo",
            "fiat",
            "porsche",
            "audi",
            "tesla",
            "bmw",
            "kia",
            "mercedes (benz)?",
            "skoda",
            "citroen",
            "hyundai",
            "honda",
            "nissan",
            "alfa.?romeo",
            "mazda",
            "suzuki",
            "chevrolet",
            "jaguar",
            "aston.?martin",
        ]
    )


@dataclass
class YTS:
    FUEL = exact_match(
        [
            "shell",
            "bp",
            "esso",
            "texaco",
            "tinq",
            "avia",
            "tamoil",
            "gulf",
            "firezone",
            "argos oil",
            "lukoil",
            "travelcard",
            "benzine",
            "brandstof",
            "diesel",
        ]
    )
    CAR_RELATED = [
        # carwash
        "carwash",
        "autowas",
    ]
    SUPERMARKETS = exact_match(
        [
            "albert heijn",
            "jumbo",
            "vomar",
        ]
    )
    POSTAL_COMPANIES = exact_match(
        f"\\b{brand}\\b"
        for brand in [
            "postnl",
            "dhl",
            "ups",
        ]
    )
    AIRLINES = exact_match(
        [
            "corendon",
            "aegean",
            "vueling",
            "tui airlines",
            "turkish airlines",
            "brussels airlines",
            "czech airlines",
            "ryanair",
            "transavia",
            "iberia",
            "klm",
        ]
    )
    STORES = exact_match(
        [
            "catawiki",
            "lego",
            "flying tiger",
            "bijenkorf",
            "forever twenty",
            "primark",
            "kleding",
        ]
    )
    SECURITY_COMPANIES = exact_match(
        [
            "hs pas",
        ]
    )
    E_COMMERCE = exact_match(
        [
            "marktplaats",
            "bol com",
            "coolblue",
            "amzn",
            "amazon",
            "zalando",
            "alibaba",
            "asos",
            "zara",
        ]
    )
    FOOD_GENERIC = exact_match(
        [
            "eten",
            "drinks",
            "drank",
            "food",
            "eat",
            "supermark(t)?",
            "pizza",
            "coffee",
            "borrel",
            "slagerij",
            "bakker(ij)?",
            "bagels",
            "wine",
            "wijn",
            "bier",
            "beer",
            "sandwich",
            "bakery",
            "candy",
            "pita",
            "broodje",
            "groente",
            "snack",
            "pub",
            "bar",
            "mossel",
            "terras",
            "cafetaria",
            "snoep",
        ]
    )
    CLEANING_COMPANIES = [
        "casa limpa",
    ]
    GROCERY_COUNTERPARTIES = [
        "^hoogvliet",
        "albert heijn",
        r"^ah$",
        r"^jumbo\b",
        r"^dekamarkt",
        "^deen",
        r"^dirk vdbroek fil$",
        r"^aldi$",
        r"\bvomar\b",
        "toogoodtogo",
    ]
    UTILITY_COUNTERPARTIES = [
        "eneco (zakelijk|services)",
        "vitens nv",
    ]
    TELECOM_COUNTERPARTIES = [
        "transip b v",
        "t mobile",
        r"^youfone nederland",
        r"^lebara$",
        r"^hollandsnieuwe$",
        r"^ben nederland$",
        r"^tele$",
    ]
    INSURANCE_COUNTERPARTIES = [
        "actua assurantien",
        "harmony service center",
    ]
    FOOD_COUNTERPARTIES = [
        r"ma?c(\ )?donald(\ )?s?",
        "ccv de knollentuin",
    ]
    FREELANCE_BROKERS = [
        "fiverr com",
    ]
    POSTAL_COUNTERPARTIES = ["^dpd nederland b v$"]
    REAL_ESTATE_COUNTERPARTIES = [
        "vastgoed",
        "lenf finance holding",
    ]
    ADVERTISING_COUNTERPARTIES = [
        "adtraction marketing",
        "amazon media",
        "facebook",
        "shotz productions",
    ]
    VEHICLE_LEASE_COUNTERPARTIES = [
        "leaseplan",
        "van mossel",
        "abn amro lease",
    ]
    FUEL_COUNTERPARTIES = [
        r"^travelcard",
    ]
    WHOLESALE_COUNTERPARTIES = [
        "delta wines",
        "sligro food group",
        "interlinex",
        "von harras agf bv",
        "csc gastro",
        "horeca solutions group",
    ]
    CHARITY_COUNTERPARTIES = [
        "stichting ronald",
    ]
    TAX_COUNTERPARTIES = [
        "belastingdienst",
    ]
    TRAVEL_COUNTERPARTIES = [
        "ns groep iz ns reizigers",
    ]
    E_COMMERCE_COUNTERPARTIES = [
        r"^coolblue",
    ]


@dataclass
class YoltApp:
    ATM = [
        "bmach",
        "all leics",
        "notemachine",
        r"\batm\b",
        "notemac",
        "cash bnkm",
        "cardtronics",
        r"\bcash\b",
        "withdrawal",
        r"\blnk\b",
        "bancomat",
    ]

    SUPERMARKETS = [
        "aldi",
        "co op",
        "asda",
        "sainsbury",
        "lidl",
        "tesco",
        "waitrose",
        "londis",
        "westbourne nogs",
        "sainsby",
        "lidl",
        r"\bspar\b",
    ]

    FOOD_GENERIC = [
        r"\bbakery\b",
        "cookies",
        "patisserie",
        "burger",
        "tavern",
        "lunch",
        r"\bpub\b",
        r"\bpubs\b",
        "restaurant",
        "canteen",
        "kiosk",
        r"\bcoffee\b",
        r"\bbar\b",
        r"\bcafe\b",
        r"\bpasta\b",
        r"\bpizza\b",
        r"\beat\b",
        r"\bcafe\b",
        r"\bdeli\b",
        "sandwich",
        "restauran",
        r"\bmeat\b",
        r"\binn\b",
        r"\bfood\b",
        r"\bramen\b",
        "tonkotsu",
        "juice",
        r"\bgrill\b",
        "caffe",
        "sushi",
        r"\bcandy\b",
        r"\bbeer\b",
        "diner",
        r"\bwine\b",
        r"\bwines\b",
        "supermark",
        "karaoke",
        r"\btaste\b",
    ]

    RESTAURANTS = [
        r"burger.?king",
        "dominos",
        "hellofresh",
        r"\bkfc\b",
        "mcdonald",
        r"papa.?john",
        "papajohns",
        "subway",
        "pret a manger",
        "porkys",
        "lucky voice",
    ]

    POSTAL_SERVICES = [
        "dpdgroup",
        "hermesparcelnet",
        "parcelhub",
        r"\bups\b",
        r"\bxdp\b",
        r"\bdhl\b",
        "pb purchase power",
        "parcelforce",
        r"\btnt\b",
        "aftership",
        "royal mail",
        "mailboxes",
    ]

    SOFTWARE_SUPPLIERS = [
        "adobe",
        "msbill",
        "microsoft",
        "google",
        "avast",
        "godaddy com",
        "timetastic",
        r"\bapple com\b",
        "ringcentral",
        "uattend",
        "avalara europe",
        r"\bxero\b",
        "linnsystems",
        "dnh media temple",
    ]

    HARDWARE_SUPPLIERS = [
        "upper street hardw",
        "istore",
        "toolstation co",
        "express group limited",
    ]

    LEGAL_SERVICES = [
        "osbornes",
        "hfw lexington",
        "rossendales",
    ]

    FINANCIAL_SERVICES = [
        "deloitte",
        "torr waterfield",
        "silver levene",
    ]

    STORES = [
        "primark",
        "tiger london",
        "river island",
        "simply be",
        "zara",
        r"\bhermes\b",
        "sonofatailorcom",
        "westfield shopping",
        "flair flooring",
        "the range",
        "charles tyrwhitt",
        "marks spencer",
        "bluebell",
        "mf penfold",
        "home bargains",
        "kitchen ideas",
        "zakti islington",
        r"\bboots\b",
        "menswear",
        "sportsdirect",
        "finisterre",
        "j crew",
        "holland barrett",
    ]

    LEISURE = [
        "royal wimbledon",
        "chelsea fc",
        "london palladium",
        "warner bros",
        "victoria palace",
        "hammersmith apollo",
        "inner temple",
        "picturehouse",
        "national galle",
    ]

    PERSONAL_CARE = [
        "kiehls co uk",
        "bannatyne health",
        "holmes place",
        "virgin active",
        "ruffians",
        "specsavers",
        "superdrug",
    ]

    FURNITURE = [
        "supersave",
        "birlea furniture",
        "furniture to go",
        "core products ltd",
        "cabinet maker",
    ]

    CHARITY = [
        "the anthony nolan",
        "justgiving",
    ]

    EVENTS = [
        "okm events",
        "catalyst event",
        "act active",
    ]

    WHOLESALERS = [
        "seconique plc",
        "veema uk",
        "costco",
        "priceminister",
        "b q digits",
    ]

    GAMBLING = [
        "lottery",
        "who internet",
        "casino",
        "poker",
        "jackpot",
        "videoslots",
        "bwin",
        "starspins",
        "junglespin",
        "lotteries",
        r"lottery",
        r"eu.?lotto",
        "lottoland",
        r"sky.?bet",
        "betbull",
        "betfred",
        "betfair",
        "betway",
        "unibet",
        "virgin bet",
        "betvictor",
        "livescorebet",
        "bet internet",
        r"bet.?365",
        "netbet",
        "gentingbet",
        "betstars",
        "betind",
        "betconnect",
        "betson",
        "betstars",
        r"32.?red",
        r"aspire.?global",
        r"ag.?commun",
        "anakatech",
        "annexio",
        r"star.?spo.*bet",
        r"bet.?stars",
        r"ny.?spins",
        "fafabet",
        r"tgp.?europ",
        "sportpesa",
        "12bet",
        "tlcbet",
        "fun88",
        r"football.?pools",
        "tombola",
        r"tony.?bet",
        r"apollo.*entertainment",
        "redzonesports",
        r"sport.?nation",
        "dafabet",
        "quinnbet",
        "betable",
        r"bet.?at.?home",
        "betiton",
        r"bgo.?enterta",
        r"blue.?star.?planet",
        r"10.?bet",
        r"bonne.?terre",
        r"boyle.?sport",
        r"broadway.?gaming",
        r"buzz.?group",
        "casumo",
        r"corbett.?sport",
        r"copy.?bet",
        r"chisholm.?bookmaker",
        r"daub.?alderney",
        r"dazzletag",
        r"buzz.?bingo",
        r"play.?sunny",
        r"draft.?king",
        "midnite",
        r"eaton.?gate.?gaming",
        "kwiff",
        "electraworks",
        "fairload",
        "lvbet",
        "fitzdares",
        r"football.?picks",
        "betvoyager",
        "gamesys",
        r"rainbow.?riches",
        r"heart.?bingo",
        r"virgin.?game",
        r"sports.?advis",
        r"geoff.?banks",
        r"genesis.?global",
        "genting",
        r"grace.*media",
        "greentube",
        r"hillside.*sport",
        r"football.?pools",
        r"in.?touch.?games",
        "jumpman",
        r"l&l.*euro",
        r"lc.?international",
        "ladbrokes",
        "betdaq",
        r"gala.?(bingo|spin|coral)",
        r"lci.*coral",
        "leovegas",
        r"lindar.*media",
        "mrq.com",
        r"\blow6\b",
        "lottomart",
        r"marathon.?bet",
        "conquestador",
        "mobinc",
        "mrgreen",
        r"network.?gam",
        r"novi.?group",
        r"one.?click.?(lim|ltd)",
        r"palace.?bingo",
        "petfre",
        "oddsking",
        r"bingo.com",
        r"paddy.?power",
        r"energy.?bet",
        r"probe.?inv",
        r"progress.?play",
        r"rank.?digi",
        r"mecca.?bingo",
        "alderney",
        r"raw.?nut.?gam",
        r"match.?bingo",
        r"run.?it.?once",
        "fanteam",
        r"scout.?and",
        r"skill.?on.?net",
        r"slots.?mil",
        "smarkets",
        "socialicity",
        r"sporting.?in",
        r"sportito",
        "spreadex",
        r"star.?rac",
        r"stech(\*|space)",
        r"voodoo.?dre",
        r"^\btote\b",
        r"match.?book",
        r"tse.?malta",
        r"\btyche.?tech",
        r"vf.?2011",
        r"bet.?600\b",
        r"virgin.?bet",
        "vivaro",
        r"whg.*(lim|ltd)",
        r"white.?hat.?ga",
        r"who.?know",
        "rizk.com",
        r"zecure.?gam",
        "zweeler",
    ]
