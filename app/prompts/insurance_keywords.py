############################################
# EXACT MATCH KEYWORDS
############################################
exact_match_keywords = {
    "personal_auto": [
        "private car",
        "private auto",
        "personal car",
        "personal auto",
    ],
    "personal_property": [
        "private property",
        "personal property",
    ],
    "commercial_auto": [
        "commercial car",
        "commercial auto",
        "commercial fleet",
        "business auto",
    ],
    "commercial_property": [
        "commercial property",
    ],
}


############################################
# PERSONAL KEYWORDS
############################################
personal_words = ["dwelling", "homeowner", "spouse", "renter"]

personal_bigrams = [
    "personal auto",
    "personal car",
    "private car",
    "primary residence",
    "principal residence",
    "personal use",
    "occupancy type",
    "uninsured motorist",
    "comprehensive coverage",
    "replacement cost",
    "single family",
    "private passenger",
]


personal_trigrams = [
    "homeowner's policy number",
    "medical payments to",
    "any household members",
    "additional interest mortgagee",
    "social domestic pleasure",
    "auto liability bodily",
    "homeowners insurance policy",
    "personal articles floater",
]


############################################
# COMMERCIAL KEYWORDS
############################################
commercial_words = [
    "fiduciary",
    "naics",
    "employee/empl",
    "workers",
    "payroll",
    "client",
    "inventory",
    "proprietor",
    "entity",
    "vendor",
    "revenues",
    "wholesale",
    "contractors",
    "llc",
]


commercial_bigrams = [
    "commercial general",
    "business auto",
    "gl code",
    "sic code",
    "business owner's",
    "general liability",
    "commercial auto",
    "workers' comp",
    "business interruption",
    "additional insured",
    "aggregate limit",
    "acord 125",
    "motor carrier",
    "part time",
    "annual revenues",
]


commercial_trigrams = [
    "business personal property",
    "masonry veneer construction",
    "certificate of insurance",
    "commercial general liability",
    "fein or soc",
    "description of operations" "boiler & machinery",
]
