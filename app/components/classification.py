import os
from pathlib import Path
import logging
from nltk import ngrams
from statistics import mode
import pandas as pd
import pprint

# Import local modules
from app.components.search_functions import (
    keyword_matches,
    semantic_search,
    search_logo,
)
from app.prompts.insurance_keywords import *
from app.utils.txt_utils import clean_ocr_text
from app.utils.pdf_utils import extract_pdf_text


# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def quote_classification(
    pdf_bytes: bytes,
):
    """
    Processes the PDF bytes using pypdfium2 to extract text from the first page.

    Args:
        pdf_bytes: The PDF content as bytes.
    """
    #     "quote_classification": {
    #         "quote_submission": True/False,
    #         "quote_class": "personal/commercial",
    #         "quote_type": "auto/property",
    #         "broker_name": "acord/other",
    #         "form_id": "acord125/custom"
    #     },
    quote_submission = False
    quote_class = set()
    quote_type = set()
    broker_name = set()
    form_id = set()

    # Create empty DataFrame with classification status
    quote_status = {
        "regex_quote_submission": None,
        "regex_quote_class": None,
        "regex_quote_type": None,
        "broker_name": None,
        "form_id": None,
    }

    ##############################
    # STEP 1: Load the PDF document directly from the bytes object
    clean_text = extract_pdf_text(pdf_bytes)

    ##############################
    # STEP 2: REGEX Match
    if keyword_matches(clean_text, exact_match_keywords["personal_auto"]) > 0:
        quote_status["regex_quote_submission"] = True
        quote_status["regex_quote_class"] = "personal"
        quote_status["regex_quote_type"] = "auto"

    if keyword_matches(clean_text, exact_match_keywords["personal_property"]) > 0:
        quote_status["regex_quote_submission"] = True
        quote_status["regex_quote_class"] = "personal"
        quote_status["regex_quote_type"] = "property"

    if keyword_matches(clean_text, exact_match_keywords["commercial_auto"]) > 0:
        quote_status["regex_quote_submission"] = True
        quote_status["regex_quote_class"] = "commercial"
        quote_status["regex_quote_type"] = "auto"

    if keyword_matches(clean_text, exact_match_keywords["commercial_property"]) > 0:
        quote_status["regex_quote_submission"] = True
        quote_status["regex_quote_class"] = "commercial"
        quote_status["regex_quote_type"] = "property"

    ##############################
    # STEP 3: Semantic Similarity
    # Create bi/tri-grams
    two_word_phrases = [" ".join(pair) for pair in ngrams(clean_text, 2)]
    three_word_phrases = [" ".join(trio) for trio in ngrams(clean_text, 3)]

    # Classify personal vs commercial
    if (
        semantic_search(personal_words, clean_text)
        or semantic_search(personal_bigrams, two_word_phrases)
        or semantic_search(personal_trigrams, three_word_phrases)
    ):
        quote_status["semantic_quote_submission"] = True
        quote_status["semantic_quote_class"] = "personal"

    if (
        semantic_search(commercial_words, clean_text)
        or semantic_search(commercial_bigrams, two_word_phrases)
        or semantic_search(commercial_trigrams, three_word_phrases)
    ):
        quote_status["semantic_quote_submission"] = True
        quote_status["semantic_quote_class"] = "commercial"

    ##############################
    # STEP 4: Logo Search

    logger.info(f"\n{pprint.pformat(quote_status)}")

    return quote_status
