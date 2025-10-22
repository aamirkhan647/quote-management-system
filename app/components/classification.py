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
    logo_search,
    footer_search,
    gemini_classify,
)
from app.prompts.insurance_keywords import *
from app.utils.txt_utils import clean_ocr_text
from app.utils.pdf_utils import extract_pdf_text, convert_pdf_to_images


# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def quote_classification(
    pdf_name: str,
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
    quote_class = None
    quote_type = None
    acord_form = False
    form_id = None

    # Create classification status in-memory database
    quote_status = {
        "regex_quote_submission": False,
        "regex_quote_class": "",
        "regex_quote_type": "",
        "semantic_quote_submission": False,
        "semantic_quote_class": "",
        "semantic_quote_type": "",
        "logo_quote_submission": False,
        "logo_acord_form": False,
        "footer_quote_submission": False,
        "footer_form_id": "",
        "gemini_quote_submission": False,
        "gemini_quote_class": "",
        "gemini_quote_type": "",
        "gemini_acord_form": False,
        "gemini_form_id": "",
    }

    ##############################
    # STEP 1: Load the PDF document directly from the bytes object
    ##############################
    clean_text = extract_pdf_text(pdf_bytes)

    ##############################
    # STEP 2: REGEX Match
    ##############################
    # Match Personal Auto
    if keyword_matches(clean_text, exact_match_keywords["personal_auto"]) > 0:
        quote_status["regex_quote_submission"] = True
        quote_status["regex_quote_class"] = "personal"
        quote_status["regex_quote_type"] = "auto"

    # Match Personal Property
    if keyword_matches(clean_text, exact_match_keywords["personal_property"]) > 0:
        quote_status["regex_quote_submission"] = True
        quote_status["regex_quote_class"] = "personal"
        quote_status["regex_quote_type"] = "property"

    # Match Commercial Auto
    if keyword_matches(clean_text, exact_match_keywords["commercial_auto"]) > 0:
        quote_status["regex_quote_submission"] = True
        quote_status["regex_quote_class"] = "commercial"
        quote_status["regex_quote_type"] = "auto"

    # Match Commercial Property
    if keyword_matches(clean_text, exact_match_keywords["commercial_property"]) > 0:
        quote_status["regex_quote_submission"] = True
        quote_status["regex_quote_class"] = "commercial"
        quote_status["regex_quote_type"] = "property"

    # Match Personal Only
    if keyword_matches(clean_text, personal_words) > 0:
        quote_status["regex_quote_submission"] = True
        quote_status["regex_quote_class"] = "personal"

    # Match Commercial Only
    if keyword_matches(clean_text, commercial_words) > 0:
        quote_status["regex_quote_submission"] = True
        quote_status["regex_quote_class"] = "commercial"

    ##############################
    # STEP 3: Semantic Similarity
    ##############################
    # Create bi/tri-grams
    two_word_phrases = [" ".join(pair) for pair in ngrams(clean_text, 2)]
    three_word_phrases = [" ".join(trio) for trio in ngrams(clean_text, 3)]

    # Classify personal vs commercial
    if semantic_search(two_word_phrases, personal_bigrams) or semantic_search(
        three_word_phrases, personal_trigrams
    ):
        quote_status["semantic_quote_submission"] = True
        quote_status["semantic_quote_class"] = "personal"

    if semantic_search(two_word_phrases, commercial_bigrams) or semantic_search(
        three_word_phrases, commercial_trigrams
    ):
        quote_status["semantic_quote_submission"] = True
        quote_status["semantic_quote_class"] = "commercial"

    ##############################
    # STEP 4: Logo Search (Acord forms)
    ##############################
    # Convert PDF to Images
    image_paths = convert_pdf_to_images(pdf_name, pdf_bytes, 300)

    # Get all logo image templates for image search
    root_path = Path("./logos/")
    logo_files = list(root_path.glob("**/*.jpg"))

    if not logo_files:
        logger.warning("No logo files found.")
        exit(0)

    # logger.info(f"Found {len(logo_files)} logo file(s).")
    logo_paths = list(map(str, logo_files))

    # match_found, confidence, bounding_box = logo_search(
    #     image_paths, logo_paths, threshold=0.8
    # )

    # if match_found:
    #     quote_status["logo_quote_submission"] = True
    #     quote_status["logo_acord_form"] = True

    ##############################
    # STEP 5: Footer Search (Acord Form Number)
    ##############################
    form_id = footer_search(image_paths)

    if form_id:
        quote_status["footer_quote_submission"] = True
        quote_status["footer_form_id"] = form_id

    ##############################
    # STEP 6: Gemini Classify
    ##############################
    # Convert PDF to Images
    gemini_output = gemini_classify(pdf_name, pdf_bytes)

    if gemini_output:
        quote_status["gemini_quote_submission"] = gemini_output["quote_submission"]
        quote_status["gemini_quote_class"] = gemini_output["quote_class"]
        quote_status["gemini_quote_type"] = gemini_output["quote_type"]
        quote_status["gemini_acord_form"] = gemini_output["acord_form"]
        quote_status["gemini_form_id"] = gemini_output["acord_form_id"]

        ##############################
        # STEP 7: Consolidate Results
        ##############################
        if 
        quote_status["regex_quote_submission"]
        quote_status["semantic_quote_submission"]
        quote_status["logo_quote_submission"]
        quote_status["footer_quote_submission"]
        quote_status["gemini_quote_submission"]

        quote_status["regex_quote_class"]
        quote_status["regex_quote_type"]

        quote_status["semantic_quote_class"]
        quote_status["semantic_quote_type"]

        quote_status["logo_acord_form"]

        quote_status["footer_form_id"]

        quote_status["gemini_quote_class"]
        quote_status["gemini_quote_type"]
        quote_status["gemini_acord_form"]
        quote_status["gemini_form_id"]

    #############################################
    logger.info(f"\n{pprint.pformat(quote_status)}")
    return quote_status
