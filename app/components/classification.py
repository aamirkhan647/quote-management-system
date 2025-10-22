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
    quote_classification = {
        "quote_submission": False,
        "quote_class": None,
        "quote_type": None,
        "acord_form": False,
        "form_id": None,
        "human_referral": False,
    }

    # Create classification status in-memory database
    quote_status = {
        "quote_submission_regex": False,
        "quote_submission_semantic": False,
        "quote_submission_logo": False,
        "quote_submission_footer": False,
        "quote_submission_gemini": False,
        "quote_class_regex": "",
        "quote_class_semantic": "",
        "quote_class_gemini": "",
        "quote_type_regex": "",
        "quote_type_semantic": "",
        "quote_type_gemini": "",
        "acord_form_logo": False,
        "acord_form_gemini": False,
        "form_id_footer": "",
        "form_id_gemini": "",
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
        quote_status["quote_submission_regex"] = True
        quote_status["quote_class_regex"] = "personal"
        quote_status["quote_type_regex"] = "auto"

    # Match Personal Property
    if keyword_matches(clean_text, exact_match_keywords["personal_property"]) > 0:
        quote_status["quote_submission_regex"] = True
        quote_status["quote_class_regex"] = "personal"
        quote_status["quote_type_regex"] = "property"

    # Match Commercial Auto
    if keyword_matches(clean_text, exact_match_keywords["commercial_auto"]) > 0:
        quote_status["quote_submission_regex"] = True
        quote_status["quote_class_regex"] = "commercial"
        quote_status["quote_type_regex"] = "auto"

    # Match Commercial Property
    if keyword_matches(clean_text, exact_match_keywords["commercial_property"]) > 0:
        quote_status["quote_submission_regex"] = True
        quote_status["quote_class_regex"] = "commercial"
        quote_status["quote_type_regex"] = "property"

    # Match Personal Only
    if keyword_matches(clean_text, personal_words) > 0:
        quote_status["quote_submission_regex"] = True
        quote_status["quote_class_regex"] = "personal"

    # Match Commercial Only
    if keyword_matches(clean_text, commercial_words) > 0:
        quote_status["quote_submission_regex"] = True
        quote_status["quote_class_regex"] = "commercial"

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
        quote_status["quote_submission_semantic"] = True
        quote_status["quote_class_semantic"] = "personal"

    if semantic_search(two_word_phrases, commercial_bigrams) or semantic_search(
        three_word_phrases, commercial_trigrams
    ):
        quote_status["quote_submission_semantic"] = True
        quote_status["quote_class_semantic"] = "commercial"

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

    match_found, confidence, bounding_box = logo_search(
        image_paths, logo_paths, threshold=0.8
    )

    if match_found:
        quote_status["quote_submission_logo"] = True
        quote_status["acord_form_logo"] = True

    ##############################
    # STEP 5: Footer Search (Acord Form Number)
    ##############################
    form_id = footer_search(image_paths)

    if form_id:
        quote_status["quote_submission_footer"] = True
        quote_status["form_id_footer"] = form_id

    ##############################
    # STEP 6: Gemini Classify
    ##############################
    # Convert PDF to Images
    gemini_output = gemini_classify(pdf_name, pdf_bytes)

    if gemini_output:
        quote_status["quote_submission_gemini"] = gemini_output["quote_submission"]
        quote_status["quote_class_gemini"] = gemini_output["quote_class"]
        quote_status["quote_type_gemini"] = gemini_output["quote_type"]
        quote_status["acord_form_gemini"] = gemini_output["acord_form"]
        quote_status["form_id_gemini"] = gemini_output["acord_form_id"]

    ##############################
    # STEP 7: Consolidate Results
    ##############################
    if not (
        quote_status["quote_submission_regex"]
        or quote_status["quote_submission_semantic"]
        or quote_status["quote_submission_logo"]
        or quote_status["quote_submission_footer"]
        or quote_status["quote_submission_gemini"]
    ):
        quote_classification = {
            "quote_submission": False,
            "quote_class": None,
            "quote_type": None,
            "acord_form": False,
            "form_id": None,
        }

        if {"personal", "commercial", "auto", "property", True}.issubset(
            quote_status.values()
        ):
            quote_classification["human_referral"] = True

        return quote_classification

    else:
        # Quote Submission
        quote_classification["quote_submission"] = True

        # Quote Class
        if "personal" in [
            quote_status["quote_class_regex"],
            quote_status["quote_class_semantic"],
            quote_status["quote_class_gemini"],
        ] and not (
            "commercial"
            in [
                quote_status["quote_class_regex"],
                quote_status["quote_class_semantic"],
                quote_status["quote_class_gemini"],
            ]
        ):
            quote_classification["quote_class"] = "personal"

        elif "commercial" in [
            quote_status["quote_class_regex"],
            quote_status["quote_class_semantic"],
            quote_status["quote_class_gemini"],
        ] and not (
            "personal"
            in [
                quote_status["quote_class_regex"],
                quote_status["quote_class_semantic"],
                quote_status["quote_class_gemini"],
            ]
        ):
            quote_classification["quote_class"] = "commercial"

        else:
            quote_classification["human_referral"] = True

        # Quote Type
        if "auto" in [
            quote_status["quote_type_regex"],
            quote_status["quote_type_semantic"],
            quote_status["quote_type_gemini"],
        ] and not (
            "property"
            in [
                quote_status["quote_type_regex"],
                quote_status["quote_type_semantic"],
                quote_status["quote_type_gemini"],
            ]
        ):
            quote_classification["quote_type"] = "auto"

        elif "property" in [
            quote_status["quote_type_regex"],
            quote_status["quote_type_semantic"],
            quote_status["quote_type_gemini"],
        ] and not (
            "auto"
            in [
                quote_status["quote_type_regex"],
                quote_status["quote_type_semantic"],
                quote_status["quote_type_gemini"],
            ]
        ):
            quote_classification["quote_type"] = "property"
        elif quote_status["quote_type_gemini"] == "both":
            quote_classification["quote_type"] = "both"
        else:
            quote_classification["human_referral"] = True

        # Acord Form
        if (
            quote_status["acord_form_logo"] == True
            and quote_status["acord_form_gemini"] == False
        ) or (
            quote_status["acord_form_logo"] == False
            and quote_status["acord_form_gemini"] == True
        ):
            quote_classification["human_referral"] = True
        elif quote_status["acord_form_logo"] or quote_status["acord_form_gemini"]:
            quote_classification["acord_form"] = True
        else:
            quote_classification["acord_form"] = False

        # Form ID
        if (
            quote_status["form_id_footer"] != ""
            and quote_status["form_id_gemini"] != ""
            and quote_status["form_id_footer"] != quote_status["form_id_gemini"]
        ):
            if quote_status["form_id_footer"] in quote_status["form_id_gemini"]:
                quote_classification["form_id"] = quote_status["form_id_footer"]
            else:
                quote_classification["human_referral"] = True

        elif quote_status["form_id_footer"] != "":
            quote_classification["form_id"] = quote_status["form_id_footer"]

        elif quote_status["form_id_gemini"] != "":
            quote_classification["form_id"] = quote_status["form_id_gemini"]

        else:
            quote_classification["form_id"] = ""

    #############################################
    logger.info(f"\n{pprint.pformat(quote_classification)}")
    return quote_classification
