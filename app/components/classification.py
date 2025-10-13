import os
from pathlib import Path
import re
import string
import unicodedata
import math
import logging
from nltk import ngrams
from statistics import mode
import pandas as pd
import pytesseract
from PIL import Image
import pypdfium2
import io


# --- 1. Logging Configuration ---
# Set up basic configuration for logging
# level=logging.DEBUG ensures that DEBUG messages (the lowest level) are captured
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


from app.components.semantic_similarity import (
    cosine_similarity,
    semantic_search,
    personal_words,
    personal_bigrams,
    personal_trigrams,
    commercial_words,
    commercial_bigrams,
    commercial_trigrams,
)
from app.utils.txt_utils import clean_ocr_text
from app.utils.pdf_utils import extract_pdf_text


def quote_classification(pdf_bytes: bytes):
    """
    Processes the PDF bytes using pypdfium2 to extract text from the first page.

    Args:
        pdf_bytes: The PDF content as bytes.
    """
    # print("Processing PDF content with pypdfium2...")

    # 1. Load the PDF document directly from the bytes object
    # pypdfium2's PdfDocument.open() handles byte streams naturally.

    form_type = set()
    claim_type = set()
    claim_cat = set()

    clean_text = extract_pdf_text(pdf_bytes)

    # Create bi/tri-grams
    two_word_phrases = [" ".join(pair) for pair in ngrams(clean_text, 2)]
    three_word_phrases = [" ".join(trio) for trio in ngrams(clean_text, 3)]

    # Classify personal vs commercial
    if ("personal" not in claim_cat) and (
        semantic_search(personal_words, clean_text)
        or semantic_search(personal_bigrams, two_word_phrases)
        or semantic_search(personal_trigrams, three_word_phrases)
    ):
        claim_cat.add("personal")

    if ("commercial" not in claim_cat) and (
        semantic_search(commercial_words, clean_text)
        or semantic_search(commercial_bigrams, two_word_phrases)
        or semantic_search(commercial_trigrams, three_word_phrases)
    ):
        claim_cat.add("commercial")

        # if ("personal auto" not in claim_type) and (semantic_search(personal_auto_bigrams, two_word_phrases)):
        #     claim_type.add("personal auto")
        # elif ("personal auto" not in claim_type) and (semantic_search(commercial_auto_bigrams, two_word_phrases) or semantic_search(commercial_auto_trigrams, three_word_phrases)):
        #     claim_type.add("commercial auto")
        # elif ("personal auto" not in claim_type) and (semantic_search(personal_property_bigrams, two_word_phrases)):
        #     claim_type.add("personal property")
        # elif ("personal auto" not in claim_type) and (semantic_search(commercial_property_bigrams, two_word_phrases)):
        #     claim_type.add("commercial property")

    return claim_cat


def search_logo(pdf_bytes):

    try:
        pdf_document = pypdfium2.PdfDocument(pdf_bytes)
    except Exception as e:
        logger.error(
            f"Failed to open PDF from bytes. Ensure the file is a valid PDF. Error: {e}"
        )
        return

    num_pages = len(pdf_document)
    # print(f"Document successfully loaded. Total pages: {num_pages}")

    if num_pages == 0:
        logger.warning("Document is empty.")
        return

    # 2. Access the first page
    page_index = 0
    page = pdf_document.get_page(page_index)

    form_type = set()
    claim_type = set()
    claim_cat = set()

    # 3. Use the PDF Text Page object for Logo detection
    for i in [70, 80, 90, 100, 150, 200]:
        bitmap = page.render(scale=300 / i)
        image = bitmap.to_pil()

        # Extract Text using Tesseract
        text1 = pytesseract.image_to_string(image)

        # Clean Text
        text2 = clean_ocr_text(text1.lower())
        clean_text = text2.split()

        # Create bi/tri-grams
        two_word_phrases = [" ".join(pair) for pair in ngrams(clean_text, 2)]
        three_word_phrases = [" ".join(trio) for trio in ngrams(clean_text, 3)]

        # Classify personal vs commercial
        if ("personal" not in claim_cat) and (
            semantic_search(personal_words, clean_text)
            or semantic_search(personal_bigrams, two_word_phrases)
            or semantic_search(personal_trigrams, three_word_phrases)
        ):
            claim_cat.add("personal")

        if ("commercial" not in claim_cat) and (
            semantic_search(commercial_words, clean_text)
            or semantic_search(commercial_bigrams, two_word_phrases)
            or semantic_search(commercial_trigrams, three_word_phrases)
        ):
            claim_cat.add("commercial")
