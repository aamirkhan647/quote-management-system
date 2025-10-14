import re
import logging
import pypdfium2
import pytesseract
from nltk import ngrams
import numpy as np
from collections import Counter
from typing import Dict, List
import os
from pathlib import Path
import re
import string
import unicodedata
import math
from nltk import ngrams
from statistics import mode
import pandas as pd
import pytesseract
from PIL import Image
import pypdfium2
import logging

# Import local modules
from app.utils.txt_utils import clean_ocr_text, get_char_vector


# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def keyword_matches(paragraph: str, keywords: list) -> int:
    """
    Finds the exact, whole-word matches for a list of keywords in a given paragraph,
    ignoring case.

    Args:
        paragraph: The block of text to search within.
        keywords: A list of phrases to search for.

    Returns:
        An integer indicating the count of total keywords found.
    """

    # We use re.IGNORECASE flag for case-insensitivity
    flags = re.IGNORECASE
    count = 0

    for keyword in keywords:
        # 1. Escape the keyword:
        # This handles cases where the keyword might contain special regex characters
        safe_keyword = re.escape(keyword)

        # 2. Construct the pattern:
        # \b is the word boundary anchor. It ensures that the match must start
        # and end at a boundary (like a space, punctuation, or start/end of string).
        pattern = r"\b" + safe_keyword + r"\b"

        # 3. Find all matches:
        # re.findall returns a list of all non-overlapping matches found.
        matches = re.findall(pattern, paragraph, flags=flags)

        # 4. Update Search Count
        count = count + len(matches)

    return count


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

        # # Classify personal vs commercial
        # if ("personal" not in claim_cat) and (
        #     semantic_search(personal_words, clean_text)
        #     or semantic_search(personal_bigrams, two_word_phrases)
        #     or semantic_search(personal_trigrams, three_word_phrases)
        # ):
        #     claim_cat.add("personal")

        # if ("commercial" not in claim_cat) and (
        #     semantic_search(commercial_words, clean_text)
        #     or semantic_search(commercial_bigrams, two_word_phrases)
        #     or semantic_search(commercial_trigrams, three_word_phrases)
        # ):
        #     claim_cat.add("commercial")


def cosine_similarity(word1: str, word2: str) -> float:
    """
    Calculates the cosine similarity between two words based on their
    Bag-of-Characters (character frequency vectors).

    The cosine similarity measures the cosine of the angle between two
    non-zero vectors, which is a common metric for similarity.

    Args:
        word1: The first word (string).
        word2: The second word (string).

    Returns:
        A float representing the cosine similarity, ranging from 0.0 (no similarity)
        to 1.0 (identical words/vectors).
    """

    # 1. Handle edge cases (empty strings)
    if not word1 or not word2:
        return 0.0

    # 2. Create frequency vectors (Bag-of-Characters)
    vec1 = get_char_vector(word1)
    vec2 = get_char_vector(word2)

    # 3. Determine the set of all unique characters across both words
    all_chars = set(vec1.keys()) | set(vec2.keys())

    # 4. Create the full numerical vectors (lists) ordered by all_chars
    #    This ensures both vectors are aligned in the same dimension space.

    # Dot product components
    numerator = 0

    # Squared sum components (for the denominator magnitude calculation)
    sum_sq1 = 0
    sum_sq2 = 0

    for char in all_chars:
        count1 = vec1.get(char, 0)
        count2 = vec2.get(char, 0)

        # Calculate Dot Product (A * B)
        numerator += count1 * count2

        # Calculate Magnitude Squared (|A|^2 and |B|^2)
        sum_sq1 += count1**2
        sum_sq2 += count2**2

    # 5. Calculate Magnitudes (|A| and |B|)
    # We use np.sqrt, but math.sqrt works too if you prefer not to use numpy
    # If the function must strictly avoid numpy, use math.sqrt:
    magnitude1 = math.sqrt(sum_sq1)
    magnitude2 = math.sqrt(sum_sq2)

    # 6. Calculate Cosine Similarity

    # Check for zero magnitude (shouldn't happen if edge cases handled, but safe practice)
    denominator = magnitude1 * magnitude2
    if denominator == 0:
        return 0.0

    return numerator / denominator


def semantic_search(
    text_list: list,
    target_list: list,
):
    temp = []
    for target_label in target_list:
        for txt in text_list:
            threshold = cosine_similarity(target_label, txt)
            if threshold > 0.97:
                temp.append(txt)
                logger.info(
                    f"Match found: {target_label} <-> {txt} (Similarity: {threshold:.4f})"
                )

        if len(temp) > 0:
            return True

    return False
