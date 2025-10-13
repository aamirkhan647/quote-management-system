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


# --- 1. Logging Configuration ---
# Set up basic configuration for logging
# level=logging.DEBUG ensures that DEBUG messages (the lowest level) are captured
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


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
    "commercial",
    "fiduciary",
    "naics",
    "employee/empl",
    "workers",
    "payroll",
    "client",
    "fleet",
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


def get_char_vector(word: str) -> Dict[str, int]:
    """
    Creates a character frequency vector (Bag-of-Characters) for a given word.
    Example: 'apple' -> {'a': 1, 'p': 2, 'l': 1, 'e': 1}
    """
    return Counter(word.lower())


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


def semantic_search(target_list: list, text_list: list):
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
