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
from collections import Counter
from typing import Dict, List


# --- 1. Normalization Functions ---
def remove_html_tags(text: str) -> str:
    """Removes HTML markup (e.g., <p>, <a> tags) that might be in the text."""
    # Pattern to match anything between < and > non-greedily
    clean = re.compile("<.*?>")
    return re.sub(clean, "", text)


def remove_whitespace(text: str) -> str:
    """
    Removes leading/trailing spaces and replaces multiple internal spaces,
    tabs, or newlines with a single space.
    """
    # Replace all sequences of whitespace characters with a single space
    text = re.sub(r"\s+", " ", text)
    # Strip any leading or trailing space left over
    return text.strip()


def normalize_accents(text: str) -> str:
    """
    Normalizes Unicode characters by decomposing accented characters into
    base characters and their diacritics, then removes the diacritics.

    Example: 'résumé' -> 'resume'
             'François' -> 'Francois'
             'â' -> 'a'
    """
    # 1. Normalize the string to the 'NFKD' form (Compatibility Decomposition)
    # This separates base characters from diacritical marks.
    normalized = unicodedata.normalize("NFKD", text)

    # 2. Encode to ASCII, ignoring errors (this drops the diacritics)
    # Then decode back to UTF-8
    # This specifically removes non-spacing marks that were separated by NFKD.
    return normalized.encode("ascii", "ignore").decode("utf-8")


# --- 2. Removal Functions ---


def remove_punctuation(text: str) -> str:
    """Removes all standard English punctuation marks."""
    # Creates a translation table mapping every punctuation character to None (removal)
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    return text.replace("\n", " ").replace("\r", " ").replace("'", "'").strip()


def remove_digits(text: str) -> str:
    """Removes all numerical digits (0-9) from the text."""
    # \d+ matches one or more digits
    return re.sub(r"\d+", "", text)


def remove_stopwords(text: str) -> str:
    """
    Removes common English stop words.

    NOTE: For production NLP, use a library like NLTK or spaCy for a more
    comprehensive and efficient stop word list.
    """
    # Simple, small list of very common stop words
    stop_words = set(
        [
            "a",
            "an",
            "the",
            "is",
            "are",
            "and",
            "or",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "it",
            "that",
            "this",
            "but",
            "by",
        ]
    )

    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)


# Clean OCR Function
def clean_ocr_text(text: str) -> str:
    text = normalize_accents(text)
    text = remove_punctuation(text)
    text = remove_whitespace(text)
    return text


def get_char_vector(word: str) -> Dict[str, int]:
    """
    Creates a character frequency vector (Bag-of-Characters) for a given word.
    Example: 'apple' -> {'a': 1, 'p': 2, 'l': 1, 'e': 1}
    """
    return Counter(word.lower())


def get_gemini_key(file_path: str = "gemini_key.txt") -> str:
    """
    Reads the API key from the specified text file.

    Args:
        file_path: The path to the file containing the API key.

    Returns:
        The cleaned API key string.

    Raises:
        FileNotFoundError: If the key file does not exist.
    """
    try:
        # 'r' mode for reading, encoding='utf-8' recommended
        with open(file_path, "r", encoding="utf-8") as f:
            # Read the entire content and use .strip() to remove leading/trailing
            # whitespace or newlines, ensuring only the raw key is used.
            api_key = f.read().strip()
            if not api_key:
                raise ValueError("Key file is empty.")
            return api_key
    except FileNotFoundError:
        print(f"Error: Authentication file not found at {file_path}")
        raise
    except Exception as e:
        print(f"Error reading API key from file: {e}")
        raise
