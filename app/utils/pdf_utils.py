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
import fitz
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


def load_pdf(file_path: str) -> bytes:
    """
    Loads a PDF file from a local repository into a bytes object.

    Args:
        file_path: The path to the PDF file.

    Returns:
        The content of the PDF file as bytes.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Error: The file was not found at {file_path}")

    # print(f"Loading file: {path.name} ({os.path.getsize(path)} bytes)")

    with open(path, "rb") as f:
        pdf_bytes = f.read()

    return pdf_bytes


def extract_pdf_text(pdf_bytes) -> str:
    """
    Extracts all text content from a PDF provided as a bytes object using PyMuPDF.

    Args:
        pdf_bytes: The PDF content as a bytes object.

    Returns:
        A string containing all extracted text, or None if an error occurs.
    """
    full_text = []

    try:
        # 1. Open the PDF document directly from the bytes stream
        # filetype="pdf" is crucial here
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        print(f"Successfully loaded PDF from bytes. Total pages: {doc.page_count}")

        # 2. Iterate through each page
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)

            # 3. Extract text
            text = page.get_text("text")

            # Add a separator between pages for readability
            full_text.append(f"\n--- Page {page_num + 1} ---\n")
            full_text.append(text)

        # 4. Close the document (important for resource management)
        doc.close()

        return "".join(full_text)

    except fitz.FileDataError as e:
        # This catches errors like invalid PDF structure or password protection
        logger.error(f"Error reading PDF data (possibly corrupt or encrypted): {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during text extraction: {e}")
        return None
