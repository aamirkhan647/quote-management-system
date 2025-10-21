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
import fitz  # PyMuPDF
import os
from typing import List, Optional


# Logging Configuration
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


def convert_pdf_to_images(
    pdf_name,
    pdf_bytes: bytes,
    max_pages: int,
    dpi: int = 300,
) -> Optional[List[str]]:
    """
    Converts the first two pages of a PDF document (provided as bytes)
    into high-resolution images.

    Args:
        pdf_bytes (bytes): The byte content of the PDF file.
        document_name (str): A descriptive name used for saving the output images.
        output_folder (str): The directory where the resulting images will be saved.
        dpi (int): The resolution (DPI) for the output images. 300 DPI recommended.

    Returns:
        Optional[List[str]]: A list of file paths to the generated images (page 0 and page 1),
                             or None if an error occurs.
    """

    if not pdf_bytes:
        print("Error: PDF bytes object is empty.")
        return None

    # 1. Prepare output directory
    os.makedirs("./temp_files", exist_ok=True)

    image_paths = []

    try:
        # Open the PDF document directly from the byte buffer
        # This is the key difference from the file path method
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        # Calculate the scaling factor for the desired DPI
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        # Determine how many pages to process (max 2)
        pages_to_process = min(max_pages, len(doc))

        print(
            f"Found {len(doc)} pages. Converting {pages_to_process} page(s) to images..."
        )

        # 2. Iterate through the first two pages (index 0 and 1)
        for page_num in range(pages_to_process):
            page = doc.load_page(page_num)

            # Create a high-resolution pixmap
            pix = page.get_pixmap(matrix=matrix, alpha=False)

            # Define the output file name using the provided document_name
            output_filename = f"{pdf_name}_page_{page_num + 1}.png"
            output_path = os.path.join("./temp_files", output_filename)

            # Save the image
            pix.save(output_path)
            image_paths.append(output_path)

            print(f"Successfully created: {output_path}")

    except Exception as e:
        print(f"An error occurred during PDF conversion: {e}")
        return None

    finally:
        # Ensure the document is closed
        if "doc" in locals() and doc:
            doc.close()

    return image_paths
