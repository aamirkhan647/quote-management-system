"""
PDF Text Extractor with OCR Support
Extracts text from scanned PDF files using Tesseract OCR

Requirements:
pip install pytesseract pdf2image Pillow

System requirements:
- Tesseract OCR (https://github.com/tesseract-ocr/tesseract)
- Poppler (for pdf2image)
  - Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/
  - Mac: brew install poppler
  - Linux: sudo apt-get install poppler-utils
"""

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import logging
import os

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

# If tesseract is not in PATH, specify the path (Windows example)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def extract_text_from_scanned_pdf(pdf_path, output_txt_path=None, dpi=300, lang="eng"):
    """
    Extract text from a scanned PDF using OCR.

    Args:
        pdf_path (str): Path to the PDF file
        output_txt_path (str, optional): Path to save extracted text. If None, returns text only
        dpi (int): DPI for image conversion (higher = better quality but slower)
        lang (str): Language for OCR (e.g., 'eng', 'spa', 'fra', 'deu')

    Returns:
        str: Extracted text from all pages
    """

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    logger.info(f"Converting PDF to images...")
    # Convert PDF to list of images
    images = convert_from_path(pdf_path, dpi=dpi)

    logger.info(f"Processing {len(images)} pages...")
    extracted_text = []

    # Process each page
    for i, image in enumerate(images, start=1):
        print(f"Processing page {i}/{len(images)}...")

        # Perform OCR on the image
        text = pytesseract.image_to_string(image, lang=lang)
        extracted_text.append(f"--- Page {i} ---\n{text}\n")

    # Combine all text
    full_text = "\n".join(extracted_text)

    # Save to file if output path is provided
    if output_txt_path:
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"Text saved to: {output_txt_path}")

    print("Extraction complete!")
    return full_text


def extract_text_from_image(image_path, lang="eng"):
    """
    Extract text from a single image file.

    Args:
        image_path (str): Path to the image file
        lang (str): Language for OCR

    Returns:
        str: Extracted text
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang=lang)

    return text
