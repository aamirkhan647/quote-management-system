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

from app.components.classification import quote_classification
from app.utils.pdf_utils import load_pdf

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

NPY_DEFAULT_LOG_DOUBLES = 1


def main():
    ########################################################
    # STEP 0: Validate Request
    ########################################################
    # {
    #     "message_id",
    #     "date_received",
    #     "file_name",
    #     "gcs_url",
    # }

    root_path = Path("./sample_forms/")

    if not root_path.is_dir():
        logger.error(f"Error: Root directory not found")
        exit(1)

    logger.info(f"Starting recursive search in: {root_path.resolve()}")

    # Use glob with '**/*.pdf' for recursive search for all files ending in .pdf
    pdf_files = list(root_path.glob("**/*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found.")
        exit(0)

    logger.info(f"Found {len(pdf_files)} PDF file(s).")
    pdf_paths = list(map(str, pdf_files))
    # file_classify = pd.DataFrame(pdf_paths, columns=["File_Path"])
    # file_classify["Claim_Type"] = None
    # file_classify["Form_Type"] = None

    for i in range(len(pdf_paths)):
        logger.info(f"\n--- File {i+1}/{len(pdf_paths)}: {pdf_paths[i]} ---")

        try:
            ########################################################
            # STEP 1: Load Attachment Files
            ########################################################
            # Load the entire file content into a bytes object using pathlib's method
            pdf_bytes = load_pdf(pdf_paths[i])

            ########################################################
            # STEP 2: Perform Quote Form Classification
            ########################################################
            # In-memory Database
            # {
            #     "message_id",
            #     "date_received",
            #     "file_name",
            #     "gcs_url",
            #     "quote_classification": {
            #         "quote_submission": True/False,
            #         "quote_class": "personal/commercial",
            #         "quote_type": "auto/property",
            #         "broker_name": "acord/other",
            #         "form_id": "acord125/custom"
            #     },
            # }

            # Process the loaded bytes
            # print(process_pdf(pdf_bytes))
            path_name = Path(pdf_paths[i]).stem
            val = quote_classification(path_name, pdf_bytes)
            # logger.info(val)
            # file_classify.at[i,"Claim_Category"] = val

        except PermissionError:
            logger.error(f"    [SKIP] Permission denied when accessing {pdf_paths[i]}")
            return False
        except Exception as e:
            logger.error(
                f"    [ERROR] An unexpected error occurred while reading {pdf_paths[i]}: {e}"
            )
            return False
    return True


if __name__ == "__main__":
    main()
