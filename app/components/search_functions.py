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
import cv2
import pytesseract
from PIL import Image
import re
import os
from typing import Optional
import os
import json
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# Set the path to the Tesseract executable if it's not in your PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Import local modules
from app.utils.txt_utils import clean_ocr_text, get_char_vector, get_gemini_key


# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


##############################
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


##############################
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


##############################
def semantic_search(
    text_list: list,
    target_list: list,
):
    temp = []
    for target_label in target_list:
        for txt in text_list:
            threshold = cosine_similarity(target_label, txt)
            if threshold > 0.95:
                temp.append(txt)
                logger.info(
                    f"Match found: {target_label} <-> {txt} (Similarity: {threshold:.4f})"
                )

        if len(temp) > 0:
            return True

    return False


##############################
def logo_search(image_paths, logo_paths, threshold=0.8):
    """
    Detects a specific pattern (logo/header) within a document image using
    OpenCV Template Matching.

    Args:
        image_path (str): Path to the full document image (the ACORD form).
        template_path (str): Path to the cropped logo template image.
        threshold (float): Minimum correlation score required to confirm a match.

    Returns:
        tuple: (bool, float, bounding_box) - Match status, confidence score,
               and coordinates if found.
    """
    # Initialize a global best match tracker (if searching all pages/logos for the single best match)
    global_best_val = -np.inf
    global_result = (False, 0.0, None)

    # Define scale steps
    scales = np.linspace(0.2, 1.0, 10)[::-1]

    for image_path in image_paths:
        # Load and convert image ONCE per page
        img = cv2.imread(image_path)
        if img is None:
            continue  # Skip if image loading failed
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h_img, w_img = img_gray.shape

        for logo_path in logo_paths:
            # Load template ONCE per logo
            template_orig = cv2.imread(logo_path)
            if template_orig is None:
                continue
            template_gray_orig = cv2.cvtColor(template_orig, cv2.COLOR_BGR2GRAY)
            h_temp_orig, w_temp_orig = template_gray_orig.shape

            best_match_val = -np.inf
            best_match_loc = None
            best_match_scale = None

            # Loop over the scales
            for scale in scales:
                # 1. Resize the template
                w_resized = int(w_temp_orig * scale)
                h_resized = int(h_temp_orig * scale)

                resized_template = cv2.resize(
                    template_gray_orig, (w_resized, h_resized)
                )

                # 2. Check size constraints
                if h_resized > h_img or w_resized > w_img:
                    continue

                # 3. Perform template matching
                result = cv2.matchTemplate(
                    img_gray, resized_template, cv2.TM_CCOEFF_NORMED
                )
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                # 4. Update best match (local to this logo/page)
                if max_val > best_match_val:
                    best_match_val = max_val
                    best_match_loc = max_loc
                    best_match_scale = scale
                    best_match_page = image_path
                    best_match_logo = logo_path

            # --- After Scale Loop ---

            # 5. Check if the best match found exceeds the threshold
            if best_match_val >= threshold and best_match_loc is not None:

                # Update global best match if current match is better
                if best_match_val > global_best_val:
                    global_best_val = best_match_val

                    # 6. Correctly calculate the bounding box using the best_match_scale
                    final_w = int(w_temp_orig * best_match_scale)
                    final_h = int(h_temp_orig * best_match_scale)

                    top_left = best_match_loc
                    bottom_right = (top_left[0] + final_w, top_left[1] + final_h)

                    # Save IMG (FOR TESTING)
                    # img = cv2.imread(best_match_page)
                    # cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), 4)
                    # cv2.imwrite("TEST_IMG.jpg", img)

                    # Update global result
                    global_result = (True, global_best_val, (top_left, bottom_right))

                    # (Optional) If you want to return the first successful match immediately:
                    # return global_result

    # Return the single best match found across all pages/logos
    return global_result


##############################
def footer_search(image_paths) -> Optional[str]:
    """
    Identifies the ACORD form type (e.g., ACORD 125) by performing OCR
    on the expected footer region of the first page image.

    Args:
        image_paths (str): Path to the image file (should be Page 1 of the document).

    Returns:
        Optional[str]: The identified form number (e.g., "ACORD 125"), or None if not found.
    """
    for image_path in image_paths:
        # Load and convert image ONCE per page
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return None

        try:
            img = Image.open(image_path)
        except Exception as e:
            print(f"Error opening image: {e}")
            return None

        # 1. Define the Region of Interest (ROI) for the footer
        width, height = img.size

        # We focus on the bottom 10% of the image, as ACORD IDs are typically there.
        # Adjusting this ROI may be necessary based on your typical document resolution.
        crop_area_percentage = 0.90  # Start 90% down the image

        # (left, top, right, bottom)
        footer_roi_box = (0, int(height * crop_area_percentage), width, height)

        # 2. Crop the image to the ROI
        try:
            footer_roi = img.crop(footer_roi_box)
        except ValueError as e:
            print(f"Error cropping image (ROI dimensions error): {e}")
            return None

        # 3. Perform OCR on the restricted area
        # Use config='--psm 6' (Assume a single uniform block of text) for better accuracy
        # on small regions.
        ocr_text = pytesseract.image_to_string(footer_roi, config="--psm 6").strip()

        if not ocr_text:
            # Fallback: Try a slightly higher ROI just in case
            print("No text found in primary footer ROI. Trying broader area.")
            footer_roi_box = (0, int(height * 0.85), width, height)  # Start 85% down
            footer_roi = img.crop(footer_roi_box)
            ocr_text = pytesseract.image_to_string(footer_roi, config="--psm 6").strip()

        # 4. Use Regular Expressions to find the ACORD pattern
        # Pattern explanation:
        # r'ACORD\s*(\d{3})'
        # - ACORD: literally "ACORD"
        # - \s*: zero or more whitespace characters (handles OCR spacing errors)
        # - (\d{3}): exactly three digits (captures the form number, e.g., 125)

        match = re.search(r"ACORD\s*(\d{3})", ocr_text, re.IGNORECASE)

        if match is None:
            match = re.search(r"ACORD\s*(\d{2})", ocr_text, re.IGNORECASE)

        if match:
            form_number = match.group(1)  # The captured three digits
            return f"ACORD {form_number}"
        else:
            # Check for the ACORD trademark text, which might appear separately
            if re.search(r"ACORD", ocr_text, re.IGNORECASE):
                print(
                    "Text contains 'ACORD' but specific form number (###) not identified."
                )

            return None


##############################
class ClassificationOutput(BaseModel):
    """Schema for structured extraction of insurance form details."""

    quote_submission: bool = Field(
        description="Returns true if the PDF file is a valid insurance quote form or application. Else returns false."
    )
    quote_class: str = Field(
        description="Returns 'personal' for personal auto and property quote requests or 'commercial' for commercial auto and property quote request forms. Return 'NA' if the class is not clearly determined."
    )
    quote_type: str = Field(
        description="Return 'auto' for all types of auto quotes (both personal and commercial) and return 'property' for all sorts of property quotes (including Homeowners, Commercial Property, etc.). Return 'NA' if nothing is found."
    )
    acord_form: bool = Field(
        description="Return true if the form is one of the ACORD standard forms (e.g., ACORD 125, 130). Else return false."
    )
    acord_form_id: str = Field(
        description="If the form is a standard ACORD form, return its standard form number (e.g., 'ACORD 125', 'ACORD 140') exactly as found on the document. Else return 'NA'."
    )


##############################
def gemini_classify(
    pdf_name: str,
    pdf_bytes: bytes,
) -> ClassificationOutput | None:
    """
    Analyzes a PDF provided as bytes using the Gemini API and extracts structured information.

    Args:
        pdf_name: A descriptive name for the file, used for uploading.
        pdf_bytes: The byte content of the PDF file.

    Returns:
        An instance of InsuranceFormAnalysis populated with the results, or None on failure.
    """
    try:
        # Load Gemini API Key
        api_key = get_gemini_key("../gemini_key.txt")

        # Initialize the Gemini Client
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(
            f"Error initializing Gemini client. Check GEMINI_API_KEY environment variable. Error: {e}"
        )
        return None

    # uploaded_file = None
    try:
        # Define the analysis prompt
        prompt = (
            "Analyze the attached insurance document (PDF). Based on the form's "
            "content, title, and markings, determine the following characteristics. "
            "Strictly adhere to the provided JSON schema for the output. "
            "If information cannot be conclusively determined, use 'NA'."
        )

        # Configure the model to return structured JSON matching the Pydantic schema
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ClassificationOutput,
        )

        # Generate Content
        response = client.models.generate_content(
            model="gemini-2.5-flash",  # Use Pro for complex document reasoning
            contents=[
                types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type="application/pdf",
                ),
                prompt,
            ],
            config=config,
        )

        # Parse and Validate the Structured JSON output
        # The response text will be a JSON string that conforms to the schema
        json_data = json.loads(response.text)

        # Pydantic validates the JSON structure
        analysis_result = ClassificationOutput(**json_data)

        return dict(analysis_result)

    except Exception as e:
        print(f"An error occurred during Gemini analysis: {e}")
        return None
