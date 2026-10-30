"""Efficient wildlife detection script using PyTorch Wildlife and MegaDetector.

This script provides a resource-efficient alternative to LLM-based classification,
using computer vision models specifically trained for wildlife camera trap images.

Main approach:
- Uses PyTorch Wildlife with MegaDetector for animal detection
- Detects animals, persons, and vehicles in camera trap images
- Processes images from wildlife camera directory structure
- Outputs results compatible with labelling.py format
- OCR support: macOS Vision (primary) with EasyOCR fallback

Dependencies:
- PytorchWildlife
- torch
- pandas
- opencv-python
- matplotlib
- pyobjc-framework-Vision pyobjc-framework-Quartz (for macOS Vision OCR)
- easyocr (optional, for OCR fallback)

pip install PytorchWildlife torch pandas opencv-python matplotlib pillow lightning omegaconf pyarrow

# For macOS Vision OCR (macOS only):
pip install pyobjc-framework-Vision pyobjc-framework-Quartz

# For EasyOCR fallback (cross-platform):
pip install easyocr

Usage:
    python detect_animals.py
"""

import gc
import logging
import os
import re
import shutil
import sys
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

# Suppress pkg_resources deprecation warning from dependencies (must be before imports)
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")

# Suppress ultralytics/YOLO verbose output
os.environ["YOLO_VERBOSE"] = "False"
logging.getLogger("ultralytics").setLevel(logging.ERROR)

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle
from PIL import Image
from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife.models import detection as pw_detection
from tqdm import tqdm

try:
    from .organize_images import image_paths, nfc
except ImportError:
    from organize_images import image_paths, nfc

# Configuration (must be defined before OCR framework checks)
repo_dir = Path(__file__).resolve().parents[2]
# MegaDetector model options: "MegaDetectorV5" or "MegaDetectorV6"
# MegaDetectorV6 versions: "MDV6-yolov9-c", "MDV6-yolov9-e", "MDV6-yolov10-c", "MDV6-yolov10-e", "MDV6-rtdetr-c"
model_version = "MegaDetectorV6"
model_name = "MDV6-yolov10-e"  # Best accuracy

# Enable species classification (uses DeepFaune classifier for European wildlife)
use_classification = True
classification_threshold = 0.1  # Minimum confidence for classification

# Save annotated images (disable to reduce memory usage and speed up processing)
save_annotated_images = True

# Enable OCR for timestamp/temperature extraction (disable to save memory)
enable_ocr = True

# Enable OCR fallback: if macOS Vision fails, try EasyOCR
# Requires: pip install easyocr
enable_ocr_fallback = True

# OCR Framework availability checks
# macOS Vision is only available on macOS — skip the import entirely on other platforms
# to avoid "ModuleNotFoundError: No module named 'objc'" on Windows/Linux.
if sys.platform == "darwin":
    try:
        import Vision
        from Foundation import NSURL
        from Quartz import CIImage

        MACOS_VISION_AVAILABLE = True
    except ImportError:
        MACOS_VISION_AVAILABLE = False
        print("Warning: macOS Vision not available (PyObjC not installed)")
        if enable_ocr and not enable_ocr_fallback:
            print("Error: OCR enabled but no OCR framework available. Install with:")
            print("  pip install pyobjc-framework-Vision pyobjc-framework-Quartz")
            print("  or enable EasyOCR fallback: pip install easyocr")
            sys.exit(1)
else:
    MACOS_VISION_AVAILABLE = False

# Try to import EasyOCR for fallback
try:
    import easyocr

    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Check OCR availability
if enable_ocr:
    if not MACOS_VISION_AVAILABLE and not EASYOCR_AVAILABLE:
        print("Error: OCR enabled but no OCR framework available.")
        print("Install one of the following:")
        print(
            "  macOS Vision: pip install pyobjc-framework-Vision pyobjc-framework-Quartz"
        )
        print("  EasyOCR:      pip install easyocr")
        sys.exit(1)

# MegaDetector class names
# MegaDetector detects: animal, person, vehicle
MEGADETECTOR_CLASS_NAMES = {
    0: "animal",
    1: "person",
    2: "vehicle",
}

# German wildlife species we want to detect (matching labelling.py)
WILDLIFE_SPECIES = [
    "roe deer",
    "wild boar",
    "pigeon",
    "badger",
    "marten",
    "dog",
    "fox",
    "hare",
    "squirrel",
    "jay",
    "owl",
    "crow",
    "human",
    "none",
    "unknown",
]


def crop_detection(img_pil, bbox):
    """Crop a detected animal from the image for classification.

    Parameters
    ----------
    img_pil : PIL.Image
        Already loaded PIL Image object.
    bbox : list
        Bounding box in format [x_min, y_min, width, height] (normalized).

    Returns
    -------
    PIL.Image or None
        Cropped image as PIL Image, or None if cropping fails.
    """
    try:
        img_w, img_h = img_pil.size

        x_min, y_min, width, height = bbox
        # Convert normalized coordinates to pixel coordinates
        x1 = int(x_min * img_w)
        y1 = int(y_min * img_h)
        x2 = int((x_min + width) * img_w)
        y2 = int((y_min + height) * img_h)

        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)

        # Crop the image and copy it
        cropped = img_pil.crop((x1, y1, x2, y2)).copy()
        return cropped
    except Exception as e:
        print(f"Error cropping image: {e}")
        return None


def map_classifier_to_wildlife(classifier_name):
    """Map DeepFaune classifier output to our wildlife categories.

    Parameters
    ----------
    classifier_name : str
        Class name from the classifier.

    Returns
    -------
    str
        Mapped wildlife category matching labelling.py classes.
    """
    # DeepFaune to our categories mapping
    # DeepFaune has European wildlife classes
    classifier_name_lower = classifier_name.lower()

    # Direct matches
    if "roe deer" in classifier_name_lower or "chevreuil" in classifier_name_lower:
        return "roe deer"
    elif "wild boar" in classifier_name_lower or "sanglier" in classifier_name_lower:
        return "wild boar"
    elif "badger" in classifier_name_lower or "blaireau" in classifier_name_lower:
        return "badger"
    elif (
        "marten" in classifier_name_lower
        or "martre" in classifier_name_lower
        or "stone marten" in classifier_name_lower
        or "fouine" in classifier_name_lower
    ):
        return "marten"
    elif "dog" in classifier_name_lower or "chien" in classifier_name_lower:
        return "dog"
    elif "fox" in classifier_name_lower or "renard" in classifier_name_lower:
        return "fox"
    elif (
        "hare" in classifier_name_lower
        or "rabbit" in classifier_name_lower
        or "lièvre" in classifier_name_lower
        or "lievre" in classifier_name_lower
        or "lapin" in classifier_name_lower
    ):
        return "hare"
    elif (
        "squirrel" in classifier_name_lower
        or "écureuil" in classifier_name_lower
        or "ecureuil" in classifier_name_lower
    ):
        return "squirrel"
    elif "jay" in classifier_name_lower or "geai" in classifier_name_lower:
        return "jay"
    elif (
        "owl" in classifier_name_lower
        or "chouette" in classifier_name_lower
        or "hibou" in classifier_name_lower
    ):
        return "owl"
    elif (
        "crow" in classifier_name_lower
        or "corbeau" in classifier_name_lower
        or "raven" in classifier_name_lower
    ):
        return "crow"
    elif (
        "pigeon" in classifier_name_lower
        or "dove" in classifier_name_lower
        or "colombe" in classifier_name_lower
    ):
        return "pigeon"
    elif (
        "human" in classifier_name_lower
        or "person" in classifier_name_lower
        or "humain" in classifier_name_lower
    ):
        return "human"
    elif "cat" in classifier_name_lower or "chat" in classifier_name_lower:
        # Domestic cats in wildlife context might be feral, but keep as separate
        return "cat"
    elif "bird" in classifier_name_lower or "oiseau" in classifier_name_lower:
        # Generic bird - default to jay
        return "jay"
    elif "deer" in classifier_name_lower or "cerf" in classifier_name_lower:
        # Generic deer - default to roe deer
        return "roe deer"
    else:
        # Unknown species
        return "unknown"


def extract_text_from_image(image_path):
    """Extract text from an image using macOS Vision framework or EasyOCR fallback.

    Parameters
    ----------
    image_path : Path
        Path to the image file.

    Returns
    -------
    str
        Extracted text as a string.
    """
    # Try macOS Vision first if available
    if MACOS_VISION_AVAILABLE:
        try:
            # Create image URL
            image_url = NSURL.fileURLWithPath_(str(image_path.absolute()))

            # Create CIImage
            ci_image = CIImage.imageWithContentsOfURL_(image_url)
            if ci_image is None:
                raise Exception("Could not create CIImage from file")

            # Create text recognition request
            request = Vision.VNRecognizeTextRequest.alloc().init()
            request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
            request.setUsesLanguageCorrection_(True)
            request.setRecognitionLanguages_(["en-US"])

            # Create request handler
            handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(
                ci_image, None
            )

            # Perform request
            success = handler.performRequests_error_([request], None)

            if not success[0]:
                raise Exception("Vision request failed")

            # Extract text from results
            results = request.results()
            if not results:
                raise Exception("No text detected")

            # Combine all recognized text
            text_lines = []
            for observation in results:
                text = observation.text()
                text_lines.append(text)

            text = "\n".join(text_lines)
            if text:
                return text
            else:
                raise Exception("Empty text result")

        except Exception as e:
            if enable_ocr_fallback and EASYOCR_AVAILABLE:
                print(f"    macOS Vision failed ({e}), trying EasyOCR fallback...")
            else:
                print(f"    OCR error: {e}")
                return ""

    # Fall back to EasyOCR if macOS Vision failed or is not available
    if enable_ocr_fallback and EASYOCR_AVAILABLE:
        try:
            # Initialize EasyOCR reader (cached globally for performance)
            if not hasattr(extract_text_from_image, "easyocr_reader"):
                print("    Initializing EasyOCR reader (one-time setup)...")
                extract_text_from_image.easyocr_reader = easyocr.Reader(
                    ["en"],
                    gpu=torch.cuda.is_available() or torch.backends.mps.is_available(),
                )

            # Read text from image
            result = extract_text_from_image.easyocr_reader.readtext(
                str(image_path), detail=0
            )

            # Combine all detected text
            text = "\n".join(result)
            return text

        except Exception as e:
            print(f"    EasyOCR error: {e}")
            return ""

    return ""


def parse_camera_metadata(
    ocr_text, image_path=None, ocr_failures_log=None, camera="seissiger"
):
    """Parse timestamp and temperature from camera metadata text.

    Supported formats:
    SEISSIGER (slash-separated, DD/MM/YYYY) — default:
      SEISSIGER
      WILDKAMERA®
      10/08/2024 16:28:22
      • 025°C
    ZEISS (dot-separated, weekday prefix):
      ZEISS
      AMPHIKANZEL
      • 3°C
      Mo 10.11.2025 07:41:41

    Parameters
    ----------
    ocr_text : str
        Text extracted from image via OCR.
    image_path : Path, optional
        Path to the image file (for logging failures).
    ocr_failures_log : Path, optional
        Path to the OCR failures log file.

    Returns
    -------
    dict
        Dictionary with 'timestamp' and 'temperature_celsius' keys.
    """
    timestamp = None
    temperature = None

    try:
        # Normalize linebreaks to spaces for more robust parsing
        normalized_text = re.sub(r"\s+", " ", ocr_text)

        # Replace common OCR artifacts before temperature parsing
        # Replace letter O with 0 when followed by C (e.g., "OC" -> "0C")
        normalized_text = re.sub(
            r"\bO(?=\s*[°С]?C)", "0", normalized_text, flags=re.IGNORECASE
        )
        # Replace Cyrillic characters that look like temperature numbers/units
        normalized_text = (
            normalized_text.replace("б", "6").replace("С", "C").replace("з", "3")
        )

        # Parse temperature (e.g., "3°C", "-5°C", "15°C", "-12°C", "• 3°C", "-1°", "12C", "-1C", "025°C", "006*C")
        # Match the number immediately before °C, *C, or C, regardless of what precedes it
        # Uses \D (non-digit) or start of string to avoid matching numbers that aren't temperatures
        # Allow optional leading zeros and asterisk in place of degree symbol
        temp_match = re.search(
            r"(?:^|\D)-?0*(\d{1,2})\s*(?:[°*]?C|°)", normalized_text, re.IGNORECASE
        )
        if temp_match:
            temperature = int(temp_match.group(1))
        else:
            # Debug: show what we're trying to parse if temperature pattern doesn't match
            if normalized_text and (
                "°" in normalized_text or "C" in normalized_text
            ):  # Likely contains a temperature
                print(
                    f"    Temperature pattern did not match in: '{normalized_text[:100]}'"
                )
                # Log to file with image path
                if image_path and ocr_failures_log:
                    with open(ocr_failures_log, "a", encoding="utf-8") as f:
                        f.write(f"\n[TEMP FAIL] {image_path.name}\n")
                        f.write(f"Image Path: {image_path.resolve()}\n")
                        f.write(f"OCR Text: {normalized_text}\n")
                        f.write(f"Original: {ocr_text}\n")

        # Parse timestamp
        # Zeiss format: weekday DD.MM.YYYY HH:MM:SS (e.g. "Mo 10.11.2025 07:41:41")
        # Seissiger format: DD/MM/YYYY HH:MM:SS (colons, dots, or asterisks as separators)
        is_seissiger = camera.lower() == "seissiger"
        if is_seissiger:
            date_match = re.search(
                r"(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{1,2})[:.* ](\d{2})[:.* ](\d{2})",
                normalized_text,
            )
        else:
            # Zeiss: allow OCR artifacts between weekday abbreviation, date, and time
            date_match = re.search(
                r"\w+\s*.*?(\d{1,2})\.(\d{1,2})\.(\d{4}).*?(\d{1,2}):(\d{2}):(\d{2})",
                normalized_text,
            )
        if date_match:
            day, month, year, hour, minute, second = date_match.groups()
            try:
                dt = datetime(
                    int(year), int(month), int(day), int(hour), int(minute), int(second)
                )
                timestamp = dt.isoformat()
            except ValueError as ve:
                print(
                    f"    Invalid date/time values: day={day}, month={month}, year={year}, "
                    f"hour={hour}, minute={minute}, second={second} - {ve}"
                )
        else:
            # Debug: show what we're trying to parse if date pattern doesn't match
            if normalized_text and "202" in normalized_text:  # Likely contains a date
                print(f"    Date pattern did not match in: '{normalized_text[:100]}'")
                # Log to file with image path
                if image_path and ocr_failures_log:
                    with open(ocr_failures_log, "a", encoding="utf-8") as f:
                        f.write(f"\n[DATE FAIL] {image_path.name}\n")
                        f.write(f"Image Path: {image_path.resolve()}\n")
                        f.write(f"OCR Text: {normalized_text}\n")
                        f.write(f"Original: {ocr_text}\n")
    except Exception as e:
        print(f"    Metadata parsing error: {e}")

    return {
        "timestamp": timestamp,
        "temperature_celsius": temperature,
    }


def extract_metadata_ocr(image_path, ocr_failures_log=None, camera=None):
    """Extract timestamp and temperature from image using macOS OCR with EasyOCR fallback.

    Tries macOS Vision first, then falls back to EasyOCR if:
    1. macOS Vision fails to extract text, OR
    2. Extracted text cannot be fully parsed (missing timestamp OR temperature)

    Parameters
    ----------
    image_path : Path
        Path to the image file.
    ocr_failures_log : Path, optional
        Path to the OCR failures log file.

    Returns
    -------
    dict
        Dictionary with 'timestamp' and 'temperature_celsius' keys.
    """
    # Try macOS Vision first if available
    if MACOS_VISION_AVAILABLE:
        try:
            # Create image URL
            image_url = NSURL.fileURLWithPath_(str(image_path.absolute()))

            # Create CIImage
            ci_image = CIImage.imageWithContentsOfURL_(image_url)
            if ci_image is None:
                raise Exception("Could not create CIImage from file")

            # Create text recognition request
            request = Vision.VNRecognizeTextRequest.alloc().init()
            request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
            request.setUsesLanguageCorrection_(True)
            request.setRecognitionLanguages_(["en-US"])

            # Create request handler
            handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(
                ci_image, None
            )

            # Perform request
            success = handler.performRequests_error_([request], None)

            if not success[0]:
                raise Exception("Vision request failed")

            # Extract text from results
            results = request.results()
            if not results:
                raise Exception("No text detected")

            # Combine all recognized text
            text_lines = []
            for observation in results:
                text = observation.text()
                text_lines.append(text)

            ocr_text = "\n".join(text_lines)
            if not ocr_text:
                raise Exception("Empty text result")

            # Try to parse the extracted text
            metadata = parse_camera_metadata(
                ocr_text, image_path, ocr_failures_log, camera
            )

            # Check if parsing was successful (both fields extracted)
            if (
                metadata["timestamp"] is not None
                and metadata["temperature_celsius"] is not None
            ):
                return metadata
            else:
                # Parsing failed - at least one field missing
                if enable_ocr_fallback and EASYOCR_AVAILABLE:
                    print(
                        "    macOS Vision text incomplete, trying EasyOCR fallback..."
                    )
                else:
                    return metadata  # Return partial result

        except Exception as e:
            if enable_ocr_fallback and EASYOCR_AVAILABLE:
                print(f"    macOS Vision failed ({e}), trying EasyOCR fallback...")
            else:
                return {"timestamp": None, "temperature_celsius": None}

    # Fall back to EasyOCR if macOS Vision failed or parsing failed
    if enable_ocr_fallback and EASYOCR_AVAILABLE:
        try:
            # Initialize EasyOCR reader (cached globally for performance)
            if not hasattr(extract_text_from_image, "easyocr_reader"):
                print("    Initializing EasyOCR reader (one-time setup)...")
                extract_text_from_image.easyocr_reader = easyocr.Reader(
                    ["en"],
                    gpu=torch.cuda.is_available() or torch.backends.mps.is_available(),
                )

            # Read text from image
            result = extract_text_from_image.easyocr_reader.readtext(
                str(image_path), detail=0
            )

            # Combine all detected text
            ocr_text = "\n".join(result)

            # Parse the EasyOCR result
            metadata = parse_camera_metadata(
                ocr_text, image_path, ocr_failures_log, camera
            )

            # Check if EasyOCR parsing was successful
            missing_fields = []
            if metadata["timestamp"] is None:
                missing_fields.append("timestamp")
            if metadata["temperature_celsius"] is None:
                missing_fields.append("temperature")

            if missing_fields:
                print(
                    f"    ⚠️  Both OCR methods failed to extract: {', '.join(missing_fields)}"
                )

            return metadata

        except Exception as e:
            print(f"    EasyOCR error: {e}")
            return {"timestamp": None, "temperature_celsius": None}

    # No OCR available or all attempts failed
    return {"timestamp": None, "temperature_celsius": None}


CAMERA_TIMEZONE = ZoneInfo("Europe/Berlin")


TIMESTAMP_TOLERANCE = timedelta(minutes=1)


def extract_exif_timestamp(image_path):
    """Read the EXIF capture time as German local wall time (same format as OCR).

    Returns "" if the file or timestamp is missing.
    """
    if not image_path.exists():
        return ""
    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
        exif_ifd = exif.get_ifd(0x8769)
        raw = exif_ifd.get(0x9003) or exif_ifd.get(0x9004) or exif.get(0x0132)
        if not raw:
            return ""
        dt = datetime.strptime(str(raw).strip().rstrip("\x00"), "%Y:%m:%d %H:%M:%S")
        return dt.isoformat()
    except Exception as e:
        print(f"    EXIF timestamp error for {image_path}: {e}")
        return ""


def in_dst_transition(dt):
    """True if a naive local time is ambiguous or non-existent due to a DST switch."""
    return (
        dt.replace(tzinfo=CAMERA_TIMEZONE, fold=0).utcoffset()
        != dt.replace(tzinfo=CAMERA_TIMEZONE, fold=1).utcoffset()
    )


def resolve_exif_timestamp(exif_ts, ocr_ts, image_path, ocr_failures_log=None):
    """Validate the EXIF timestamp against the OCR timestamp and return the value to store.

    Mismatches beyond TIMESTAMP_TOLERANCE are logged and the EXIF value is kept.
    During DST transition hours the OCR timestamp is used.
    """
    if not exif_ts or not ocr_ts:
        return exif_ts
    exif_dt = datetime.fromisoformat(exif_ts)
    ocr_dt = datetime.fromisoformat(ocr_ts)
    if in_dst_transition(exif_dt) or in_dst_transition(ocr_dt):
        return ocr_ts
    if abs(exif_dt - ocr_dt) > TIMESTAMP_TOLERANCE:
        print(
            f"    ⚠️  Timestamp mismatch for {image_path}: EXIF={exif_ts}, OCR={ocr_ts}"
        )
        if ocr_failures_log:
            with open(ocr_failures_log, "a", encoding="utf-8") as f:
                f.write(f"\n[TIMESTAMP MISMATCH] {image_path.name}\n")
                f.write(f"Image Path: {image_path.resolve()}\n")
                f.write(f"EXIF: {exif_ts}\n")
                f.write(f"OCR: {ocr_ts}\n")
    return exif_ts


def detect_lighting(img_input, dim=10, thresh=0.5):
    """Detect whether image is bright or dark using LAB color space.

    Uses the LAB color space to extract the luminous channel (L) which is independent
    of colors. This is much more reliable than grayscale for classifying day vs. night
    images from wildlife cameras.

    Based on: https://github.com/imneonizer/How-to-find-if-an-image-is-bright-or-dark

    Parameters
    ----------
    img_input : PIL.Image, numpy.ndarray, or Path
        Image as PIL Image, numpy array, or path to image file.
    dim : int, optional
        Resize dimension for faster computation (default: 10).
    thresh : float, optional
        Threshold for brightness classification (default: 0.5).
        Range 0-1, where higher values mean stricter bright classification.

    Returns
    -------
    tuple
        (classification, brightness_value) where classification is 'bright' or 'dark',
        and brightness_value is the normalized mean L channel value (0-1)
    """
    # Convert input to cv2 format
    if isinstance(img_input, Image.Image):
        # Convert PIL to numpy array in RGB format, then to BGR for cv2
        img = cv2.cvtColor(np.array(img_input), cv2.COLOR_RGB2BGR)
    elif isinstance(img_input, np.ndarray):
        img = img_input
    else:
        # Assume it's a path
        img = cv2.imread(str(img_input))

    if img is None:
        return "unknown", None

    # Resize image to reduce computation
    img = cv2.resize(img, (dim, dim))

    # Convert color space to LAB format and extract L channel
    L, A, B = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))

    # Normalize L channel by dividing all pixel values with maximum pixel value
    L = L / np.max(L)

    # Calculate mean brightness
    brightness = np.mean(L)

    # Classify based on threshold
    if brightness > thresh:
        return "bright", brightness
    else:
        return "dark", brightness


def show_image_with_detection(
    image_path,
    image_file,
    img_class,
    box=None,
    save_path=None,
    classification_info=None,
    metadata=None,
    lighting=None,
):
    """Display an image with its predicted class and bounding box.

    Parameters
    ----------
    image_path : Path
        Path to the image file.
    image_file : str
        Name of the image file.
    img_class : str
        Predicted class of the animal.
    box : list, optional
        Bounding box in YOLO format [x_center, y_center, width, height].
    save_path : Path, optional
        Path to save the annotated image.
    classification_info : dict, optional
        Additional classification information to display.
    metadata : dict, optional
        OCR metadata (timestamp, temperature) to display.
    lighting : str, optional
        Lighting classification ('bright' or 'dark') to display.
    """
    img = mpimg.imread(str(image_path))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    ax.axis("off")

    # Build title with classification details
    title = f"{image_file}\nclass: {img_class}"
    if classification_info:
        species = classification_info.get("species")
        conf = classification_info.get("confidence")
        if species and conf is not None:
            title += f"\nclassified as: {species} ({conf:.2%})"

    # Add lighting info
    if lighting:
        title += f"\nlighting: {lighting}"

    # Add metadata info
    if metadata:
        if metadata.get("timestamp"):
            title += f"\ntimestamp: {metadata['timestamp']}"
        if metadata.get("temperature_celsius") is not None:
            title += f"\ntemp: {metadata['temperature_celsius']}°C"

    ax.set_title(title)

    # Draw bounding box if present
    if box is not None and isinstance(box, list) and len(box) == 4:
        x_center, y_center, width, height = box
        img_h, img_w = img.shape[:2]
        x = (x_center - width / 2) * img_w
        y = (y_center - height / 2) * img_h
        w = width * img_w
        h = height * img_h
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor="r", facecolor="none")
        ax.add_patch(rect)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    else:
        plt.show()
    # Always close the figure to free memory
    plt.close(fig)


def latest_classifier_weights(models_dir=None):
    models_dir = Path(models_dir) if models_dir is not None else repo_dir / "models"
    candidates = []
    for weights in models_dir.glob("deepfaune_finetuned_*/weights.pt"):
        match = re.fullmatch(
            r"deepfaune_finetuned_(\d{8}-\d{6})_.+", weights.parent.name
        )
        if match and weights.is_file():
            candidates.append((match.group(1), weights.parent.name, weights))
    return max(candidates)[2] if candidates else None


class SpeciesClassifier:
    def __init__(self, weights_path, device):
        self.device = device
        self.weights_path = weights_path
        self.pretrained = weights_path is None
        if self.pretrained:
            print("No fine-tuned checkpoint found; using pretrained DeepFaune")
            self.model = pw_classification.DeepfauneClassifier(
                device=device, class_name_lang="en"
            )
        else:
            try:
                from .finetune_classifier import load_finetuned_classifier
            except ImportError:
                from finetune_classifier import load_finetuned_classifier
            print(f"Loading latest species classifier: {weights_path}")
            self.model, self.class_names, self.transform = load_finetuned_classifier(
                weights_path, device=device
            )

    def single_image_classification(self, image, img_id):
        if self.pretrained:
            result = self.model.single_image_classification(image, img_id=img_id)
            return {
                **result,
                "prediction": map_classifier_to_wildlife(result["prediction"]),
            }
        with Image.fromarray(image).convert("RGB") as crop:
            tensor = self.transform(crop).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = self.model(tensor).softmax(dim=1)[0]
            confidence, index = probabilities.max(dim=0)
        return {
            "prediction": self.class_names[index.item()],
            "confidence": confidence.item(),
        }


def load_species_classifier(device):
    return SpeciesClassifier(latest_classifier_weights(), device)


def process_images_with_pytorch_wildlife(camera=None, reprocess_missing=False):
    """Process images using PyTorch Wildlife MegaDetector for efficient animal detection.

    Returns
    -------
    None
        Saves results to CSV and parquet files.
    """
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # Print OCR framework being used
    if enable_ocr:
        if MACOS_VISION_AVAILABLE:
            ocr_framework = "macOS Vision"
            if enable_ocr_fallback and EASYOCR_AVAILABLE:
                ocr_framework += " (with EasyOCR fallback)"
        elif EASYOCR_AVAILABLE:
            ocr_framework = "EasyOCR"
        else:
            ocr_framework = "None"
        print(f"Using OCR framework: {ocr_framework}")

    # Load MegaDetector model
    print(f"Loading {model_version} {model_name} model...")
    if model_version == "MegaDetectorV6":
        detection_model = pw_detection.MegaDetectorV6(
            device=device, pretrained=True, version=model_name
        )
    elif model_version == "MegaDetectorV5":
        detection_model = pw_detection.MegaDetectorV5(device=device, pretrained=True)
    else:
        raise ValueError(f"Unknown model version: {model_version}")

    # Load classification model if enabled
    classification_model = None
    if use_classification:
        classification_model = load_species_classifier(device)

    if classification_model is None:
        print("WARNING: Classification is disabled or failed to load!")
        print("You will only see 'animal', 'human', 'vehicle', or 'none' as classes.")

    # Create output directories
    classifier_suffix = "_classified" if use_classification else ""
    classifier_run = (
        classification_model.weights_path.parent.name
        if classification_model is not None
        and classification_model.weights_path is not None
        else "deepfaune_pretrained" if use_classification else ""
    )
    run_suffix = f"_{classifier_run}" if classifier_run else ""
    labels_dir = (
        repo_dir / "labels"
        / f"labels_{model_version}_{model_name}{classifier_suffix}{run_suffix}"
    )
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_output_dir = labels_dir / "images"
    images_output_dir.mkdir(exist_ok=True)

    # Initialize OCR failures log file in labels_dir
    ocr_failures_log = labels_dir / "ocr_failures.txt"
    if enable_ocr:
        with open(ocr_failures_log, "w", encoding="utf-8") as f:
            f.write(f"OCR Parsing Failures Log - {datetime.now().isoformat()}\\n")
            f.write("=" * 80 + "\\n")

    # Collect all images with location info from images/<year>/<location>/
    indexed_paths = image_paths()
    image_info_list = [
        {"path": path, "name": name, "location_id": location_id}
        for (location_id, name), path in indexed_paths.items()
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    locations = {info["location_id"] for info in image_info_list}

    image_info_list.sort(key=lambda x: (x["location_id"], x["name"]))

    # Load existing results to avoid reprocessing
    csv_path = (
        labels_dir
        / f"detection_results_{model_version}_{model_name}{classifier_suffix}.csv"
    )
    if not csv_path.exists():
        source_csvs = list((repo_dir / "labels").glob(f"*/{csv_path.name}"))
        if source_csvs:
            source_csv = max(source_csvs, key=lambda path: path.stat().st_mtime_ns)
            shutil.copy2(source_csv, csv_path)
            print(f"Copied existing labels (including manual labels): {source_csv}")
    print(f"Writing labels to: {csv_path}")

    # Define explicit dtypes to avoid FutureWarning about NA entries
    COLUMN_DTYPES = {
        "location_id": str,
        "timestamp": str,
        "timestamp_exif": str,
        "image_file": str,
        "class": str,
        "box": str,
        "lighting": str,
        "confidence": float,
        "classification_confidence": float,
        "temperature_celsius": float,
        "manual_label": str,
    }

    existing_df = None
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path, keep_default_na=False, dtype=str)
        if "manual_label" not in existing_df:
            existing_df["manual_label"] = ""
        backup = csv_path.with_name(
            f"{csv_path.name}.before-relabel-{datetime.now():%Y%m%d-%H%M%S-%f}.bak"
        )
        shutil.copy2(csv_path, backup)
        print(f"Full CSV backup (including manual labels): {backup}")

    images_to_process = image_info_list

    print(
        f"Found {len(image_info_list)} total images in {len(locations)} locations, "
        f"rerunning detection and classification for all {len(images_to_process)} images\n"
    )

    results = []
    start_time = time.time()

    # Batch size for incremental saves (to avoid memory issues)
    BATCH_SIZE = 50  # Reduced from 100 to save memory

    # Process images one by one
    for idx, img_info in enumerate(tqdm(images_to_process, desc="Processing images")):
        location_id = img_info["location_id"]
        image_file = img_info["name"]
        image_path = img_info["path"]

        # Load image once and reuse for multiple operations (with statement ensures cleanup)
        try:
            # Detect incomplete JPEGs (missing FF D9 end marker) before PIL silently loads them
            if image_path.suffix.lower() in (".jpg", ".jpeg"):
                with open(image_path, "rb") as _f:
                    _f.seek(-2, 2)
                    if _f.read(2) != b"\xff\xd9":
                        raise OSError("corrupt JPEG: premature end of data")
            with Image.open(image_path) as img_pil:
                img_w, img_h = img_pil.size

                # Run single image detection (suppress verbose output)
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    detection_result = detection_model.single_image_detection(
                        str(image_path), det_conf_thres=0.6
                    )

                # Get detections
                img_class = "none"
                box = None
                confidence = 0.0
                classification_confidence = None
                classified_species = None  # Store the actual classified species

                if detection_result and "detections" in detection_result:
                    detections = detection_result["detections"]
                    if detections and len(detections) > 0:
                        # PyTorch Wildlife returns detections as a supervision Detections object
                        # Access the underlying arrays
                        if hasattr(detections, "confidence"):
                            # supervision.Detections object
                            confidences = detections.confidence
                            best_idx = confidences.argmax()
                            confidence = float(confidences[best_idx])

                            # Get class ID
                            class_id = int(detections.class_id[best_idx])
                            megadetector_class = MEGADETECTOR_CLASS_NAMES.get(
                                class_id, "unknown"
                            )

                            # Get bounding box in format [x_min, y_min, x_max, y_max]
                            bbox_xyxy = detections.xyxy[best_idx]
                            # Convert to [x_min, y_min, width, height] normalized
                            # Use already loaded image dimensions
                            x_min = float(bbox_xyxy[0]) / img_w
                            y_min = float(bbox_xyxy[1]) / img_h
                            x_max = float(bbox_xyxy[2]) / img_w
                            y_max = float(bbox_xyxy[3]) / img_h
                            width = x_max - x_min
                            height = y_max - y_min
                            bbox = [x_min, y_min, width, height]
                        else:
                            # Fallback: try to handle as dict/list
                            try:
                                best_detection = max(
                                    detections, key=lambda x: x.get("conf", 0)
                                )
                                confidence = float(best_detection["conf"])
                                class_id = int(best_detection["category"])
                                megadetector_class = MEGADETECTOR_CLASS_NAMES.get(
                                    class_id, "unknown"
                                )
                                bbox = best_detection["bbox"]
                            except:
                                # Skip this detection if we can't parse it
                                megadetector_class = None
                                bbox = None

                        # If it's an animal and we have classification enabled, classify the species
                        if (
                            megadetector_class == "animal"
                            and classification_model is not None
                            and bbox is not None
                        ):
                            # Crop the detected animal from already-loaded image
                            cropped_img = crop_detection(img_pil, bbox)

                            if cropped_img is not None:
                                try:
                                    # Run classification on the cropped image
                                    classification_result = classification_model.single_image_classification(
                                        np.array(cropped_img), img_id=image_file
                                    )

                                    # Get top prediction - classification_result is already a dict, not a list!
                                    if classification_result and isinstance(
                                        classification_result, dict
                                    ):
                                        classifier_class = classification_result.get(
                                            "prediction", "unknown"
                                        )
                                        classification_confidence = (
                                            classification_result.get("confidence", 0.0)
                                        )

                                        # Map the classifier output to our categories
                                        classified_species = classifier_class

                                        # Always use the classified species, even if confidence is low
                                        # This way users can see what the model thinks it is
                                        img_class = classified_species
                                    else:
                                        img_class = "animal"
                                except Exception as e:
                                    raise RuntimeError(
                                        f"Classification failed for {image_path}"
                                    ) from e
                                finally:
                                    # Close the cropped image to free memory
                                    cropped_img.close()
                            else:
                                img_class = "animal"
                        elif megadetector_class == "person":
                            img_class = "human"
                        elif megadetector_class == "vehicle":
                            img_class = "vehicle"
                        elif megadetector_class is not None:
                            img_class = "unknown"

                        # Convert bbox to YOLO format [x_center, y_center, width, height]
                        if bbox is not None:
                            x_min, y_min, width, height = bbox
                            x_center = x_min + width / 2
                            y_center = y_min + height / 2
                            box = [x_center, y_center, width, height]

                # Detect lighting using already-loaded PIL image (inside with block)
                lighting, brightness_value = detect_lighting(img_pil)

                # Extract metadata using OCR (only if enabled, still needs file path for Vision API)
                if enable_ocr:
                    metadata = extract_metadata_ocr(
                        image_path, ocr_failures_log, camera
                    )
                else:
                    metadata = {"timestamp": None, "temperature_celsius": None}
        except Exception as e:
            raise RuntimeError(
                f"Labelling failed; original image retained: {image_path}"
            ) from e

        result_dict = {
            "location_id": location_id,
            "timestamp": metadata["timestamp"] if metadata["timestamp"] else "",
            "timestamp_exif": resolve_exif_timestamp(
                extract_exif_timestamp(image_path),
                metadata["timestamp"] or "",
                image_path,
                ocr_failures_log,
            ),
            "image_file": image_file,
            "class": img_class,
            "box": str(box) if box is not None else "",
            "lighting": lighting,
            "confidence": float(confidence),
            "classification_confidence": float(classification_confidence)
            if classification_confidence is not None
            else 0.0,
            "temperature_celsius": float(metadata["temperature_celsius"])
            if metadata["temperature_celsius"] is not None
            else np.nan,
            "manual_label": "",
        }
        results.append(result_dict)

        # Save to CSV in batches to avoid memory issues
        # Only save every BATCH_SIZE images or on the last image
        if (idx + 1) % BATCH_SIZE == 0 or (idx + 1) == len(images_to_process):
            # Create DataFrame from accumulated results with explicit dtypes
            new_rows_df = pd.DataFrame(results).astype(COLUMN_DTYPES)

            if existing_df is not None and len(existing_df) > 0:
                keys = ["location_id", "image_file"]
                existing_df = existing_df.set_index(keys)
                existing_df.index = pd.MultiIndex.from_tuples(
                    [(nfc(location), name) for location, name in existing_df.index],
                    names=keys,
                )
                new_rows_df = new_rows_df.set_index(keys)
                retained = existing_df.loc[~existing_df.index.isin(new_rows_df.index)]
                refreshed = existing_df.loc[
                    existing_df.index.isin(new_rows_df.index)
                ].copy()
                for column in new_rows_df.columns:
                    if column != "manual_label":
                        refreshed[column] = (
                            new_rows_df[column].reindex(refreshed.index).to_numpy()
                        )
                added = new_rows_df.loc[~new_rows_df.index.isin(existing_df.index)]
                existing_df = pd.concat([retained, refreshed, added]).reset_index()
            else:
                existing_df = new_rows_df
            temporary_csv = csv_path.with_suffix(".csv.tmp")
            existing_df.to_csv(temporary_csv, index=False)
            temporary_csv.replace(csv_path)

            # Clear results to free memory
            results = []

            # Force garbage collection after each batch
            gc.collect()

            # Clear torch cache if using GPU/MPS
            if device in ["cuda", "mps"]:
                if device == "cuda":
                    torch.cuda.empty_cache()
                elif device == "mps":
                    torch.mps.empty_cache()

            # Additional aggressive cleanup for macOS Vision framework
            import objc

            objc.recycleAutoreleasePool()

        # Save annotated image (only if enabled)
        if save_annotated_images:
            label_save_path = images_output_dir / f"{location_id}_{image_file}"

            # Prepare classification info for display
            classification_info = None
            if classified_species and classification_confidence is not None:
                classification_info = {
                    "species": classified_species,
                    "confidence": classification_confidence,
                }

            show_image_with_detection(
                image_path,
                image_file,
                img_class,
                box=box,
                save_path=label_save_path,
                classification_info=classification_info,
                metadata=metadata,
                lighting=lighting,
            )

    # Print statistics
    end_time = time.time()
    total_time = end_time - start_time
    num_images = len(images_to_process)
    avg_time_per_image = total_time / num_images if num_images > 0 else 0

    print(
        f"\nTotal execution time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)"
    )
    print(f"Average time per image: {avg_time_per_image:.2f} seconds")
    print(f"Images processed this run: {num_images}")

    # Print class distribution
    if results:
        df_results = pd.DataFrame(results)
        print("\nClass distribution:")
        print(df_results["class"].value_counts())

    # Print OCR failures log location
    if enable_ocr and ocr_failures_log.exists():
        print(f"\n✓ OCR failures logged to: {ocr_failures_log}")


if __name__ == "__main__":
    process_images_with_pytorch_wildlife()
