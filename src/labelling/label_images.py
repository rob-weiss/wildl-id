"""Efficient wildlife detection script using PyTorch Wildlife and MegaDetector.

This script provides a resource-efficient alternative to LLM-based classification,
using computer vision models specifically trained for wildlife camera trap images.

Main approach:
- Uses PyTorch Wildlife with MegaDetector for animal detection
- Detects animals, persons, and vehicles in camera trap images
- Processes images from wildlife camera directory structure
- Outputs results compatible with labelling.py format

Dependencies:
- PytorchWildlife
- torch
- pandas
- opencv-python
- matplotlib

pip install PytorchWildlife torch pandas opencv-python matplotlib pillow lightning omegaconf pyarrow

Usage:
    python detect_animals.py
"""

import re
import sys
import time
from datetime import datetime
from pathlib import Path

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
    import Vision
    from Foundation import NSURL
    from Quartz import CIImage
except ImportError:
    print("Error: PyObjC not installed. Install with:")
    print("  pip install pyobjc-framework-Vision pyobjc-framework-Quartz")
    sys.exit(1)

# Configuration
image_dir = Path(__file__).parent.parent.parent / "data"
# MegaDetector model options: "MegaDetectorV5" or "MegaDetectorV6"
# MegaDetectorV6 versions: "MDV6-yolov9-c", "MDV6-yolov9-e", "MDV6-yolov10-c", "MDV6-yolov10-e", "MDV6-rtdetr-c"
model_version = "MegaDetectorV6"
model_name = "MDV6-yolov10-e"  # Best accuracy

# Enable species classification (uses DeepFaune classifier for European wildlife)
use_classification = True
classification_threshold = 0.1  # Minimum confidence for classification

# MegaDetector class names
# MegaDetector detects: animal, person, vehicle
MEGADETECTOR_CLASS_NAMES = {
    0: "animal",
    1: "person",
    2: "vehicle",
}

# German wildlife species we want to detect (matching labelling.py)
WILDLIFE_SPECIES = [
    "roe_deer",
    "wild_boar",
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


def crop_detection(image_path, bbox):
    """Crop a detected animal from the image for classification.

    Parameters
    ----------
    image_path : Path
        Path to the image file.
    bbox : list
        Bounding box in format [x_min, y_min, width, height] (normalized).

    Returns
    -------
    PIL.Image or None
        Cropped image as PIL Image, or None if cropping fails.
    """
    try:
        img = Image.open(image_path)
        img_w, img_h = img.size

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

        # Crop the image
        cropped = img.crop((x1, y1, x2, y2))
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
        return "roe_deer"
    elif "wild boar" in classifier_name_lower or "sanglier" in classifier_name_lower:
        return "wild_boar"
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
        or "lièvre" in classifier_name_lower
        or "lievre" in classifier_name_lower
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
        return "roe_deer"
    else:
        # Unknown species
        return "unknown"


def extract_text_from_image(image_path):
    """Extract text from an image using macOS Vision framework.

    Parameters
    ----------
    image_path : Path
        Path to the image file.

    Returns
    -------
    str
        Extracted text as a string.
    """
    try:
        # Create image URL
        image_url = NSURL.fileURLWithPath_(str(image_path.absolute()))

        # Create CIImage
        ci_image = CIImage.imageWithContentsOfURL_(image_url)
        if ci_image is None:
            return ""

        # Create text recognition request
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)

        # Create request handler
        handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(
            ci_image, None
        )

        # Perform request
        success = handler.performRequests_error_([request], None)

        if not success[0]:
            return ""

        # Extract text from results
        results = request.results()
        if not results:
            return ""

        # Combine all recognized text
        text_lines = []
        for observation in results:
            text = observation.text()
            text_lines.append(text)

        return "\n".join(text_lines)
    except Exception as e:
        print(f"    OCR error: {e}")
        return ""


def parse_camera_metadata(ocr_text):
    """Parse timestamp and temperature from camera metadata text.

    Expected format:
    ZEISS
    AMPHIKANZEL
    • 3°C
    Mo 10.11.2025 07:41:41

    Parameters
    ----------
    ocr_text : str
        Text extracted from image via OCR.

    Returns
    -------
    dict
        Dictionary with 'timestamp' and 'temperature_celsius' keys.
    """
    timestamp = None
    temperature = None

    try:
        # Parse temperature (e.g., "3°C", "-5°C", "15°C", "-12°C", "• 3°C")
        # Matches one or two digit numbers (positive or negative) before °C
        temp_match = re.search(r'(-?\d{1,2})\s*°C', ocr_text, re.IGNORECASE)
        if temp_match:
            temperature = int(temp_match.group(1))

        # Parse timestamp (e.g., "Mo 10.11.2025 07:41:41")
        # Format: weekday DD.MM.YYYY HH:MM:SS
        date_match = re.search(
            r"\w+\s+(\d{1,2})\.(\d{1,2})\.(\d{4})\s+(\d{1,2}):(\d{2}):(\d{2})", ocr_text
        )
        if date_match:
            day, month, year, hour, minute, second = date_match.groups()
            dt = datetime(
                int(year), int(month), int(day), int(hour), int(minute), int(second)
            )
            timestamp = dt.isoformat()
    except Exception as e:
        print(f"    Metadata parsing error: {e}")

    return {
        "timestamp": timestamp,
        "temperature_celsius": temperature,
    }


def extract_metadata_ocr(image_path):
    """Extract timestamp and temperature from image using macOS OCR.

    Parameters
    ----------
    image_path : Path
        Path to the image file.

    Returns
    -------
    dict
        Dictionary with 'timestamp' and 'temperature_celsius' keys.
    """
    ocr_text = extract_text_from_image(image_path)
    return parse_camera_metadata(ocr_text)


def detect_lighting(image_path):
    """Detect whether image is bright or dark based on average intensity.

    Parameters
    ----------
    image_path : Path
        Path to the image file.

    Returns
    -------
    str
        'bright' or 'dark'
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return "unknown"

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)

    # Check if image is mostly grayscale (black and white camera)
    # Calculate color variance
    if len(img.shape) == 3:
        std_color = np.std(img, axis=2).mean()
        is_bw = std_color < 10  # Low color variance indicates B&W
    else:
        is_bw = True

    if is_bw or avg_brightness < 80:
        return "dark"
    else:
        return "bright"


def show_image_with_detection(
    image_path,
    image_file,
    img_class,
    box=None,
    save_path=None,
    classification_info=None,
    metadata=None,
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
        plt.close()
    else:
        plt.show()


def process_images_with_pytorch_wildlife():
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

    print("Using macOS Vision framework for OCR")

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
        print("Loading DeepFaune classifier for species identification...")
        try:
            classification_model = pw_classification.DeepfauneClassifier(
                device=device, class_name_lang="en"
            )
            print("✓ Classification model loaded successfully")
        except Exception as e:
            print(f"✗ ERROR: Could not load classification model: {e}")
            import traceback

            traceback.print_exc()
            print("Continuing with detection only...")
            classification_model = None

    if classification_model is None:
        print("WARNING: Classification is disabled or failed to load!")
        print("You will only see 'animal', 'human', 'vehicle', or 'none' as classes.")

    # Create output directories
    classifier_suffix = "_classified" if use_classification else ""
    labels_dir = image_dir / f"labels_{model_version}_{model_name}{classifier_suffix}"
    labels_dir.mkdir(exist_ok=True)
    images_output_dir = labels_dir / "images"
    images_output_dir.mkdir(exist_ok=True)

    # Get all subfolders (locations)
    subfolders = [
        d
        for d in image_dir.iterdir()
        if d.is_dir() and not d.name.startswith("labels_")
    ]

    # Collect all images with location info
    image_info_list = []
    for subfolder in subfolders:
        location_id = subfolder.name
        for image_file in subfolder.iterdir():
            if image_file.is_file() and image_file.name.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                image_info_list.append(
                    {
                        "path": image_file,
                        "name": image_file.name,
                        "location_id": location_id,
                    }
                )

    image_info_list.sort(key=lambda x: (x["location_id"], x["name"]))

    # Load existing results to avoid reprocessing
    csv_path = (
        labels_dir
        / f"detection_results_{model_version}_{model_name}{classifier_suffix}.csv"
    )
    processed_images = set()
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)

        # Define expected columns with default values
        expected_columns = {
            "location_id": "",
            "image_file": "",
            "class": "none",
            "box": None,
            "lighting": "unknown",
            "confidence": 0.0,
            "classification_confidence": None,
            "timestamp": None,
            "temperature_celsius": None,
        }

        # Add missing columns with default values
        for col, default_value in expected_columns.items():
            if col not in existing_df.columns:
                print(
                    f"Adding missing column '{col}' with default value: {default_value}"
                )
                existing_df[col] = default_value

        # Save the updated CSV with all columns
        existing_df.to_csv(csv_path, index=False)

        processed_images = set(
            zip(existing_df["location_id"], existing_df["image_file"])
        )

    images_to_process = [
        img_info
        for img_info in image_info_list
        if (img_info["location_id"], img_info["name"]) not in processed_images
    ]

    print(
        f"Found {len(image_info_list)} total images in {len(subfolders)} locations, "
        f"{len(processed_images)} already processed, {len(images_to_process)} to process\n"
    )

    results = []
    start_time = time.time()

    # Process images one by one
    for img_info in tqdm(images_to_process, desc="Processing images"):
        location_id = img_info["location_id"]
        image_file = img_info["name"]
        image_path = img_info["path"]

        # Run single image detection
        detection_result = detection_model.single_image_detection(
            str(image_path), det_conf_thres=0.2
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
                    # First get image dimensions from the detection result
                    img = Image.open(image_path)
                    img_w, img_h = img.size
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
                        best_detection = max(detections, key=lambda x: x.get("conf", 0))
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
                    print(f"  Classifying animal in {image_file}...")
                    # Crop the detected animal
                    cropped_img = crop_detection(image_path, bbox)

                    if cropped_img is not None:
                        print(f"    Cropped image size: {cropped_img.size}")
                        try:
                            # Run classification on the cropped image
                            classification_result = (
                                classification_model.single_image_classification(
                                    np.array(cropped_img), img_id=image_file
                                )
                            )

                            print(f"    Classification result: {classification_result}")

                            # Get top prediction - classification_result is already a dict, not a list!
                            if classification_result and isinstance(
                                classification_result, dict
                            ):
                                classifier_class = classification_result.get(
                                    "prediction", "unknown"
                                )
                                classification_confidence = classification_result.get(
                                    "confidence", 0.0
                                )

                                print(
                                    f"    Classifier says: {classifier_class} "
                                    f"(confidence: {classification_confidence:.2%})"
                                )

                                # Map the classifier output to our categories
                                classified_species = map_classifier_to_wildlife(
                                    classifier_class
                                )

                                print(f"    Mapped to: {classified_species}")

                                # Always use the classified species, even if confidence is low
                                # This way users can see what the model thinks it is
                                img_class = classified_species

                                # Add a note if confidence was low
                                if classification_confidence < classification_threshold:
                                    print(
                                        f"    WARNING: Low confidence ({classification_confidence:.2%}) "
                                        f"classification: {classified_species}"
                                    )
                            else:
                                print("    No classification result returned")
                                img_class = "animal"
                        except Exception as e:
                            print(f"    Classification error: {e}")
                            import traceback

                            traceback.print_exc()
                            img_class = "animal"
                    else:
                        print("    Failed to crop image")
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

        # Detect lighting
        lighting = detect_lighting(image_path)

        # Extract metadata using OCR
        print("  Extracting metadata via OCR...")
        metadata = extract_metadata_ocr(image_path)

        result_dict = {
            "location_id": location_id,
            "image_file": image_file,
            "class": img_class,
            "box": box,
            "lighting": lighting,
            "confidence": confidence,
            "classification_confidence": classification_confidence,
            "timestamp": metadata["timestamp"],
            "temperature_celsius": metadata["temperature_celsius"],
        }
        results.append(result_dict)

        # Save to CSV incrementally
        df = pd.DataFrame([result_dict])
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        df.to_csv(csv_path, mode="a", header=write_header, index=False)

        # Save annotated image
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
        )

        # Convert to parquet every 100 images
        if len(results) % 100 == 0 and csv_path.exists():
            df_all = pd.read_csv(csv_path)
            df_all.to_parquet(
                labels_dir
                / f"detection_results_{model_version}_{model_name}{classifier_suffix}.parquet",
                index=False,
            )

    # Final conversion to parquet
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.to_parquet(
            labels_dir
            / f"detection_results_{model_version}_{model_name}{classifier_suffix}.parquet",
            index=False,
        )
        print(f"\nConverted results to parquet: {len(df)} total images")

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


if __name__ == "__main__":
    process_images_with_pytorch_wildlife()
