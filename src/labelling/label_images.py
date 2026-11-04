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

import gc
import logging
import os
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
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
import numpy as np
import pandas as pd
import torch
from PIL import Image
from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife.models import detection as pw_detection
from tqdm import tqdm

try:
    from . import label_data
    from .organize_images import image_paths, nfc
    from .species import normalize_species
except ImportError:
    import label_data
    from organize_images import image_paths, nfc
    from species import normalize_species

# Configuration
repo_dir = Path(__file__).resolve().parents[2]
# MegaDetector model options: "MegaDetectorV5" or "MegaDetectorV6"
# MegaDetectorV6 versions: "MDV6-yolov9-c", "MDV6-yolov9-e", "MDV6-yolov10-c", "MDV6-yolov10-e", "MDV6-rtdetr-c"
model_version = "MegaDetectorV6"
model_name = "MDV6-yolov10-e"  # Best accuracy

detection_threshold = 0.4  # Minimum MegaDetector confidence for the best detection
# Further boxes need more confidence: a missed animal beats a false positive.
additional_detection_threshold = 0.8

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


def parse_detections(detections, img_w, img_h):
    """Return confident (confidence, MegaDetector class, [x_min, y_min, w, h]) tuples.

    Sorted by confidence; only the best detection may use the lower threshold.
    """
    if detections is None or len(detections) == 0:
        return []
    parsed = []
    if hasattr(detections, "confidence"):
        for confidence, class_id, (x1, y1, x2, y2) in zip(
            detections.confidence, detections.class_id, detections.xyxy, strict=True
        ):
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            parsed.append(
                (
                    float(confidence),
                    MEGADETECTOR_CLASS_NAMES.get(int(class_id), "unknown"),
                    [x1 / img_w, y1 / img_h, (x2 - x1) / img_w, (y2 - y1) / img_h],
                )
            )
    else:
        for detection in detections:
            try:
                parsed.append(
                    (
                        float(detection["conf"]),
                        MEGADETECTOR_CLASS_NAMES.get(
                            int(detection["category"]), "unknown"
                        ),
                        [float(value) for value in detection["bbox"]],
                    )
                )
            except KeyError, TypeError, ValueError:
                continue
    parsed.sort(key=lambda detection: detection[0], reverse=True)
    return [
        detection
        for index, detection in enumerate(parsed)
        if detection[0]
        >= (detection_threshold if index == 0 else additional_detection_threshold)
    ]


def classify_detection(img_pil, bbox, megadetector_class, classifier, image_file):
    """Return (label, classification confidence) for one detection."""
    if megadetector_class == "person":
        return "human", None
    if megadetector_class == "vehicle":
        return "vehicle", None
    if megadetector_class != "animal":
        return "unknown", None
    if classifier is None:
        return "animal", None
    cropped_img = crop_detection(img_pil, bbox)
    if cropped_img is None:
        return "animal", None
    try:
        result = classifier.single_image_classification(
            np.array(cropped_img), img_id=image_file
        )
    finally:
        cropped_img.close()
    if result and isinstance(result, dict):
        return result.get("prediction", "unknown"), result.get("confidence", 0.0)
    return "animal", None


def detections_for_primary_class(labelled):
    """Keep detections matching the class of the highest-confidence detection.

    `labelled` holds (label, classification confidence, confidence, bbox) tuples
    ordered by detection confidence.
    """
    if not labelled:
        return []
    winner = max(labelled, key=lambda detection: detection[2])[0]
    return sorted(
        (detection for detection in labelled if detection[0] == winner),
        key=lambda detection: detection[2],
        reverse=True,
    )


def yolo_box(bbox):
    """Convert [x_min, y_min, w, h] to YOLO [x_center, y_center, w, h]."""
    x_min, y_min, width, height = bbox
    return [x_min + width / 2, y_min + height / 2, width, height]


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
    return normalize_species(classifier_name)








CAMERA_TIMEZONE = ZoneInfo("Europe/Berlin")


def extract_exif_timestamp(image_path):
    """Read the EXIF capture time as German local wall time.

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


def latest_classifier_weights(models_dir=None):
    models_dir = Path(models_dir) if models_dir is not None else repo_dir / "models"
    return label_data.latest_weights(models_dir)


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
        prediction = self.class_names[index.item()]
        return {
            "prediction": normalize_species(prediction),
            "confidence": confidence.item(),
        }


def load_species_classifier(device):
    return SpeciesClassifier(latest_classifier_weights(), device)


def images_without_manual_labels(image_info_list, existing_df):
    manually_labelled = set()
    if existing_df is not None and "manual_label" in existing_df.columns:
        manually_labelled = {
            (nfc(row["location_id"]), row["image_file"])
            for row in existing_df.to_dict("records")
            if str(row.get("manual_label") or "").strip()
        }
    return [
        image
        for image in image_info_list
        if (nfc(image["location_id"]), image["name"]) not in manually_labelled
    ]


def images_requiring_processing(image_info_list, existing_df, reprocess_missing=False):
    completed = set()
    if existing_df is not None:
        for row in existing_df.to_dict("records"):
            manual = str(row.get("manual_label") or "").strip()
            automatic = str(row.get("class") or "").strip().casefold()
            inferred = str(row.get("classifier_model") or "").strip()
            metadata_complete = all(
                pd.notna(row.get(field)) and str(row.get(field, "")).strip()
                for field in ("timestamp_exif", "temperature_celsius")
            )
            if manual or (
                automatic
                and (inferred or automatic not in {"unknown", "animal"})
                and (not reprocess_missing or metadata_complete)
            ):
                completed.add((nfc(row["location_id"]), row["image_file"]))
    return [
        image
        for image in image_info_list
        if (nfc(image["location_id"]), image["name"]) not in completed
    ]


def sort_images_newest_first(images, metadata_rows=()):
    timestamps = {
        label_data.image_key(row): row.get("timestamp_exif", "")
        for row in metadata_rows
    }

    def sort_key(image):
        key = nfc(image["location_id"]), image["name"]
        saved = timestamps.get(key, "")
        for value in (saved, None):
            if value is None:
                value = extract_exif_timestamp(image["path"])
            try:
                taken = datetime.fromisoformat(value)
            except ValueError, TypeError:
                continue
            if taken.tzinfo is None:
                taken = taken.replace(tzinfo=CAMERA_TIMEZONE)
            return (0, -taken.timestamp(), *key)
        return (1, 0, *key)

    images.sort(key=sort_key)


def process_images_with_pytorch_wildlife(
    camera=None,
    reprocess_missing=False,
    unlabelled_only=False,
    output_csv_path=None,
    image_keys=None,
    force=False,
):
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

    classifier_run = (
        classification_model.weights_path.parent.name
        if classification_model is not None
        and classification_model.weights_path is not None
        else "deepfaune_pretrained"
        if use_classification
        else ""
    )
    # Collect all images with location info from images/<year>/<location>/
    indexed_paths = image_paths()
    if image_keys is not None:
        requested = {(nfc(location), name) for location, name in image_keys}
        indexed_paths = {
            key: path for key, path in indexed_paths.items() if key in requested
        }
    image_info_list = [
        {"path": path, "name": name, "location_id": location_id}
        for (location_id, name), path in indexed_paths.items()
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    locations = {info["location_id"] for info in image_info_list}

    # Load existing results to avoid reprocessing
    separated = (
        output_csv_path is None or Path(output_csv_path).name == "human_labels.csv"
    )
    if separated:
        model_dir = (
            classification_model.weights_path.parent
            if classification_model is not None
            and classification_model.weights_path is not None
            else repo_dir / "models" / (classifier_run or "megadetector")
        )
        csv_path = label_data.predictions_path(repo_dir, model_dir.name)
    else:
        csv_path = Path(output_csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing labels to: {csv_path}")

    # Define explicit dtypes to avoid FutureWarning about NA entries
    COLUMN_DTYPES = {
        "location_id": str,
        "timestamp_exif": str,
        "image_file": str,
        "class": str,
        "box": str,
        "additional_boxes": str,
        "lighting": str,
        "confidence": float,
        "classification_confidence": float,
        "temperature_celsius": float,
        "manual_label": str,
        "classifier_model": str,
    }

    existing_df = None
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path, keep_default_na=False, dtype=str)
        existing_df = existing_df.drop(columns="timestamp", errors="ignore")
        if not separated and "manual_label" not in existing_df:
            existing_df["manual_label"] = ""

    original_boxes = (
        {
            (nfc(row["location_id"]), row["image_file"]): row.get("box", "")
            for row in existing_df.to_dict("records")
        }
        if existing_df is not None
        else {}
    )
    if force:
        images_to_process = image_info_list
    else:
        images_to_process = images_requiring_processing(
            image_info_list, existing_df, reprocess_missing=reprocess_missing
        )

    sort_images_newest_first(
        images_to_process,
        label_data.read_records(repo_dir / "labels/image_metadata.csv"),
    )

    print(
        f"Found {len(image_info_list)} total images in {len(locations)} locations, "
        f"processing {len(images_to_process)} images "
        f"(skipping {len(image_info_list) - len(images_to_process)} completed images)\n"
    )

    if images_to_process and csv_path.exists():
        backup = label_data.backup_csv(csv_path, "relabel")
        if backup is not None:
            print(f"Full CSV backup: {backup}")

    results = []
    start_time = time.time()

    def save_results():
        nonlocal existing_df
        if not results:
            return
        # Create DataFrame from accumulated results with explicit dtypes
        new_rows_df = pd.DataFrame(results).astype(COLUMN_DTYPES)
        if separated:
            new_rows_df = new_rows_df.drop(columns="manual_label")
        if csv_path.exists():
            existing_df = pd.read_csv(csv_path, keep_default_na=False, dtype=str)
            existing_df = existing_df.drop(columns="timestamp", errors="ignore")
            if not separated and "manual_label" not in existing_df:
                existing_df["manual_label"] = ""
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
            saved_boxes = refreshed.get(
                "box", pd.Series("", index=refreshed.index)
            ).copy()
            for column in new_rows_df.columns:
                if column != "manual_label":
                    values = new_rows_df[column].reindex(refreshed.index).to_numpy()
                    if column == "box":
                        values = [
                            saved if saved != original_boxes.get(key, "") else automatic
                            for key, saved, automatic in zip(
                                refreshed.index, saved_boxes, values, strict=True
                            )
                        ]
                    refreshed[column] = values
            added = new_rows_df.loc[~new_rows_df.index.isin(existing_df.index)]
            existing_df = pd.concat([retained, refreshed, added]).reset_index()
        else:
            existing_df = new_rows_df
        temporary_csv = csv_path.with_suffix(".csv.tmp")
        existing_df.to_csv(temporary_csv, index=False)
        temporary_csv.replace(csv_path)
        if separated:
            label_data.save_image_metadata(
                repo_dir, pd.DataFrame(results).fillna("").to_dict("records")
            )
        # Clear results to free memory
        results.clear()

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
                        str(image_path), det_conf_thres=detection_threshold
                    )

                img_class = "none"
                box = None
                additional_boxes = []
                confidence = 0.0
                classification_confidence = None
                detections = (
                    detection_result.get("detections")
                    if isinstance(detection_result, dict)
                    else None
                )
                labelled = detections_for_primary_class(
                    [
                        (
                            *classify_detection(
                                img_pil,
                                bbox,
                                megadetector_class,
                                classification_model,
                                image_file,
                            ),
                            detection_confidence,
                            bbox,
                        )
                        for detection_confidence, megadetector_class, bbox in (
                            parse_detections(detections, img_w, img_h)
                        )
                    ]
                )
                if labelled:
                    img_class, classification_confidence, confidence, bbox = labelled[0]
                    box = yolo_box(bbox)
                    additional_boxes = [
                        yolo_box(detection[3]) for detection in labelled[1:]
                    ]

                # Detect lighting using already-loaded PIL image (inside with block)
                lighting, brightness_value = detect_lighting(img_pil)

                exif_timestamp = extract_exif_timestamp(image_path)
        except (Exception, KeyboardInterrupt) as e:
            save_results()
            if isinstance(e, KeyboardInterrupt):
                raise
            raise RuntimeError(
                f"Labelling failed; original image retained: {image_path}"
            ) from e

        result_dict = {
            "location_id": location_id,
            "timestamp_exif": exif_timestamp,
            "image_file": image_file,
            "class": img_class,
            "box": str(box) if box is not None else "",
            "additional_boxes": str(additional_boxes) if additional_boxes else "",
            "lighting": lighting,
            "confidence": float(confidence),
            "classification_confidence": float(classification_confidence)
            if classification_confidence is not None
            else 0.0,
            "temperature_celsius": np.nan,
            "manual_label": "",
            "classifier_model": classifier_run,
        }
        results.append(result_dict)

        # Save to CSV in batches to avoid memory issues
        # Only save every BATCH_SIZE images or on the last image
        if (idx + 1) % BATCH_SIZE == 0 or (idx + 1) == len(images_to_process):
            save_results()

            # Force garbage collection after each batch
            gc.collect()

            # Clear torch cache if using GPU/MPS
            if device in ["cuda", "mps"]:
                if device == "cuda":
                    torch.cuda.empty_cache()
                elif device == "mps":
                    torch.mps.empty_cache()

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
