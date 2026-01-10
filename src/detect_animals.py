"""Efficient wildlife detection script using YOLOv8 and image classification models.

This script provides a resource-efficient alternative to LLM-based classification,
using computer vision models optimized for object detection and classification.

Main approach:
- Uses YOLOv8 for object detection and localization
- Maps detected COCO classes to wildlife categories
- Processes images from wildlife camera directory structure
- Outputs results compatible with labelling.py format

Dependencies:
- ultralytics (YOLOv8)
- torch
- torchvision
- PIL
- pandas
- opencv-python

pip install ultralytics torch torchvision opencv-python pandas matplotlib pillow

Usage:
    python detect_animals.py
"""

import time
from pathlib import Path

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from tqdm import tqdm
from ultralytics import YOLO

# Configuration
image_dir = Path("/Users/wri2lr/Downloads")
model_name = "yolov8x"  # Use yolov8x for best accuracy, or yolov8n for fastest speed

# Map COCO classes to wildlife categories
# COCO class IDs for animals we care about
ANIMAL_CLASS_MAP = {
    0: "human",  # person
    15: "jay",  # bird -> jay (approximation)
    16: "dog",  # dog
    17: "hare",  # cat -> hare (approximation, as COCO has no hare)
    18: "dog",  # dog
    19: "cow",  # cow (might appear in rural areas)
    20: "unknown",  # elephant
    21: "hare",  # bear -> hare (no bear in Germany typically)
    22: "squirrel",  # zebra -> squirrel (approximation)
    23: "unknown",  # giraffe
    24: "unknown",  # backpack (ignore)
}

# German wildlife mapping (attempt to map COCO classes to German wildlife)
WILDLIFE_CATEGORIES = {
    "roe_deer": ["deer"],  # COCO has "deer" but often misclassified
    "wild_boar": [],  # Not in COCO, might be detected as "bear" or "cow"
    "pigeon": ["bird"],
    "badger": [],  # Not in COCO, might be "bear" or "cat"
    "marten": ["cat"],
    "dog": ["dog"],
    "fox": ["dog", "cat"],  # Might be confused with dog or cat
    "hare": ["cat", "rabbit"],  # COCO has rabbit as class 23 in some versions
    "squirrel": [],  # Not in standard COCO
    "jay": ["bird"],
    "owl": ["bird"],
    "crow": ["bird"],
    "human": ["person"],
    "none": [],
    "unknown": [],
}

# Reverse mapping from COCO to likely wildlife class
COCO_TO_WILDLIFE = {
    "person": "human",
    "bird": "jay",  # Default bird to jay (could be crow, owl, pigeon)
    "dog": "dog",
    "cat": "marten",  # Cats in wildlife context more likely marten or fox
    "bear": "wild_boar",  # Bears don't exist in Germany, likely wild boar
    "cow": "roe_deer",  # Might be roe deer
    "sheep": "roe_deer",
    "horse": "roe_deer",
}


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
    image_path, image_file, img_class, box=None, save_path=None
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
    """
    img = mpimg.imread(str(image_path))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"{image_file}\nclass: {img_class}")

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


def process_images_with_yolo():
    """Process images using YOLOv8 for efficient animal detection.

    Returns
    -------
    None
        Saves results to CSV and parquet files.
    """
    # Load YOLOv8 model
    print(f"Loading {model_name} model...")
    model = YOLO(f"{model_name}.pt")  # Will download if not present

    # Create output directories
    labels_dir = image_dir / f"labels_{model_name}"
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
    csv_path = labels_dir / f"detection_results_{model_name}.csv"
    processed_images = set()
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
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

    # Process images in batches for efficiency
    batch_size = 16  # Adjust based on GPU memory
    for i in tqdm(
        range(0, len(images_to_process), batch_size), desc="Processing batches"
    ):
        batch = images_to_process[i : i + batch_size]
        batch_paths = [str(img_info["path"]) for img_info in batch]

        # Run batch inference
        predictions = model.predict(
            batch_paths,
            conf=0.25,  # Confidence threshold
            verbose=False,
            device="mps",  # Use 'cuda' for NVIDIA GPU, 'cpu' for CPU, 'mps' for Apple Silicon
        )

        # Process each image's results
        for img_info, result in zip(batch, predictions):
            location_id = img_info["location_id"]
            image_file = img_info["name"]
            image_path = img_info["path"]

            # Get detections
            img_class = "none"
            box = None
            confidence = 0.0

            if len(result.boxes) > 0:
                # Get the detection with highest confidence
                confidences = result.boxes.conf.cpu().numpy()
                best_idx = np.argmax(confidences)
                confidence = float(confidences[best_idx])

                # Get class name
                class_id = int(result.boxes.cls[best_idx].cpu().numpy())
                coco_class = result.names[class_id]

                # Map to wildlife category
                img_class = COCO_TO_WILDLIFE.get(coco_class, "unknown")

                # Get bounding box in YOLO format (normalized)
                bbox = result.boxes.xywhn[best_idx].cpu().numpy()
                box = bbox.tolist()  # [x_center, y_center, width, height]

            # Detect lighting
            lighting = detect_lighting(image_path)

            result_dict = {
                "location_id": location_id,
                "image_file": image_file,
                "class": img_class,
                "box": box,
                "lighting": lighting,
                "confidence": confidence,
            }
            results.append(result_dict)

            # Save to CSV incrementally
            df = pd.DataFrame([result_dict])
            write_header = not csv_path.exists() or csv_path.stat().st_size == 0
            df.to_csv(csv_path, mode="a", header=write_header, index=False)

            # Save annotated image
            label_save_path = images_output_dir / f"{location_id}_{image_file}"
            show_image_with_detection(
                image_path, image_file, img_class, box=box, save_path=label_save_path
            )

        # Convert to parquet periodically
        if (i // batch_size) % 10 == 0 and csv_path.exists():
            df_all = pd.read_csv(csv_path)
            df_all.to_parquet(
                labels_dir / f"detection_results_{model_name}.parquet", index=False
            )

    # Final conversion to parquet
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.to_parquet(
            labels_dir / f"detection_results_{model_name}.parquet", index=False
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
    process_images_with_yolo()
