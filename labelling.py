# %%
"""This script automates the classification and optional bounding box extraction of wildlife camera images using an LLM (Large Language Model) via the Ollama API.

It processes images from a specified directory, sends each image (base64-encoded) to the model with a prompt tailored for wildlife detection in Germany, and expects a JSON response containing the detected class and optionally a bounding box in YOLO format.

Main functionalities:
- Loads images from a directory and encodes them for LLM input.
- Sends each image to the specified LLM model with a detailed prompt.
- Parses the model's JSON response to extract the predicted class and bounding box.
- Optionally displays each image with its predicted class and bounding box.
- Collects all results and saves them to a parquet file for further analysis.

Dependencies:
- base64, json, pathlib, matplotlib, pandas, ollama, tqdm

Functions:
- show_image_with_class(image_path, image_file, img_class): Displays an image with its predicted class as the title.
- process_images(): Main loop for processing images, sending them to the LLM, parsing responses, and saving results.

Usage:
- Configure the image directory and model name as needed.
- Run the script to process all images and save results to 'labelling_results.parquet'.
"""

import base64
import json
import os
import time
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from ollama import Client
from tqdm import tqdm

# Set proxy environment variables
os.environ["no_proxy"] = (
    "localhost,127.0.0.1,127.*,::1,172.16.*,172.17.*,172.18.*,172.19.*,172.20.*,172.21.*,172.22.*,172.23.*,172.24.*,172.25.*,172.26.*,172.27.*,172.28.*,172.29.*,172.30.*,172.31.*,192.168.*,10.*,de.bosch.com,apac.bosch.com,emea.bosch.com,us.bosch.com,inside.bosch.cloud,rb-artifactory.bosch.com,sourcecode01.de.bosch.com,sourcecode06.dev.bosch.com,sourcecode.socialcoding.bosch.com,rb-tracker.bosch.com,mirror-osd.de.bosch.com"
)
os.environ["NO_PROXY"] = (
    "localhost,127.0.0.1,127.*,::1,172.16.*,172.17.*,172.18.*,172.19.*,172.20.*,172.21.*,172.22.*,172.23.*,172.24.*,172.25.*,172.26.*,172.27.*,172.28.*,172.29.*,172.30.*,172.31.*,192.168.*,10.*,de.bosch.com,apac.bosch.com,emea.bosch.com,us.bosch.com,inside.bosch.cloud,rb-artifactory.bosch.com,sourcecode01.de.bosch.com,sourcecode06.dev.bosch.com,sourcecode.socialcoding.bosch.com,rb-tracker.bosch.com,mirror-osd.de.bosch.com"
)

# Echo proxy environment variables
print("Proxy Environment Variables:")
for var in [
    "http_proxy",
    "HTTP_PROXY",
    "https_proxy",
    "HTTPS_PROXY",
    "no_proxy",
    "NO_PROXY",
]:
    print(f"{var}: {os.environ.get(var, 'Not set')}")
print("\n")

model = "qwen3-vl:235b-a22b-thinking"

client = Client(
    host="http://localhost:11434",
)

client.list()
client.show(model=model)
image_dir = Path(os.environ["HOME"] + "/mnt/wildlife")


prompt = (
    "Determine the class of creature in this image.\n"
    "The photo was taken by a wildlife camera in Germany.\n"
    "Common animals include roe deer, wild boar, pigeon, badger, marten, dog, fox, hare, squirrel, jay, owl, and crow.\n"
    "Either there is one or multiple animals of the above species, a human, or no creature in the image.\n"
    "If multiple animals are visible, echo the one that appears most often.\n"
    "If there is an animal, please specify the species.\n"
    "If there is no creature, return the class 'none'.\n"
    "If there is a creature but you don't know what kind of creature, choose class 'unknown'.\n"
    "Return in JSON format just the class with key 'class'.\n"
    "Optionally, if you can identify the location of the creature in the image, also return the bounding box in YOLO format with key 'box'.\n"
    "YOLO format is [x_center, y_center, width, height] with all values normalized between 0 and 1.\n"
    "If there are multiple creatures, return the bounding box of the most prominent one.\n"
    "Also return whether it's bright or dark in the image with key 'lighting' and value either 'bright' or 'dark'.\n"
    "If it is a black and white image, label it as dark.\n"
    "There is a gray bar in the bottom of the image. It contains the date and time in the lower right corner.\n"
    "Add the timestamp as an additional key 'timestamp' in ISO 8601 format (YYYY-MM-DDTHH:MM:SS) if you can read it from the gray bar using OCR.\n"
    "There is a temperature value in the lower right corner of the gray bar. Add it as an additional key 'temperature_celsius' in degrees Celsius if you can read it from the gray bar using OCR.\n"
)

# Collect results in a list
results = []

# Start timing
start_time = time.time()


def show_image_with_class(
    image_path, image_file, img_class, json_data=None, save_path=None
):
    """Display an image with its predicted class as the title.

    Parameters
    ----------
    image_path : Path or str
        The path to the image file.
    image_file : str
        The name of the image file.
    img_class : str
        The predicted class of the image.
    json_data : dict, optional
        The complete JSON response data to display.
    save_path : Path or str, optional
        The path to save the labelled image.
    """
    img = mpimg.imread(str(image_path))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"{image_file} \n class: {img_class}")

    # Draw bounding box if present in json_data
    if json_data and "box" in json_data:
        box = json_data["box"]
        if box is not None and isinstance(box, list) and len(box) == 4:
            x_center, y_center, width, height = box
            img_h, img_w = img.shape[:2]
            x = (x_center - width / 2) * img_w
            y = (y_center - height / 2) * img_h
            w = width * img_w
            h = height * img_h
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor="r", facecolor="none")
            ax.add_patch(rect)

    # Add JSON data as text underneath the image
    if json_data:
        json_text = json.dumps(json_data, indent=2)
        fig.text(
            0.5,
            0.02,
            json_text,
            ha="center",
            va="bottom",
            fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()  # Close the figure without displaying
    else:
        plt.show()


def process_images():
    """Processes images from subfolders in the specified directory, sends them to the LLM for classification,
    parses the responses, and collects the results. Uses subfolder name as location_id.

    Returns:
    -------
    None
        The function appends results to the global 'results' list and prints progress.
    """
    # Create labels directory in image_dir (not in subfolders)
    labels_dir = image_dir / f"labels_{model}"
    labels_dir.mkdir(exist_ok=True)

    # Create images subdirectory for labelled images
    images_output_dir = labels_dir / "images"
    images_output_dir.mkdir(exist_ok=True)

    # Get all subfolders in image_dir
    subfolders = [
        d
        for d in image_dir.iterdir()
        if d.is_dir() and not d.name.startswith("labels_")
    ]

    # Collect all images from subfolders with their location_id (subfolder name)
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

    # Load existing CSV to check which images are already in the dataset
    csv_path = labels_dir / f"labelling_results_{model}.csv"
    processed_images = set()
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        # Create unique key from location_id and image_file
        processed_images = set(
            zip(existing_df["location_id"], existing_df["image_file"])
        )

    # Count images that need processing
    images_to_process = [
        img_info
        for img_info in image_info_list
        if (img_info["location_id"], img_info["name"]) not in processed_images
        or not (
            images_output_dir / f"{img_info['location_id']}_{img_info['name']}"
        ).exists()
    ]
    print(
        f"Found {len(image_info_list)} total images in {len(subfolders)} subfolders, "
        f"{len(processed_images)} already processed, {len(images_to_process)} to process\n"
    )

    for img_info in tqdm(image_info_list, desc="Processing images"):
        location_id = img_info["location_id"]
        image_file = img_info["name"]
        image_path = img_info["path"]

        # Skip only if BOTH CSV entry AND labelled image exist
        label_save_path = images_output_dir / f"{location_id}_{image_file}"
        if (location_id, image_file) in processed_images and label_save_path.exists():
            continue
        with image_path.open("rb") as img_file:
            image_bytes = img_file.read()
            img_base64 = base64.b64encode(image_bytes).decode("utf-8")

        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt, "images": [img_base64]}],
            options={"seed": 42, "temperature": 0},
        )

        img_class = None
        box = None
        try:
            message_content = response["message"]["content"]
            if message_content.startswith("```json"):
                message_content = message_content[len("```json") :].lstrip()
            if message_content.endswith("```"):
                message_content = message_content[: -len("```")].rstrip()
            result = json.loads(message_content)
            if isinstance(result, dict):
                img_class = result.get("class")
                # YOLO format: [class, x_center, y_center, width, height] (all normalized 0-1)
                box = result.get("box")
        except json.JSONDecodeError as e:
            print(f"Could not parse bounding box: {e}")

        results.append(
            {
                "location_id": location_id,
                "image_file": image_file,
                "class": img_class,
                "box": box,
                "lighting": result.get("lighting"),
                "timestamp": result.get("timestamp"),
                "temperature_celsius": result.get("temperature_celsius"),
            }
        )

        # Save to CSV after each image (fast incremental write)
        # Only append if not already in processed_images to avoid duplicates
        if (location_id, image_file) not in processed_images:
            df = pd.DataFrame([results[-1]])  # Only the last result
            csv_path = labels_dir / f"labelling_results_{model}.csv"
            # Check if CSV exists and has content to determine if we need header
            write_header = not csv_path.exists() or csv_path.stat().st_size == 0
            df.to_csv(csv_path, mode="a", header=write_header, index=False)

        # Convert to parquet every 100 images
        if len(results) % 100 == 0:
            df_all = pd.read_csv(csv_path)
            df_all.to_parquet(
                labels_dir / f"labelling_results_{model}.parquet", index=False
            )

        # Save labelled image to labels directory
        show_image_with_class(
            image_path,
            image_file,
            img_class,
            json_data=result,
            save_path=label_save_path,
        )


process_images()

# Convert CSV to parquet for efficient storage
labels_dir = image_dir / f"labels_{model}"
csv_path = labels_dir / f"labelling_results_{model}.csv"
if csv_path.exists():
    df = pd.read_csv(csv_path)
    df.to_parquet(labels_dir / f"labelling_results_{model}.parquet", index=False)
    print(f"Converted results to parquet: {len(df)} total images in dataset")

# Print total execution time
end_time = time.time()
total_time = end_time - start_time
num_images = len(results)
avg_time_per_image = total_time / num_images if num_images > 0 else 0
print(
    f"\nTotal execution time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)"
)
print(f"Average time per image: {avg_time_per_image:.2f} seconds")
print(f"Images processed this run: {num_images}")

# %%
