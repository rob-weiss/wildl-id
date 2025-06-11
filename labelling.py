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
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import pytesseract
from matplotlib.patches import Rectangle
from ollama import Client
from tqdm import tqdm

model = "gemma3:4b"

client = Client(
    host="http://localhost:11434",
)

client.pull(model=model)
client.list()
client.show(model=model)
image_dir = Path("/home/wri2lr/repos/home/mnt/images")


prompt = (
    "Determine the class of creature in this image.\n"
    "The photo was taken by a wildlife camera in Germany.\n"
    "Common animals include roe deer, wild boar, dove, badger, marten, dog, fox, hare, and crow.\n"
    "Either there is an animal, a human, or no creature in the image.\n"
    "If multiple animals are visible, echo the one that appears most often.\n"
    "If there is an animal, please specify the species.\n"
    "If there is no creature, just return the class 'nothing'.\n"
    "If there is a creature but you don't know what kind of creature, choose class 'unknown'.\n"
    "Return in JSON format just the class with key 'class' but nothing else.\n"
    # and the bounding box of the creature with key 'box'
    # "Return the bounding box in the YOLO format."
)

# Collect results in a list
results = []


def show_image_with_class(image_path, image_file, img_class):
    """Display an image with its predicted class as the title.

    Parameters
    ----------
    image_path : Path or str
        The path to the image file.
    image_file : str
        The name of the image file.
    img_class : str
        The predicted class of the image.
    """
    img = mpimg.imread(str(image_path))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{image_file} \n class: {img_class}")
    plt.show()


def process_images():
    """Processes images in the specified directory, sends them to the LLM for classification,
    parses the responses, and collects the results.

    Returns:
    -------
    None
        The function appends results to the global 'results' list and prints progress.
    """
    image_files = [
        f.name
        for f in image_dir.iterdir()
        if f.is_file() and f.name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    image_files.sort()
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = image_dir / image_file
        with image_path.open("rb") as img_file:
            image_bytes = img_file.read()
            img_base64 = base64.b64encode(image_bytes).decode("utf-8")
        # Load the image array for shape and visualization
        image = mpimg.imread(str(image_path))

        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt, "images": [img_base64]}],
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
                if box is not None and isinstance(box, list) and len(box) == 4:
                    x_center, y_center, width, height = box
                    img_h, img_w = image.shape[:2]
                    x = (x_center - width / 2) * img_w
                    y = (y_center - height / 2) * img_h
                    w = width * img_w
                    h = height * img_h
                    rect = Rectangle((x, y), w, h, linewidth=2, edgecolor="r", facecolor="none")
                    plt.gca().add_patch(rect)
        except json.JSONDecodeError as e:
            print(f"Could not parse bounding box: {e}")

        results.append({"image_file": image_file, "class": img_class, "box": box})

        show_image_with_class(image_path, image_file, img_class)

        ocr_text = pytesseract.image_to_string(str(image_path), lang="eng+deu")
        print(f"OCR annotation for {image_file}: {ocr_text.strip()}")

        print(
            f"Processed {image_file}: class={img_class}, box={box}\n OCR annotation for {image_file}: {ocr_text.strip()}"
        )


process_images()

# Save results to parquet
df = pd.DataFrame(results)
df.to_parquet(f"labelling_results_{model}.parquet", index=False)

# %%
