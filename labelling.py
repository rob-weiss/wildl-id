# %%
import base64
import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from ollama import Client

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


# Move the for loop code into a function to collect results
def process_images():
    image_files = [
        f.name
        for f in image_dir.iterdir()
        if f.is_file() and f.name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    image_files.sort()
    for image_file in image_files:
        image_path = image_dir / image_file
        with image_path.open("rb") as img:
            image_bytes = img.read()
            img_base64 = base64.b64encode(image_bytes).decode("utf-8")

        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt, "images": [img_base64]}],
        )

        img = mpimg.imread(str(image_path))
        plt.imshow(img)

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
                    img_h, img_w = img.shape[:2]
                    x = (x_center - width / 2) * img_w
                    y = (y_center - height / 2) * img_h
                    w = width * img_w
                    h = height * img_h
                    rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor="r", facecolor="none")
                    plt.gca().add_patch(rect)
        except Exception as e:
            print(f"Could not parse bounding box: {e}")

        plt.axis("off")
        plt.title(f"{image_file} \n class: {img_class}")
        plt.show()

        results.append({"image_file": image_file, "class": img_class, "box": box})


process_images()

# Save results to parquet
df = pd.DataFrame(results)
df.to_parquet("labelling_results.parquet", index=False)
# %%
