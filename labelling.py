import base64
from pathlib import Path

from ollama import Client

model = "gemma3:12b"

client = Client(
    host="http://localhost:11434",
)

client.pull(model=model)
client.list()
client.show(model=model)
image_dir = Path("/home/wri2lr/repos/home/mnt/images")
image_files = [
    f.name for f in image_dir.iterdir() if f.is_file() and f.name.lower().endswith((".jpg", ".jpeg", ".png"))
]

for image_file in image_files:
    image_path = image_dir / image_file
    with image_path.open("rb") as img:
        image_bytes = img.read()
        # Convert image to base64 for Ollama
        img_base64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = "Label the content of this image. The photo was taken by a wildlife camera in Germany. Either there is an animal, a human, or nothing in the image. If there is an animal, please specify the species. If there is nothing, just say 'nothing'."
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": [img_base64]}],
    )
    print(f"{image_file}: {response['message']['content']}")
