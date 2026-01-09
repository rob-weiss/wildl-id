from ollama_ocr import OCRProcessor

# Create an instance
ocr = OCRProcessor(model_name="llama3.2-vision:11b")

# Test with an image (replace with your image path)
result = ocr.process_image(
    image_path="/home/wri2lr/repos/home/mnt/wildlife/Alusitz/0ad6bfa5-c2fa-49b5-bea9-86fb3559ddf6.jpg",
    format_type="text",
    custom_prompt="Extract all text in the gray bar in the bottom of the image.",
    language="eng",
)
print(result)
