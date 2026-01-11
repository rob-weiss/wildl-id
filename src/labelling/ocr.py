from ollama_ocr import OCRProcessor

# Create an instance
ocr = OCRProcessor(model_name="llama3.2-vision:11b")

# Test with an image (replace with your image path)
result = ocr.process_image(
    image_path="/Users/wri2lr/repos/wildl-id/data/Alu Sitz/Alu Sitz_2025-11-11_04-45-00_72c13c12d21390f7.jpg",
    format_type="text",
    custom_prompt="There is a gray bar in the bottom of the image. It contains the date and time in the lower right corner.\n"
    "Add the timestamp as an additional key 'timestamp' in ISO 8601 format (YYYY-MM-DDTHH:MM:SS) if you can read it from the gray bar using OCR.\n"
    "There is a temperature value in the lower right corner of the gray bar. Add it as an additional key 'temperature_celsius' in degrees Celsius if you can read it from the gray bar using OCR.\n"
    "In the gray bar, there is also a rectangular battery symbol somewhere left of the temperature value. Inside the symbol there is either a percentage value with integer multiples of ten or up to four bars indicating the remaining battery level of the camera.\n"
    "If you can read the battery percentage value or count the bars, add it as an additional key 'battery_level' with value in percent (0-100).\n"
    "That is, if there are four bars, assume 100%, three bars 75%, two bars 50%, one bar 25%, and no bars 0%.\n"
    "Respond only with the JSON object, nothing else.\n",
    language="eng",
)

print("Raw result:", result)

# Parse the JSON string
import json
import re

# Extract JSON from the response (model may add extra text)
json_match = re.search(r"\{.*\}", result, re.DOTALL)
if json_match:
    json_str = json_match.group(0)
    result = json.loads(json_str)
else:
    # Fallback: try to parse as-is
    result = json.loads(result)

print("Parsed result:", result)
