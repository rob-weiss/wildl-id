import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import requests

#!/usr/bin/env python3


def list_image_files(directory: str) -> List[Path]:
    """Find all image files in the given directory."""
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(directory).glob(f"**/*{ext}"))
        image_files.extend(Path(directory).glob(f"**/*{ext.upper()}"))

    return sorted(image_files)


def encode_image_to_base64(image_path: Path) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def get_available_models() -> List[str]:
    """Get list of available models from Ollama API."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        return []
    except requests.RequestException:
        print("Warning: Could not connect to Ollama API. Is Ollama running?")
        return []


def label_image_with_ollama(image_path: Path, model: str, stream: bool = False) -> Dict[str, Any]:
    """Send image to Ollama API for labeling and return the response."""
    base64_image = encode_image_to_base64(image_path)

    # Prepare the API request
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": "What's in this image? Provide a short description and relevant labels, separated by commas.",
        "images": [base64_image],
        "stream": stream,
    }

    try:
        # Make the API call
        response = requests.post(url, json=payload, stream=stream)

        if response.status_code == 200:
            if stream:
                # Handle streaming response
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        line_json = json.loads(line.decode("utf-8"))
                        response_chunk = line_json.get("response", "")
                        full_response += response_chunk
                        print(response_chunk, end="", flush=True)
                print()  # Add a newline at the end
                return {"image_path": str(image_path), "labels": full_response, "success": True}
            # Handle non-streaming response
            response_json = response.json()
            return {
                "image_path": str(image_path),
                "labels": response_json.get("response", ""),
                "success": True,
            }
        return {
            "image_path": str(image_path),
            "error": f"API call failed with status code {response.status_code}",
            "response": response.text,
            "success": False,
        }
    except requests.RequestException as e:
        return {"image_path": str(image_path), "error": f"Request failed: {e!s}", "success": False}


def process_directory(directory: str, model: str, output_file: str = None, stream: bool = False) -> None:
    """Process all images in a directory and label them."""
    # Check if model is available
    available_models = get_available_models()
    if available_models and model not in available_models:
        print(f"Warning: Model '{model}' not found in available models.")
        print(f"Available models: {', '.join(available_models)}")
        if not input("Continue anyway? (y/n): ").lower().startswith("y"):
            return

    image_files = list_image_files(directory)
    results = []

    print(f"Found {len(image_files)} images to process.")

    for i, image_path in enumerate(image_files):
        print(f"Processing image {i + 1}/{len(image_files)}: {image_path}")
        result = label_image_with_ollama(image_path, model, stream)
        results.append(result)

        # Print result for immediate feedback if not streaming
        if not stream:
            if result["success"]:
                print(f"Labels: {result['labels']}")
            else:
                print(f"Error: {result['error']}")

    # Save results to file if requested
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Label images using Ollama API")
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default="/home/wri2lr/repos/home/mnt/images",
        help="Directory containing images to label",
    )
    parser.add_argument(
        "--model", "-m", type=str, default="llava", help="Ollama model to use for labeling (default: llava)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="image_labels.json",
        help="Output file to save results (default: image_labels.json)",
    )
    parser.add_argument("--list-models", action="store_true", help="List available Ollama models and exit")
    parser.add_argument("--stream", "-s", action="store_true", help="Use streaming mode for responses")

    args = parser.parse_args()

    # Check internet connection.
    response = requests.get("http://www.google.com", timeout=5)
    # If a proxy is set and the response is a 407 (proxy auth required) or 502/503/504 (proxy errors), fail
    if response.status_code in [407, 502, 503, 504]:
        # Echo proxy variables.
        if "http_proxy" in os.environ:
            print(f"http_proxy: {os.environ['http_proxy']}")
        if "https_proxy" in os.environ:
            print(f"https_proxy: {os.environ['https_proxy']}")
        if "HTTP_PROXY" in os.environ:
            print(f"HTTP_PROXY: {os.environ['HTTP_PROXY']}")
        if "HTTPS_PROXY" in os.environ:
            print(f"HTTPS_PROXY: {os.environ['HTTPS_PROXY']}")

        raise requests.ConnectionError(f"Proxy error detected. Response: {response.text}")

    # List models if requested
    if args.list_models:
        models = get_available_models()
        if models:
            print("Available Ollama models:")
            for model in models:
                print(f"- {model}")
        else:
            print("No models found or could not connect to Ollama API.")
        return

    # Check if directory exists
    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist.")
        return

    process_directory(args.directory, args.model, args.output, args.stream)


if __name__ == "__main__":
    main()
