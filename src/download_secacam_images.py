#!/usr/bin/env python3
"""
Script to extract and download all Zeiss Secacam images from the app's databases.

Requirements:
    pip install requests
"""

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List
from urllib.parse import unquote

import requests

# Paths
CONTAINER_PATH = Path(
    "/Users/wri2lr/Library/Containers/4225BDAD-C278-4C1E-81B0-726B325A096D/Data"
)

DOCUMENTS_DIR = CONTAINER_PATH / "Documents"

# Find the realm database file
REALM_DB = None
for file in DOCUMENTS_DIR.glob("user_*_realm.realm"):
    REALM_DB = file
    break

# Output directory
OUTPUT_DIR = Path.home() / "SecacamImages"


def extract_urls_from_realm_db() -> List[Dict]:
    """Extract image URLs and metadata from the Realm database using strings command."""
    if not REALM_DB or not REALM_DB.exists():
        print("Realm database not found")
        return []

    print(f"Reading Realm database: {REALM_DB}")
    print("Extracting URLs and metadata using strings command...")

    try:
        # Use strings command to extract URLs from the binary Realm database
        result = subprocess.run(
            ["strings", str(REALM_DB)], capture_output=True, text=True, timeout=60
        )

        if result.returncode != 0:
            print(f"Error running strings command: {result.stderr}")
            return []

        # Find all image URLs with context
        lines = result.stdout.split("\n")
        urls = []
        url_pattern = r'https://media\.secacam\.com/getImage/param/[^\s"&<>]+'

        # Look for potential camera IDs or location info
        camera_id_pattern = (
            r"\b[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}\b"
        )
        serial_pattern = r"\bSEC[0-9]+-[WR][0-9]+-[0-9]+\b"

        for i, line in enumerate(lines):
            matches = re.findall(url_pattern, line)
            for url in matches:
                # Clean up the URL
                url = url.rstrip('"&<>')

                # Try to find associated metadata in nearby lines
                camera_id = None
                serial = None

                # Check surrounding lines for metadata (within 5 lines before/after)
                for j in range(max(0, i - 5), min(len(lines), i + 6)):
                    context_line = lines[j]

                    # Look for camera IDs
                    camera_matches = re.findall(camera_id_pattern, context_line)
                    if camera_matches:
                        camera_id = camera_matches[0]

                    # Look for serial numbers
                    serial_matches = re.findall(serial_pattern, context_line)
                    if serial_matches:
                        serial = serial_matches[0]

                urls.append(
                    {
                        "url": url,
                        "camera_id": camera_id,
                        "serial": serial,
                        "source": "realm_db",
                    }
                )

        # Deduplicate URLs
        unique_urls = []
        seen = set()
        for item in urls:
            url = item["url"]
            if url not in seen:
                seen.add(url)
                unique_urls.append(item)

        print(f"Found {len(unique_urls)} unique image URLs in Realm database")

        # Print camera info summary
        cameras = set()
        serials = set()
        for item in unique_urls:
            if item.get("camera_id"):
                cameras.add(item["camera_id"])
            if item.get("serial"):
                serials.add(item["serial"])

        if cameras:
            print(
                f"Found {len(cameras)} camera ID(s): {', '.join(sorted(cameras)[:3])}{'...' if len(cameras) > 3 else ''}"
            )
        if serials:
            print(
                f"Found {len(serials)} serial number(s): {', '.join(sorted(serials))}"
            )

        return unique_urls

    except subprocess.TimeoutExpired:
        print("Timeout while extracting URLs from Realm database")
        return []
    except Exception as e:
        print(f"Error reading Realm database: {e}")
        return []


def extract_timestamp_from_url(url: str) -> str:
    """Try to extract timestamp from the encoded URL parameter."""
    try:
        # The URL contains encoded parameters - try to decode and find date info
        decoded = unquote(url)

        # Look for date patterns in the decoded URL (format: YYYY-MM-DD or YYYYMMDD)
        date_match = re.search(r"20\d{2}[-:]?\d{2}[-:]?\d{2}", decoded)
        if date_match:
            return date_match.group(0).replace(":", "-")

        # Look for timestamp patterns
        time_match = re.search(
            r"20\d{2}[-:]?\d{2}[-:]?\d{2}\s+\d{2}:\d{2}:\d{2}", decoded
        )
        if time_match:
            return time_match.group(0).replace(" ", "_").replace(":", "-")
    except:
        pass

    return None


def generate_filename_from_url(url: str, camera_info: Dict = None) -> str:
    """Generate a consistent filename from URL and camera metadata."""
    # Try to extract timestamp
    timestamp = extract_timestamp_from_url(url)

    # Create a short hash of the URL for uniqueness
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]

    # Add camera serial if available
    prefix = ""
    if camera_info:
        serial = camera_info.get("serial")
        if serial:
            # Extract just the last part of the serial (e.g., "0062026" from "SEC5-W100-0062026")
            serial_parts = serial.split("-")
            if len(serial_parts) >= 2:
                prefix = f"{serial_parts[0]}_{serial_parts[-1]}_"

    if timestamp:
        return f"{prefix}{timestamp}_{url_hash}.jpg"
    else:
        return f"{prefix}{url_hash}.jpg"


def download_image(url: str, output_path: Path) -> bool:
    """Download an image from URL."""
    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def download_all_images(urls: List[Dict]):
    """Download all images from URLs, skipping existing files."""
    if not urls:
        print("No URLs to download")
        return

    print(f"\n{'=' * 60}")
    print(f"Found {len(urls)} images to process")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'=' * 60}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save URL list for reference
    urls_file = OUTPUT_DIR / "image_urls.json"
    with open(urls_file, "w") as f:
        json.dump(urls, f, indent=2, default=str)
    print(f"Saved URL list to: {urls_file}\n")

    # Check existing files to avoid re-downloading
    existing_files = set(f.name for f in OUTPUT_DIR.glob("*.jpg"))
    print(f"Found {len(existing_files)} existing images in output directory\n")

    success_count = 0
    fail_count = 0
    skip_count = 0

    for i, item in enumerate(urls, 1):
        url = item["url"]

        # Generate filename from URL and metadata
        filename = generate_filename_from_url(url, item)
        output_path = OUTPUT_DIR / filename

        # Skip if already exists
        if output_path.exists():
            skip_count += 1
            if skip_count % 100 == 0:  # Print progress every 100 skipped
                print(f"[{i}/{len(urls)}] Skipped {skip_count} existing images...")
            continue

        print(f"[{i}/{len(urls)}] Downloading: {filename}")

        if download_image(url, output_path):
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'=' * 60}")
    print("Download complete!")
    print(f"  Downloaded: {success_count}")
    print(f"  Skipped (already exist): {skip_count}")
    print(f"  Failed:  {fail_count}")
    print(f"  Total:   {len(urls)}")
    print(f"{'=' * 60}\n")


def main():
    print("Zeiss Secacam Image Extractor")
    print("=" * 60)

    # Extract all URLs from Realm database
    unique_urls = extract_urls_from_realm_db()

    print(f"\nTotal unique URLs found: {len(unique_urls)}")

    if not unique_urls:
        print("\nNo image URLs found in databases.")
        print("This might mean:")
        print("  1. The app hasn't synced any images yet")
        print("  2. The Realm database uses a different schema")
        print("  3. Images are only stored temporarily in the cache")
        return

    # Step 4: Download all images (skipping existing ones)
    download_all_images(unique_urls)


if __name__ == "__main__":
    main()
