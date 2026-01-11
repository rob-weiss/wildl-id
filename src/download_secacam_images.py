#!/usr/bin/env python3
"""
Script to extract and download all Zeiss Secacam images from the app's databases.

Requirements:
    pip install requests
"""

import hashlib
import json
import os
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Dict, List
from urllib.parse import unquote

import requests

# Paths
CONTAINER_PATH = Path(
    "/Users/wri2lr/Library/Containers/4225BDAD-C278-4C1E-81B0-726B325A096D/Data"
)
CACHE_DB = CONTAINER_PATH / "Library/Caches/de.zeiss.cop.caledonia/Cache.db"
FS_CACHE_DIR = CONTAINER_PATH / "Library/Caches/de.zeiss.cop.caledonia/fsCachedData"
DOCUMENTS_DIR = CONTAINER_PATH / "Documents"

# Find the realm database file
REALM_DB = None
for file in DOCUMENTS_DIR.glob("user_*_realm.realm"):
    REALM_DB = file
    break

# Output directory
OUTPUT_DIR = Path.home() / "SecacamImages"


def extract_urls_from_cache_db() -> List[Dict]:
    """Extract image URLs from the Cache.db SQLite database."""
    print(f"Reading Cache.db: {CACHE_DB}")

    if not CACHE_DB.exists():
        print(f"Cache.db not found at {CACHE_DB}")
        return []

    urls = []
    conn = sqlite3.connect(str(CACHE_DB))
    cursor = conn.cursor()

    try:
        # Get all cached image URLs with their file references
        query = """
        SELECT 
            r.entry_ID,
            r.request_key as url,
            r.time_stamp,
            hex(d.receiver_data) as file_uuid_hex,
            d.isDataOnFS
        FROM cfurl_cache_response r 
        JOIN cfurl_cache_receiver_data d ON r.entry_ID = d.entry_ID 
        WHERE r.request_key LIKE '%getImage%'
        ORDER BY r.time_stamp DESC
        """

        cursor.execute(query)
        rows = cursor.fetchall()

        for row in rows:
            entry_id, url, timestamp, file_uuid_hex, is_on_fs = row

            # Decode the UUID from hex
            file_uuid = None
            if file_uuid_hex:
                try:
                    file_uuid = bytes.fromhex(file_uuid_hex).decode("utf-8")
                except:
                    pass

            urls.append(
                {
                    "entry_id": entry_id,
                    "url": url,
                    "timestamp": timestamp,
                    "file_uuid": file_uuid,
                    "is_on_fs": is_on_fs,
                    "source": "cache_db",
                }
            )

        print(f"Found {len(urls)} image URLs in Cache.db")

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    finally:
        conn.close()

    return urls


def extract_urls_from_realm_db() -> List[Dict]:
    """Extract image URLs from the Realm database using strings command."""
    if not REALM_DB or not REALM_DB.exists():
        print("Realm database not found")
        return []

    print(f"Reading Realm database: {REALM_DB}")
    print("Extracting URLs using strings command...")

    try:
        # Use strings command to extract URLs from the binary Realm database
        result = subprocess.run(
            ["strings", str(REALM_DB)], capture_output=True, text=True, timeout=60
        )

        if result.returncode != 0:
            print(f"Error running strings command: {result.stderr}")
            return []

        # Find all image URLs
        urls = []
        url_pattern = r'https://media\.secacam\.com/getImage/param/[^\s"&<>]+'

        for line in result.stdout.split("\n"):
            matches = re.findall(url_pattern, line)
            for url in matches:
                # Clean up the URL (remove any trailing characters)
                url = url.rstrip('"&<>')
                urls.append({"url": url, "source": "realm_db"})

        # Deduplicate URLs
        unique_urls = []
        seen = set()
        for item in urls:
            url = item["url"]
            if url not in seen:
                seen.add(url)
                unique_urls.append(item)

        print(f"Found {len(unique_urls)} unique image URLs in Realm database")
        return unique_urls

    except subprocess.TimeoutExpired:
        print("Timeout while extracting URLs from Realm database")
        return []
    except Exception as e:
        print(f"Error reading Realm database: {e}")
        return []


def copy_cached_files():
    """Copy currently cached files from fsCachedData."""
    print(f"\nCopying cached files from: {FS_CACHE_DIR}")

    if not FS_CACHE_DIR.exists():
        print("fsCachedData directory not found")
        return

    cached_dir = OUTPUT_DIR / "cached_files"
    cached_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for file in FS_CACHE_DIR.iterdir():
        if file.is_file():
            # Check if it's an image
            result = os.popen(f'file -b "{file}"').read()
            if "JPEG" in result or "PNG" in result:
                # Copy with .jpg extension
                dest = cached_dir / f"{file.name}.jpg"
                os.system(f'cp "{file}" "{dest}"')
                copied += 1

    print(f"Copied {copied} cached image files to {cached_dir}")


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


def generate_filename_from_url(url: str, index: int) -> str:
    """Generate a consistent filename from URL."""
    # Try to extract timestamp
    timestamp = extract_timestamp_from_url(url)

    # Create a short hash of the URL for uniqueness
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]

    if timestamp:
        return f"{timestamp}_{url_hash}.jpg"
    else:
        return f"image_{index:04d}_{url_hash}.jpg"


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

        # Generate filename from URL
        filename = generate_filename_from_url(url, i)
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

    # Step 1: Copy currently cached files
    copy_cached_files()

    # Step 2: Extract URLs from Cache.db
    cache_urls = extract_urls_from_cache_db()

    # Step 3: Extract URLs from Realm database
    realm_urls = extract_urls_from_realm_db()

    # Step 4: Combine and deduplicate URLs
    all_urls = cache_urls + realm_urls
    unique_urls = []
    seen_urls = set()

    for item in all_urls:
        url = item["url"]
        if url not in seen_urls:
            seen_urls.add(url)
            unique_urls.append(item)

    print(f"\nTotal unique URLs found: {len(unique_urls)}")

    if not unique_urls:
        print("\nNo image URLs found in databases.")
        print("This might mean:")
        print("  1. The app hasn't synced any images yet")
        print("  2. The Realm database uses a different schema")
        print("  3. Images are only stored temporarily in the cache")
        return

    # Step 5: Download all images (skipping existing ones)
    download_all_images(unique_urls)


if __name__ == "__main__":
    main()
