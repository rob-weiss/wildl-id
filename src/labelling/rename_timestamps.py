#!/usr/bin/env python3
"""
Rename files with old timestamp format (YYYY-MM-DD_HH-MM-SS) to ISO format (YYYY-MM-DDTHH-MM-SS).
For images, fixes timestamps using EXIF metadata to correct DST offset issues.
"""

import re
from datetime import datetime
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS


def get_exif_timestamp(image_path):
    """Extract timestamp from image EXIF metadata."""
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == "DateTimeOriginal" or tag == "DateTime":
                        # Parse date format: "YYYY:MM:DD HH:MM:SS"
                        # Camera stores time in local Berlin time (already accounts for DST)
                        date_obj = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                        # Format as ISO 8601 compatible filename: YYYY-MM-DDTHH-MM-SS
                        return date_obj.strftime("%Y-%m-%dT%H-%M-%S")
    except Exception:
        pass
    return None


def main():
    """Find and rename files with old timestamp format."""
    data_dir = Path("/Users/wri2lr/repos/wildl-id/data")

    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return

    print(f"Scanning directory: {data_dir}")
    print("=" * 60)

    # Pattern to match: YYYY-MM-DD_HH-MM-SS
    # Looking for date format followed by underscore and time format
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})")

    # Supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

    renamed_count = 0
    deleted_count = 0
    fixed_count = 0
    skipped_count = 0

    # Walk through all subdirectories
    for file_path in data_dir.rglob("*"):
        if not file_path.is_file():
            continue

        filename = file_path.name

        # Check if filename contains the old timestamp format
        match = pattern.search(filename)
        if match:
            # For images, try to get the correct timestamp from EXIF
            new_filename = None
            is_image = file_path.suffix.lower() in image_extensions

            if is_image:
                exif_timestamp = get_exif_timestamp(file_path)
                if exif_timestamp:
                    # Extract the camera name/location prefix from the original filename
                    # Format is typically: Location_YYYY-MM-DD_HH-MM-SS_hash.jpg
                    # We want to keep everything before the date and everything after the time
                    before_date = filename[: match.start()]
                    after_time = filename[match.end() :]
                    new_filename = f"{before_date}{exif_timestamp}{after_time}"
                    fixed_count += 1

            # If not an image or no EXIF found, just replace _ with T
            if not new_filename:
                new_filename = pattern.sub(r"\1T\2", filename)

            if new_filename == filename:
                # No change needed (shouldn't happen, but safety check)
                skipped_count += 1
                continue

            new_path = file_path.parent / new_filename

            # Check if target already exists
            if new_path.exists():
                # Delete the old file since new one already exists
                try:
                    file_path.unlink()
                    print(f"🗑️  Deleted (target exists): {filename}")
                    print(f"   → {new_filename} (already exists)")
                    deleted_count += 1
                except Exception as e:
                    print(f"❌ Error deleting {filename}: {e}")
                    skipped_count += 1
                continue

            # Rename the file
            try:
                file_path.rename(new_path)
                action = (
                    "✓ Fixed from EXIF" if is_image and exif_timestamp else "✓ Renamed"
                )
                print(f"{action}: {filename}")
                print(f"  → {new_filename}")
                renamed_count += 1
            except Exception as e:
                print(f"❌ Error renaming {filename}: {e}")
                skipped_count += 1

    # Summary
    print("=" * 60)
    print("✓ Complete!")
    print(f"  Files renamed: {renamed_count}")
    print(f"  Files fixed from EXIF: {fixed_count}")
    print(f"  Files deleted: {deleted_count}")
    print(f"  Files skipped: {skipped_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
