#!/usr/bin/env python3
"""
Move images from subfolders containing DCIM directories to the data directory.
Optionally downsamples images if enabled.
"""

import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image
from PIL.ExifTags import TAGS
from tqdm import tqdm


def process_images(source_dir, data_dir, downsample=False, target_width=1920):
    """
    Move images from source subfolders to data directory.

    For each subfolder in source_dir:
    - Finds all images in <subfolder>/DCIM/*
    - Moves them to <data_dir>/<subfolder_name>/
    - Optionally downsamples if enabled
    - Uses lowercase filenames for output

    Args:
        source_dir: Path to directory containing subfolders with DCIM directories
        data_dir: Path to target data directory
        downsample: Whether to downsample images (default: False)
        target_width: Target width in pixels if downsampling (height calculated to maintain aspect ratio)
    """
    source_path = Path(source_dir)
    data_path = Path(data_dir)

    source_path = Path(source_dir)
    data_path = Path(data_dir)

    if not source_path.exists():
        print(f"Error: {source_dir} does not exist")
        return

    if not data_path.exists():
        print(f"Error: {data_dir} does not exist")
        return

    # Supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

    # Find all subfolders with DCIM directories
    dcim_folders = []
    for subfolder in source_path.iterdir():
        if subfolder.is_dir():
            dcim_path = subfolder / "DCIM"
            if dcim_path.exists() and dcim_path.is_dir():
                dcim_folders.append((subfolder.name, dcim_path))

    if not dcim_folders:
        print(f"No DCIM folders found in {source_dir}")
        return

    # Sort folders alphabetically
    dcim_folders.sort(key=lambda x: x[0])

    print(f"Found {len(dcim_folders)} DCIM folder(s)")
    print(f"Downsample: {'enabled' if downsample else 'disabled'}")

    total_processed = 0
    # Track serial numbers per location-month combination
    serial_counters = defaultdict(int)

    for folder_name, dcim_path in dcim_folders:
        print(f"\n{'=' * 60}")
        print(f"Processing: {folder_name}")
        print(f"{'=' * 60}")

        # Create output directory in data folder
        output_dir = data_path / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all images in all DCIM subfolders
        image_files = []
        # Sort subfolders alphabetically
        subfolders = sorted(
            [sf for sf in dcim_path.iterdir() if sf.is_dir()], key=lambda x: x.name
        )
        for subfolder in subfolders:
            for file_path in subfolder.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    image_files.append(file_path)

        if not image_files:
            print(f"  No images found in {dcim_path}")
            continue

        # Sort images by filename
        image_files.sort(key=lambda x: x.name)

        print(f"  Found {len(image_files)} images")

        for image_file in tqdm(image_files, desc=f"  {folder_name}", unit="img"):
            try:
                # Extract date from EXIF data and convert to Berlin timezone
                timestamp_str = None
                try:
                    with Image.open(image_file) as img:
                        exif_data = img._getexif()
                        if exif_data:
                            for tag_id, value in exif_data.items():
                                tag = TAGS.get(tag_id, tag_id)
                                if tag == "DateTimeOriginal" or tag == "DateTime":
                                    # Parse date format: "YYYY:MM:DD HH:MM:SS"
                                    # Camera stores time in standard time (CET, UTC+1) without DST
                                    date_obj = datetime.strptime(
                                        value, "%Y:%m:%d %H:%M:%S"
                                    )
                                    # Treat EXIF time as CET (standard time, UTC+1, no DST)
                                    from datetime import timezone, timedelta
                                    cet = timezone(timedelta(hours=1))
                                    date_obj_cet = date_obj.replace(tzinfo=cet)
                                    # Convert to Berlin time (which handles DST)
                                    berlin_tz = ZoneInfo("Europe/Berlin")
                                    date_obj_berlin = date_obj_cet.astimezone(berlin_tz)
                                    # Format as ISO 8601 compatible filename: YYYY-MM-DDTHH-MM-SS
                                    timestamp_str = date_obj_berlin.strftime(
                                        "%Y-%m-%dT%H-%M-%S"
                                    )
                                    break
                except Exception:
                    pass

                # Fallback to "unknown_timestamp" if no EXIF date found
                if not timestamp_str:
                    timestamp_str = "unknown_timestamp"
                    )

                # Create output filename: Location_YYYY-MM-DDTHH-MM-SS.ext
                file_ext = image_file.suffix.lower()
                output_filename = f"{folder_name}_{timestamp_str}{file_ext}"
                output_path = output_dir / output_filename

                if downsample:
                    # Open and downsample image
                    with Image.open(image_file) as img:
                        # Convert RGBA to RGB if necessary (for JPEG compatibility)
                        if img.mode == "RGBA":
                            img = img.convert("RGB")

                        # Calculate new height maintaining aspect ratio
                        aspect_ratio = img.height / img.width
                        target_height = int(target_width * aspect_ratio)

                        # Only resize if image is larger than target
                        if img.width > target_width:
                            img = img.resize(
                                (target_width, target_height), Image.Resampling.LANCZOS
                            )

                        # Save downsampled version with optimizations:
                        # - Strip EXIF metadata (exif=None)
                        # - Progressive JPEG for better compression
                        # - Quality 80 (good balance)
                        img.save(
                            output_path,
                            quality=80,
                            optimize=True,
                            progressive=True,
                            exif=b"",
                        )
                else:
                    # Move original file
                    shutil.move(str(image_file), str(output_path))

                total_processed += 1

            except Exception as e:
                tqdm.write(
                    f"  Error processing {image_file.name}: {e}", file=sys.stderr
                )

        print(f"  Completed {folder_name}: {len(image_files)} images")

    print(f"\n{'=' * 60}")
    print(f"Total images processed: {total_processed}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Move images from DCIM directories to data directory"
    )
    parser.add_argument(
        "-s",
        "--source",
        default="/Users/wri2lr/Pictures/Wildkameras",
        help="Path to folder containing subfolders with DCIM directories (default: /Users/wri2lr/Pictures/Wildkameras)",
    )
    parser.add_argument(
        "-d",
        "--data",
        default="/Users/wri2lr/repos/wildl-id/data",
        help="Path to target data directory (default: /Users/wri2lr/repos/wildl-id/data)",
    )
    parser.add_argument(
        "--downsample",
        action="store_true",
        help="Enable downsampling (default: disabled)",
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=1920,
        help="Target width in pixels when downsampling (default: 1920)",
    )

    args = parser.parse_args()

    process_images(args.source, args.data, args.downsample, args.width)
