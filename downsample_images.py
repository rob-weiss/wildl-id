#!/usr/bin/env python3
"""
Process images from subfolders containing DCIM directories.
Copies originals to DCIM/sendlist and saves downsampled versions to DCIM/sendlist_small.
"""

import shutil
import sys
from pathlib import Path

from PIL import Image


def process_images(base_dir, target_width=1920):
    """
    Process all images in subfolders containing DCIM directories.

    For each subfolder in base_dir:
    - Finds all images in <subfolder>/DCIM/*
    - Copies originals to <subfolder>/DCIM/sendlist
    - Saves downsampled versions to <subfolder>/DCIM/sendlist_small
    - Uses lowercase filenames for output

    Args:
        base_dir: Path to directory containing subfolders with DCIM directories
        target_width: Target width in pixels (height calculated to maintain aspect ratio)
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Error: {base_dir} does not exist")
        return

    # Supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

    # Find all subfolders with DCIM directories
    dcim_folders = []
    for subfolder in base_path.iterdir():
        if subfolder.is_dir():
            dcim_path = subfolder / "DCIM"
            if dcim_path.exists() and dcim_path.is_dir():
                dcim_folders.append((subfolder.name, dcim_path))

    if not dcim_folders:
        print(f"No DCIM folders found in {base_dir}")
        return

    print(f"Found {len(dcim_folders)} DCIM folder(s)")

    total_processed = 0

    for folder_name, dcim_path in dcim_folders:
        print(f"\n{'=' * 60}")
        print(f"Processing: {folder_name}")
        print(f"{'=' * 60}")

        # Delete output directories if they exist
        sendlist_dir = dcim_path / "sendlist"
        sendlist_small_dir = dcim_path / "sendlist_small"

        if sendlist_dir.exists():
            shutil.rmtree(sendlist_dir)
        if sendlist_small_dir.exists():
            shutil.rmtree(sendlist_small_dir)

        # Create fresh output directories
        sendlist_dir.mkdir(parents=True, exist_ok=True)
        sendlist_small_dir.mkdir(parents=True, exist_ok=True)

        # Find all images in all DCIM subfolders
        image_files = []
        for subfolder in dcim_path.iterdir():
            if subfolder.is_dir() and subfolder.name not in [
                "sendlist",
                "sendlist_small",
            ]:
                for file_path in subfolder.rglob("*"):
                    if (
                        file_path.is_file()
                        and file_path.suffix.lower() in image_extensions
                    ):
                        image_files.append(file_path)

        if not image_files:
            print(f"  No images found in {dcim_path}")
            continue

        print(f"  Found {len(image_files)} images")

        for idx, image_file in enumerate(image_files, 1):
            # Use lowercase filename for output
            output_filename = image_file.name.lower()

            try:
                # Copy original to sendlist
                original_output = sendlist_dir / output_filename
                shutil.copy2(image_file, original_output)

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

                    # Save downsampled version with lower quality for smaller file size
                    downsampled_output = sendlist_small_dir / output_filename
                    img.save(downsampled_output, quality=85, optimize=True)

                print(
                    f"  [{idx}/{len(image_files)}] Processed: {image_file.name} -> {output_filename}"
                )
                total_processed += 1

            except Exception as e:
                print(f"  Error processing {image_file.name}: {e}", file=sys.stderr)

        print(f"  Completed {folder_name}: {len(image_files)} images")

    print(f"\n{'=' * 60}")
    print(f"Total images processed: {total_processed}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process images from subfolders with DCIM directories"
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=str(Path.home() / "Downloads"),
        help="Path to folder containing subfolders with DCIM directories",
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=1920,
        help="Target width in pixels (default: 1920)",
    )

    args = parser.parse_args()

    process_images(args.folder, args.width)
