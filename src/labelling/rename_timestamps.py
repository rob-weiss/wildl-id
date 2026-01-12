#!/usr/bin/env python3
"""
Rename files with old timestamp format (YYYY-MM-DD_HH-MM-SS) to ISO format (YYYY-MM-DDTHH-MM-SS).
"""

import re
from pathlib import Path


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

    renamed_count = 0
    deleted_count = 0
    skipped_count = 0

    # Walk through all subdirectories
    for file_path in data_dir.rglob("*"):
        if not file_path.is_file():
            continue

        filename = file_path.name

        # Check if filename contains the old timestamp format
        match = pattern.search(filename)
        if match:
            # Replace the underscore between date and time with T
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
                print(f"✓ Renamed: {filename}")
                print(f"  → {new_filename}")
                renamed_count += 1
            except Exception as e:
                print(f"❌ Error renaming {filename}: {e}")
                skipped_count += 1

    # Summary
    print("=" * 60)
    print("✓ Complete!")
    print(f"  Files renamed: {renamed_count}")
    print(f"  Files deleted: {deleted_count}")
    print(f"  Files skipped: {skipped_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
