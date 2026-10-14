#!/bin/bash

# Target directory (default is the current directory if not specified)
TARGET_DIR="${1:-.}"

echo "Scanning for videos to delete in: $TARGET_DIR"

# List of common video extensions to look for (case-insensitive)
VIDEO_EXTENSIONS=("mp4" "mkv" "avi" "mov" "flv" "wmv" "webm" "m4v")

for ext in "${VIDEO_EXTENSIONS[@]}"; do
    echo "Deleting all *.$ext files..."
    # -type f ensures only files are deleted
    # -iname makes the search case-insensitive (.MP4 and .mp4 will both match)
    # -delete deletes the matched files automatically
    find "$TARGET_DIR" -type f -iname "*.$ext" -delete
done

echo "Cleanup complete!"

