#!/usr/bin/env python3
"""
Downsample images from a source directory to a target directory.
Resizes images to 1920x1440 while maintaining aspect ratio.
"""

from pathlib import Path
from PIL import Image
import sys


def downsample_images(source_dir, target_dir, target_width=1920):
    """
    Downsample all images from source_dir to target_dir.
    
    Args:
        source_dir: Path to source directory containing images
        target_dir: Path to target directory for downsampled images
        target_width: Target width in pixels (height calculated to maintain aspect ratio)
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    # Get all image files
    image_files = [f for f in source_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {source_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    for idx, image_file in enumerate(image_files, 1):
        try:
            # Open image
            with Image.open(image_file) as img:
                # Convert RGBA to RGB if necessary (for JPEG compatibility)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                # Calculate new height maintaining aspect ratio
                aspect_ratio = img.height / img.width
                target_height = int(target_width * aspect_ratio)
                
                # Resize image
                img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                
                # Save to target directory with same filename
                output_path = target_path / image_file.name
                img.save(output_path, quality=95, optimize=True)
                
                print(f"[{idx}/{len(image_files)}] Processed: {image_file.name}")
                
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}", file=sys.stderr)
    
    print(f"\nCompleted! Processed images saved to {target_dir}")


if __name__ == "__main__":
    source_dir = "/Users/wri2lr/Downloads/Amphitheater/DCIM/THUMB/sendlist/"
    target_dir = "/Users/wri2lr/Downloads/Amphitheater/DCIM/THUMB/sendlist_small"
    
    downsample_images(source_dir, target_dir)
