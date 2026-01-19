#!/usr/bin/env python3
"""
Wildlife Camera Image Analysis - Main Interface

This is the main entry point for the wildlife camera analysis system.
It provides a user-friendly interface to:
1. Download new images from camera gallery
2. Label/classify images using MegaDetector and species classifier
3. Generate visualizations and analysis reports
4. Run all steps in sequence

Usage:
    python main.py
    
Then follow the interactive menu prompts.
"""

import subprocess
import sys
from pathlib import Path


def print_header():
    """Print a nice header."""
    print("\n" + "=" * 70)
    print("🦌  WILDLIFE CAMERA IMAGE ANALYSIS SYSTEM  🦌")
    print("=" * 70)
    print()


def print_menu():
    """Print the main menu."""
    print("\n📋 MAIN MENU")
    print("-" * 70)
    print("1. Download new images from camera gallery")
    print("2. Label/classify images (MegaDetector + Species Classifier)")
    print("3. Generate visualizations and analysis reports")
    print("4. Run all steps in sequence (Download → Label → Visualize)")
    print("5. Exit")
    print("-" * 70)


def download_images():
    """Download images from camera gallery."""
    print("\n" + "=" * 70)
    print("📥  DOWNLOADING IMAGES FROM CAMERA GALLERY")
    print("=" * 70)
    print()
    print("⚠️  IMPORTANT: Make sure the ZEISS Secacam carousel is open in Safari!")
    print()
    
    proceed = input("Ready to proceed? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Download cancelled.")
        return False
    
    script_path = Path(__file__).parent / "src" / "labelling" / "download_images.py"
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=Path(__file__).parent
        )
        print("\n✓ Download completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Download failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        return False


def label_images():
    """Label and classify images using MegaDetector."""
    print("\n" + "=" * 70)
    print("🏷️  LABELING IMAGES")
    print("=" * 70)
    print()
    print("This will run MegaDetector for animal detection and DeepFaune for species")
    print("classification. It will also extract metadata (timestamp, temperature) and")
    print("analyze lighting conditions.")
    print()
    
    proceed = input("Ready to proceed? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Labeling cancelled.")
        return False
    
    script_path = Path(__file__).parent / "src" / "labelling" / "label_images.py"
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=Path(__file__).parent
        )
        print("\n✓ Labeling completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Labeling failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Labeling interrupted by user")
        return False


def generate_visualizations():
    """Generate visualizations and analysis reports."""
    print("\n" + "=" * 70)
    print("📊  GENERATING VISUALIZATIONS")
    print("=" * 70)
    print()
    print("This will create comprehensive visualizations including:")
    print("  • Activity patterns by hour of day")
    print("  • Species distribution charts")
    print("  • Calendar heatmaps")
    print("  • Location-based statistics")
    print("  • Day/night activity patterns")
    print("  • Temperature correlations")
    print()
    
    proceed = input("Ready to proceed? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Visualization cancelled.")
        return False
    
    script_path = Path(__file__).parent / "src" / "visualisation" / "evaluate_labels.py"
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=Path(__file__).parent
        )
        print("\n✓ Visualizations generated successfully!")
        print(f"📁 Check the docs/diagrams/ folder for output files")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Visualization failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Visualization interrupted by user")
        return False


def run_all():
    """Run all steps in sequence."""
    print("\n" + "=" * 70)
    print("🚀  RUNNING COMPLETE PIPELINE")
    print("=" * 70)
    print()
    print("This will run all steps in sequence:")
    print("  1. Download images")
    print("  2. Label/classify images")
    print("  3. Generate visualizations")
    print()
    
    proceed = input("Ready to proceed? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Pipeline cancelled.")
        return
    
    # Step 1: Download
    print("\n\n" + "=" * 70)
    print("STEP 1/3: DOWNLOADING IMAGES")
    print("=" * 70)
    success = download_images()
    if not success:
        print("\n⚠️  Pipeline stopped due to download failure")
        return
    
    # Step 2: Label
    print("\n\n" + "=" * 70)
    print("STEP 2/3: LABELING IMAGES")
    print("=" * 70)
    success = label_images()
    if not success:
        print("\n⚠️  Pipeline stopped due to labeling failure")
        return
    
    # Step 3: Visualize
    print("\n\n" + "=" * 70)
    print("STEP 3/3: GENERATING VISUALIZATIONS")
    print("=" * 70)
    success = generate_visualizations()
    if not success:
        print("\n⚠️  Pipeline completed with visualization errors")
        return
    
    print("\n\n" + "=" * 70)
    print("✓  PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print()


def main():
    """Main function with interactive menu."""
    print_header()
    
    while True:
        print_menu()
        
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                download_images()
            elif choice == '2':
                label_images()
            elif choice == '3':
                generate_visualizations()
            elif choice == '4':
                run_all()
            elif choice == '5':
                print("\n👋  Goodbye!")
                break
            else:
                print("\n⚠️  Invalid choice. Please enter 1-5.")
        
        except KeyboardInterrupt:
            print("\n\n👋  Goodbye!")
            break
        except Exception as e:
            print(f"\n✗ An error occurred: {e}")
            print("Please try again or report this issue.")
    
    print()


if __name__ == "__main__":
    main()
