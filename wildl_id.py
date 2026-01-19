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
    # Interactive menu:
    python wildl_id.py

    # Command-line arguments:
    python wildl_id.py --download         # Download images
    python wildl_id.py --label            # Label images
    python wildl_id.py --visualize        # Generate visualizations
    python wildl_id.py --all              # Run all steps
    python wildl_id.py -d -l -v           # Same as --all (short form)
"""

import argparse
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the main functions from each module
from labelling.download_images import main as download_main
from labelling.label_images import process_images_with_pytorch_wildlife
from visualisation.evaluate_labels import main as visualisation_main


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


def download_images(skip_prompt=False):
    """Download images from camera gallery."""
    print("\n" + "=" * 70)
    print("📥  DOWNLOADING IMAGES FROM CAMERA GALLERY")
    print("=" * 70)
    print()
    print("⚠️  IMPORTANT: Make sure the ZEISS Secacam carousel is open in Safari!")
    print()

    if not skip_prompt:
        proceed = input("Ready to proceed? (y/n): ").strip().lower()
        if proceed != "y":
            print("Download cancelled.")
            return False

    try:
        download_main()
        print("\n✓ Download completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        return False


def label_images(skip_prompt=False):
    """Label and classify images using MegaDetector."""
    print("\n" + "=" * 70)
    print("🏷️  LABELING IMAGES")
    print("=" * 70)
    print()
    print("This will run MegaDetector for animal detection and DeepFaune for species")
    print("classification. It will also extract metadata (timestamp, temperature) and")
    print("analyze lighting conditions.")
    print()

    if not skip_prompt:
        proceed = input("Ready to proceed? (y/n): ").strip().lower()
        if proceed != "y":
            print("Labeling cancelled.")
            return False

    try:
        process_images_with_pytorch_wildlife()
        print("\n✓ Labeling completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Labeling failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Labeling interrupted by user")
        return False


def generate_visualizations(skip_prompt=False):
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

    if not skip_prompt:
        proceed = input("Ready to proceed? (y/n): ").strip().lower()
        if proceed != "y":
            print("Visualization cancelled.")
            return False

    try:
        visualisation_main()
        print("\n✓ Visualizations generated successfully!")
        print("📁 Check the docs/diagrams/ folder for output files")
        return True
    except Exception as e:
        print(f"\n✗ Visualization failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Visualization interrupted by user")
        return False


def run_all(skip_prompt=False):
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

    if not skip_prompt:
        proceed = input("Ready to proceed? (y/n): ").strip().lower()
        if proceed != "y":
            print("Pipeline cancelled.")
            return

    # Step 1: Download
    print("\n\n" + "=" * 70)
    print("STEP 1/3: DOWNLOADING IMAGES")
    print("=" * 70)
    success = download_images(skip_prompt=True)
    if not success:
        print("\n⚠️  Pipeline stopped due to download failure")
        return

    # Step 2: Label
    print("\n\n" + "=" * 70)
    print("STEP 2/3: LABELING IMAGES")
    print("=" * 70)
    success = label_images(skip_prompt=True)
    if not success:
        print("\n⚠️  Pipeline stopped due to labeling failure")
        return

    # Step 3: Visualize
    print("\n\n" + "=" * 70)
    print("STEP 3/3: GENERATING VISUALIZATIONS")
    print("=" * 70)
    success = generate_visualizations(skip_prompt=True)
    if not success:
        print("\n⚠️  Pipeline completed with visualization errors")
        return

    print("\n\n" + "=" * 70)
    print("✓  PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print()


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Wildlife Camera Image Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    Run interactive menu
  %(prog)s --download         Download new images
  %(prog)s --label            Label/classify images
  %(prog)s --visualize        Generate visualizations
  %(prog)s --all              Run all steps in sequence
  %(prog)s -d -l -v           Same as --all (using short options)
        """,
    )

    parser.add_argument(
        "-d",
        "--download",
        action="store_true",
        help="Download new images from camera gallery",
    )

    parser.add_argument(
        "-l",
        "--label",
        action="store_true",
        help="Label/classify images using MegaDetector and species classifier",
    )

    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="Generate visualizations and analysis reports",
    )

    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Run all steps in sequence (download, label, visualize)",
    )

    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompts (auto-proceed)",
    )

    return parser.parse_args()


def main():
    """Main function with command-line argument support and interactive menu."""
    args = parse_arguments()

    # Check if any action flags were provided
    has_action = args.download or args.label or args.visualize or args.all

    if has_action:
        # Command-line mode: run specified actions
        print_header()

        skip_prompt = args.yes

        if args.all:
            # Run all steps in sequence
            run_all(skip_prompt=skip_prompt)
        else:
            # Run individual steps as specified
            if args.download:
                download_images(skip_prompt=skip_prompt)

            if args.label:
                label_images(skip_prompt=skip_prompt)

            if args.visualize:
                generate_visualizations(skip_prompt=skip_prompt)

        print()
    else:
        # Interactive menu mode
        print_header()

        while True:
            print_menu()

            try:
                choice = input("\nEnter your choice (1-5): ").strip()

                if choice == "1":
                    download_images()
                elif choice == "2":
                    label_images()
                elif choice == "3":
                    generate_visualizations()
                elif choice == "4":
                    run_all()
                elif choice == "5":
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
