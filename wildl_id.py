#!/usr/bin/env python3
"""
Wildlife Camera Image Analysis - Main Interface

This is the main entry point for the wildlife camera analysis system.
It provides a user-friendly interface to:
1. Organise new images from data/ into images/<year>/<location>/
2. Label/classify images using MegaDetector and species classifier
3. Generate visualisations and analysis reports
4. Run all supported steps in sequence

Usage:
    # Interactive menu:
    python wildl_id.py

    # Command-line arguments:
    python wildl_id.py --download         # Download new camera images from email
    python wildl_id.py --notify           # Push alerts for new target species
    python wildl_id.py --organize         # Organise images
    python wildl_id.py --label            # Label images
    python wildl_id.py --visualize        # Generate visualisations
    python wildl_id.py --all              # Run all steps
    python wildl_id.py -o -l -v           # Organise, label, and visualise
"""

import argparse
import csv
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the main functions from each module
from alerting.push_alerts import send_push_alerts as alerts_main
from downloading.download_email_images import download_email_images as download_main
from labelling.label_images import process_images_with_pytorch_wildlife
from labelling.organize_images import image_paths, nfc
from labelling.organize_images import organize_images as organize_main
from labelling.review_labels import DEFAULT_CSV
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
    print("1. Organise new images (data/ → images/<year>/<location>/)")
    print("2. Label/classify images (MegaDetector + Species Classifier)")
    print("3. Generate visualisations and analysis reports")
    print("4. Run all supported steps in sequence (Organise → Label → Visualise)")
    print("5. Download new images from email")
    print("6. Exit")
    print("-" * 70)


def organize_images():
    """Move new images from data/ into images/<year>/<location>/."""
    print("\n" + "=" * 70)
    print("🗂️  ORGANISING IMAGES")
    print("=" * 70)
    print()

    try:
        organize_main()
        print("\n✓ Organising completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Organising failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Organising interrupted by user")
        return False


def download_images(organize=True):
    """Download matching camera images and organise staged images when needed."""
    print("\n" + "=" * 70)
    print("📨  DOWNLOADING IMAGES FROM EMAIL")
    print("=" * 70)
    print()

    try:
        result = download_main()
        for failure in result.failures:
            print(f"✗ {failure}")
        success = not result.failures
        print(
            f"Downloaded {result.saved_images} images; "
            f"skipped {result.duplicates} duplicates"
        )
        if result.pending_organization:
            if organize:
                organization_success = organize_images()
                success = organization_success and success
            else:
                print("Staged images will be organised by the requested workflow.")
        return success
    except Exception as e:
        print(f"\n✗ Email download failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Email download interrupted by user")
        return False


def send_alerts():
    """Push alerts for target species in newly organised images."""
    try:
        result = alerts_main()
    except Exception as e:
        print(f"\n✗ Push alerts failed: {e}")
        return False
    if not result.enabled:
        print("Push alerts are not configured; skipping.")
        return True
    for failure in result.failures:
        print(f"✗ Push alert: {failure}")
    print(f"Sent {result.sent} push alerts")
    return not result.failures


def label_new_images(camera=None):
    try:
        completed = set()
        if DEFAULT_CSV.exists():
            with DEFAULT_CSV.open(newline="", encoding="utf-8") as source:
                for row in csv.DictReader(source):
                    manual = (row.get("manual_label") or "").strip()
                    automatic = (row.get("class") or "").strip().casefold()
                    if manual or automatic not in {"", "unknown", "animal"}:
                        completed.add((nfc(row["location_id"]), row["image_file"]))
        pending = {
            key
            for key, path in image_paths().items()
            if path.suffix.casefold() in {".jpg", ".jpeg", ".png"}
        } - completed
        if not pending:
            print("No new images require automatic labelling.")
            return True
        print(
            f"Automatically labelling {len(pending)} pending images into {DEFAULT_CSV}"
        )
        process_images_with_pytorch_wildlife(
            camera=camera, output_csv_path=DEFAULT_CSV, image_keys=pending
        )
        return True
    except Exception as error:
        print(f"Automatic labelling failed: {error}")
        return False


def label_images(camera=None, reprocess_missing=False):
    """Label and classify images using MegaDetector."""
    print("\n" + "=" * 70)
    print("🏷️  LABELLING IMAGES")
    print("=" * 70)
    print()
    print("This will run MegaDetector for animal detection and DeepFaune for species")
    print("classification. It will also extract metadata (timestamp, temperature) and")
    print("analyse lighting conditions.")
    print()

    try:
        process_images_with_pytorch_wildlife(
            camera=camera, reprocess_missing=reprocess_missing
        )
        print("\n✓ Labelling completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Labelling failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Labelling interrupted by user")
        return False


def generate_visualizations():
    """Generate visualisations and analysis reports."""
    print("\n" + "=" * 70)
    print("📊  GENERATING VISUALISATIONS")
    print("=" * 70)
    print()
    print("This will create comprehensive visualisations including:")
    print("  • Activity patterns by hour of day")
    print("  • Species distribution charts")
    print("  • Calendar heatmaps")
    print("  • Location-based statistics")
    print("  • Day/night activity patterns")
    print("  • Temperature correlations")
    print()

    try:
        if visualisation_main() is False:
            print("\n✗ Visualisation failed")
            return False
        print("\n✓ Visualisations generated successfully!")
        print("📁 Check the docs/diagrams/ folder for output files")
        return True
    except Exception as e:
        print(f"\n✗ Visualisation failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Visualisation interrupted by user")
        return False


def run_all(camera=None, reprocess_missing=False):
    """Run all supported steps in sequence."""
    print("\n" + "=" * 70)
    print("🚀  RUNNING COMPLETE PIPELINE")
    print("=" * 70)
    print()
    print("This will run all supported steps in sequence:")
    print("  1. Organise images")
    print("  2. Label/classify images")
    print("  3. Generate visualisations")
    print()

    steps = (
        ("ORGANISING IMAGES", organize_images),
        (
            "LABELLING IMAGES",
            lambda: label_images(camera=camera, reprocess_missing=reprocess_missing),
        ),
        ("GENERATING VISUALISATIONS", generate_visualizations),
    )
    for index, (description, action) in enumerate(steps, start=1):
        print("\n\n" + "=" * 70)
        print(f"STEP {index}/{len(steps)}: {description}")
        print("=" * 70)
        if not action():
            print(f"\n⚠️  Pipeline stopped during {description.lower()}")
            return False

    print("\n\n" + "=" * 70)
    print("✓  PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print()
    return True


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Wildlife Camera Image Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                    Run interactive menu
    %(prog)s --download         Download new images from email
    %(prog)s --download --notify   Download, then push alerts for new sightings
    %(prog)s --organize         Organise new images
    %(prog)s --label            Label/classify images
    %(prog)s --visualize        Generate visualisations
    %(prog)s --all              Run all supported steps in sequence
    %(prog)s -o -l -v           Organise, label, and visualise
    %(prog)s -l --camera seissiger   Label with explicit Seissiger camera format
    %(prog)s -l --camera zeiss       Label with explicit Zeiss camera format
    %(prog)s -l --reprocess-missing  Reprocess incomplete/missing label entries
        """,
    )

    parser.add_argument(
        "-c",
        "--camera",
        choices=["zeiss", "seissiger"],
        default="seissiger",
        help="Camera brand for timestamp parsing (default: seissiger)",
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help="Download new camera images from email",
    )

    parser.add_argument(
        "--notify",
        action="store_true",
        help="Push ntfy alerts for target species in newly organised images",
    )

    parser.add_argument(
        "-o",
        "--organize",
        action="store_true",
        help="Move new images from data/ into images/<year>/<location>/",
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
        help="Generate visualisations and analysis reports",
    )

    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Run all supported steps in sequence (organise, label, visualise)",
    )

    parser.add_argument(
        "--reprocess-missing",
        action="store_true",
        help="When labelling, reprocess entries with missing class/timestamp/temperature",
    )

    return parser.parse_args()


def main():
    """Run requested command-line actions or the interactive menu."""
    args = parse_arguments()

    # Check if any action flags were provided
    has_action = (
        args.download
        or args.notify
        or args.organize
        or args.label
        or args.visualize
        or args.all
    )

    if has_action:
        # Command-line mode: run specified actions
        print_header()

        if args.all:
            download_success = True
            if args.download:
                download_success = download_images(organize=False)
            pipeline_success = run_all(
                camera=args.camera, reprocess_missing=args.reprocess_missing
            )
            success = pipeline_success and download_success
            if args.notify:
                success = send_alerts() and success
        else:
            success = True
            # Run individual steps as specified
            if args.download:
                success = download_images(organize=not args.organize)

            if args.organize:
                organization_success = organize_images()
                success = organization_success and success

            if args.label:
                success = (
                    label_images(
                        camera=args.camera, reprocess_missing=args.reprocess_missing
                    )
                    and success
                )
            elif args.download:
                success = label_new_images(camera=args.camera) and success

            if args.notify:
                success = send_alerts() and success

            if args.visualize:
                success = generate_visualizations() and success

        print()
        return 0 if success else 1
    else:
        # Interactive menu mode
        print_header()
        exit_status = 0

        while True:
            print_menu()

            try:
                choice = input("\nEnter your choice (1-5): ").strip()

                if choice == "1":
                    success = organize_images()
                elif choice == "2":
                    success = label_images()
                elif choice == "3":
                    success = generate_visualizations()
                elif choice == "4":
                    success = run_all()
                elif choice == "5":
                    success = download_images()
                elif choice == "6":
                    print("\n👋  Goodbye!")
                    break
                else:
                    print("\n⚠️  Invalid choice. Please enter 1-6.")
                    continue
                if not success:
                    exit_status = 1

            except KeyboardInterrupt:
                print("\n\n👋  Goodbye!")
                break
            except Exception as e:
                print(f"\n✗ An error occurred: {e}")
                print("Please try again or report this issue.")

        print()
        return exit_status


if __name__ == "__main__":
    sys.exit(main())
