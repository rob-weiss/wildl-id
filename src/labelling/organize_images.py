"""Move camera images from data/<location>/ into images/<year>/<location>/.

Steps:
1. Delete videos and images that cannot be decoded from data/.
2. Sort remaining images by EXIF capture time and rename them to image_XXXXXXX.jpg.
   Numbering continues after the highest number already present in images/.
   Images without an EXIF timestamp stay in data/.
3. Update image_file in the detection CSV and rename the matching annotated images
   (<location>_<image_file>) so both stay consistent with the moved files.

A mapping of old to new paths is written to logs/ before anything is moved.

Usage:
    uv run src/labelling/organize_images.py [--dry-run]
"""

import argparse
import csv
import os
import re
import shutil
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from PIL import Image
from tqdm import tqdm

REPO_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_DIR / "data"
IMAGES_DIR = REPO_DIR / "images"
LOG_DIR = REPO_DIR / "logs"
LABEL_DIR = REPO_DIR / "labels" / "labels_MegaDetectorV6_MDV6-yolov10-e_classified"
CSV_PATH = LABEL_DIR / "detection_results_MegaDetectorV6_MDV6-yolov10-e_classified.csv"
ANNOTATED_DIR = LABEL_DIR / "images"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".flv",
    ".wmv",
    ".webm",
    ".m4v",
    ".3gp",
}
NAME_PATTERN = re.compile(r"^image_(\d{7})\.[^.]+$")


def nfc(text):
    # macOS stores names like "Madenhäule" decomposed; the CSV uses composed form.
    return unicodedata.normalize("NFC", text)


def parse_exif_time(raw):
    return datetime.strptime(str(raw).strip().rstrip("\x00"), "%Y:%m:%d %H:%M:%S")


def inspect_image(path):
    """Return (path, timestamp or None, error or None) after fully decoding the image."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
            exif = img.getexif()
        exif_ifd = exif.get_ifd(0x8769)
        raw = exif_ifd.get(0x9003) or exif_ifd.get(0x9004) or exif.get(0x0132)
        timestamp = parse_exif_time(raw) if raw else None
        return path, timestamp, None
    except Exception as e:
        return path, None, str(e) or type(e).__name__


def natural_key(name):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def next_free_number():
    numbers = [
        int(match.group(1))
        for path in IMAGES_DIR.rglob("image_*")
        if (match := NAME_PATTERN.match(path.name))
    ]
    return max(numbers, default=0) + 1


def image_paths(images_dir=IMAGES_DIR):
    """Map (location, image_file) to images/<year>/<location>/<file>."""
    return {
        (nfc(path.parent.name), path.name): path
        for path in images_dir.glob("*/*/*")
        if path.is_file() and not path.name.startswith(".")
    }


def collect_files():
    images, videos, others = [], [], []
    if not DATA_DIR.is_dir():
        return images, videos, others
    for location_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        for path in sorted(location_dir.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            suffix = path.suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                images.append(path)
            elif suffix in VIDEO_EXTENSIONS:
                videos.append(path)
            else:
                others.append(path)
    return images, videos, others


def location_of(path):
    return nfc(path.relative_to(DATA_DIR).parts[0])


def rename_row(row, renames):
    location = nfc(row["location_id"])
    row["location_id"] = location
    row["image_file"] = renames.get((location, row["image_file"]), row["image_file"])
    return row


def annotated_name(location, image_file):
    return f"{location}_{image_file}"


def rewrite_csv(path, rows, fieldnames):
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def update_csv(renames, removed, dry_run):
    """Rewrite image_file for moved images and drop rows of deleted images."""
    if not CSV_PATH.exists():
        print(f"CSV not found, skipping: {CSV_PATH}")
        return

    fieldnames, rows = read_csv(CSV_PATH)
    kept, renamed, dropped = [], 0, 0
    for row in rows:
        key = (nfc(row["location_id"]), row["image_file"])
        if key in removed:
            dropped += 1
            continue
        renamed += key in renames
        kept.append(rename_row(row, renames))

    print(f"CSV: {renamed} rows renamed, {dropped} rows dropped (deleted images)")
    if dry_run:
        return
    backup = CSV_PATH.with_name(
        f"{CSV_PATH.name}.before-organize-{datetime.now():%Y%m%d-%H%M%S}.bak"
    )
    shutil.copy2(CSV_PATH, backup)
    rewrite_csv(CSV_PATH, kept, fieldnames)
    print(f"CSV updated, backup at {backup.name}")


def rename_files(directory, renames, name_fn, dry_run, removed=()):
    """Rename files named name_fn(location, old) below directory; delete removed ones."""
    targets = {
        name_fn(location, old): name_fn(location, new)
        for (location, old), new in renames.items()
    }
    doomed = {name_fn(location, old) for location, old in removed}
    moved = deleted = 0
    for path in sorted(directory.rglob("*")):
        name = nfc(path.name)
        if not path.is_file():
            continue
        if name in targets:
            moved += 1
            if not dry_run:
                path.rename(path.with_name(targets[name]))
        elif name in doomed:
            deleted += 1
            if not dry_run:
                path.unlink()
    return moved, deleted


def rename_annotated(renames, removed, dry_run):
    """Keep annotated images (<location>_<image_file>) in sync with the CSV."""
    if not ANNOTATED_DIR.is_dir():
        return
    moved, deleted = rename_files(
        ANNOTATED_DIR, renames, annotated_name, dry_run, removed
    )
    print(f"Annotated images: {moved} renamed, {deleted} deleted")


def organize_images(dry_run=False, workers=None):
    images, videos, others = collect_files()
    print(
        f"Found {len(images)} images, {len(videos)} videos, "
        f"{len(others)} other files in {DATA_DIR}"
    )
    for path in others:
        print(f"  Ignoring unknown file type: {path.relative_to(REPO_DIR)}")

    for path in videos:
        if not dry_run:
            path.unlink()
    print(f"{'Would delete' if dry_run else 'Deleted'} {len(videos)} videos")

    with ThreadPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
        results = list(
            tqdm(
                pool.map(inspect_image, images),
                total=len(images),
                desc="Checking images",
            )
        )

    removed = set()
    valid, no_timestamp = [], []
    for path, timestamp, error in results:
        key = (location_of(path), path.name)
        if error:
            print(f"  Corrupt: {path.relative_to(REPO_DIR)} ({error})")
            removed.add(key)
            if not dry_run:
                path.unlink()
            continue
        if timestamp is None:
            no_timestamp.append(path)
            continue
        valid.append((timestamp, key[0], natural_key(path.name), path))

    print(f"{'Would delete' if dry_run else 'Deleted'} {len(removed)} corrupt images")
    for path in no_timestamp:
        print(f"  No timestamp, left in place: {path.relative_to(REPO_DIR)}")

    valid.sort(key=lambda item: item[:3])
    number = next_free_number()
    plan, renames = [], {}
    for timestamp, location, _, path in valid:
        new_name = f"image_{number:07d}{path.suffix.lower()}"
        target = IMAGES_DIR / str(timestamp.year) / location / new_name
        if target.exists():
            raise FileExistsError(target)
        plan.append((path, target))
        renames[(location, path.name)] = new_name
        number += 1

    if plan:
        print(
            f"{'Would move' if dry_run else 'Moving'} {len(plan)} images to "
            f"{IMAGES_DIR.relative_to(REPO_DIR)}/ "
            f"({plan[0][1].name} .. {plan[-1][1].name})"
        )

    if not dry_run and (plan or removed):
        LOG_DIR.mkdir(exist_ok=True)
        log_path = LOG_DIR / f"organize_images_{datetime.now():%Y%m%d-%H%M%S}.csv"
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["old_path", "new_path"])
            for source, target in plan:
                writer.writerow(
                    [source.relative_to(REPO_DIR), target.relative_to(REPO_DIR)]
                )
            for location, name in sorted(removed):
                writer.writerow([f"data/{location}/{name}", ""])
        print(f"Mapping written to {log_path.relative_to(REPO_DIR)}")

        for source, target in tqdm(plan, desc="Moving images"):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(source, target)

    if not (renames or removed):
        return
    update_csv(renames, removed, dry_run)
    rename_annotated(renames, removed, dry_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run", action="store_true", help="report planned changes only"
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(), help="parallel image checks"
    )
    args = parser.parse_args()
    organize_images(args.dry_run, args.workers)


if __name__ == "__main__":
    main()
