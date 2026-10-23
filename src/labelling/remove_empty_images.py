"""Delete images confirmed empty by a human (class == "none" and manual_label == "none").

Removes the original image in images/<year>/<location>/, its annotated image and
its row in the detection CSV. Deleted paths are written to logs/ and the CSV is
backed up before it is rewritten.

Usage:
    uv run src/labelling/remove_empty_images.py [--dry-run]
"""

import argparse
import csv
import shutil
from datetime import datetime

try:
    from .organize_images import (
        ANNOTATED_DIR,
        CSV_PATH,
        IMAGES_DIR,
        LOG_DIR,
        REPO_DIR,
        annotated_name,
        image_paths,
        nfc,
        read_csv,
        rewrite_csv,
    )
except ImportError:
    from organize_images import (
        ANNOTATED_DIR,
        CSV_PATH,
        IMAGES_DIR,
        LOG_DIR,
        REPO_DIR,
        annotated_name,
        image_paths,
        nfc,
        read_csv,
        rewrite_csv,
    )

EMPTY_LABEL = "none"


def is_confirmed_empty(row):
    return (
        row["class"].strip() == EMPTY_LABEL
        and (row.get("manual_label") or "").strip() == EMPTY_LABEL
    )


def annotated_paths(removed):
    if not ANNOTATED_DIR.is_dir():
        return []
    names = {annotated_name(location, name) for location, name in removed}
    return [
        path
        for path in sorted(ANNOTATED_DIR.rglob("*"))
        if path.is_file() and nfc(path.name) in names
    ]


def format_size(size):
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def remove_empty_images(dry_run=False):
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        return

    fieldnames, rows = read_csv(CSV_PATH)
    if "manual_label" not in fieldnames:
        print("CSV has no manual_label column, nothing to remove")
        return

    kept, removed = [], set()
    for row in rows:
        if is_confirmed_empty(row):
            removed.add((nfc(row["location_id"]), row["image_file"]))
        else:
            kept.append(row)

    paths = image_paths(IMAGES_DIR)
    files = [paths[key] for key in sorted(removed) if key in paths]
    annotated = annotated_paths(removed)
    size = sum(path.stat().st_size for path in files + annotated)
    verb = "Would delete" if dry_run else "Deleting"
    print(
        f"{verb} {len(removed)} confirmed empty images "
        f"({len(files)} found in {IMAGES_DIR.relative_to(REPO_DIR)}/, "
        f"{len(annotated)} annotated images)"
    )
    if dry_run:
        print(f"Would free {format_size(size)}")
        return
    if not removed:
        return

    LOG_DIR.mkdir(exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    log_path = LOG_DIR / f"remove_empty_images_{stamp}.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["location_id", "image_file", "path"])
        for location, name in sorted(removed):
            path = paths.get((location, name))
            relative = path.relative_to(REPO_DIR) if path else ""
            writer.writerow([location, name, relative])
    print(f"Deleted paths written to {log_path.relative_to(REPO_DIR)}")

    for path in files + annotated:
        path.unlink()

    backup = CSV_PATH.with_name(f"{CSV_PATH.name}.before-remove-empty-{stamp}.bak")
    shutil.copy2(CSV_PATH, backup)
    rewrite_csv(CSV_PATH, kept, fieldnames)
    print(f"CSV: {len(removed)} rows dropped, backup at {backup.name}")
    print(f"Freed {format_size(size)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run", action="store_true", help="report planned changes only"
    )
    args = parser.parse_args()
    remove_empty_images(args.dry_run)


if __name__ == "__main__":
    main()
