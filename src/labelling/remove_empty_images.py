"""Delete images confirmed empty by a human (class == "none" and manual_label == "none").

Removes the original image in images/<year>/<location>/, its annotated image and
its row in the detection CSV. Deleted paths are written to logs/ and the CSV is
backed up before it is rewritten.

Usage:
    uv run src/labelling/remove_empty_images.py [--dry-run]
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

try:
    from . import label_data
    from .label_data import backup_csv
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
    import label_data
    from label_data import backup_csv
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
    return (row.get("class") or "").strip() == EMPTY_LABEL and (
        row.get("manual_label") or ""
    ).strip() == EMPTY_LABEL


def annotated_paths(removed, annotated_dir=None):
    annotated_dir = Path(annotated_dir) if annotated_dir is not None else ANNOTATED_DIR
    if not annotated_dir.is_dir():
        return []
    names = {annotated_name(location, name) for location, name in removed}
    return [
        path
        for path in sorted(annotated_dir.rglob("*"))
        if path.is_file() and nfc(path.name) in names
    ]


def format_size(size):
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def remove_empty_images(
    dry_run=False, *, csv_path=None, annotated_dir=None, expected_bytes=None
):
    csv_path = Path(csv_path) if csv_path is not None else CSV_PATH
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return 0

    original_bytes = csv_path.read_bytes()
    if expected_bytes is not None and original_bytes != expected_bytes:
        raise RuntimeError(
            "CSV changed outside this reviewer. "
            "Close and reopen it before removing images."
        )
    fieldnames, rows = read_csv(csv_path)
    if "manual_label" not in fieldnames:
        print("CSV has no manual_label column, nothing to remove")
        return 0

    removed = set()
    review_rows = (
        label_data.load_rows(csv_path) if csv_path.name == "human_labels.csv" else rows
    )
    for row in review_rows:
        if is_confirmed_empty(row):
            removed.add((nfc(row["location_id"]), row["image_file"]))
    kept = [
        row
        for row in rows
        if (nfc(row["location_id"]), row["image_file"]) not in removed
    ]

    paths = image_paths(IMAGES_DIR)
    files = [paths[key] for key in sorted(removed) if key in paths]
    annotated = annotated_paths(removed, annotated_dir)
    size = sum(path.stat().st_size for path in files + annotated)
    verb = "Would delete" if dry_run else "Deleting"
    print(
        f"{verb} {len(removed)} confirmed empty images "
        f"({len(files)} found in {IMAGES_DIR.relative_to(REPO_DIR)}/, "
        f"{len(annotated)} annotated images)"
    )
    if dry_run:
        print(f"Would free {format_size(size)}")
        return len(removed)
    if not removed:
        return 0

    if csv_path.read_bytes() != original_bytes:
        raise RuntimeError(
            "CSV changed during cleanup. Close and reopen it before removing images."
        )

    LOG_DIR.mkdir(exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d-%H%M%S-%f}"
    backup = backup_csv(csv_path, "remove-empty")
    log_path = LOG_DIR / f"remove_empty_images_{stamp}.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["location_id", "image_file", "path"])
        for location, name in sorted(removed):
            path = paths.get((location, name))
            relative = path.relative_to(REPO_DIR) if path else ""
            writer.writerow([location, name, relative])
    print(f"Deleted paths written to {log_path.relative_to(REPO_DIR)}")

    if csv_path.name == "human_labels.csv":
        deleted_path = csv_path.parent / "deleted_images.csv"
        deleted = {
            label_data.image_key(row) for row in label_data.read_records(deleted_path)
        } | removed
        label_data.write_records(
            deleted_path,
            [
                {"location_id": location, "image_file": name}
                for location, name in sorted(deleted)
            ],
            ("location_id", "image_file"),
        )

    for path in files + annotated:
        path.unlink()

    rewrite_csv(csv_path, kept, fieldnames)
    print(
        f"CSV: {len(removed)} rows dropped; backup: {backup.name if backup else 'existing identical snapshot'}"
    )
    print(f"Freed {format_size(size)}")
    return len(removed)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run", action="store_true", help="report planned changes only"
    )
    args = parser.parse_args()
    remove_empty_images(args.dry_run)


if __name__ == "__main__":
    main()
