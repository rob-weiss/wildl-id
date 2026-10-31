"""Human annotations, image metadata, and model-specific predictions."""

import csv
import io
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unicodedata import normalize

REPO_DIR = Path(__file__).resolve().parents[2]
HUMAN_COLUMNS = ("location_id", "image_file", "box", "manual_label")
METADATA_COLUMNS = (
    "location_id",
    "image_file",
    "timestamp",
    "timestamp_exif",
    "lighting",
    "temperature_celsius",
)
PREDICTION_COLUMNS = (
    "location_id",
    "image_file",
    "class",
    "box",
    "confidence",
    "classification_confidence",
    "classifier_model",
    "timestamp",
    "timestamp_exif",
    "lighting",
    "temperature_celsius",
)
LEGACY_RELATIVE_PATH = (
    "labels/labels_MegaDetectorV6_MDV6-yolov10-e_classified/"
    "detection_results_MegaDetectorV6_MDV6-yolov10-e_classified.csv"
)
DEFAULT_HUMAN_CSV = REPO_DIR / "labels/human_labels.csv"


def image_key(row):
    return normalize("NFC", row["location_id"]), row["image_file"]


def latest_weights(models_dir):
    candidates = [
        path
        for path in Path(models_dir).glob("deepfaune_finetuned_*/weights.pt")
        if path.is_file()
        and re.fullmatch(r"deepfaune_finetuned_\d{8}-\d{6}_.+", path.parent.name)
    ]
    return max(candidates, key=lambda path: path.parent.name, default=None)


def predictions_path(repo_dir=REPO_DIR):
    weights = latest_weights(Path(repo_dir) / "models")
    run = weights.parent if weights else Path(repo_dir) / "models/deepfaune_pretrained"
    return run / "predictions.csv"


def read_records(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_records(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=path.parent, delete=False
        ) as target:
            temporary_path = Path(target.name)
            writer = csv.DictWriter(
                target,
                fieldnames=columns,
                extrasaction="ignore",
                lineterminator="\n",
                quoting=csv.QUOTE_ALL,
            )
            writer.writeheader()
            writer.writerows(rows)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def save_image_metadata(repo_dir, rows):
    path = Path(repo_dir) / "labels/image_metadata.csv"
    indexed = {image_key(row): row for row in read_records(path)}
    for row in rows:
        key = image_key(row)
        previous = indexed.get(key, {})
        indexed[key] = {
            column: row.get(column, "")
            if row.get(column, "") not in ("", None)
            else previous.get(column, "")
            for column in METADATA_COLUMNS
        }
    write_records(path, indexed.values(), METADATA_COLUMNS)


def migrate_human_labels(repo_dir=REPO_DIR):
    repo_dir = Path(repo_dir)
    human_path = repo_dir / "labels/human_labels.csv"
    displaced = []
    displaced_bytes = None
    if human_path.exists():
        with human_path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames == list(HUMAN_COLUMNS):
                return human_path
            if not {*HUMAN_COLUMNS, "class", "classifier_model"}.issubset(
                reader.fieldnames or []
            ):
                raise ValueError("Existing human-label CSV has an unexpected schema")
            displaced = list(reader)
        if any(row.get("manual_label", "").strip() for row in displaced):
            raise ValueError(
                "Mixed human-label file has manual annotations; resolve them before migration"
            )
        displaced_bytes = human_path.read_bytes()
        backup = human_path.with_name(
            f"{human_path.name}.before-separation-{datetime.now():%Y%m%d-%H%M%S-%f}.bak"
        )
        shutil.copy2(human_path, backup)
    legacy = repo_dir / LEGACY_RELATIVE_PATH
    if not legacy.exists():
        raise FileNotFoundError(f"Original human-labelled dataset not found: {legacy}")
    original = legacy.read_bytes()
    reader = csv.DictReader(io.StringIO(original.decode("utf-8")))
    if not set(HUMAN_COLUMNS).issubset(reader.fieldnames or []):
        raise ValueError("Original dataset is missing human annotation columns")
    rows = list(reader)
    save_image_metadata(repo_dir, [*displaced, *rows])
    prediction_path = predictions_path(repo_dir)
    verified = {image_key(row): row for row in read_records(prediction_path)}
    for row in [*displaced, *rows]:
        if (
            row.get("classifier_model") == prediction_path.parent.name
            and row.get("class", "").strip()
        ):
            verified.setdefault(image_key(row), row)
    write_records(prediction_path, verified.values(), PREDICTION_COLUMNS)
    if legacy.read_bytes() != original:
        raise RuntimeError(
            "Original dataset changed during migration; retry after stopping its writer"
        )
    if displaced_bytes is not None and human_path.read_bytes() != displaced_bytes:
        raise RuntimeError(
            "Automatic output changed during migration; stop its writer and retry"
        )
    write_records(human_path, rows, HUMAN_COLUMNS)
    return human_path


def load_rows(csv_path):
    csv_path = Path(csv_path)
    if csv_path.name != "human_labels.csv":
        return read_records(csv_path)
    with csv_path.open(newline="", encoding="utf-8") as source:
        if csv.DictReader(source).fieldnames != list(HUMAN_COLUMNS):
            raise ValueError(
                "Human-label CSV must contain only the four annotation columns"
            )
    repo_dir = csv_path.parent.parent
    humans = read_records(csv_path)
    metadata = {
        image_key(row): row
        for row in read_records(repo_dir / "labels/image_metadata.csv")
    }
    predictions = {
        image_key(row): row for row in read_records(predictions_path(repo_dir))
    }
    rows = []
    seen = set()
    for human in humans:
        key = image_key(human)
        prediction = predictions.get(key, {})
        row = dict.fromkeys((*PREDICTION_COLUMNS, "manual_label"), "")
        row.update(metadata.get(key, {}))
        row.update(
            {column: value for column, value in prediction.items() if value != ""}
        )
        row.update(
            {
                column: human.get(column, "")
                for column in HUMAN_COLUMNS
                if column != "box"
            }
        )
        if human.get("box", ""):
            row["box"] = human["box"]
        rows.append(row)
        seen.add(key)
    for key, prediction in predictions.items():
        if key not in seen:
            rows.append({**metadata.get(key, {}), **prediction, "manual_label": ""})
    return rows
