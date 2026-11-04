"""Batch-enrich canonical image metadata with interpolated Open-Meteo estimates."""

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unicodedata import normalize
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.labelling import label_data
from src.visits import exif_datetime

API_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
REPO_DIR = Path(__file__).resolve().parents[1]
METADATA_REQUIRED_COLUMNS = (
    "location_id",
    "image_file",
    "timestamp_exif",
    "temperature_celsius",
)
PREDICTION_REQUIRED_COLUMNS = (
    "location_id",
    "image_file",
    "temperature_celsius",
)


def fetch_hourly_temperature(
    latitude: float, longitude: float, start_date: date, end_date: date
) -> dict[datetime, float]:
    query = urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "timezone": "UTC",
        }
    )
    request = Request(f"{API_URL}?{query}")
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)

    hourly = payload.get("hourly") if isinstance(payload, dict) else None
    if not isinstance(hourly, dict):
        raise ValueError("Open-Meteo response is missing hourly data")
    times = hourly.get("time")
    values = hourly.get("temperature_2m")
    if not isinstance(times, list) or not isinstance(values, list):
        raise ValueError("Open-Meteo hourly time and temperature arrays are required")
    if len(times) != len(values):
        raise ValueError("Open-Meteo hourly arrays must have matching lengths")

    temperatures = {}
    for timestamp, value in zip(times, values):
        if not isinstance(timestamp, str):
            raise ValueError("Open-Meteo returned an invalid hourly timestamp")
        try:
            taken = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(
                f"Open-Meteo returned an invalid hourly timestamp: {timestamp!r}"
            ) from error
        if taken.tzinfo is None:
            taken = taken.replace(tzinfo=UTC)
        else:
            taken = taken.astimezone(UTC)
        hour = taken.replace(minute=0, second=0, microsecond=0)

        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Open-Meteo returned an invalid temperature")
        try:
            temperature = float(value)
        except OverflowError as error:
            raise ValueError(
                f"Open-Meteo returned an invalid temperature: {value!r}"
            ) from error
        if not math.isfinite(temperature):
            raise ValueError(
                f"Open-Meteo returned an invalid temperature: {value!r}"
            )
        temperatures[hour] = temperature
    return temperatures


def _read_csv(path, required_columns):
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames or []
    missing = set(required_columns) - set(columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {', '.join(sorted(missing))}"
        )
    rows = label_data.read_records(path)
    if any(None in row for row in rows):
        raise ValueError(f"{path} contains rows with more fields than its header")
    if any(not row.get("location_id") or not row.get("image_file") for row in rows):
        raise ValueError(f"{path} contains a row without an image identity")
    return rows, columns


def _location_key(value):
    return normalize("NFC", str(value or ""))


def _temperature_at(
    taken: datetime, temperatures: dict[datetime, float]
) -> float | None:
    hour = taken.replace(minute=0, second=0, microsecond=0)
    current = temperatures.get(hour)
    if current is None or taken == hour:
        return current
    following = temperatures.get(hour + timedelta(hours=1))
    if following is None:
        return None
    fraction = (taken - hour).total_seconds() / 3600
    return round(current + (following - current) * fraction, 1)


def _load_cameras(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cameras = payload.get("cameras") if isinstance(payload, dict) else None
    if not isinstance(cameras, list):
        raise ValueError("Camera locations must contain a cameras list")

    configured = {}
    for camera in cameras:
        if not isinstance(camera, dict):
            raise ValueError("Camera location entries must be objects")
        location = _location_key(camera.get("location_id"))
        if not location:
            raise ValueError("Camera location entry is missing location_id")
        if location in configured:
            raise ValueError(f"Duplicate camera location: {location}")
        latitude = camera.get("latitude")
        longitude = camera.get("longitude")
        if latitude is None or longitude is None:
            configured[location] = None
            continue
        try:
            latitude = float(latitude)
            longitude = float(longitude)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid coordinates for camera {location}") from error
        if (
            not math.isfinite(latitude)
            or not math.isfinite(longitude)
            or not -90 <= latitude <= 90
            or not -180 <= longitude <= 180
        ):
            raise ValueError(f"Invalid coordinates for camera {location}")
        configured[location] = (latitude, longitude)
    return configured


def enrich_temperatures(repo_dir: Path, dry_run: bool = False) -> tuple[int, int]:
    repo_dir = Path(repo_dir)
    labels_dir = repo_dir / "labels"
    metadata_path = labels_dir / "image_metadata.csv"
    metadata_rows, metadata_columns = _read_csv(
        metadata_path, METADATA_REQUIRED_COLUMNS
    )
    prediction_outputs = []
    for path in sorted(labels_dir.glob("*/predictions.csv")):
        rows, columns = _read_csv(path, PREDICTION_REQUIRED_COLUMNS)
        prediction_outputs.append((path, rows, columns))
    cameras = _load_cameras(repo_dir / "config/camera_locations.json")

    keys = [label_data.image_key(row) for row in metadata_rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Image metadata contains duplicate image identities")

    row_times = []
    hours_by_camera = defaultdict(list)
    for row in metadata_rows:
        location = _location_key(row["location_id"])
        camera = cameras.get(location)
        taken = exif_datetime(row)
        if camera is None or taken is None:
            row_times.append(None)
            continue
        taken = taken.astimezone(UTC)
        hour = taken.replace(minute=0, second=0, microsecond=0)
        row_times.append((location, taken))
        hours_by_camera[location].append(hour)
        if taken != hour:
            hours_by_camera[location].append(hour + timedelta(hours=1))

    weather_by_camera = {}
    for location, hours in hours_by_camera.items():
        latitude, longitude = cameras[location]
        weather_by_camera[location] = fetch_hourly_temperature(
            latitude,
            longitude,
            min(hour.date() for hour in hours),
            max(hour.date() for hour in hours),
        )

    metadata_temperatures = {}
    updated_count = 0
    unresolved_count = 0
    for row, row_time in zip(metadata_rows, row_times):
        temperature = (
            _temperature_at(row_time[1], weather_by_camera[row_time[0]])
            if row_time is not None
            else None
        )
        if temperature is None:
            value = ""
            unresolved_count += 1
        else:
            value = str(temperature)
            updated_count += 1
        row["temperature_celsius"] = value
        metadata_temperatures[label_data.image_key(row)] = value

    for _, rows, _ in prediction_outputs:
        for row in rows:
            key = label_data.image_key(row)
            if key in metadata_temperatures:
                row["temperature_celsius"] = metadata_temperatures[key]

    if not dry_run:
        label_data.write_records(
            metadata_path,
            metadata_rows,
            metadata_columns,
            quoting=csv.QUOTE_ALL,
        )
        for path, rows, columns in prediction_outputs:
            label_data.write_records(path, rows, columns)
    return updated_count, unresolved_count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report update counts without writing any CSV files",
    )
    args = parser.parse_args(argv)
    updated_count, unresolved_count = enrich_temperatures(
        REPO_DIR, dry_run=args.dry_run
    )
    action = "Would update" if args.dry_run else "Updated"
    print(
        f"{action} {updated_count} metadata rows; "
        f"{unresolved_count} unresolved rows."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
