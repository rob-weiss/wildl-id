"""Send ntfy push alerts for selected species in newly organised images."""

import csv
import json
import os
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from PIL import Image

try:
    from ..labelling.organize_images import IMAGES_DIR, image_paths
    from ..labelling.species import RECENT_SPECIES
except ImportError:
    from labelling.organize_images import IMAGES_DIR, image_paths
    from labelling.species import RECENT_SPECIES

CONFIG_PATH = Path.home() / ".config/wildl-id/push-alerts.json"
STATE_PATH = Path.home() / "Library/Application Support/wildl-id/push-alerts/state.json"
SUMMARY_STATE_PATH = STATE_PATH.with_name("daily-summary.json")
KEYCHAIN_SERVICE = "wildl-id-ntfy"
KEYCHAIN_ACCOUNT = "topic"
DEFAULTS = {
    "server": "https://ntfy.sh",
    "species": ["wild boar", "raccoon"],
    "summary_species": sorted(RECENT_SPECIES),
    "min_confidence": 0.5,
    "cooldown_minutes": 30,
    "max_age_minutes": 180,
    "attach_image": True,
    "daily_summary": False,
}
DETECTION_THRESHOLD = 0.6
# Organised EXIF times are timezone-free fixed CET.
CAMERA_TIME = timezone(timedelta(hours=1))
LOCAL_TIME = ZoneInfo("Europe/Berlin")
PUBLISH_TIMEOUT_SECONDS = 30
TAGS = {"wild boar": "boar", "raccoon": "raccoon"}


@dataclass(frozen=True)
class Alert:
    location: str
    species: str
    count: int
    captured: datetime
    image_path: Path | None


@dataclass(frozen=True)
class AlertResult:
    enabled: bool
    sent: int
    failures: tuple[str, ...]


def _read_keychain_secret(service, account):
    try:
        result = subprocess.run(
            [
                "/usr/bin/security",
                "find-generic-password",
                "-w",
                "-s",
                service,
                "-a",
                account,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.rstrip("\r\n")


def _read_config(path):
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("push alert configuration must be a JSON object")
    config = {**DEFAULTS, **loaded}
    server = config["server"]
    if not isinstance(server, str) or urlsplit(server).scheme != "https":
        raise ValueError("push alert server must be an https URL")
    for key in ("species", "summary_species"):
        species = config[key]
        if (
            not isinstance(species, list)
            or not species
            or not all(isinstance(name, str) and name.strip() for name in species)
        ):
            raise ValueError(f"push alert {key} must be a non-empty list of names")
        config[key] = {name.strip() for name in species}
    for key in ("min_confidence", "cooldown_minutes", "max_age_minutes"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
            raise ValueError(f"push alert {key} must be a non-negative number")
    if not 0 <= config["min_confidence"] <= 1:
        raise ValueError("push alert min_confidence must be between 0 and 1")
    if not isinstance(config["attach_image"], bool):
        raise ValueError("push alert attach_image must be true or false")
    if not isinstance(config["daily_summary"], bool):
        raise ValueError("push alert daily_summary must be true or false")
    return {
        **config,
        "server": server.rstrip("/"),
    }


def _load_state(path):
    if not path.exists():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError("push alert state must be a JSON object")
    seen = state.get("seen", {})
    last_alerts = state.get("last_alerts", {})
    if not isinstance(seen, dict) or not isinstance(last_alerts, dict):
        raise ValueError("push alert state has an invalid structure")
    return {
        "seen": {location: set(names) for location, names in seen.items()},
        "last_alerts": last_alerts,
    }


def _write_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    serialisable = {
        "seen": {location: sorted(names) for location, names in state["seen"].items()},
        "last_alerts": state["last_alerts"],
    }
    temporary.write_text(
        json.dumps(serialisable, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _capture_time(path):
    try:
        with Image.open(path) as image:
            exif = image.getexif()
        exif_ifd = exif.get_ifd(0x8769)
        raw = exif_ifd.get(0x9003) or exif_ifd.get(0x9004) or exif.get(0x0132)
        if not raw:
            return None
        captured = datetime.strptime(
            str(raw).strip().rstrip("\x00"), "%Y:%m:%d %H:%M:%S"
        )
    except (OSError, ValueError):
        return None
    return captured.replace(tzinfo=CAMERA_TIME)


def _load_analyser(min_confidence):
    """Return analyse(path) -> species names using the labelling models."""
    import numpy as np
    import torch

    try:
        from ..labelling import label_images
    except ImportError:
        from labelling import label_images

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    if torch.backends.mps.is_available():
        device = "mps"
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        detector = label_images.pw_detection.MegaDetectorV6(
            device=device, pretrained=True, version=label_images.model_name
        )
    classifier = label_images.load_species_classifier(device)

    def analyse(path):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            result = detector.single_image_detection(
                str(path), det_conf_thres=DETECTION_THRESHOLD
            )
        detections = (result or {}).get("detections")
        species = []
        if detections is None or not hasattr(detections, "confidence"):
            return species
        with Image.open(path) as image:
            width, height = image.size
            for xyxy, class_id in zip(detections.xyxy, detections.class_id):
                kind = label_images.MEGADETECTOR_CLASS_NAMES.get(int(class_id))
                if kind != "animal":
                    continue
                x_min, y_min, x_max, y_max = (float(value) for value in xyxy)
                bbox = [
                    x_min / width,
                    y_min / height,
                    (x_max - x_min) / width,
                    (y_max - y_min) / height,
                ]
                crop = label_images.crop_detection(image, bbox)
                if crop is None:
                    continue
                with crop:
                    prediction = classifier.single_image_classification(
                        np.array(crop), img_id=path.name
                    )
                if prediction["confidence"] >= min_confidence:
                    species.append(prediction["prediction"])
        return species

    return analyse


def ntfy_request(server, topic, alert):
    local_time = alert.captured.astimezone(LOCAL_TIME)
    title = f"{alert.species.capitalize()} at {alert.location}"
    message = (
        f"{alert.count} {alert.species} at {local_time:%H:%M} ({local_time:%d.%m.})"
    )
    params = {"title": title, "priority": "high"}
    if alert.species in TAGS:
        params["tags"] = TAGS[alert.species]
    if alert.image_path is None:
        body = message.encode("utf-8")
    else:
        params["message"] = message
        params["filename"] = alert.image_path.name
        body = alert.image_path.read_bytes()
    url = f"{server}/{quote(topic, safe='')}?{urlencode(params)}"
    return Request(url, data=body, method="PUT")


def _publish(server, topic, alert):
    with urlopen(ntfy_request(server, topic, alert), timeout=PUBLISH_TIMEOUT_SECONDS):
        pass


def _send_species_alerts(
    config_path: Path | None = None,
    state_path: Path | None = None,
    images_dir: Path | None = None,
    *,
    keychain_reader: Callable[[str, str], str] | None = None,
    analyser_factory: Callable[[float], Callable] | None = None,
    publisher: Callable[[str, str, Alert], None] | None = None,
    now: datetime | None = None,
) -> AlertResult:
    config_path = Path(config_path or CONFIG_PATH).expanduser()
    state_path = Path(state_path or STATE_PATH).expanduser()
    if not config_path.exists():
        return AlertResult(False, 0, ())
    try:
        config = _read_config(config_path)
        topic = (keychain_reader or _read_keychain_secret)(
            KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT
        )
        if not topic:
            raise RuntimeError("ntfy topic is missing from macOS Keychain")
        images = image_paths(Path(images_dir or IMAGES_DIR))
        state = _load_state(state_path)
        if state is None:
            # First run: existing images are history, not news.
            state = {"seen": {}, "last_alerts": {}}
            for location, name in images:
                state["seen"].setdefault(location, set()).add(name)
            _write_state(state_path, state)
            return AlertResult(True, 0, ())
    except Exception as error:
        return AlertResult(True, 0, (f"{type(error).__name__}: {error}",))

    now = now or datetime.now(UTC)
    oldest = now - timedelta(minutes=config["max_age_minutes"])
    cooldown = timedelta(minutes=config["cooldown_minutes"])
    publish = publisher or _publish
    analyse = None
    sent = 0
    failures = []

    pending = []
    skipped = False
    for (location, name), path in images.items():
        if name in state["seen"].get(location, ()):
            continue
        captured = None
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            captured = _capture_time(path)
        if captured is None or captured < oldest:
            state["seen"].setdefault(location, set()).add(name)
            skipped = True
            continue
        pending.append((captured, location, name, path))

    for captured, location, name, path in sorted(pending):
        try:
            if analyse is None:
                analyse = (analyser_factory or _load_analyser)(config["min_confidence"])
            species = analyse(path)
            for kind in sorted(set(species) & config["species"]):
                key = f"{location}|{kind}"
                previous = state["last_alerts"].get(key)
                if previous and captured - datetime.fromisoformat(previous) < cooldown:
                    continue
                alert = Alert(
                    location,
                    kind,
                    species.count(kind),
                    captured,
                    path if config["attach_image"] else None,
                )
                publish(config["server"], topic, alert)
                state["last_alerts"][key] = captured.isoformat()
                sent += 1
            state["seen"].setdefault(location, set()).add(name)
        except Exception as error:
            failures.append(f"{location}/{name}: {type(error).__name__}: {error}")
        finally:
            _write_state(state_path, state)

    if skipped and not pending:
        _write_state(state_path, state)
    return AlertResult(True, sent, tuple(failures))


def send_daily_summary(
    config_path=None,
    state_path=None,
    csv_path=None,
    *,
    keychain_reader=None,
    publisher=None,
    now=None,
):
    config_path = Path(config_path or CONFIG_PATH).expanduser()
    if not config_path.exists():
        return AlertResult(False, 0, ())
    try:
        config = _read_config(config_path)
        if not config["daily_summary"]:
            return AlertResult(False, 0, ())
        local_now = (now or datetime.now(UTC)).astimezone(LOCAL_TIME)
        scheduled = local_now.replace(hour=7, minute=0, second=0, microsecond=0)
        if local_now < scheduled:
            return AlertResult(True, 0, ())
        day = scheduled.date().isoformat()
        state_path = Path(state_path or SUMMARY_STATE_PATH).expanduser()
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("last_sent", "") >= day:
                return AlertResult(True, 0, ())
        end = scheduled.astimezone(UTC)
        start = end - timedelta(hours=24)
        if csv_path is None:
            try:
                from ..labelling.review_labels import DEFAULT_CSV
            except ImportError:
                from labelling.review_labels import DEFAULT_CSV

            csv_path = DEFAULT_CSV
        counts = defaultdict(Counter)
        seen = set()
        with Path(csv_path).open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if not {"location_id", "image_file", "class"}.issubset(
                reader.fieldnames or ()
            ):
                raise ValueError(
                    "summary CSV needs location_id, image_file, and class columns"
                )
            for row in reader:
                stamp = (
                    row.get("timestamp_exif") or row.get("timestamp") or ""
                ).strip()
                if not stamp:
                    continue
                try:
                    captured = datetime.fromisoformat(stamp)
                except ValueError:
                    continue
                if captured.tzinfo is None:
                    captured = captured.replace(tzinfo=CAMERA_TIME)
                if not start <= captured.astimezone(UTC) < end:
                    continue
                key = (row["location_id"], row["image_file"])
                if key in seen:
                    continue
                seen.add(key)
                manual = (row.get("manual_label") or "").strip()
                species = (manual or row["class"] or "").strip().casefold()
                if species in config["summary_species"]:
                    counts[row["location_id"]][species] += 1
        species_totals = Counter()
        for location_counts in counts.values():
            species_totals.update(location_counts)
        total = sum(species_totals.values())
        local_start = start.astimezone(LOCAL_TIME)
        lines = [
            f"{local_start:%d.%m.%Y %H:%M %Z} - {scheduled:%d.%m.%Y %H:%M %Z}",
            f"Total: {total} {'sighting' if total == 1 else 'sightings'} in the last 24 hours.",
        ]
        for label, species_counts in [
            ("By species", species_totals),
            *sorted(counts.items()),
        ]:
            if not species_counts:
                continue
            details = " ".join(
                f"{count} {species}"
                for species, count in sorted(
                    species_counts.items(), key=lambda item: (-item[1], item[0])
                )
            )
            lines.append(f"{label}: {details}")
        if not total:
            lines.append("No relevant species sightings in this period.")
        topic = (keychain_reader or _read_keychain_secret)(
            KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT
        )
        if not topic:
            raise RuntimeError("ntfy topic is missing from macOS Keychain")
        params = urlencode({"title": "Wildl-ID daily summary", "priority": "default"})
        request = Request(
            f"{config['server']}/{quote(topic, safe='')}?{params}",
            data="\n".join(lines).encode("utf-8"),
            method="PUT",
        )
        if publisher is not None:
            publisher(request)
        else:
            with urlopen(request, timeout=PUBLISH_TIMEOUT_SECONDS):
                pass
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = state_path.with_name(f".{state_path.name}.tmp")
        temporary.write_text(json.dumps({"last_sent": day}) + "\n", encoding="utf-8")
        os.replace(temporary, state_path)
        return AlertResult(True, 1, ())
    except Exception as error:
        return AlertResult(
            True, 0, (f"Daily summary: {type(error).__name__}: {error}",)
        )


def send_push_alerts(
    config_path: Path | None = None,
    state_path: Path | None = None,
    images_dir: Path | None = None,
    *,
    keychain_reader: Callable[[str, str], str] | None = None,
    analyser_factory: Callable[[float], Callable] | None = None,
    publisher: Callable[[str, str, Alert], None] | None = None,
    now: datetime | None = None,
) -> AlertResult:
    summary = send_daily_summary(config_path, keychain_reader=keychain_reader, now=now)
    alerts = _send_species_alerts(
        config_path,
        state_path,
        images_dir,
        keychain_reader=keychain_reader,
        analyser_factory=analyser_factory,
        publisher=publisher,
        now=now,
    )
    return AlertResult(
        summary.enabled or alerts.enabled,
        summary.sent + alerts.sent,
        summary.failures + alerts.failures,
    )
