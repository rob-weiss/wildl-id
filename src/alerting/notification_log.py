"""History of sent ntfy notifications, shown on the web app's landing page."""

import csv
import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "push_alerts.csv"
FIELDS = ("sent", "title", "message")


def _notification_key(title, message, sent):
    return hashlib.sha256(
        json.dumps([title, message, sent], ensure_ascii=True).encode()
    ).hexdigest()


def _write_attachment_file(path, content):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as target:
            temporary = Path(target.name)
            target.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _store_attachment(path, title, message, sent, image_path):
    directory = path.parent / "notification_images"
    directory.mkdir(parents=True, exist_ok=True)
    key = _notification_key(title, message, sent)
    filename = None
    metadata = directory / f"{key}.json"
    _write_attachment_file(metadata, json.dumps({"image_file": None}).encode())
    if image_path is not None:
        source = Path(image_path)
        if source.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            filename = key + source.suffix.lower()
            _write_attachment_file(directory / filename, source.read_bytes())
            _write_attachment_file(
                metadata, json.dumps({"image_file": filename}).encode()
            )


def notification_image_path(filename, path=None):
    if (
        not isinstance(filename, str)
        or re.fullmatch(r"[a-f0-9]{64}\.(?:jpg|jpeg|png)", filename) is None
    ):
        return None
    candidate = Path(path or LOG_PATH).parent / "notification_images" / filename
    return candidate if candidate.is_file() and not candidate.is_symlink() else None


def _attachment_details(row, path):
    key = _notification_key(row["title"], row["message"], row["sent"])
    metadata = path.parent / "notification_images" / f"{key}.json"
    try:
        filename = json.loads(metadata.read_text(encoding="utf-8"))["image_file"]
    except OSError, ValueError, KeyError, TypeError:
        return {"image_file": None, "attachment_recorded": metadata.exists()}
    return {
        "image_file": filename
        if notification_image_path(filename, path) is not None
        else None,
        "attachment_recorded": True,
    }


def _display_message(title, message):
    match = re.fullmatch(
        r"\d+ (?P<species>.+) at (?P<timestamp>\d{2}:\d{2} \(\d{2}-\d{2}-\d{4}\))",
        message,
    )
    if match and title.casefold().startswith(f"{match['species']} at ".casefold()):
        try:
            datetime.strptime(match["timestamp"], "%H:%M (%d-%m-%Y)")
        except ValueError:
            return message
        return match["timestamp"]
    return message


def record_notification(title, message, sent, path=None, *, image_path=None):
    path = Path(path or LOG_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _store_attachment(path, title, message, sent.isoformat(), image_path)
        except OSError:
            pass
        is_new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as target:
            writer = csv.writer(target, lineterminator="\n")
            if is_new:
                writer.writerow(FIELDS)
            writer.writerow([sent.isoformat(), title, message])
    except OSError:
        # Display-only history: a failed write must not make alerts resend.
        pass


def recent_notifications(limit=5, path=None, *, hours=None, now=None):
    """Return the newest notifications first."""
    path = Path(path or LOG_PATH)
    try:
        with path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
    except FileNotFoundError:
        return []
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=ZoneInfo("Europe/Berlin"))
    now = now.astimezone(UTC)
    cutoff = now - timedelta(hours=hours) if hours is not None else None
    notifications = []
    for row in reversed(rows):
        try:
            sent = datetime.fromisoformat(row["sent"])
        except KeyError, TypeError, ValueError:
            continue
        if sent.tzinfo is None:
            sent = sent.replace(tzinfo=ZoneInfo("Europe/Berlin"))
        if cutoff is not None and not cutoff <= sent.astimezone(UTC) <= now:
            continue
        notifications.append(
            {
                "sent": sent,
                "title": row["title"],
                "message": _display_message(row["title"], row["message"]),
                **_attachment_details(row, path),
            }
        )
    notifications.sort(key=lambda item: item["sent"].astimezone(UTC), reverse=True)
    return notifications[:limit] if limit is not None else notifications


def record_status(enabled, sent, failures, checked, path=None):
    path = Path(path or Path(LOG_PATH).with_name("push_alerts_status.json"))
    status = {
        "checked": checked.isoformat(),
        "enabled": enabled,
        "sent": sent,
        "failures": [_status_failure(failure) for failure in failures],
    }
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(status, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except OSError:
        pass
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def latest_status(path=None):
    path = Path(path or Path(LOG_PATH).with_name("push_alerts_status.json"))
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
        checked = datetime.fromisoformat(status["checked"])
        enabled = status["enabled"]
        sent = status["sent"]
        failures = status["failures"]
    except OSError, KeyError, TypeError, ValueError:
        return None
    if (
        not isinstance(enabled, bool)
        or isinstance(sent, bool)
        or not isinstance(sent, int)
        or sent < 0
        or not isinstance(failures, list)
        or not all(isinstance(failure, str) for failure in failures)
    ):
        return None
    return {
        "checked": checked,
        "enabled": enabled,
        "sent": sent,
        "failures": failures,
    }


def _status_failure(failure):
    if not isinstance(failure, str):
        return "Alert failed"
    if "ntfy topic is missing from macOS Keychain" in failure:
        return "Keychain topic missing"
    match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*Error)\b", failure)
    return match.group(1) if match else "Alert failed"
