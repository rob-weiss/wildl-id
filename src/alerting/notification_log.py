"""History of sent ntfy notifications, shown on the web app's landing page."""

import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "push_alerts.csv"
FIELDS = ("sent", "title", "message")


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


def record_notification(title, message, sent, path=None):
    path = Path(path or LOG_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as target:
            writer = csv.writer(target, lineterminator="\n")
            if is_new:
                writer.writerow(FIELDS)
            writer.writerow([sent.isoformat(), title, message])
    except OSError:
        # Display-only history: a failed write must not make alerts resend.
        pass


def recent_notifications(limit=5, path=None):
    """Return the newest notifications first."""
    path = Path(path or LOG_PATH)
    try:
        with path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
    except FileNotFoundError:
        return []
    notifications = []
    for row in reversed(rows):
        try:
            sent = datetime.fromisoformat(row["sent"])
        except KeyError, TypeError, ValueError:
            continue
        notifications.append(
            {
                "sent": sent,
                "title": row["title"],
                "message": _display_message(row["title"], row["message"]),
            }
        )
        if len(notifications) == limit:
            break
    return notifications


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
