"""History of sent ntfy notifications, shown on the web app's landing page."""

import csv
from datetime import datetime
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "push_alerts.csv"
FIELDS = ("sent", "title", "message")


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
        except (KeyError, TypeError, ValueError):
            continue
        notifications.append(
            {"sent": sent, "title": row["title"], "message": row["message"]}
        )
        if len(notifications) == limit:
            break
    return notifications
