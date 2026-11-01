import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from importlib import import_module
from pathlib import Path

notification_log = import_module("src.alerting.notification_log")


class NotificationLogTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "logs" / "push_alerts.csv"

    def test_recent_notifications_are_newest_first_and_limited(self):
        start = datetime(2026, 11, 1, 12, tzinfo=UTC)
        for minute in range(7):
            notification_log.record_notification(
                f"Title {minute}",
                f"Message, with comma\nline {minute}",
                start + timedelta(minutes=minute),
                self.path,
            )

        recent = notification_log.recent_notifications(5, self.path)

        self.assertEqual(
            [item["title"] for item in recent],
            ["Title 6", "Title 5", "Title 4", "Title 3", "Title 2"],
        )
        self.assertEqual(recent[0]["message"], "Message, with comma\nline 6")
        self.assertEqual(recent[0]["sent"], start + timedelta(minutes=6))

    def test_missing_log_has_no_notifications(self):
        self.assertEqual(notification_log.recent_notifications(5, self.path), [])

    def test_unwritable_log_does_not_raise(self):
        self.path.parent.parent.joinpath("logs").write_text("not a directory")

        notification_log.record_notification(
            "Title", "Message", datetime.now(UTC), self.path
        )


if __name__ == "__main__":
    unittest.main()
