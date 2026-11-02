import csv
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

    def test_attachment_snapshot_survives_source_deletion_and_preserves_csv_schema(
        self,
    ):
        sent = datetime(2026, 11, 2, 20, tzinfo=UTC)
        source = self.path.parent.parent / "photo.jpg"
        source.write_bytes(b"exact sent image")
        notification_log.record_notification(
            "Old", "Message", sent - timedelta(minutes=1), self.path
        )
        notification_log.record_notification(
            "New", "Message", sent, self.path, image_path=source
        )
        source.unlink()

        recent = notification_log.recent_notifications(None, self.path)
        cached = notification_log.notification_image_path(
            recent[0]["image_file"], self.path
        )

        self.assertEqual(cached.read_bytes(), b"exact sent image")
        self.assertTrue(recent[0]["attachment_recorded"])
        self.assertEqual(len(recent), 2)
        with self.path.open(newline="", encoding="utf-8") as handle:
            self.assertEqual(
                tuple(csv.DictReader(handle).fieldnames), notification_log.FIELDS
            )
        self.assertIsNone(
            notification_log.notification_image_path("../photo.jpg", self.path)
        )

    def test_failed_image_copy_is_recorded_without_allowing_legacy_guessing(self):
        notification_log.record_notification(
            "Image",
            "Message",
            datetime.now(UTC),
            self.path,
            image_path=self.path.parent / "missing.jpg",
        )

        recent = notification_log.recent_notifications(1, self.path)

        self.assertEqual(len(recent), 1)
        self.assertTrue(recent[0]["attachment_recorded"])
        self.assertIsNone(recent[0]["image_file"])

    def test_text_only_notifications_explicitly_have_no_image(self):
        notification_log.record_notification(
            "Text", "Message", datetime.now(UTC), self.path
        )

        recent = notification_log.recent_notifications(1, self.path)

        self.assertTrue(recent[0]["attachment_recorded"])
        self.assertIsNone(recent[0]["image_file"])

    def test_rolling_window_counts_all_recent_notifications_and_excludes_future(self):
        now = datetime(2026, 11, 2, 20, tzinfo=UTC)
        for index in range(7):
            notification_log.record_notification(
                f"Recent {index}", "Message", now - timedelta(minutes=index), self.path
            )
        notification_log.record_notification(
            "Boundary", "Message", now - timedelta(hours=24), self.path
        )
        notification_log.record_notification(
            "Old", "Message", now - timedelta(hours=24, seconds=1), self.path
        )
        notification_log.record_notification(
            "Future", "Message", now + timedelta(seconds=1), self.path
        )

        recent = notification_log.recent_notifications(
            None, self.path, hours=24, now=now
        )

        self.assertEqual(
            [item["title"] for item in recent],
            [f"Recent {index}" for index in range(7)] + ["Boundary"],
        )

    def test_legacy_species_body_displays_only_capture_timestamp_without_rewriting_log(
        self,
    ):
        sent = datetime(2026, 11, 2, 19, 14, tzinfo=UTC)
        notification_log.record_notification(
            "🐗 Wild boar at Grenzweg",
            "1 🐗 wild boar at 20:10 (02-11-2026)",
            sent,
            self.path,
        )
        original = self.path.read_bytes()

        recent = notification_log.recent_notifications(5, self.path)

        self.assertEqual(recent[0]["title"], "🐗 Wild boar at Grenzweg")
        self.assertEqual(recent[0]["message"], "20:10 (02-11-2026)")
        self.assertEqual(self.path.read_bytes(), original)

    def test_summary_and_unrelated_bodies_are_not_shortened(self):
        sent = datetime(2026, 11, 2, 19, 14, tzinfo=UTC)
        for title, message in (
            ("Wildl-ID daily summary", "Total: 3 sightings\nBy species: 3 wild boar"),
            ("Test notification", "1 wild boar at 20:10 (02-11-2026)"),
            ("🐗 Wild boar at Grenzweg", "20:10 (02-11-2026)"),
        ):
            with self.subTest(title=title):
                notification_log.record_notification(title, message, sent, self.path)
                self.assertEqual(
                    notification_log.recent_notifications(1, self.path)[0]["message"],
                    message,
                )

    def test_notification_status_is_saved_and_read(self):
        status_path = self.path.with_name("push_alerts_status.json")
        checked = datetime(2026, 11, 1, 12, 30, tzinfo=UTC)

        notification_log.record_status(
            True,
            2,
            ("RuntimeError: request failed for https://ntfy.sh/private-topic",),
            checked,
            status_path,
        )

        self.assertEqual(
            notification_log.latest_status(status_path),
            {
                "checked": checked,
                "enabled": True,
                "sent": 2,
                "failures": ["RuntimeError"],
            },
        )

    def test_unwritable_log_does_not_raise(self):
        self.path.parent.parent.joinpath("logs").write_text("not a directory")

        notification_log.record_notification(
            "Title", "Message", datetime.now(UTC), self.path
        )

    def test_unwritable_status_does_not_raise(self):
        blocked_path = self.path.parent.parent / "blocked"
        blocked_path.write_text("not a directory", encoding="utf-8")

        notification_log.record_status(
            True, 1, (), datetime.now(UTC), blocked_path / "status.json"
        )


if __name__ == "__main__":
    unittest.main()
