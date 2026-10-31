import csv
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from importlib import import_module
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from unittest.mock import patch
from zoneinfo import ZoneInfo

from PIL import Image

alerts = import_module("src.alerting.push_alerts")

CET = timezone(timedelta(hours=1))
NOW = datetime(2026, 10, 30, 21, 0, tzinfo=CET)


class PushAlertTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.images = root / "images"
        self.config = root / "push-alerts.json"
        self.state = root / "state.json"
        self.config.write_text("{}", encoding="utf-8")
        self.analysed = []
        self.results = {}
        self.published = []
        self.factory_calls = 0

    def image(self, name, captured, location="Wendeplatte", result=("wild boar",)):
        path = self.images / "2026" / location / name
        path.parent.mkdir(parents=True, exist_ok=True)
        exif = Image.Exif()
        exif[0x0132] = captured.strftime("%Y:%m:%d %H:%M:%S")
        Image.new("RGB", (8, 8)).save(path, exif=exif)
        self.results[path] = result
        return path

    def analyser_factory(self, min_confidence):
        self.factory_calls += 1

        def analyse(path):
            self.analysed.append(path.name)
            return self.results[path]

        return analyse

    def run_alerts(self, publisher=None, keychain=lambda service, account: "topic-x"):
        return alerts.send_push_alerts(
            self.config,
            self.state,
            self.images,
            keychain_reader=keychain,
            analyser_factory=self.analyser_factory,
            publisher=publisher or (lambda *args: self.published.append(args)),
            now=NOW,
        )

    def test_missing_config_disables_alerts(self):
        self.config.unlink()

        result = self.run_alerts()

        self.assertFalse(result.enabled)
        self.assertFalse(self.state.exists())

    def test_first_run_marks_existing_images_seen_without_alerts(self):
        self.image("image_0000001.jpg", NOW)

        result = self.run_alerts()
        second = self.run_alerts()

        self.assertEqual((result.sent, second.sent), (0, 0))
        self.assertEqual(self.published, [])
        self.assertEqual(self.factory_calls, 0)

    def test_new_target_species_is_published_once(self):
        self.run_alerts()
        self.image("image_0000002.jpg", NOW - timedelta(minutes=5))
        self.image(
            "image_0000003.jpg",
            NOW - timedelta(minutes=4),
            location="Douglasien",
            result=["raccoon", "raccoon", "fox"],
        )
        self.image("image_0000004.jpg", NOW, result=["roe deer"])

        result = self.run_alerts()
        again = self.run_alerts()

        self.assertEqual((result.sent, result.failures), (2, ()))
        self.assertEqual(again.sent, 0)
        sent = [
            (server, topic, alert.location, alert.species, alert.count)
            for server, topic, alert in self.published
        ]
        self.assertEqual(
            sent,
            [
                ("https://ntfy.sh", "topic-x", "Wendeplatte", "wild boar", 1),
                ("https://ntfy.sh", "topic-x", "Douglasien", "raccoon", 2),
            ],
        )
        self.assertEqual(self.factory_calls, 1)

    def test_cooldown_suppresses_repeat_alerts_per_site_and_species(self):
        self.run_alerts()
        self.image("image_0000002.jpg", NOW - timedelta(minutes=40))
        self.image("image_0000003.jpg", NOW - timedelta(minutes=20))
        self.image("image_0000004.jpg", NOW)

        result = self.run_alerts()

        self.assertEqual(result.sent, 2)
        self.assertEqual(
            [alert.captured for _, _, alert in self.published],
            [NOW - timedelta(minutes=40), NOW],
        )

    def test_old_images_are_skipped_without_analysis(self):
        self.run_alerts()
        self.image("image_0000002.jpg", NOW - timedelta(hours=4))

        result = self.run_alerts()

        self.assertEqual(result.sent, 0)
        self.assertEqual(self.analysed, [])

    def test_image_is_attached_by_default(self):
        self.run_alerts()
        path = self.image("image_0000002.jpg", NOW)

        self.run_alerts()

        self.assertEqual(self.published[0][2].image_path, path)

    def test_failed_publish_is_retried_next_run(self):
        self.run_alerts()
        self.image("image_0000002.jpg", NOW)

        def failing(*args):
            raise OSError("offline")

        failed = self.run_alerts(publisher=failing)
        retried = self.run_alerts()

        self.assertEqual(len(failed.failures), 1)
        self.assertIn("offline", failed.failures[0])
        self.assertEqual(retried.sent, 1)

    def test_missing_topic_is_reported(self):
        result = self.run_alerts(keychain=lambda service, account: "")

        self.assertEqual(len(result.failures), 1)
        self.assertIn("Keychain", result.failures[0])

    def test_config_rejects_plain_http_server(self):
        self.config.write_text(
            json.dumps({"server": "http://ntfy.example"}), encoding="utf-8"
        )

        result = self.run_alerts()

        self.assertIn("https", result.failures[0])

    def test_ntfy_request_attaches_image_with_utf8_query(self):
        path = self.image("image_0000002.jpg", NOW, location="Madenhäule")
        alert = alerts.Alert("Madenhäule", "wild boar", 3, NOW, path)

        request = alerts.ntfy_request("https://ntfy.sh", "topic x", alert)

        url = urlsplit(request.full_url)
        query = parse_qs(url.query)
        self.assertEqual(request.get_method(), "PUT")
        self.assertEqual(url.path, "/topic%20x")
        self.assertEqual(query["title"], ["Wild boar at Madenhäule"])
        self.assertEqual(query["message"], ["3 wild boar at 21:00 (30.10.)"])
        self.assertEqual(query["tags"], ["boar"])
        self.assertEqual(query["filename"], ["image_0000002.jpg"])
        self.assertEqual(request.data, path.read_bytes())

    def test_ntfy_request_without_image_sends_message_body(self):
        alert = alerts.Alert("Wendeplatte", "raccoon", 1, NOW, None)

        request = alerts.ntfy_request("https://ntfy.sh", "topic", alert)

        self.assertEqual(request.data, b"1 raccoon at 21:00 (30.10.)")
        self.assertNotIn("filename", parse_qs(urlsplit(request.full_url).query))


class DailySummaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.config = root / "config.json"
        self.config.write_text('{"daily_summary": true}', encoding="utf-8")
        self.state = root / "summary-state.json"
        self.labels = root / "labels.csv"
        self.now = datetime(2026, 10, 31, 7, tzinfo=ZoneInfo("Europe/Berlin"))
        self.published = []
        self.write_rows([])

    def write_rows(self, rows):
        with self.labels.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "location_id", "image_file", "timestamp_exif", "timestamp",
                    "class", "manual_label",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    def row(self, name, stamp, species="fox", manual="", location="Wendeplatte"):
        return dict(
            location_id=location, image_file=name, timestamp_exif=stamp,
            timestamp="", **{"class": species}, manual_label=manual,
        )

    def run_summary(self, **kwargs):
        return alerts.send_daily_summary(
            self.config, self.state, self.labels,
            now=kwargs.pop("now", self.now),
            keychain_reader=lambda *args: "topic-x",
            publisher=kwargs.pop("publisher", lambda request: self.published.append(request)),
            **kwargs,
        )

    def test_digest_counts_unique_photos_with_manual_labels_and_exact_window(self):
        row = self.row("one.jpg", "2026-10-30 07:00:00", manual="roe deer")
        self.write_rows([
            row, row,
            self.row("two.jpg", "2026-10-31 06:59:59", location="Douglasien"),
            self.row("old.jpg", "2026-10-30 06:59:59"),
            self.row("future.jpg", "2026-10-31 07:00:00"),
            self.row("empty.jpg", "2026-10-31 06:00:00", manual="none"),
            self.row("human.jpg", "2026-10-31 06:00:00", species="human"),
            self.row("invalid.jpg", "not a timestamp"),
            self.row("undated.jpg", ""),
        ])

        result = self.run_summary()
        again = self.run_summary()

        self.assertEqual((result.sent, result.failures, again.sent), (1, (), 0))
        request = self.published[0]
        self.assertIn("Wendeplatte: 1 sighting", request.data.decode())
        self.assertIn("Douglasien: 1 sighting", request.data.decode())
        self.assertIn("Total: 2 sightings", request.data.decode())
        self.assertEqual(urlsplit(request.full_url).path, "/topic-x")
        self.assertEqual(parse_qs(urlsplit(request.full_url).query)["priority"], ["default"])

    def test_not_due_before_seven_berlin_even_on_utc_host(self):
        result = self.run_summary(now=datetime(2026, 10, 31, 5, 59, tzinfo=timezone.utc))
        self.assertEqual(result.sent, 0)
        self.assertFalse(self.state.exists())

    def test_only_landing_page_species_contribute_to_total_and_location_counts(self):
        relevant = ["roe deer", "wild boar", "badger", "marten", "fox", "hare", "raccoon"]
        rows = [
            self.row(f"relevant-{index}.jpg", "2026-10-31 06:00:00", species=species)
            for index, species in enumerate(relevant)
        ]
        rows.extend([
            self.row("fox.jpg", "2026-10-31 06:00:00", location="Douglasien"),
            self.row("manual.jpg", "2026-10-31 06:00:00", species="woodpecker", manual="hare", location="Douglasien"),
            self.row("excluded.jpg", "2026-10-31 06:00:00", species="fox", manual="squirrel", location="Bird camera"),
            self.row("bird.jpg", "2026-10-31 06:00:00", species="woodpecker", location="Bird camera"),
            self.row("unknown.jpg", "2026-10-31 06:00:00", species="unknown"),
        ])
        self.write_rows(rows)

        result = self.run_summary()

        self.assertEqual((result.sent, result.failures), (1, ()))
        body = self.published[0].data.decode()
        self.assertIn("Total: 9 sightings", body)
        self.assertIn("Wendeplatte: 7 sightings", body)
        self.assertIn("Douglasien: 2 sightings", body)
        self.assertNotIn("Bird camera", body)
        self.assertNotIn("woodpecker", body)
        self.assertNotIn("roe deer", body)

    def test_summer_seven_is_five_utc(self):
        result = self.run_summary(now=datetime(2026, 7, 1, 5, tzinfo=timezone.utc))
        self.assertEqual(result.sent, 1)

    def test_dst_transition_still_uses_twenty_four_elapsed_hours(self):
        self.now = datetime(2026, 10, 25, 7, tzinfo=ZoneInfo("Europe/Berlin"))
        self.write_rows([
            self.row("old.jpg", "2026-10-24 06:59:59"),
            self.row("included.jpg", "2026-10-24 07:00:00"),
        ])
        self.run_summary()
        self.assertIn("Total: 1 sighting", self.published[0].data.decode())

    def test_wake_after_seven_uses_scheduled_window(self):
        self.write_rows([
            self.row("early.jpg", "2026-10-30 07:01:00"),
            self.row("late.jpg", "2026-10-31 08:00:00"),
        ])
        self.run_summary(now=self.now + timedelta(hours=3))
        self.assertIn("Total: 1 sighting", self.published[0].data.decode())

    def test_failed_publish_retries_without_marking_day_sent(self):
        def failing(request):
            raise OSError("offline")

        failed = self.run_summary(publisher=failing)
        self.assertIn("offline", failed.failures[0])
        self.assertFalse(self.state.exists())
        self.assertEqual(self.run_summary().sent, 1)

    def test_missing_labels_reports_failure_instead_of_empty_summary(self):
        self.labels.unlink()
        result = self.run_summary()
        self.assertTrue(result.failures)
        self.assertFalse(self.state.exists())

    def test_disabled_summary_does_not_read_labels_or_publish(self):
        self.config.write_text("{}", encoding="utf-8")
        self.labels.unlink()
        result = self.run_summary()
        self.assertFalse(result.enabled)
        self.assertEqual(self.published, [])

    def test_empty_day_is_reported_and_next_day_sends_again(self):
        self.run_summary()
        self.run_summary(now=self.now + timedelta(days=1))
        self.assertEqual(len(self.published), 2)
        self.assertIn("Total: 0 sightings", self.published[0].data.decode())
        self.assertIn("No relevant species sightings", self.published[0].data.decode())

    def test_existing_notification_entry_point_runs_digest(self):
        expected = alerts.AlertResult(True, 1, ())
        with (
            patch.object(alerts, "send_daily_summary", return_value=expected) as daily,
            patch.object(alerts, "_send_species_alerts", return_value=alerts.AlertResult(True, 0, ())),
        ):
            result = alerts.send_push_alerts(config_path=self.config, now=self.now)
        self.assertEqual(result.sent, 1)
        daily.assert_called_once_with(self.config, keychain_reader=None, now=self.now)


if __name__ == "__main__":
    unittest.main()
