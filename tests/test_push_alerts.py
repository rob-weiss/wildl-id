import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from importlib import import_module
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

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

    def image(
        self, name, captured, location="Wendeplatte", result=("wild boar",)
    ):
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


if __name__ == "__main__":
    unittest.main()
