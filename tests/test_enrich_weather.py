import io
import json
import tempfile
import unittest
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from src.enrich_weather import enrich_temperatures, fetch_hourly_temperature
from src.labelling import label_data


class FetchHourlyTemperatureTests(unittest.TestCase):
    def test_requests_and_parses_utc_hourly_temperatures(self):
        response = {
            "hourly": {
                "time": ["2025-01-01T00:00", "2025-01-01T01:00"],
                "temperature_2m": [8.5, None],
            }
        }
        with patch(
            "src.enrich_weather.urlopen",
            return_value=io.BytesIO(json.dumps(response).encode()),
        ) as urlopen:
            temperatures = fetch_hourly_temperature(
                48.65, 8.98, date(2025, 1, 1), date(2025, 1, 2)
            )

        request = urlopen.call_args.args[0]
        query = parse_qs(urlsplit(request.full_url).query)
        self.assertEqual(
            urlsplit(request.full_url).netloc,
            "historical-forecast-api.open-meteo.com",
        )
        self.assertEqual(query["hourly"], ["temperature_2m"])
        self.assertEqual(query["timezone"], ["UTC"])
        self.assertEqual(query["latitude"], ["48.65"])
        self.assertEqual(query["longitude"], ["8.98"])
        self.assertEqual(query["start_date"], ["2025-01-01"])
        self.assertEqual(query["end_date"], ["2025-01-02"])
        self.assertEqual(
            temperatures,
            {datetime(2025, 1, 1, tzinfo=UTC): 8.5},
        )

    def test_rejects_mismatched_hourly_arrays(self):
        response = {
            "hourly": {
                "time": ["2025-01-01T00:00"],
                "temperature_2m": [8.5, 9.0],
            }
        }
        with (
            patch(
                "src.enrich_weather.urlopen",
                return_value=io.BytesIO(json.dumps(response).encode()),
            ),
            self.assertRaisesRegex(ValueError, "matching lengths"),
        ):
            fetch_hourly_temperature(
                48.65, 8.98, date(2025, 1, 1), date(2025, 1, 1)
            )

    def test_rejects_invalid_non_null_temperature(self):
        response = {
            "hourly": {
                "time": ["2025-01-01T00:00"],
                "temperature_2m": ["not a temperature"],
            }
        }
        with (
            patch(
                "src.enrich_weather.urlopen",
                return_value=io.BytesIO(json.dumps(response).encode()),
            ),
            self.assertRaisesRegex(ValueError, "temperature"),
        ):
            fetch_hourly_temperature(
                48.65, 8.98, date(2025, 1, 1), date(2025, 1, 1)
            )


class EnrichTemperaturesTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.repo_dir = Path(temporary_directory.name)
        labels_dir = self.repo_dir / "labels"
        self.metadata_path = labels_dir / "image_metadata.csv"
        self.predictions_path = labels_dir / "labels_model" / "predictions.csv"
        (self.repo_dir / "config").mkdir()
        (self.repo_dir / "config/camera_locations.json").write_text(
            json.dumps(
                {
                    "cameras": [
                        {
                            "location_id": "camera1",
                            "latitude": 48.65,
                            "longitude": 8.98,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        self.metadata_rows = [
            self.metadata_row("camera1", "utc.jpg", "2025-01-01T00:45:00+00:00"),
            self.metadata_row("camera1", "local.jpg", "2025-01-01T01:15:00"),
            self.metadata_row(
                "camera1", "late-hour.jpg", "2025-01-01T23:30:00+00:00"
            ),
            self.metadata_row(
                "camera1", "missing-hour.jpg", "2025-01-01T02:15:00+00:00"
            ),
            self.metadata_row(
                "unknown-camera", "unknown.jpg", "2025-01-01T00:15:00+00:00"
            ),
            self.metadata_row("camera1", "invalid-time.jpg", "not a timestamp"),
            self.metadata_row("camera1", "missing-time.jpg", ""),
            self.metadata_row("camera1", "dst-fold.jpg", "2025-10-26T02:30:00"),
        ]
        label_data.write_records(
            self.metadata_path,
            self.metadata_rows,
            (*label_data.METADATA_COLUMNS, "source"),
        )
        self.prediction_rows = [
            self.prediction_row(self.metadata_rows[0], reviewed="true"),
            self.prediction_row(self.metadata_rows[2], temperature_celsius="42"),
            self.prediction_row(self.metadata_rows[3], temperature_celsius="31"),
            {
                "location_id": "camera1",
                "image_file": "not-in-metadata.jpg",
                "class": "fox",
                "temperature_celsius": "5",
                "reviewed": "",
                "annotation": "keep",
            },
        ]
        label_data.write_records(
            self.predictions_path,
            self.prediction_rows,
            (*label_data.PREDICTION_COLUMNS, "annotation"),
        )
        self.original_metadata = label_data.read_records(self.metadata_path)
        self.original_predictions = label_data.read_records(self.predictions_path)

    @staticmethod
    def metadata_row(location_id, image_file, timestamp_exif):
        return {
            "location_id": location_id,
            "image_file": image_file,
            "timestamp_exif": timestamp_exif,
            "lighting": "dark",
            "temperature_celsius": "old value",
            "source": "preserve",
        }

    @staticmethod
    def prediction_row(metadata, temperature_celsius="old value", reviewed=""):
        return {
            "location_id": metadata["location_id"],
            "image_file": metadata["image_file"],
            "class": "fox",
            "timestamp_exif": metadata["timestamp_exif"],
            "lighting": "bright",
            "temperature_celsius": temperature_celsius,
            "reviewed": reviewed,
            "annotation": "preserve",
        }

    def fake_fetch(self, latitude, longitude, start_date, end_date):
        self.assertEqual((latitude, longitude), (48.65, 8.98))
        self.assertEqual(
            (start_date, end_date), (date(2025, 1, 1), date(2025, 1, 2))
        )
        return {
            datetime(2025, 1, 1, tzinfo=UTC): 8.5,
            datetime(2025, 1, 1, 1, tzinfo=UTC): 10.5,
            datetime(2025, 1, 1, 23, tzinfo=UTC): 8.0,
            datetime(2025, 1, 2, tzinfo=UTC): 10.0,
        }

    @staticmethod
    def non_temperature_fields(rows):
        return [
            {
                key: value
                for key, value in row.items()
                if key != "temperature_celsius"
            }
            for row in rows
        ]

    def test_updates_by_image_identity_and_preserves_other_fields(self):
        with patch(
            "src.enrich_weather.fetch_hourly_temperature",
            side_effect=self.fake_fetch,
        ) as fetch:
            updated_count, unresolved_count = enrich_temperatures(self.repo_dir)

        self.assertEqual((updated_count, unresolved_count), (3, 5))
        fetch.assert_called_once()
        metadata = label_data.read_records(self.metadata_path)
        self.assertEqual(
            [row["temperature_celsius"] for row in metadata],
            ["10.0", "9.0", "9.0", "", "", "", "", ""],
        )
        self.assertEqual(
            self.non_temperature_fields(metadata),
            self.non_temperature_fields(self.original_metadata),
        )
        self.assertTrue(all(row["source"] == "preserve" for row in metadata))
        predictions = label_data.read_records(self.predictions_path)
        self.assertEqual(
            [row["temperature_celsius"] for row in predictions],
            ["10.0", "9.0", "", "5"],
        )
        self.assertEqual(
            self.non_temperature_fields(predictions),
            self.non_temperature_fields(self.original_predictions),
        )
        self.assertEqual(predictions[0]["reviewed"], "true")
        self.assertEqual(predictions[0]["class"], "fox")
        self.assertEqual(predictions[0]["annotation"], "preserve")
        self.assertEqual(predictions[-1]["annotation"], "keep")

    def test_dry_run_does_not_write_any_csv(self):
        originals = {
            path: path.read_bytes()
            for path in (self.metadata_path, self.predictions_path)
        }
        with patch(
            "src.enrich_weather.fetch_hourly_temperature",
            side_effect=self.fake_fetch,
        ):
            counts = enrich_temperatures(self.repo_dir, dry_run=True)

        self.assertEqual(counts, (3, 5))
        self.assertEqual(
            {path: path.read_bytes() for path in originals},
            originals,
        )

    def test_api_failure_leaves_all_csv_bytes_unchanged(self):
        originals = {
            path: path.read_bytes()
            for path in (self.metadata_path, self.predictions_path)
        }
        with (
            patch(
                "src.enrich_weather.fetch_hourly_temperature",
                side_effect=RuntimeError("API unavailable"),
            ),
            self.assertRaisesRegex(RuntimeError, "API unavailable"),
        ):
            enrich_temperatures(self.repo_dir)

        self.assertEqual(
            {path: path.read_bytes() for path in originals},
            originals,
        )


if __name__ == "__main__":
    unittest.main()
