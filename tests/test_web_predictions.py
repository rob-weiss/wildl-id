import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from web import predictions
from web.gallery import RECENT_SPECIES


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.csv_path = self.root / "labels.csv"
        self.model_path = self.root / "metadata.json"
        self.now = pd.Timestamp("2026-10-31T06:20Z")
        self.species = sorted(RECENT_SPECIES | {"crow", "pigeon"})
        self.site = {
            "location_id": "camera",
            "latitude": 48.63969,
            "longitude": 9.01149,
            "local_timezone": "Europe/Berlin",
            "timestamp_timezone": "UTC+01:00",
            "ingestion_delay_hours": 1,
        }
        self.bundle = {
            "schema": 1,
            "kind": "poisson",
            "include_weather": False,
            "include_moon": False,
            "species": self.species,
            "sites": ["camera"],
            "supported": [True] * 9,
            "columns": [
                "horizon",
                "hour_sin",
                "hour_cos",
                "year_sin",
                "year_cos",
                "solar_elevation",
                "daylight_hours",
            ],
            "models": [
                {
                    "mean": [0.0] * 10,
                    "scale": [1.0] * 10,
                    "coef": [0.0] * 11,
                    "intercept": math.log(2),
                }
                for _ in self.species
            ],
            "references": {"camera": [4.0] * 9},
            "training_cutoff": "2026-08-01T00:00:00+00:00",
            "config": {"sites": [self.site]},
            "experiment_status": "exploratory_assumed_coverage",
        }
        self.model_path.write_text(json.dumps(self.bundle))
        self.write_rows()

    def write_rows(self, stale=False):
        self.rows = [
            {
                "location_id": "camera",
                "image_file": "start.jpg",
                "class": "fox",
                "timestamp": "2026-10-22T00:00Z",
            },
            {
                "location_id": "camera",
                "image_file": "last.jpg",
                "class": "human",
                "timestamp": "2026-10-29T06:10Z" if stale else "2026-10-31T06:10Z",
            },
        ]
        pd.DataFrame(self.rows).to_csv(self.csv_path, index=False)

    def test_next_24_hours_matches_home_species_and_numeric_model(self):
        result = predictions.forecast_next24(
            self.csv_path, self.model_path, "camera", now=self.now
        )
        self.assertEqual(set(result.frame.species), RECENT_SPECIES)
        self.assertEqual(len(result.frame), 24 * 7)
        self.assertEqual(result.origin, pd.Timestamp("2026-10-31T07:00Z"))
        self.assertEqual(
            result.frame.valid_time.max(), result.origin + pd.Timedelta(hours=23)
        )
        np.testing.assert_allclose(result.frame.expected_count, 2.0)
        np.testing.assert_allclose(result.frame.activity_index, 0.5)
        self.assertFalse(result.assumed_gap)

    def test_stale_context_requires_explicit_assumption(self):
        self.write_rows(stale=True)
        with self.assertRaisesRegex(predictions.ForecastUnavailable, "coverage"):
            predictions.forecast_next24(
                self.csv_path, self.model_path, "camera", now=self.now
            )
        result = predictions.forecast_next24(
            self.csv_path,
            self.model_path,
            "camera",
            assume_continuous=True,
            now=self.now,
        )
        self.assertTrue(result.assumed_gap)
        self.assertEqual(len(result.frame), 168)

    def test_manual_labels_deduplication_and_future_records(self):
        index = self.species.index("roe deer")
        self.bundle["models"][index]["coef"][8] = 1.0
        self.model_path.write_text(json.dumps(self.bundle))
        row = {
            "location_id": "camera",
            "image_file": "deer.jpg",
            "class": "human",
            "manual_label": "roe deer",
            "timestamp": "2026-10-31T05:30Z",
        }
        self.rows += [
            row,
            row.copy(),
            dict(row, image_file="future.jpg", timestamp="2026-11-02T00:00Z"),
        ]
        pd.DataFrame(self.rows).to_csv(self.csv_path, index=False)
        result = predictions.forecast_next24(
            self.csv_path, self.model_path, "camera", now=self.now
        )
        np.testing.assert_allclose(
            result.frame[result.frame.species == "roe deer"].expected_count,
            2 * math.exp(1 / 24),
        )
        self.assertEqual(result.latest_record, pd.Timestamp("2026-10-31T06:10Z"))

    def test_invalid_missing_models_and_references(self):
        with self.assertRaises(predictions.ForecastUnavailable):
            predictions.forecast_next24(
                self.csv_path, self.root / "missing.json", "camera", now=self.now
            )
        self.bundle["models"][0]["coef"] = []
        self.model_path.write_text(json.dumps(self.bundle))
        with self.assertRaises(predictions.ForecastUnavailable):
            predictions.forecast_next24(
                self.csv_path, self.model_path, "camera", now=self.now
            )

    def test_dst_still_has_24_distinct_elapsed_hours(self):
        self.now = pd.Timestamp("2026-10-25T00:20Z")
        self.rows[-1]["timestamp"] = "2026-10-25T00:10Z"
        self.rows[0]["timestamp"] = "2026-10-10T00:00Z"
        pd.DataFrame(self.rows).to_csv(self.csv_path, index=False)
        result = predictions.forecast_next24(
            self.csv_path, self.model_path, "camera", now=self.now
        )
        self.assertEqual(result.frame.valid_time.nunique(), 24)

    def test_container_includes_prediction_module(self):
        dockerfile = Path(__file__).resolve().parents[1] / "web" / "Dockerfile"
        self.assertIn("web/predictions.py", dockerfile.read_text(encoding="utf-8"))

    def test_count_chart_explicitly_clears_relative_axis_limit(self):
        options = predictions.chart_options("fox", relative=False)
        self.assertIn("max", options["yAxis"])
        self.assertIsNone(options["yAxis"]["max"])

    def test_record_in_last_complete_hour_is_fresh_enough(self):
        self.rows[-1]["timestamp"] = "2026-10-31T05:55Z"
        pd.DataFrame(self.rows).to_csv(self.csv_path, index=False)
        result = predictions.forecast_next24(
            self.csv_path, self.model_path, "camera", now=self.now
        )
        self.assertFalse(result.assumed_gap)
