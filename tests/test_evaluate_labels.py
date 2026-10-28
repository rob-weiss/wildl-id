import csv
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import matplotlib.dates as mdates
from src.visualisation import evaluate_labels as evaluate_module
from src.visualisation.evaluate_labels import (
    filter_baiting_classes,
    filter_visualization_classes,
)


class VisualizationClassFilterTests(unittest.TestCase):
    def test_keeps_only_requested_wildlife_classes(self):
        classes = [
            "roe deer",
            "wild boar",
            "pigeon",
            "badger",
            "marten",
            "fox",
            "hare",
            "raccoon",
            "crow",
            "horse",
            "none",
            "unknown",
        ]
        dataframe = pd.DataFrame({"class": classes})

        filtered = filter_visualization_classes(dataframe)

        self.assertEqual(filtered["class"].tolist(), classes[:9])

    def test_uses_manual_labels_and_falls_back_to_class_when_blank(self):
        dataframe = pd.DataFrame(
            {
                "class": ["horse", "roe deer", "fox", "crow"],
                "manual_label": ["roe deer", "", None, "horse"],
            }
        )

        filtered = filter_visualization_classes(dataframe)

        self.assertEqual(filtered["class"].tolist(), ["roe deer", "roe deer", "fox"])

    def test_baiting_filter_keeps_humans_and_target_species(self):
        dataframe = pd.DataFrame(
            {
                "class": ["horse", "human", "crow", "fox", "wild boar"],
                "manual_label": ["human", "", "roe deer", "wild boar", "squirrel"],
            }
        )

        filtered = filter_baiting_classes(dataframe)

        self.assertEqual(
            filtered["class"].tolist(), ["human", "human", "roe deer", "wild boar"]
        )


class VisualizationOutputSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.repo_dir = Path(self.temp_dir.name)
        self.script_path = (
            self.repo_dir / "src" / "visualisation" / "evaluate_labels.py"
        )
        self.output_dir = self.repo_dir / "docs" / "diagrams"
        self.output_dir.mkdir(parents=True)
        self.sentinel = self.output_dir / "keep.svg"
        self.sentinel.write_text("existing output", encoding="utf-8")
        model = "MegaDetectorV6_MDV6-yolov10-e_classified"
        self.csv_path = (
            self.repo_dir
            / "labels"
            / f"labels_{model}"
            / f"detection_results_{model}.csv"
        )

    def run_visualisation(self):
        with patch.object(evaluate_module, "__file__", self.script_path):
            return evaluate_module.main()

    def test_missing_csv_preserves_existing_visualisations(self):
        result = self.run_visualisation()

        self.assertIs(result, False)
        self.assertTrue(self.sentinel.exists())

    def test_invalid_csv_preserves_existing_visualisations(self):
        self.csv_path.parent.mkdir(parents=True)
        self.csv_path.write_text("class\nfox\n", encoding="utf-8")

        error = None
        result = None
        try:
            result = self.run_visualisation()
        except (KeyError, ValueError) as caught:
            error = caught

        self.assertIsNone(error)
        self.assertIs(result, False)
        self.assertTrue(self.sentinel.exists())


class VisualizationCustomPathTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.repo_dir = Path(self.temp_dir.name)
        self.script_path = (
            self.repo_dir / "src" / "visualisation" / "evaluate_labels.py"
        )
        self.csv_path = self.repo_dir / "fixture.csv"
        self.output_dir = self.repo_dir / "web-output"
        self.default_output = self.repo_dir / "docs" / "diagrams"
        self.default_output.mkdir(parents=True)
        self.sentinel = self.default_output / "keep.svg"
        self.sentinel.write_text("existing output", encoding="utf-8")

    def write_valid_csv(self):
        fieldnames = [
            "location_id",
            "timestamp",
            "timestamp_exif",
            "image_file",
            "class",
            "manual_label",
            "lighting",
            "temperature_celsius",
        ]
        classes = ["roe deer", "wild boar", "human", "horse"]
        start = datetime(2024, 4, 1)
        with self.csv_path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            for day in range(28):
                for index, label in enumerate(classes):
                    timestamp = start + timedelta(days=day, hours=index * 4 + 1)
                    writer.writerow(
                        {
                            "location_id": f"camera-{day % 2 + 1}",
                            "timestamp": timestamp.isoformat(),
                            "timestamp_exif": timestamp.isoformat(),
                            "image_file": f"image-{day:02d}-{index}.jpg",
                            "class": label,
                            "manual_label": "fox" if label == "horse" else "",
                            "lighting": "bright"
                            if 6 <= timestamp.hour < 19
                            else "dark",
                            "temperature_celsius": -5 + (day * 3 + index * 7) % 41,
                        }
                    )

    def run_visualisation(self, csv_path=None, output_dir=None):
        with patch.object(evaluate_module, "__file__", self.script_path):
            return evaluate_module.main(csv_path=csv_path, output_dir=output_dir)

    def test_main_uses_custom_paths_and_manual_labels(self):
        self.write_valid_csv()

        result = self.run_visualisation(self.csv_path, self.output_dir)

        self.assertIs(result, True)
        summary = self.output_dir / "summary_statistics.txt"
        self.assertTrue(summary.is_file())
        self.assertTrue(list(self.output_dir.glob("*.svg")))
        summary_text = summary.read_text(encoding="utf-8")
        self.assertIn("fox", summary_text)
        self.assertNotIn("horse", summary_text)
        self.assertEqual(self.sentinel.read_text(encoding="utf-8"), "existing output")

    def test_missing_custom_csv_preserves_existing_output(self):
        self.output_dir.mkdir()
        output_sentinel = self.output_dir / "keep.svg"
        output_sentinel.write_text("existing output", encoding="utf-8")

        result = self.run_visualisation(self.csv_path, self.output_dir)

        self.assertIs(result, False)
        self.assertEqual(output_sentinel.read_text(encoding="utf-8"), "existing output")
        self.assertEqual(self.sentinel.read_text(encoding="utf-8"), "existing output")

    def test_invalid_custom_csv_preserves_existing_output(self):
        self.csv_path.write_text("class\nfox\n", encoding="utf-8")
        self.output_dir.mkdir()
        output_sentinel = self.output_dir / "keep.svg"
        output_sentinel.write_text("existing output", encoding="utf-8")

        result = self.run_visualisation(self.csv_path, self.output_dir)

        self.assertIs(result, False)
        self.assertEqual(output_sentinel.read_text(encoding="utf-8"), "existing output")
        self.assertEqual(self.sentinel.read_text(encoding="utf-8"), "existing output")


class IndividualTimelineAxisTests(unittest.TestCase):
    def test_timeline_subplots_share_the_full_date_range(self):
        figure, axes = evaluate_module._create_individual_timeline_axes(
            n_rows=1, n_cols=2, figsize=(8, 4)
        )
        self.addCleanup(evaluate_module.plt.close, figure)
        axes[0].plot(pd.to_datetime(["2024-01-01", "2024-06-01"]), [1, 2])
        axes[1].plot(pd.to_datetime(["2025-01-01", "2025-06-01"]), [1, 2])

        self.assertTrue(axes[0].get_shared_x_axes().joined(axes[0], axes[1]))
        self.assertEqual(axes[0].get_xlim(), axes[1].get_xlim())


class VisualizationDateAxisTests(unittest.TestCase):
    def test_plot_dates_use_matplotlib_units_across_pandas_resolutions(self):
        dates = pd.Series(pd.to_datetime(["2025-10-28", "2026-08-23"]))

        actual = evaluate_module._matplotlib_date_values(dates)

        self.assertEqual(
            actual.tolist(), mdates.date2num(dates.to_numpy()).tolist()
        )


if __name__ == "__main__":
    unittest.main()
