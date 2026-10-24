import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
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


if __name__ == "__main__":
    unittest.main()
