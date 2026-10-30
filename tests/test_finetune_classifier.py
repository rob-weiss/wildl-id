import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from src.labelling import finetune_classifier


class FinetuneLabelTests(unittest.TestCase):
    def test_save_confusion_matrix_exports_labelled_counts(self):
        cases = [
            ([0, 0, 1], [0, 1, 1], ["badger", "fox", "wild boar"], 3, 6),
            ([0, 0], [0, 0], ["badger"], 0, 0),
        ]
        for targets, predictions, class_names, ones, zeros in cases:
            with self.subTest(class_names=class_names):
                with tempfile.TemporaryDirectory() as temp_dir:
                    output_path = Path(temp_dir) / "confusion_matrix.svg"
                    finetune_classifier.save_confusion_matrix(
                        targets, predictions, class_names, output_path
                    )
                    root = ET.parse(output_path).getroot()

                namespace = {"svg": "http://www.w3.org/2000/svg"}
                axes = root.find(".//svg:g[@id='axes_1']", namespace)
                texts = [
                    "".join(element.itertext())
                    for element in axes.findall(".//svg:text", namespace)
                ]
                for label in ["True label", "Predicted label", *class_names]:
                    self.assertIn(label, texts)
                counts = [text for text in texts if text.isdigit()]
                self.assertEqual(len(counts), len(class_names) ** 2)
                self.assertEqual(counts.count("1"), ones)
                self.assertEqual(counts.count("0"), zeros)
                if len(class_names) == 1:
                    self.assertEqual(counts, ["2"])

    def test_load_labelled_rows_strips_manual_labels_before_counting(self):
        labels = ["pigeon", "pigeon", "pigeon", "pigeon", "pigeon ", "human ", "cat"]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "labels.csv"
            pd.DataFrame(
                {
                    "box": ["[0.5, 0.5, 0.4, 0.4]"] * len(labels),
                    "manual_label": labels,
                    "location_id": ["location"] * len(labels),
                    "image_file": [
                        f"image_{index}.jpg" for index in range(len(labels))
                    ],
                }
            ).to_csv(csv_path, index=False)
            indexed_paths = {
                ("location", f"image_{index}.jpg"): temp_path / f"image_{index}.jpg"
                for index in range(len(labels))
            }

            with (
                patch.object(finetune_classifier, "CSV_PATH", csv_path),
                patch.object(
                    finetune_classifier, "image_paths", return_value=indexed_paths
                ),
            ):
                rows = finetune_classifier.load_labelled_rows()

        self.assertEqual(rows["manual_label"].tolist(), ["pigeon"] * 5)


if __name__ == "__main__":
    unittest.main()
