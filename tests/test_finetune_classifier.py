import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
from src.labelling import finetune_classifier


class FinetuneLabelTests(unittest.TestCase):
    def test_build_model_seeds_all_native_classes_and_bird_subclasses(self):
        native_names = finetune_classifier.DeepfauneClassifier.CLASS_NAMES["en"]
        model = torch.nn.Module()
        model.head = torch.nn.Linear(2, len(native_names))
        model.blocks = torch.nn.ModuleList([torch.nn.Identity()] * 2)
        model.norm = torch.nn.Identity()
        model.fc_norm = torch.nn.Identity()
        old_head = model.head
        class_names = [
            finetune_classifier.normalize_species(name) for name in native_names
        ] + ["crow", "pigeon"]
        pretrained = SimpleNamespace(
            predictor=model, CLASS_NAMES=dict(enumerate(native_names))
        )
        with patch.object(
            finetune_classifier, "DeepfauneClassifier", return_value=pretrained
        ):
            result = finetune_classifier.build_model(class_names, "cpu")
        expected_indices = (
            list(range(len(native_names))) + [native_names.index("bird")] * 2
        )
        torch.testing.assert_close(
            result.head.weight, old_head.weight[expected_indices]
        )
        torch.testing.assert_close(result.head.bias, old_head.bias[expected_indices])

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
        labels = [
            "pigeon",
            "pigeon",
            "pigeon",
            "pigeon",
            "pigeon ",
            "human ",
            "cat",
            "crow",
            "other bird",
            "jay",
            "owl",
            "equid",
            "micromammal",
        ]
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

        self.assertEqual(
            rows["manual_label"].tolist(),
            ["pigeon"] * 5 + ["cat", "crow", "bird", "bird", "bird", "horse", "mouse"],
        )


if __name__ == "__main__":
    unittest.main()
