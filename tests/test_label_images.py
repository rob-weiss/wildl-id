import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from src.labelling import label_images


class AutomaticLabelTests(unittest.TestCase):
    def test_finetuned_classifier_keeps_native_labels(self):
        classifier = label_images.SpeciesClassifier.__new__(
            label_images.SpeciesClassifier
        )
        classifier.pretrained = False
        classifier.device = "cpu"
        classifier.class_names = ["fox", "song_bird"]
        classifier.transform = Mock(return_value=torch.zeros(3, 8, 8))
        classifier.model = Mock(return_value=torch.tensor([[0.0, 3.0]]))
        result = classifier.single_image_classification(
            np.zeros((8, 8, 3), dtype=np.uint8), "image.jpg"
        )
        self.assertEqual(result["prediction"], "song_bird")
        self.assertGreater(result["confidence"], 0.9)

    def test_latest_model_uses_timestamp_and_ignores_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            models = Path(directory)
            for name in (
                "deepfaune_finetuned_20261023-201148_old",
                "deepfaune_finetuned_20261029-221530_new",
                "deepfaune_finetuned_20261030-120000_incomplete",
            ):
                (models / name).mkdir()
            old = models / "deepfaune_finetuned_20261023-201148_old/weights.pt"
            newest = models / "deepfaune_finetuned_20261029-221530_new/weights.pt"
            newest.touch()
            old.touch()
            self.assertEqual(label_images.latest_classifier_weights(models), newest)

    def test_default_reruns_every_image_and_preserves_manual_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels_dir = root / "labels/labels_MegaDetectorV6_MDV6-yolov10-e_classified"
            labels_dir.mkdir(parents=True)
            csv_path = (
                labels_dir
                / "detection_results_MegaDetectorV6_MDV6-yolov10-e_classified.csv"
            )
            names = ["reviewed.jpg", "unreviewed.jpg", "new.jpg"] + [
                f"new_{index}.jpg" for index in range(50)
            ]
            paths = {}
            for name in names:
                path = root / name
                Image.new("RGB", (40, 40)).save(path)
                paths[("camera", name)] = path
            existing_rows = [
                {
                    "location_id": "camera",
                    "image_file": name,
                    "class": "fox",
                    "manual_label": manual,
                    "box": "[0.1, 0.1, 0.1, 0.1]",
                    "timestamp": "2026-10-29 12:00:00",
                    "timestamp_exif": "2026-10-29 12:00:00",
                    "temperature_celsius": 10,
                }
                for name, manual in zip(names[:2], ["cat ", ""], strict=True)
            ]
            existing_rows.append({**existing_rows[0], "manual_label": "fox"})
            existing_rows.append(
                {**existing_rows[0], "image_file": "missing.jpg", "manual_label": "dog"}
            )
            pd.DataFrame(existing_rows).to_csv(csv_path, index=False)
            original = csv_path.read_bytes()
            detector = Mock()
            detections = MagicMock()
            detections.__len__.return_value = 1
            detections.confidence = np.array([0.95])
            detections.class_id = np.array([0])
            detections.xyxy = np.array([[4, 4, 36, 36]])
            detector.single_image_detection.return_value = {
                "detections": detections,
            }
            classifier = Mock()
            classifier.weights_path = (
                root / "models/deepfaune_finetuned_20261029-221530_new/weights.pt"
            )
            classifier.single_image_classification.return_value = {
                "prediction": "song_bird",
                "confidence": 0.9,
            }
            with (
                patch.object(label_images, "repo_dir", root),
                patch.object(label_images, "image_paths", return_value=paths),
                patch.object(label_images, "enable_ocr", False),
                patch.object(label_images, "save_annotated_images", False),
                patch.object(
                    label_images.pw_detection, "MegaDetectorV6", return_value=detector
                ),
                patch.object(
                    label_images.pw_classification,
                    "DeepfauneClassifier",
                    return_value=classifier,
                ),
                patch.object(
                    label_images,
                    "load_species_classifier",
                    return_value=classifier,
                    create=True,
                ),
                patch.object(
                    label_images,
                    "resolve_exif_timestamp",
                    return_value="2026-10-29 12:00:00",
                ),
                patch.object(label_images, "extract_exif_timestamp", return_value=None),
                patch.dict(
                    sys.modules,
                    {"objc": SimpleNamespace(recycleAutoreleasePool=lambda: None)},
                ),
            ):
                label_images.process_images_with_pytorch_wildlife()
            output_dir = labels_dir.with_name(
                labels_dir.name + "_deepfaune_finetuned_20261029-221530_new"
            )
            output_csv = output_dir / csv_path.name
            self.assertTrue(output_csv.is_file())
            self.assertEqual(csv_path.read_bytes(), original)
            updated = pd.read_csv(output_csv, keep_default_na=False).set_index(
                "image_file"
            )
            self.assertEqual(detector.single_image_detection.call_count, len(names))
            self.assertEqual(len(updated), len(names) + 2)
            reviewed = updated.loc[["reviewed.jpg"]]
            self.assertEqual(reviewed["manual_label"].tolist(), ["cat ", "fox"])
            self.assertEqual(updated.loc["new.jpg", "manual_label"], "")
            self.assertEqual(set(updated.loc[names, "class"]), {"song_bird"})
            self.assertEqual(updated.loc["missing.jpg", "manual_label"], "dog")
            self.assertEqual(updated.loc["missing.jpg", "class"], "fox")
            self.assertTrue((reviewed["box"] != "[0.1, 0.1, 0.1, 0.1]").all())
            backups = list(output_dir.glob("*.before-relabel-*.bak"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
