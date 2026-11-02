import sys
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from src.labelling import label_data, label_images

wildl_id = import_module("wildl_id")


class AutomaticLabelTests(unittest.TestCase):
    def test_image_metadata_keeps_quoted_csv_format(self):
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            label_data.save_image_metadata(
                repo_root,
                [
                    {
                        "location_id": "camera",
                        "image_file": "photo.jpg",
                        "timestamp": "2026-11-02T12:00:00",
                        "timestamp_exif": "2026-11-02T12:00:00",
                        "lighting": "bright",
                        "temperature_celsius": "12.0",
                    }
                ],
            )

            self.assertEqual(
                (repo_root / "labels/image_metadata.csv").read_text(encoding="utf-8"),
                '"location_id","image_file","timestamp","timestamp_exif",'
                '"lighting","temperature_celsius"\n'
                '"camera","photo.jpg","2026-11-02T12:00:00",'
                '"2026-11-02T12:00:00","bright","12.0"\n',
            )

    def test_ocr_disabled_by_default_preserves_exif_timestamp(self):
        self.assertFalse(label_images.enable_ocr)
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            exif = Image.Exif()
            exif[0x0132] = "2026:10:25 02:30:00"
            Image.new("RGB", (8, 8)).save(image_path, exif=exif)

            timestamp = label_images.extract_exif_timestamp(image_path)

            self.assertEqual(
                label_images.resolve_exif_timestamp(timestamp, "", image_path),
                "2026-10-25T02:30:00",
            )

    def test_saved_generic_predictions_are_not_repeated(self):
        names = ["unknown.jpg", "animal.jpg", "pending.jpg"]
        existing = pd.DataFrame(
            [
                {
                    "location_id": "camera",
                    "image_file": name,
                    "class": automatic,
                    "manual_label": "",
                    "classifier_model": model,
                }
                for name, automatic, model in (
                    ("unknown.jpg", "unknown", "deepfaune_finetuned_latest"),
                    ("animal.jpg", "animal", "deepfaune_finetuned_latest"),
                    ("pending.jpg", "unknown", ""),
                )
            ]
        )
        images = [{"location_id": "camera", "name": name} for name in names]

        selected = label_images.images_requiring_processing(images, existing)

        self.assertEqual([image["name"] for image in selected], ["pending.jpg"])

    def test_resume_selection_skips_completed_and_manual_labels(self):
        names = [
            "reviewed.jpg",
            "complete.jpg",
            "unknown.jpg",
            "animal.jpg",
            "empty.jpg",
            "new.jpg",
        ]
        image_info = [{"location_id": "camera", "name": name} for name in names]
        existing = pd.DataFrame(
            [
                {
                    "location_id": "camera",
                    "image_file": name,
                    "class": automatic,
                    "manual_label": manual,
                }
                for name, automatic, manual in (
                    ("reviewed.jpg", "", "fox"),
                    ("complete.jpg", "none", ""),
                    ("unknown.jpg", "unknown", ""),
                    ("animal.jpg", "animal", ""),
                    ("empty.jpg", "", ""),
                )
            ]
        )

        selected = label_images.images_requiring_processing(image_info, existing)

        self.assertEqual([image["name"] for image in selected], names[2:])

    def test_reprocess_missing_retries_metadata_but_preserves_manual_labels(self):
        names = [
            "complete.jpg",
            "timestamp.jpg",
            "temperature.jpg",
            "manual.jpg",
            "new.jpg",
        ]
        image_info = [{"location_id": "camera", "name": name} for name in names]
        existing = pd.DataFrame(
            [
                {
                    "location_id": "camera",
                    "image_file": name,
                    "class": "fox",
                    "manual_label": manual,
                    "timestamp": timestamp,
                    "temperature_celsius": temperature,
                }
                for name, timestamp, temperature, manual in (
                    ("complete.jpg", "2026-10-29 12:00:00", "0", ""),
                    ("timestamp.jpg", "", "10", ""),
                    ("temperature.jpg", "2026-10-29 12:00:00", "", ""),
                    ("manual.jpg", "", "", "fox"),
                )
            ]
        )

        selected = label_images.images_requiring_processing(
            image_info, existing, reprocess_missing=True
        )

        self.assertEqual(
            [image["name"] for image in selected], names[1:3] + ["new.jpg"]
        )

    def test_unlabelled_selection_includes_new_images_and_skips_manual_labels(self):
        image_info = [
            {"location_id": "camera", "name": name}
            for name in ("reviewed.jpg", "unreviewed.jpg", "new.jpg")
        ]
        existing = pd.DataFrame(
            [
                {
                    "location_id": "camera",
                    "image_file": "reviewed.jpg",
                    "manual_label": "fox",
                },
                {
                    "location_id": "camera",
                    "image_file": "unreviewed.jpg",
                    "manual_label": " ",
                },
                {
                    "location_id": "camera",
                    "image_file": "removed.jpg",
                    "manual_label": "hare",
                },
            ]
        )

        selected = label_images.images_without_manual_labels(image_info, existing)

        self.assertEqual(
            [image["name"] for image in selected], ["unreviewed.jpg", "new.jpg"]
        )

    def test_classifier_does_not_alias_rabbit_to_hare(self):
        self.assertEqual(label_images.map_classifier_to_wildlife("rabbit"), "rabbit")

    def test_pretrained_classifier_maps_supported_labels(self):
        classifier = label_images.SpeciesClassifier.__new__(
            label_images.SpeciesClassifier
        )
        classifier.pretrained = True
        classifier.model = Mock()
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        renames = {
            "equid": "horse",
            "lagomorph": "hare",
            "micromammal": "mouse",
            "mustelid": "marten",
        }
        native_labels = label_images.pw_classification.DeepfauneClassifier.CLASS_NAMES[
            "en"
        ]
        self.assertEqual(len(native_labels), 34)
        for native_label in native_labels:
            with self.subTest(native_label=native_label):
                classifier.model.single_image_classification.return_value = {
                    "prediction": native_label,
                    "confidence": 0.9,
                }
                result = classifier.single_image_classification(image, "image.jpg")
                self.assertEqual(
                    result["prediction"], renames.get(native_label, native_label)
                )
                self.assertEqual(result["confidence"], 0.9)

    def test_classifier_preserves_unmapped_labels(self):
        for label in (
            "chevreuil", "cerf", "sanglier", "blaireau", "martre", "fouine",
            "chien", "renard", "lièvre", "lievre", "lapin", "écureuil", "ecureuil",
            "geai", "chouette", "hibou", "oiseau", "corbeau", "colombe", "humain",
            "chat",
        ):
            with self.subTest(label=label):
                self.assertEqual(label_images.map_classifier_to_wildlife(label), label)

    def test_finetuned_classifier_keeps_native_labels(self):
        classifier = label_images.SpeciesClassifier.__new__(
            label_images.SpeciesClassifier
        )
        classifier.pretrained = False
        classifier.device = "cpu"
        classifier.class_names = ["fox", "crow", "pigeon"]
        classifier.transform = Mock(return_value=torch.zeros(3, 8, 8))
        for index, label in enumerate(classifier.class_names):
            with self.subTest(label=label):
                logits = torch.zeros(1, len(classifier.class_names))
                logits[0, index] = 5.0
                classifier.model = Mock(return_value=logits)
                result = classifier.single_image_classification(
                    np.zeros((8, 8, 3), dtype=np.uint8), "image.jpg"
                )
                self.assertEqual(result["prediction"], label)
                self.assertGreater(result["confidence"], 0.9)

    def test_legacy_bird_predictions_are_reported_as_bird(self):
        self.assertEqual(label_images.map_classifier_to_wildlife("bird"), "bird")
        classifier = label_images.SpeciesClassifier.__new__(
            label_images.SpeciesClassifier
        )
        classifier.pretrained = False
        classifier.device = "cpu"
        classifier.class_names = ["fox", "owl"]
        classifier.transform = Mock(return_value=torch.zeros(3, 8, 8))
        classifier.model = Mock(return_value=torch.tensor([[0.0, 3.0]]))
        result = classifier.single_image_classification(
            np.zeros((8, 8, 3), dtype=np.uint8), "image.jpg"
        )
        self.assertEqual(result["prediction"], "bird")

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

    def test_classifier_loader_uses_latest_available_weights(self):
        newest = Path("models/deepfaune_finetuned_20261029-221530_new/weights.pt")
        with (
            patch.object(
                label_images, "latest_classifier_weights", return_value=newest
            ),
            patch.object(label_images, "SpeciesClassifier") as classifier,
        ):
            loaded = label_images.load_species_classifier("cpu")

        classifier.assert_called_once_with(newest, "cpu")
        self.assertIs(loaded, classifier.return_value)

    def test_force_reruns_every_image_and_preserves_manual_labels(self):
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
                label_images.process_images_with_pytorch_wildlife(
                    force=True, output_csv_path=csv_path
                )
            output_dir = labels_dir.with_name(
                labels_dir.name + "_deepfaune_finetuned_20261029-221530_new"
            )
            output_csv = csv_path
            self.assertTrue(output_csv.is_file())
            self.assertFalse((labels_dir / "images").exists())
            self.assertTrue(all(path.is_file() for path in paths.values()))
            self.assertFalse((output_dir / csv_path.name).exists())
            updated = pd.read_csv(output_csv, keep_default_na=False).set_index(
                "image_file"
            )
            self.assertEqual(detector.single_image_detection.call_count, len(names))
            self.assertEqual(len(updated), len(names) + 2)
            reviewed = updated.loc[["reviewed.jpg"]]
            self.assertEqual(reviewed["manual_label"].tolist(), ["cat ", "fox"])
            self.assertEqual(updated.loc["new.jpg", "manual_label"], "")
            self.assertEqual(set(updated.loc[names, "class"]), {"song_bird"})
            self.assertEqual(
                set(updated.loc[names, "classifier_model"]),
                {"deepfaune_finetuned_20261029-221530_new"},
            )
            self.assertEqual(updated.loc["missing.jpg", "manual_label"], "dog")
            self.assertEqual(updated.loc["missing.jpg", "class"], "fox")
            self.assertTrue((reviewed["box"] != "[0.1, 0.1, 0.1, 0.1]").all())
            backups = list(csv_path.parent.glob("*.before-relabel-*.bak"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)

    def test_download_then_app_and_cli_reuse_saved_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels_dir = root / "labels/labels_MegaDetectorV6_MDV6-yolov10-e_classified"
            labels_dir.mkdir(parents=True)
            csv_path = (
                labels_dir
                / "detection_results_MegaDetectorV6_MDV6-yolov10-e_classified.csv"
            )
            pd.DataFrame(
                [
                    {
                        "location_id": "camera",
                        "image_file": "reviewed.jpg",
                        "class": "fox",
                        "manual_label": "fox",
                    },
                    {
                        "location_id": "camera",
                        "image_file": "unreviewed.jpg",
                        "class": "fox",
                        "manual_label": "",
                    },
                ]
            ).to_csv(csv_path, index=False)
            paths = {}
            for name in ("reviewed.jpg", "unreviewed.jpg", "new.jpg"):
                path = root / name
                Image.new("RGB", (40, 40)).save(path)
                paths[("camera", name)] = path
            detector = Mock()
            detector.single_image_detection.return_value = {}
            classifier = Mock()
            classifier.weights_path = (
                root / "models/deepfaune_finetuned_test/weights.pt"
            )
            with (
                patch.object(sys, "argv", ["wildl_id.py", "--download"]),
                patch.object(wildl_id, "MAIL_LOCK_PATH", root / "mail-job.lock"),
                patch.object(wildl_id, "MAIL_PAUSE_PATH", root / "mail-job.paused"),
                patch.object(wildl_id, "download_images", return_value=True),
                patch.object(wildl_id, "DEFAULT_CSV", csv_path),
                patch.object(wildl_id, "image_paths", return_value=paths),
                patch.object(
                    wildl_id,
                    "process_images_with_pytorch_wildlife",
                    label_images.process_images_with_pytorch_wildlife,
                ),
                patch.object(label_images, "repo_dir", root),
                patch.object(label_images, "image_paths", return_value=paths),
                patch.object(label_images, "enable_ocr", False),
                patch.object(
                    label_images.pw_detection, "MegaDetectorV6", return_value=detector
                ),
                patch.object(
                    label_images,
                    "load_species_classifier",
                    return_value=classifier,
                ),
                patch.object(label_images, "resolve_exif_timestamp", return_value=""),
                patch.object(label_images, "extract_exif_timestamp", return_value=None),
                patch.dict(
                    sys.modules,
                    {"objc": SimpleNamespace(recycleAutoreleasePool=lambda: None)},
                ),
            ):
                self.assertEqual(wildl_id.main(), 0)
                self.assertEqual(detector.single_image_detection.call_count, 1)
                saved = csv_path.read_bytes()
                detector.reset_mock()
                self.assertEqual(wildl_id.main(), 0)
                label_images.process_images_with_pytorch_wildlife(
                    output_csv_path=csv_path
                )
                detector.single_image_detection.assert_not_called()
                self.assertEqual(csv_path.read_bytes(), saved)
                label_images.process_images_with_pytorch_wildlife(
                    output_csv_path=csv_path, unlabelled_only=True
                )
                detector.single_image_detection.assert_not_called()
                for name in ("additional_a.jpg", "additional_b.jpg"):
                    path = root / name
                    Image.new("RGB", (40, 40)).save(path)
                    paths[("camera", name)] = path
                detector.single_image_detection.side_effect = [{}, KeyboardInterrupt]
                with self.assertRaises(KeyboardInterrupt):
                    label_images.process_images_with_pytorch_wildlife(
                        output_csv_path=csv_path
                    )
                interrupted = pd.read_csv(csv_path, keep_default_na=False).set_index(
                    "image_file"
                )
                self.assertEqual(interrupted.loc["additional_a.jpg", "class"], "none")
                self.assertNotIn("additional_b.jpg", interrupted.index)
                detector.single_image_detection.side_effect = None
                detector.reset_mock()
                self.assertEqual(wildl_id.main(), 0)

            self.assertEqual(detector.single_image_detection.call_count, 1)
            updated = pd.read_csv(csv_path, keep_default_na=False).set_index(
                "image_file"
            )
            self.assertEqual(updated.loc["reviewed.jpg", "manual_label"], "fox")
            self.assertEqual(updated.loc["unreviewed.jpg", "manual_label"], "")
            self.assertEqual(updated.loc["unreviewed.jpg", "class"], "fox")
            self.assertEqual(
                updated.loc["new.jpg", "classifier_model"],
                "deepfaune_finetuned_test",
            )
            self.assertIn("new.jpg", updated.index)

    def test_targeted_run_preserves_manual_edits_made_during_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "active.csv"
            pd.DataFrame(
                [
                    {
                        "location_id": "camera",
                        "image_file": "reviewed.jpg",
                        "class": "fox",
                        "manual_label": "fox",
                    },
                    {
                        "location_id": "camera",
                        "image_file": "unreviewed.jpg",
                        "class": "fox",
                        "manual_label": "",
                    },
                ]
            ).to_csv(csv_path, index=False)
            paths = {}
            for name in ("reviewed.jpg", "unreviewed.jpg", "new.jpg"):
                path = root / name
                Image.new("RGB", (40, 40)).save(path)
                paths[("camera", name)] = path

            def detect(_path, **_kwargs):
                latest = pd.read_csv(csv_path, keep_default_na=False)
                latest.loc[latest.image_file == "reviewed.jpg", "manual_label"] = (
                    "badger"
                )
                latest.loc[len(latest)] = ["camera", "new.jpg", "", "hare"]
                latest["box"] = ""
                latest.loc[latest.image_file == "new.jpg", "box"] = (
                    "[0.5, 0.5, 0.2, 0.2]"
                )
                latest.to_csv(csv_path, index=False)
                return {}

            detector = Mock()
            detector.single_image_detection.side_effect = detect
            classifier = Mock()
            classifier.weights_path = (
                root / "models/deepfaune_finetuned_test/weights.pt"
            )
            with (
                patch.object(label_images, "repo_dir", root),
                patch.object(label_images, "image_paths", return_value=paths),
                patch.object(label_images, "enable_ocr", False),
                patch.object(
                    label_images.pw_detection, "MegaDetectorV6", return_value=detector
                ),
                patch.object(
                    label_images, "load_species_classifier", return_value=classifier
                ),
                patch.object(label_images, "resolve_exif_timestamp", return_value=""),
                patch.object(label_images, "extract_exif_timestamp", return_value=None),
                patch.dict(
                    sys.modules,
                    {"objc": SimpleNamespace(recycleAutoreleasePool=lambda: None)},
                ),
            ):
                label_images.process_images_with_pytorch_wildlife(
                    output_csv_path=csv_path, image_keys={("camera", "new.jpg")}
                )
            self.assertEqual(detector.single_image_detection.call_count, 1)
            updated = pd.read_csv(csv_path, keep_default_na=False).set_index(
                "image_file"
            )
            self.assertEqual(updated.loc["reviewed.jpg", "manual_label"], "badger")
            self.assertEqual(updated.loc["unreviewed.jpg", "class"], "fox")
            self.assertEqual(updated.loc["new.jpg", "manual_label"], "hare")
            self.assertEqual(updated.loc["new.jpg", "class"], "none")
            self.assertEqual(updated.loc["new.jpg", "box"], "[0.5, 0.5, 0.2, 0.2]")


if __name__ == "__main__":
    unittest.main()
