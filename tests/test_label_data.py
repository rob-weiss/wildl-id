import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.labelling import label_data


class LabelDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.human = self.root / "labels/human_labels.csv"
        self.legacy = self.root / label_data.LEGACY_RELATIVE_PATH
        self.model = self.root / "models/deepfaune_finetuned_20261029-221530_test"
        self.model.mkdir(parents=True)
        (self.model / "weights.pt").touch()

    def write(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_backup_retention_keeps_five_newest_distinct_states(self):
        self.write(
            self.human,
            [
                {
                    "location_id": "camera",
                    "image_file": "photo.jpg",
                    "box": "",
                    "manual_label": "fox",
                }
            ],
        )
        original = self.human.read_bytes()
        backups = []
        for index in range(8):
            path = self.human.with_name(
                f"{self.human.name}.before-organize-202610{index + 1:02d}-120000.bak"
            )
            path.write_bytes(f"state-{index}".encode())
            backups.append(path)
        duplicate = self.human.with_name(
            f"{self.human.name}.before-relabel-20261101-120000-000000.bak"
        )
        duplicate.write_bytes(backups[-1].read_bytes())
        migration = self.human.with_name(
            f"{self.human.name}.before-separation-20260101-120000.bak"
        )
        migration.write_bytes(b"migration")
        unknown = self.human.with_name(f"{self.human.name}.custom.bak")
        unknown.write_bytes(b"custom")
        linked = self.human.with_name(
            f"{self.human.name}.before-relabel-20261102-120000.bak"
        )
        linked.symlink_to(self.human)

        removed = label_data.prune_csv_backups(self.human)

        self.assertEqual(len(removed), 4)
        retained = [path for path in [*backups, duplicate] if path.exists()]
        self.assertEqual(len(retained), 5)
        self.assertEqual(
            {path.read_bytes() for path in retained},
            {f"state-{index}".encode() for index in range(3, 8)},
        )
        self.assertTrue(duplicate.exists())
        self.assertTrue(migration.exists())
        self.assertTrue(unknown.exists())
        self.assertTrue(linked.is_symlink())
        self.assertEqual(self.human.read_bytes(), original)

    def test_unchanged_csv_does_not_create_duplicate_backups(self):
        self.write(
            self.human,
            [
                {
                    "location_id": "camera",
                    "image_file": "photo.jpg",
                    "box": "",
                    "manual_label": "fox",
                }
            ],
        )

        first = label_data.backup_csv(self.human, "relabel")
        second = label_data.backup_csv(self.human, "organize")

        self.assertTrue(first.is_file())
        self.assertIsNone(second)
        self.assertEqual(len(list(self.human.parent.glob("*.bak"))), 1)

    def test_backup_cleanup_dry_run_does_not_delete_anything(self):
        self.human.parent.mkdir(parents=True)
        for index in range(7):
            path = self.human.with_name(
                f"{self.human.name}.before-relabel-202610{index + 1:02d}-120000.bak"
            )
            path.write_bytes(str(index).encode())

        removed = label_data.prune_csv_backups(self.human, dry_run=True)

        self.assertEqual(len(removed), 2)
        self.assertEqual(len(list(self.human.parent.glob("*.bak"))), 7)
        with self.assertRaises(ValueError):
            label_data.prune_csv_backups(self.human, keep=0)

    def test_organizing_without_changes_does_not_backup_or_rewrite_csv(self):
        from src.labelling import organize_images

        self.write(
            self.human,
            [
                {
                    "location_id": "camera",
                    "image_file": "photo.jpg",
                    "box": "",
                    "manual_label": "fox",
                }
            ],
        )
        original = self.human.read_bytes()
        with patch.object(organize_images, "CSV_PATH", self.human):
            organize_images.update_csv({}, set(), False)

        self.assertEqual(self.human.read_bytes(), original)
        self.assertEqual(list(self.human.parent.glob("*.bak")), [])

    def test_reviewer_skips_unchanged_saves_and_bounds_its_backups(self):
        from src.labelling.review_labels import LabelReview

        self.write(
            self.human,
            [
                {
                    "location_id": "camera",
                    "image_file": "photo.jpg",
                    "box": "",
                    "manual_label": "fox",
                }
            ],
        )
        original = self.human.read_bytes()
        review = LabelReview(self.human)
        review.save()
        self.assertEqual(self.human.read_bytes(), original)
        self.assertEqual(list(self.human.parent.glob("*.bak")), [])
        review.set_mode("All")
        for label in ("cat", "hare", "dog", "badger", "marten", "owl", "crow"):
            review.position = 0
            review.choose(label)
        self.assertEqual(len(list(self.human.parent.glob("*.bak"))), 5)
        self.assertEqual(label_data.read_records(self.human)[0]["manual_label"], "crow")

    def test_migration_preserves_human_annotations_and_timestamps_separately(self):
        rows = [
            {
                "location_id": "camera",
                "image_file": "image.jpg",
                "box": "[0.5, 0.5, 0.2, 0.2]",
                "manual_label": label,
                "class": "fox",
                "timestamp_exif": "2026-10-29 12:00:00",
                "confidence": "0.9",
            }
            for label in ("cat ", "fox")
        ]
        self.write(self.legacy, rows)
        original = self.legacy.read_bytes()

        label_data.migrate_human_labels(self.root)

        with self.human.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            human = list(reader)
            self.assertEqual(reader.fieldnames, list(label_data.HUMAN_COLUMNS))
        self.assertEqual([row["manual_label"] for row in human], ["cat ", "fox"])
        self.assertNotIn("timestamp_exif", human[0])
        self.assertEqual(self.legacy.read_bytes(), original)
        joined = label_data.load_rows(self.human)
        self.assertEqual(joined[0]["timestamp_exif"], "2026-10-29 12:00:00")
        self.assertEqual(joined[0]["class"], "")
        before = self.human.read_bytes()
        label_data.migrate_human_labels(self.root)
        self.assertEqual(self.human.read_bytes(), before)

    def test_migration_archives_accidental_model_output_and_preserves_predictions(self):
        self.write(
            self.legacy,
            [
                {
                    "location_id": "camera",
                    "image_file": "reviewed.jpg",
                    "box": "manual box",
                    "manual_label": "cat",
                    "class": "fox",
                }
            ],
        )
        self.write(
            self.human,
            [
                {
                    "location_id": "camera",
                    "image_file": "new.jpg",
                    "box": "automatic box",
                    "manual_label": "",
                    "class": "hare",
                    "classifier_model": self.model.name,
                }
            ],
        )
        original = self.human.read_bytes()

        label_data.migrate_human_labels(self.root)

        human = label_data.read_records(self.human)
        self.assertEqual(tuple(human[0]), label_data.HUMAN_COLUMNS)
        self.assertEqual(human[0]["manual_label"], "cat")
        predictions = label_data.read_records(label_data.predictions_path(self.root))
        self.assertEqual(predictions[0]["class"], "hare")
        backups = list(
            self.human.parent.glob("human_labels.csv.before-separation-*.bak")
        )
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original)

    def test_newest_model_predictions_join_manual_labels_without_using_older_model(
        self,
    ):
        self.write(
            self.human,
            [
                {
                    "location_id": "camera",
                    "image_file": "image.jpg",
                    "box": "manual box",
                    "manual_label": "cat",
                }
            ],
        )
        self.write(
            label_data.predictions_path(self.root),
            [
                {
                    "location_id": "camera",
                    "image_file": "image.jpg",
                    "class": "fox",
                    "box": "automatic box",
                }
            ],
        )
        joined = label_data.load_rows(self.human)
        self.assertEqual(joined[0]["class"], "fox")
        self.assertEqual(joined[0]["manual_label"], "cat")
        self.assertEqual(joined[0]["box"], "manual box")

        newer = self.root / "models/deepfaune_finetuned_20261031-120000_new"
        newer.mkdir()
        (newer / "weights.pt").touch()
        joined = label_data.load_rows(self.human)
        self.assertEqual(joined[0]["class"], "")
        self.assertEqual(joined[0]["manual_label"], "cat")
        self.assertEqual(
            label_data.predictions_path(self.root),
            self.root / "labels" / f"labels_{newer.name}" / "predictions.csv",
        )

    def test_prediction_only_images_are_not_lost(self):
        self.write(
            self.human,
            [
                {
                    "location_id": "camera",
                    "image_file": "reviewed.jpg",
                    "box": "",
                    "manual_label": "fox",
                }
            ],
        )
        self.write(
            label_data.predictions_path(self.root),
            [
                {
                    "location_id": "camera",
                    "image_file": "new.jpg",
                    "class": "hare",
                    "box": "",
                }
            ],
        )
        rows = label_data.load_rows(self.human)
        self.assertEqual(
            [row["image_file"] for row in rows], ["reviewed.jpg", "new.jpg"]
        )
        self.assertEqual(rows[1]["manual_label"], "")

    def test_incomplete_training_run_is_not_selected(self):
        (self.root / "models/deepfaune_finetuned_20261101-120000_incomplete").mkdir()
        self.assertEqual(
            label_data.predictions_path(self.root),
            self.root / "labels" / f"labels_{self.model.name}" / "predictions.csv",
        )

    def test_model_predictions_are_stored_under_labels(self):
        self.assertEqual(
            label_data.predictions_path(self.root),
            self.root
            / "labels"
            / f"labels_{self.model.name}"
            / "predictions.csv",
        )

    def test_validation_predictions_are_stored_under_labels(self):
        self.assertEqual(
            label_data.validation_predictions_dir(self.root, self.model.name),
            self.root
            / "labels"
            / f"labels_{self.model.name}"
            / "validation_predictions",
        )

    def test_training_keeps_burst_timestamps_outside_human_labels(self):
        from src.labelling import finetune_classifier

        rows = [
            {
                "location_id": "camera",
                "image_file": f"image_{index}.jpg",
                "box": "[0.5, 0.5, 0.2, 0.2]",
                "manual_label": "fox",
                "class": "fox",
                "timestamp_exif": f"2026-10-29 12:00:0{index}",
                "timestamp": "",
            }
            for index in range(5)
        ]
        self.write(self.legacy, rows)
        label_data.migrate_human_labels(self.root)
        paths = {
            ("camera", row["image_file"]): self.root / row["image_file"] for row in rows
        }
        with (
            patch.object(finetune_classifier, "CSV_PATH", self.human),
            patch.object(finetune_classifier, "image_paths", return_value=paths),
        ):
            training = finetune_classifier.load_labelled_rows()
            groups = finetune_classifier.assign_sequence_groups(training)

        self.assertEqual(len(training), 5)
        self.assertEqual(len(set(groups)), 1)
        self.assertNotIn("timestamp_exif", label_data.read_records(self.human)[0])

    def test_reviewer_saves_only_human_columns_and_keeps_model_predictions(self):
        from src.labelling.review_labels import LabelReview

        self.write(
            self.human,
            [
                {
                    "location_id": "camera",
                    "image_file": "image.jpg",
                    "box": "",
                    "manual_label": "",
                }
            ],
        )
        prediction = label_data.predictions_path(self.root)
        self.write(
            prediction,
            [
                {
                    "location_id": "camera",
                    "image_file": "image.jpg",
                    "class": "fox",
                    "box": "[0.5, 0.5, 0.2, 0.2]",
                }
            ],
        )
        before = prediction.read_bytes()

        review = LabelReview(self.human)
        review.choose("cat")

        saved = label_data.read_records(self.human)
        self.assertEqual(tuple(saved[0]), label_data.HUMAN_COLUMNS)
        self.assertEqual(saved[0]["manual_label"], "cat")
        self.assertEqual(prediction.read_bytes(), before)

    def test_inference_writes_only_current_model_and_never_changes_human_labels(self):
        from PIL import Image
        from src.labelling import label_images

        self.write(
            self.human,
            [
                {
                    "location_id": "camera",
                    "image_file": "reviewed.jpg",
                    "box": "",
                    "manual_label": "cat",
                }
            ],
        )
        original = self.human.read_bytes()
        paths = {}
        for name in ("reviewed.jpg", "new.jpg"):
            path = self.root / name
            Image.new("RGB", (40, 40)).save(path)
            paths[("camera", name)] = path
        classifier = Mock()
        classifier.weights_path = self.model / "weights.pt"
        detector = Mock()
        detector.single_image_detection.return_value = {}
        with (
            patch.object(label_images, "repo_dir", self.root),
            patch.object(label_images, "image_paths", return_value=paths),
            patch.object(label_images, "enable_ocr", False),
            patch.object(label_images, "save_annotated_images", False),
            patch.object(
                label_images, "load_species_classifier", return_value=classifier
            ),
            patch.object(
                label_images.pw_detection, "MegaDetectorV6", return_value=detector
            ),
            patch.object(label_images, "extract_exif_timestamp", return_value=None),
            patch.object(
                label_images,
                "resolve_exif_timestamp",
                return_value="2026-10-29 12:00:00",
            ),
            patch.dict(
                sys.modules,
                {"objc": SimpleNamespace(recycleAutoreleasePool=lambda: None)},
            ),
        ):
            label_images.process_images_with_pytorch_wildlife()
            self.assertEqual(detector.single_image_detection.call_count, 2)
            self.assertEqual(self.human.read_bytes(), original)
            saved_path = label_data.predictions_path(self.root, self.model.name)
            saved = label_data.read_records(saved_path)
            self.assertTrue(saved_path.is_relative_to(self.root / "labels"))
            self.assertEqual(len(saved), 2)
            self.assertNotIn("manual_label", saved[0])
            self.assertEqual(saved[0]["classifier_model"], self.model.name)
            detector.reset_mock()
            label_images.process_images_with_pytorch_wildlife(
                output_csv_path=self.human
            )
            detector.single_image_detection.assert_not_called()
            self.assertEqual(list(self.model.glob("*.bak")), [])
            newer = self.root / "models/deepfaune_finetuned_20261031-120000_new"
            newer.mkdir()
            (newer / "weights.pt").touch()
            classifier.weights_path = newer / "weights.pt"
            label_images.process_images_with_pytorch_wildlife()
            self.assertEqual(detector.single_image_detection.call_count, 2)
            self.assertEqual(
                len(label_data.read_records(label_data.predictions_path(self.root))),
                2,
            )
            self.assertEqual(self.human.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
