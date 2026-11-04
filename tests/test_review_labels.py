import asyncio
import csv
import tempfile
import unittest
import xml.etree.ElementTree as ET
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgba
from PIL import Image

review_module = import_module("src.labelling.review_labels")
LabelReview = review_module.LabelReview
review_app_module = import_module("src.labelling.review_app")
empty_images_module = import_module("src.labelling.remove_empty_images")


class RemoveEmptyImagesTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.csv_path = self.root / "active.csv"
        self.images_dir = self.root / "images"
        self.annotated_dir = self.root / "active-annotations"
        self.annotated_dir.mkdir()
        self.original_dir = self.images_dir / "2024" / "camera1"
        self.original_dir.mkdir(parents=True)
        self.names = ["empty.jpg", "unseen.jpg", "corrected.jpg", "animal.jpg"]
        with self.csv_path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(
                target,
                fieldnames=["location_id", "image_file", "class", "manual_label"],
            )
            writer.writeheader()
            for name, label, manual_label in zip(
                self.names,
                ["none", "none", "fox", "fox"],
                ["none", "", "none", "fox"],
            ):
                writer.writerow(
                    {
                        "location_id": "camera1",
                        "image_file": name,
                        "class": label,
                        "manual_label": manual_label,
                    }
                )
                (self.original_dir / name).write_bytes(b"original image")
                (self.annotated_dir / f"camera1_{name}").write_bytes(b"annotation")
        self.default_csv = self.root / "default.csv"
        self.default_csv.write_bytes(b"do not change")
        patches = patch.multiple(
            empty_images_module,
            REPO_DIR=self.root,
            IMAGES_DIR=self.images_dir,
            ANNOTATED_DIR=self.root / "default-annotations",
            CSV_PATH=self.default_csv,
            LOG_DIR=self.root / "logs",
        )
        patches.start()
        self.addCleanup(patches.stop)

    def build_app(self):
        app = review_app_module.ReviewApp(self.csv_path, self.annotated_dir)
        image_patch = patch.object(app, "image_path_for", return_value=None)
        image_patch.start()
        self.addCleanup(image_patch.stop)
        with review_app_module.ui.column() as container:
            app.build()
        self.addCleanup(container.delete)
        return app

    def test_remove_empty_uses_active_csv_and_keeps_unconfirmed_images(self):
        before = self.csv_path.read_bytes()

        count = empty_images_module.remove_empty_images(
            csv_path=self.csv_path,
            annotated_dir=self.annotated_dir,
            expected_bytes=before,
        )

        self.assertEqual(count, 1)
        self.assertFalse((self.original_dir / "empty.jpg").exists())
        self.assertFalse((self.annotated_dir / "camera1_empty.jpg").exists())
        for name in self.names[1:]:
            self.assertTrue((self.original_dir / name).exists())
            self.assertTrue((self.annotated_dir / f"camera1_{name}").exists())
        _, rows = empty_images_module.read_csv(self.csv_path)
        self.assertEqual([row["image_file"] for row in rows], self.names[1:])
        backup = next(self.root.glob("active.csv.before-remove-empty-*.bak"))
        self.assertEqual(backup.read_bytes(), before)
        log = next((self.root / "logs").glob("remove_empty_images_*.csv"))
        with log.open(newline="", encoding="utf-8") as source:
            logged = list(csv.DictReader(source))
        self.assertEqual([row["image_file"] for row in logged], ["empty.jpg"])
        self.assertEqual(self.default_csv.read_bytes(), b"do not change")

    def test_game_cleanup_selects_unique_human_labelled_images_only(self):
        rows = [
            {
                "location_id": "camera1",
                "image_file": name,
                "class": automatic,
                "manual_label": manual,
            }
            for name, automatic, manual in (
                ("cat.jpg", "fox", "cat"),
                ("cat.jpg", "fox", "cat"),
                ("fox.jpg", "cat", "fox"),
                ("unseen.jpg", "cat", ""),
                ("bird.jpg", "cat", "bird"),
                ("legacy-bird.jpg", "cat", "owl"),
                ("unknown.jpg", "fox", "unknown"),
                ("empty.jpg", "fox", "none"),
                ("mixed.jpg", "fox", "cat"),
                ("mixed.jpg", "cat", "fox"),
                ("squirrel.jpg", "fox", "squirrel"),
            )
        ]
        self.assertEqual(
            empty_images_module.non_game_images(rows, {"fox"}),
            {
                ("camera1", "cat.jpg"): "cat",
                ("camera1", "bird.jpg"): "bird",
                ("camera1", "legacy-bird.jpg"): "bird",
                ("camera1", "unknown.jpg"): "unknown",
                ("camera1", "empty.jpg"): "none",
                ("camera1", "squirrel.jpg"): "squirrel",
            },
        )
        protected_birds = empty_images_module.non_game_images(
            rows, review_app_module.GAME_SPECIES
        )
        self.assertNotIn(("camera1", "bird.jpg"), protected_birds)
        self.assertNotIn(("camera1", "legacy-bird.jpg"), protected_birds)

    def test_game_cleanup_removes_related_csv_rows_but_not_training_archives(self):
        data = empty_images_module.label_data
        related_paths = [
            self.root / "labels/image_metadata.csv",
            self.root / "labels/labels_model/predictions.csv",
            self.root / "labels/labels_model/validation_predictions/predictions.csv",
        ]
        rows = [{"location_id": "camera1", "image_file": name} for name in self.names]
        columns = ("location_id", "image_file")
        for path in related_paths:
            data.write_records(path, rows, columns)
        archive = self.root / "models/model/training_data/labels.csv"
        data.write_records(archive, rows, columns)
        before_archive = archive.read_bytes()
        before = self.csv_path.read_bytes()

        count = empty_images_module.remove_empty_images(
            csv_path=self.csv_path,
            annotated_dir=self.annotated_dir,
            expected_bytes=before,
            keep_species={"fox"},
        )

        self.assertEqual(count, 2)
        for name in self.names:
            kept = name in {"unseen.jpg", "animal.jpg"}
            self.assertEqual((self.original_dir / name).exists(), kept)
            self.assertEqual((self.annotated_dir / f"camera1_{name}").exists(), kept)
        self.assertEqual(
            [row["image_file"] for row in data.read_records(self.csv_path)],
            ["unseen.jpg", "animal.jpg"],
        )
        for path in related_paths:
            self.assertEqual(
                [row["image_file"] for row in data.read_records(path)],
                ["unseen.jpg", "animal.jpg"],
            )
            self.assertTrue(
                list(path.parent.glob(f"{path.name}.before-remove-empty-*.bak"))
            )
        self.assertEqual(archive.read_bytes(), before_archive)
        self.assertEqual(
            {
                data.image_key(row)
                for row in data.read_records(self.root / "labels/deleted_images.csv")
            },
            {("camera1", "empty.jpg"), ("camera1", "corrected.jpg")},
        )

    def test_remove_empty_can_include_false_detections(self):
        count = empty_images_module.remove_empty_images(
            csv_path=self.csv_path,
            annotated_dir=self.annotated_dir,
            include_false_detections=True,
        )

        self.assertEqual(count, 2)
        for name, kept in zip(self.names, (False, True, False, True)):
            self.assertEqual((self.original_dir / name).exists(), kept, name)
        _, rows = empty_images_module.read_csv(self.csv_path)
        self.assertEqual(
            [row["image_file"] for row in rows], ["unseen.jpg", "animal.jpg"]
        )

    def test_game_cleanup_dry_run_changes_nothing(self):
        before = self.csv_path.read_bytes()
        count = empty_images_module.remove_empty_images(
            dry_run=True,
            csv_path=self.csv_path,
            annotated_dir=self.annotated_dir,
            keep_species={"fox"},
        )
        self.assertEqual(count, 2)
        self.assertEqual(self.csv_path.read_bytes(), before)
        self.assertTrue(all((self.original_dir / name).exists() for name in self.names))
        self.assertFalse((self.root / "logs").exists())
        self.assertFalse((self.root / "labels/deleted_images.csv").exists())

    def test_game_cleanup_related_backup_failure_keeps_images_and_rows(self):
        data = empty_images_module.label_data
        related = self.root / "labels/image_metadata.csv"
        data.write_records(
            related,
            [{"location_id": "camera1", "image_file": "empty.jpg"}],
            data.METADATA_COLUMNS,
        )
        before = self.csv_path.read_bytes()
        before_related = related.read_bytes()
        backup_csv = empty_images_module.backup_csv

        def backup(path, reason):
            if path == related:
                raise OSError("backup failed")
            return backup_csv(path, reason)

        with (
            patch.object(empty_images_module, "backup_csv", side_effect=backup),
            self.assertRaisesRegex(OSError, "backup failed"),
        ):
            empty_images_module.remove_empty_images(
                csv_path=self.csv_path,
                annotated_dir=self.annotated_dir,
                keep_species={"fox"},
            )
        self.assertEqual(self.csv_path.read_bytes(), before)
        self.assertEqual(related.read_bytes(), before_related)
        self.assertTrue(all((self.original_dir / name).exists() for name in self.names))

    def test_gui_game_cleanup_summarizes_classes_and_requires_confirmation(self):
        with self.csv_path.open("a", newline="", encoding="utf-8") as target:
            csv.writer(target).writerows(
                [
                    ["camera1", "cat.jpg", "fox", "cat"],
                    ["camera1", "dog.jpg", "fox", "dog"],
                    ["camera1", "dog.jpg", "fox", "dog"],
                    ["camera1", "bird.jpg", "fox", "bird"],
                    ["camera1", "unknown.jpg", "fox", "unknown"],
                ]
            )
        app = self.build_app()
        before = self.csv_path.read_bytes()
        for confirmed in (False, True):

            async def answer():
                self.assertTrue(app.cleanup_dialog_open)
                return confirmed

            async def confirm():
                with app.remove_irrelevant_button.parent_slot.parent:
                    await app.confirm_remove_irrelevant()

            with (
                patch.object(
                    review_app_module.ui.dialog,
                    "__await__",
                    lambda dialog: answer().__await__(),
                ),
                patch.object(
                    review_app_module.ui, "table", wraps=review_app_module.ui.table
                ) as table,
                patch.object(
                    review_app_module.ui, "button", wraps=review_app_module.ui.button
                ) as buttons,
                patch.object(app, "remove_empty", new_callable=AsyncMock) as remove,
            ):
                asyncio.run(confirm())
                self.assertEqual(
                    table.call_args.kwargs["rows"],
                    [
                        {"class": "cat", "images": 1},
                        {"class": "dog", "images": 1},
                        {"class": "none", "images": 2},
                        {"class": "unknown", "images": 1},
                    ],
                )
                self.assertEqual(
                    table.call_args.kwargs["pagination"], {"rowsPerPage": 0}
                )
                self.assertEqual(
                    {call.args[0] for call in buttons.call_args_list},
                    {"❌ Cancel", "🗑️ Delete 5 images"},
                )
                self.assertTrue(
                    all(not call.kwargs.get("icon") for call in buttons.call_args_list)
                )
                if confirmed:
                    remove.assert_awaited_once_with(
                        keep_species=review_app_module.GAME_SPECIES
                    )
                else:
                    remove.assert_not_awaited()
            self.assertFalse(app.cleanup_dialog_open)
        self.assertEqual(self.csv_path.read_bytes(), before)
        self.assertTrue(all((self.original_dir / name).exists() for name in self.names))

    def test_review_buttons_use_emojis_and_one_cleanup_control(self):
        with patch.object(
            review_app_module.ui, "button", wraps=review_app_module.ui.button
        ) as button:
            app = self.build_app()
        self.assertEqual(app.remove_irrelevant_button.text, "🗑️ Remove irrelevant")
        self.assertFalse(hasattr(app, "remove_empty_button"))
        self.assertFalse(hasattr(app, "remove_unknown_button"))
        self.assertTrue(button.call_args_list)
        for call in button.call_args_list:
            self.assertIsNone(call.kwargs.get("icon"))

    def test_gui_game_cleanup_reloads_rows_and_resets_undo(self):
        app = self.build_app()
        app.change_mode("All")
        app.review.history.append((0, "none"))
        app.box_history.append((0, ""))
        with patch.object(review_app_module.ui, "notify"):
            asyncio.run(app.remove_empty(keep_species={"fox"}))
        self.assertEqual(
            [row["image_file"] for row in app.review.rows], ["unseen.jpg", "animal.jpg"]
        )
        self.assertEqual(app.counter.text, "1 / 2")
        self.assertEqual(app.review.mode, "All")
        self.assertEqual(app.review.history, [])
        self.assertEqual(app.box_history, [])

    def test_remove_empty_supports_human_only_csv_and_hides_deleted_predictions(self):
        data = import_module("src.labelling.label_data")
        human = self.root / "labels/human_labels.csv"
        data.write_records(
            human,
            [
                {
                    "location_id": "camera1",
                    "image_file": "empty.jpg",
                    "box": "",
                    "manual_label": "none",
                }
            ],
            data.HUMAN_COLUMNS,
        )
        prediction = data.predictions_path(self.root, "deepfaune_pretrained")
        data.write_records(
            prediction,
            [
                {
                    "location_id": "camera1",
                    "image_file": "empty.jpg",
                    "class": "none",
                    "box": "",
                }
            ],
            data.PREDICTION_COLUMNS,
        )
        original_predictions = prediction.read_bytes()

        removed = empty_images_module.remove_empty_images(
            csv_path=human, annotated_dir=self.annotated_dir
        )

        self.assertEqual(removed, 1)
        self.assertFalse((self.original_dir / "empty.jpg").exists())
        self.assertEqual(data.read_records(human), [])
        self.assertEqual(data.load_rows(human), [])
        self.assertEqual(prediction.read_bytes(), original_predictions)
        data.write_records(
            self.root / "labels/image_metadata.csv",
            [{"location_id": "camera1", "image_file": "empty.jpg"}],
            data.METADATA_COLUMNS,
        )
        self.assertEqual(data.load_rows(human), [])

    def test_remove_empty_dry_run_does_not_change_files(self):
        before = self.csv_path.read_bytes()
        count = empty_images_module.remove_empty_images(
            dry_run=True, csv_path=self.csv_path, annotated_dir=self.annotated_dir
        )
        self.assertEqual(count, 1)
        self.assertEqual(self.csv_path.read_bytes(), before)
        self.assertTrue((self.original_dir / "empty.jpg").exists())
        self.assertTrue((self.annotated_dir / "camera1_empty.jpg").exists())
        self.assertFalse((self.root / "logs").exists())
        self.assertEqual(list(self.root.glob("*.bak")), [])

    def test_remove_empty_rejects_stale_csv_before_deleting_files(self):
        with self.assertRaisesRegex(RuntimeError, "CSV changed"):
            empty_images_module.remove_empty_images(
                csv_path=self.csv_path,
                annotated_dir=self.annotated_dir,
                expected_bytes=b"stale snapshot",
            )
        self.assertTrue((self.original_dir / "empty.jpg").exists())
        self.assertTrue((self.annotated_dir / "camera1_empty.jpg").exists())
        self.assertFalse((self.root / "logs").exists())

    def test_remove_empty_backup_failure_does_not_delete_images(self):
        with (
            patch.object(
                empty_images_module,
                "backup_csv",
                side_effect=OSError("backup failed"),
            ),
            self.assertRaisesRegex(OSError, "backup failed"),
        ):
            empty_images_module.remove_empty_images(
                csv_path=self.csv_path, annotated_dir=self.annotated_dir
            )
        self.assertTrue((self.original_dir / "empty.jpg").exists())
        self.assertTrue((self.annotated_dir / "camera1_empty.jpg").exists())

    def test_remove_unknown_deletes_only_human_unknown_labels(self):
        rows = [
            ("camera1", "unknown.jpg", "fox", "unknown"),
            ("camera1", "pending.jpg", "", "unknown"),
            ("camera1", "model-unknown.jpg", "unknown", ""),
        ]
        with self.csv_path.open("a", newline="", encoding="utf-8") as target:
            csv.writer(target).writerows(rows)
        for _, name, _, _ in rows:
            (self.original_dir / name).write_bytes(b"original image")

        count = empty_images_module.remove_empty_images(
            csv_path=self.csv_path, annotated_dir=self.annotated_dir, label="unknown"
        )

        self.assertEqual(count, 2)
        self.assertFalse((self.original_dir / "unknown.jpg").exists())
        self.assertFalse((self.original_dir / "pending.jpg").exists())
        self.assertTrue((self.original_dir / "model-unknown.jpg").exists())
        self.assertTrue((self.original_dir / "empty.jpg").exists())

    def test_gui_remove_irrelevant_reports_stale_csv_without_deleting_images(self):
        app = self.build_app()
        self.csv_path.write_bytes(self.csv_path.read_bytes() + b"\n")

        with patch.object(review_app_module.ui, "notify") as notify:
            asyncio.run(app.remove_empty(keep_species=review_app_module.GAME_SPECIES))

        self.assertIn("CSV changed", notify.call_args.args[0])
        self.assertEqual(notify.call_args.kwargs["type"], "negative")
        self.assertTrue((self.original_dir / "empty.jpg").exists())

    def test_gui_shortcuts_do_nothing_while_cleanup_dialog_is_open(self):
        app = self.build_app()
        app.cleanup_dialog_open = True
        event = SimpleNamespace(
            action=SimpleNamespace(keydown=True, repeat=False),
            key=SimpleNamespace(name=" ", space=True, backspace=False, enter=False),
            modifiers=SimpleNamespace(alt=False, ctrl=False, meta=False),
        )
        with patch.object(app, "accept") as accept:
            app.on_key(event)
        accept.assert_not_called()


class LabelReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "labels.csv"
        with self.csv_path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(
                target,
                fieldnames=[
                    "location_id",
                    "image_file",
                    "class",
                    "box",
                    "manual_label",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "location_id": "camera1",
                        "image_file": "unchecked.jpg",
                        "class": "fox",
                        "box": "[0.5, 0.5, 0.4, 0.4]",
                        "manual_label": "",
                    },
                    {
                        "location_id": "camera1",
                        "image_file": "correct.jpg",
                        "class": "fox",
                        "box": "[0.5, 0.5, 0.4, 0.4]",
                        "manual_label": "fox",
                    },
                    {
                        "location_id": "camera1",
                        "image_file": "incorrect.jpg",
                        "class": "fox",
                        "box": "[0.5, 0.5, 0.4, 0.4]",
                        "manual_label": "hare",
                    },
                ]
            )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_unseen_is_default_mode(self):
        review = LabelReview(self.csv_path)

        self.assertEqual(review.mode, "Unseen")
        self.assertEqual(review.pending, [0])

    def test_latest_incorrect_excludes_true_labels_with_manual_precedence(self):
        review = LabelReview(self.csv_path)
        review.latest_predictions.update(
            {
                "camera1_unchecked.jpg": {
                    "true_label": "bird",
                    "predicted_label": "crow",
                    "status": "incorrect",
                },
                "camera1_incorrect.jpg": {
                    "true_label": "bird",
                    "predicted_label": "fox",
                    "status": "incorrect",
                },
            }
        )
        original = self.csv_path.read_bytes()
        review.set_mode("Latest model incorrect")
        self.assertEqual(set(review.pending), {0, 2})
        self.assertEqual(review.true_labels, ["bird", "hare"])

        review.set_excluded_true_labels([" HARE "])
        self.assertEqual(review.excluded_true_labels, {"hare"})
        self.assertEqual(review.pending, [0])

        review.set_excluded_true_labels(["bird"])
        self.assertEqual(review.pending, [2])

        review.set_excluded_true_labels(["bird", "hare"])
        self.assertEqual(review.pending, [])
        self.assertEqual(review.position, 0)
        self.assertEqual(review.true_labels, ["bird", "hare"])

        review.set_excluded_true_labels([])
        self.assertEqual(set(review.pending), {0, 2})
        self.assertEqual(self.csv_path.read_bytes(), original)

    def test_true_label_options_only_include_unresolved_errors(self):
        review = LabelReview(self.csv_path)
        self.assertEqual(review.true_labels, ["hare"])

        review.resolved_incorrect.add("camera1_incorrect.jpg")
        self.assertEqual(review.true_labels, [])

        review.resolved_incorrect.clear()
        review.rows[2]["reviewed"] = "true"
        self.assertEqual(review.true_labels, [])

        review.rows[2]["reviewed"] = ""
        review.rows[2]["class"] = "hare"
        self.assertEqual(review.true_labels, [])

    def test_true_label_exclusions_do_not_change_other_review_modes(self):
        review = LabelReview(self.csv_path)
        review.set_excluded_true_labels(["hare"])
        for mode, expected in (
            ("Unseen", {0}),
            ("Seen", {1, 2}),
            ("All", {0, 1, 2}),
            ("Unseen + errors", {0, 2}),
            ("Latest model incorrect", set()),
        ):
            with self.subTest(mode=mode):
                review.set_mode(mode)
                self.assertEqual(set(review.pending), expected)

    def test_undo_does_not_restore_excluded_true_labels_to_queue(self):
        review = LabelReview(self.csv_path)
        review.set_mode("Latest model incorrect")
        review.choose("fox")
        review.set_excluded_true_labels(["hare"])

        review.undo()

        self.assertEqual(review.rows[2]["manual_label"], "hare")
        self.assertEqual(review.pending, [])
        self.assertEqual(review.position, 0)
        review.set_excluded_true_labels([])
        self.assertEqual(review.pending, [2])

    def test_browser_true_label_exclusions_update_queue_and_visibility(self):
        app = review_app_module.ReviewApp(self.csv_path)
        app.review.latest_predictions["camera1_unchecked.jpg"] = {
            "true_label": "bird",
            "predicted_label": "crow",
            "status": "incorrect",
        }
        with review_app_module.ui.column() as container:
            self.addCleanup(container.delete)
            with patch.object(app, "image_path_for", return_value=None):
                app.build()
                selector = app.exclude_true_labels_select
                self.assertFalse(selector.visible)
                self.assertTrue(selector._props["multiple"])
                self.assertTrue(selector._props["clearable"])

                app.change_mode("Latest model incorrect")
                self.assertTrue(selector.visible)
                self.assertEqual(set(selector.options), {"bird", "hare"})

                selector.set_value(["hare"])
                self.assertEqual(app.review.pending, [0])
                self.assertEqual(app.current_index, 0)
                self.assertEqual(set(selector.options), {"bird", "hare"})

                selector.set_value(["bird", "hare"])
                self.assertEqual(app.review.pending, [])
                self.assertEqual(app.counter.text, "0 images")
                self.assertIsNone(app.current_index)

                selector.set_value(None)
                self.assertEqual(set(app.review.pending), {0, 2})

                selector.set_value(["hare"])
                app.change_mode("Unseen")
                self.assertFalse(selector.visible)
                self.assertEqual(app.review.pending, [0])
                app.change_mode("Latest model incorrect")
                self.assertTrue(selector.visible)
                self.assertEqual(selector.value, ["hare"])
                self.assertEqual(app.review.pending, [0])

    def test_latest_incorrect_deduplicates_resolved_images_until_undo(self):
        with self.csv_path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fieldnames = reader.fieldnames
            rows = list(reader)
        rows.append(rows[2].copy())
        with self.csv_path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        run_name = "deepfaune_finetuned_test"
        with (
            patch.object(review_module, "image_paths", return_value={}),
            patch.object(
                review_module,
                "latest_incorrect_predictions",
                return_value=(run_name, {"camera1_incorrect.jpg"}),
            ),
            patch.object(review_module, "load_validation_predictions", return_value={}),
        ):
            review = LabelReview(self.csv_path)
        review.set_mode("Latest model incorrect")

        self.assertEqual(
            [review.rows[index]["image_file"] for index in review.pending],
            ["incorrect.jpg"],
        )
        review.choose("hare")
        self.assertEqual(review.pending, [])
        review.set_mode("Unseen")
        review.set_mode("Latest model incorrect")
        self.assertEqual(review.pending, [])

        review.undo()

        self.assertEqual(
            [review.rows[index]["image_file"] for index in review.pending],
            ["incorrect.jpg"],
        )

    def test_current_inference_error_remains_when_validation_matches_manual_label(self):
        review = LabelReview(self.csv_path)
        filename = "camera1_incorrect.jpg"
        review.latest_incorrect_filenames.add(filename)
        review.latest_predictions[filename] = {
            "true_label": "bird",
            "predicted_label": "hare",
            "status": "incorrect",
        }

        review.set_mode("Latest model incorrect")

        self.assertEqual(review.pending, [2])
        review.rows[2]["class"] = "hare"
        review.set_mode("Latest model incorrect")
        self.assertEqual(review.pending, [])

    def test_reviewed_model_persists_and_new_run_starts_fresh(self):
        human_csv = Path(self.temp_dir.name) / "labels" / "human_labels.csv"
        human_csv.parent.mkdir()
        with human_csv.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(
                target,
                fieldnames=["location_id", "image_file", "box", "manual_label"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "location_id": "camera1",
                    "image_file": "incorrect.jpg",
                    "box": "",
                    "manual_label": "hare",
                }
            )
        key = "camera1_incorrect.jpg"
        prediction = {
            "location_id": "camera1",
            "image_file": "incorrect.jpg",
            "true_label": "hare",
            "predicted_label": "fox",
            "status": "incorrect",
        }

        def open_review(run_name):
            weights = human_csv.parent.parent / "models" / run_name / "weights.pt"
            weights.parent.mkdir(parents=True, exist_ok=True)
            weights.touch()
            predictions = review_module.label_data.predictions_path(
                human_csv.parent.parent, run_name
            )
            if not predictions.exists():
                review_module.label_data.write_records(
                    predictions,
                    [{"location_id": "camera1", "image_file": "incorrect.jpg"}],
                    review_module.label_data.PREDICTION_COLUMNS,
                )
            with (
                patch.object(review_module, "image_paths", return_value={}),
                patch.object(
                    review_module,
                    "latest_incorrect_predictions",
                    return_value=(run_name, {key}),
                ),
                patch.object(
                    review_module,
                    "load_validation_predictions",
                    return_value={key: prediction},
                ),
            ):
                return LabelReview(human_csv)

        first_run = "deepfaune_finetuned_20261101-120000_first"
        review = open_review(first_run)
        review.set_mode("Latest model incorrect")
        self.assertEqual(review.pending, [0])
        human_before_review = human_csv.read_bytes()
        review.choose("hare")
        self.assertEqual(human_csv.read_bytes(), human_before_review)

        saved = review_module.label_data.read_records(human_csv)
        self.assertNotIn("reviewed_model", saved[0])
        self.assertEqual(tuple(saved[0]), review_module.label_data.HUMAN_COLUMNS)
        first_predictions = review_module.label_data.predictions_path(
            human_csv.parent.parent, first_run
        )
        self.assertEqual(
            review_module.label_data.read_records(first_predictions)[0]["reviewed"],
            "true",
        )

        reopened = open_review(first_run)
        reopened.set_mode("Latest model incorrect")
        self.assertEqual(reopened.pending, [])

        second_run = "deepfaune_finetuned_20261102-120000_second"
        fresh_review = open_review(second_run)
        fresh_review.set_mode("Latest model incorrect")
        self.assertEqual(fresh_review.pending, [0])
        fresh_review.choose("marten")
        fresh_review.undo()
        self.assertEqual(fresh_review.pending, [0])
        saved = review_module.label_data.read_records(human_csv)
        self.assertNotIn("reviewed_model", saved[0])
        second_predictions = review_module.label_data.predictions_path(
            human_csv.parent.parent, second_run
        )
        self.assertEqual(
            review_module.label_data.read_records(second_predictions)[0]["reviewed"],
            "",
        )
        self.assertEqual(
            review_module.label_data.read_records(first_predictions)[0]["reviewed"],
            "true",
        )

    def test_prediction_review_save_failure_keeps_human_annotation_and_queue(self):
        human_csv = Path(self.temp_dir.name) / "labels/human_labels.csv"
        label_data = review_module.label_data
        label_data.write_records(
            human_csv,
            [
                {
                    "location_id": "camera1",
                    "image_file": "incorrect.jpg",
                    "manual_label": "hare",
                }
            ],
            label_data.HUMAN_COLUMNS,
        )
        with (
            patch.object(review_module, "image_paths", return_value={}),
            patch.object(
                review_module,
                "latest_incorrect_predictions",
                return_value=("deepfaune_finetuned_test", {"camera1_incorrect.jpg"}),
            ),
            patch.object(
                review_module,
                "load_validation_predictions",
                return_value={
                    "camera1_incorrect.jpg": {
                        "true_label": "hare",
                        "predicted_label": "fox",
                        "status": "incorrect",
                    }
                },
            ),
        ):
            review = LabelReview(human_csv)
        review.set_mode("Latest model incorrect")
        original = human_csv.read_bytes()
        with (
            patch.object(
                label_data, "save_prediction_reviews", side_effect=OSError("disk full")
            ),
            patch.object(
                Path, "write_bytes", side_effect=AssertionError("non-atomic rollback")
            ),
            self.assertRaisesRegex(OSError, "disk full"),
        ):
            review.choose("fox")
        self.assertEqual(human_csv.read_bytes(), original)
        self.assertEqual(review.rows[0]["manual_label"], "hare")
        self.assertEqual(review.pending, [0])
        self.assertEqual(review.resolved_incorrect, set())
        self.assertEqual(review.history, [])

        prediction = review.review_predictions_path
        label_data.write_records(
            prediction,
            [{"location_id": "camera1", "image_file": "incorrect.jpg", "class": "cat"}],
            label_data.PREDICTION_COLUMNS,
        )
        external = prediction.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "Predictions changed outside"):
            review.choose("fox")
        self.assertEqual(human_csv.read_bytes(), original)
        self.assertEqual(prediction.read_bytes(), external)
        self.assertEqual(review.rows[0]["manual_label"], "hare")

    def test_gallery_and_review_share_pending_errors_and_review_flags(self):
        from web import gallery

        data = review_module.label_data
        root = Path(self.temp_dir.name)
        human = root / "labels/human_labels.csv"
        run = "deepfaune_finetuned_20261102-120000_shared"
        weights = root / "models" / run / "weights.pt"
        weights.parent.mkdir(parents=True)
        weights.touch()
        humans = [
            {"location_id": "camera1", "image_file": name, "manual_label": "fox"}
            for name in (
                "inference.jpg",
                "validation.jpg",
                "reviewed.jpg",
                "correct.jpg",
            )
        ]
        data.write_records(human, [*humans, humans[0].copy()], data.HUMAN_COLUMNS)
        data.write_records(
            data.predictions_path(root, run),
            [
                {
                    **row,
                    "class": "hare"
                    if row["image_file"] in {"inference.jpg", "reviewed.jpg"}
                    else "fox",
                    "reviewed": "true" if row["image_file"] == "reviewed.jpg" else "",
                }
                for row in humans
            ],
            data.PREDICTION_COLUMNS,
        )
        data.write_records(
            data.validation_predictions_dir(root, run) / "predictions.csv",
            [
                {
                    "location_id": "camera1",
                    "image_file": "validation.jpg",
                    "true_label": "fox",
                    "predicted_label": "hare",
                    "status": "incorrect",
                }
            ],
            ("location_id", "image_file", "true_label", "predicted_label", "status"),
        )

        def gallery_names():
            matches = gallery.filter_rows(
                gallery.load_rows(human), gallery.parse_filter_prompt("misclassified")
            )
            names = {row["image_file"] for row in matches}
            self.assertEqual(len(matches), len(names))
            return names

        with patch.object(review_module, "image_paths", return_value={}):
            review = LabelReview(human)
        review.set_mode("Latest model incorrect")
        expected = {"inference.jpg", "validation.jpg"}
        self.assertEqual(
            {review.rows[index]["image_file"] for index in review.pending}, expected
        )
        self.assertEqual(gallery_names(), expected)
        self.assertEqual(review.latest_model_name, run)
        review.choose("fox")
        self.assertEqual(gallery_names(), {"validation.jpg"})
        with patch.object(review_module, "image_paths", return_value={}):
            reopened = LabelReview(human)
        reopened.set_mode("Latest model incorrect")
        self.assertEqual(
            [reopened.rows[index]["image_file"] for index in reopened.pending],
            ["validation.jpg"],
        )
        review.undo()
        self.assertEqual(gallery_names(), expected)
        self.assertEqual(
            {review.rows[index]["image_file"] for index in review.pending}, expected
        )
        review.choose("fox")
        newer = "deepfaune_finetuned_20261102-130000_new"
        newer_weights = root / "models" / newer / "weights.pt"
        newer_weights.parent.mkdir(parents=True)
        newer_weights.touch()
        data.write_records(
            data.predictions_path(root, newer),
            [
                {
                    **row,
                    "class": "hare"
                    if row["image_file"] in {"inference.jpg", "reviewed.jpg"}
                    else "fox",
                }
                for row in humans
            ],
            data.PREDICTION_COLUMNS,
        )
        with patch.object(review_module, "image_paths", return_value={}):
            fresh = LabelReview(human)
        fresh.set_mode("Latest model incorrect")
        self.assertEqual(fresh.latest_model_name, newer)
        expected_new = {"inference.jpg", "reviewed.jpg"}
        self.assertEqual(gallery_names(), expected_new)
        self.assertEqual(
            {fresh.rows[index]["image_file"] for index in fresh.pending}, expected_new
        )

    def test_pretrained_model_review_flags_persist_without_finetuned_weights(self):
        from web import gallery

        data = review_module.label_data
        root = Path(self.temp_dir.name)
        human = root / "labels/human_labels.csv"
        row = {
            "location_id": "camera1",
            "image_file": "photo.jpg",
            "manual_label": "fox",
        }
        data.write_records(human, [row], data.HUMAN_COLUMNS)
        data.write_records(
            data.predictions_path(root),
            [{**row, "class": "hare"}],
            data.PREDICTION_COLUMNS,
        )
        with (
            patch.object(review_module, "image_paths", return_value={}),
            patch.object(
                review_module,
                "latest_incorrect_predictions",
                return_value=(None, set()),
            ),
            patch.object(review_module, "load_validation_predictions", return_value={}),
        ):
            review = LabelReview(human)
        review.set_mode("Latest model incorrect")
        self.assertEqual(review.pending, [0])
        review.choose("fox")
        self.assertEqual(
            data.read_records(data.predictions_path(root))[0]["reviewed"], "true"
        )
        self.assertEqual(
            gallery.filter_rows(
                gallery.load_rows(human), gallery.parse_filter_prompt("misclassified")
            ),
            [],
        )

    def test_auto_label_unseen_button_is_not_rendered(self):
        app = review_app_module.ReviewApp(self.csv_path)
        with review_app_module.ui.column() as container:
            app.build()
        self.addCleanup(container.delete)

        self.assertFalse(
            any(
                getattr(element, "text", None) == "Auto-label unseen"
                for element in container.descendants()
            )
        )

    def test_images_are_reviewed_oldest_first_and_undated_last(self):
        with self.csv_path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(
                target,
                fieldnames=[
                    "location_id",
                    "image_file",
                    "timestamp_exif",
                    "timestamp",
                    "class",
                ],
            )
            writer.writeheader()
            for name, taken in (
                ("new.jpg", "2026-11-01T10:00:00"),
                ("undated.jpg", ""),
                ("old.jpg", "2024-08-10T02:05:15"),
                ("mid.jpg", "2025-06-01T12:00:00+02:00"),
            ):
                writer.writerow(
                    {
                        "location_id": "camera1",
                        "image_file": name,
                        "timestamp_exif": taken,
                        "timestamp": "2023-01-01T00:00:00"
                        if name == "undated.jpg"
                        else "",
                        "class": "fox",
                    }
                )
        review = LabelReview(self.csv_path)

        names = [review.rows[index]["image_file"] for index in review.pending]

        self.assertEqual(names, ["old.jpg", "mid.jpg", "new.jpg", "undated.jpg"])
        review.choose("fox")
        self.assertEqual(
            review.rows[review.pending[review.position]]["image_file"], "mid.jpg"
        )

    def test_unknown_manual_label_is_seen(self):
        with self.csv_path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fieldnames = reader.fieldnames
            rows = list(reader)
        rows.append(
            {
                "location_id": "camera1",
                "image_file": "unknown.jpg",
                "class": "",
                "box": "",
                "manual_label": "unknown",
            }
        )
        with self.csv_path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        review = LabelReview(self.csv_path)

        self.assertEqual(review.pending, [0])
        review.set_mode("Seen")
        self.assertEqual(review.pending, [1, 2, 3])

    def test_downloads_without_predictions_are_unseen_and_can_be_reviewed(self):
        original = Path(self.temp_dir.name) / "download.jpg"
        Image.new("RGB", (40, 30)).save(original)
        indexed_paths = {
            ("camera1", "correct.jpg"): original,
            ("camera1", "download.jpg"): original,
        }
        before = self.csv_path.read_bytes()
        with patch.object(review_module, "image_paths", return_value=indexed_paths):
            review = LabelReview(self.csv_path)

            self.assertEqual(review.pending, [0, 3])
            self.assertEqual(len(review.rows), 4)
            self.assertEqual(review.rows[1]["manual_label"], "fox")
            row = review.rows[3]
            self.assertEqual(row["image_file"], "download.jpg")
            self.assertEqual(row["manual_label"], "")
            self.assertEqual(row["class"], "")
            self.assertEqual(review.original_path(row), original)
            self.assertEqual(self.csv_path.read_bytes(), before)

            review.position = 1
            review.choose("hare")
            reopened = LabelReview(self.csv_path)
            self.assertEqual(len(reopened.rows), 4)
            self.assertEqual(reopened.rows[3]["manual_label"], "hare")
            self.assertEqual(reopened.pending, [0])
            review.undo()
            self.assertEqual(review.pending, [0, 3])

    def test_subject_focus_persists_between_images_and_handles_missing_boxes(self):
        app = review_app_module.ReviewApp(self.csv_path)
        with review_app_module.ui.column() as container:
            with patch.object(app, "image_path_for", return_value=None):
                app.build()
        self.addCleanup(container.delete)
        with patch.object(app, "image_path_for", return_value=None):
            self.assertFalse(app.focus_subject)
            app.toggle_subject_focus()
            self.assertTrue(app.viewer._props["focusSubject"])
            app.change_mode("All")
            app.choose_label("fox")
            self.assertEqual(app.current_index, 1)
            self.assertTrue(app.focus_subject)
            self.assertEqual(app.viewer._props["subject"], [0.5, 0.5, 0.4, 0.4])
            self.assertEqual(app.viewer._props["imageKey"], "camera1/correct.jpg")

            app.review.rows[1]["box"] = ""
            app.refresh()
            self.assertIsNone(app.viewer._props["subject"])
            self.assertTrue(app.viewer._props["focusSubject"])

            app.fit_image()
            self.assertFalse(app.focus_subject)
            self.assertFalse(app.viewer._props["focusSubject"])

    def test_pan_mode_does_not_edit_bounding_boxes(self):
        app = review_app_module.ReviewApp(self.csv_path)
        with review_app_module.ui.column() as container:
            with patch.object(app, "image_path_for", return_value=None):
                app.build()
        self.addCleanup(container.delete)
        app.image_size = (100, 100)
        original_box = app.review.rows[0]["box"]
        app.set_image_mode("pan")
        with patch.object(app.review, "save") as save:
            app.on_image_mouse(
                SimpleNamespace(type="mousedown", image_x=10, image_y=10)
            )
            app.on_image_mouse(SimpleNamespace(type="mouseup", image_x=80, image_y=80))
        save.assert_not_called()
        self.assertEqual(app.review.rows[0]["box"], original_box)
        self.assertIsNone(app.drag_start)
        app.set_image_mode("box")
        self.assertEqual(app.viewer._props["interactionMode"], "box")

    def test_zoomed_overlay_keeps_four_pixel_outline(self):
        row = {"class": "fox", "box": "[0.5, 0.5, 0.4, 0.4]"}
        overlay = review_app_module.box_overlay_svg(
            row, (2000, 1500), display_size=(2400, 1800), zoom_scale=3
        )
        root = ET.fromstring(f"<svg>{overlay}</svg>")
        self.assertAlmostEqual(
            float(root.find("rect").get("stroke-width")) * 3, 4, places=4
        )
        self.assertEqual(root.find("g/text").get("font-size"), "16")

    def test_browser_label_buttons_sort_game_species_first_and_hide_vehicle(self):
        with self.csv_path.open("a", newline="", encoding="utf-8") as target:
            csv.writer(target).writerows(
                [
                    ["camera1", "deer.jpg", "red deer", "", "red deer"],
                    ["camera1", "mouflon.jpg", "mouflon", "", "mouflon"],
                    ["camera1", "custom.jpg", "custom animal", "", ""],
                ]
            )
        app = review_app_module.ReviewApp(self.csv_path)
        with review_app_module.ui.column() as container:
            with patch.object(app, "image_path_for", return_value=None):
                app.build()
        self.addCleanup(container.delete)
        labels = [
            button.text.split(" ", 1)[1].rsplit(" [", 1)[0]
            for button in app.label_buttons
        ]
        available = set(app.review.shortcuts.values())
        expected = (
            sorted(available & review_app_module.GAME_SPECIES)
            + sorted(
                available
                - review_app_module.GAME_SPECIES
                - {"vehicle", "unknown", "none"}
            )
            + ["unknown", "none"]
        )
        self.assertEqual(labels, expected)
        self.assertNotIn("vehicle", labels)

    def test_browser_label_buttons_include_unique_nonempty_manual_classes(self):
        with self.csv_path.open("a", newline="", encoding="utf-8") as target:
            csv.writer(target).writerows(
                [
                    ["camera1", "manual.jpg", "machine class", "", " red deer "],
                    ["camera1", "duplicate.jpg", "fox", "", "red deer"],
                    ["camera1", "blank.jpg", "fox", "", "   "],
                ]
            )
        app = review_app_module.ReviewApp(self.csv_path)
        with review_app_module.ui.column() as container:
            with patch.object(app, "image_path_for", return_value=None):
                app.build()
        self.addCleanup(container.delete)
        captions = [button.text for button in app.label_buttons]

        self.assertEqual(captions.count("🦌 red deer [i]"), 1)
        self.assertEqual(captions.count("🦊 fox [f]"), 1)
        self.assertIn("🐾 machine class [e]", captions)
        self.assertTrue(all(caption.strip() for caption in captions))
        self.assertNotIn(" red deer ", captions)
        self.assertEqual(app.accept_button.text, "✅ Accept prediction")
        self.assertIsNotNone(getattr(app, "keep_manual_button", None))
        self.assertIs(
            app.accept_button.parent_slot.parent,
            app.keep_manual_button.parent_slot.parent,
        )
        self.assertFalse(app.keep_manual_button.visible)
        self.assertEqual(app.undo_button.text, "↩️ Undo label")
        self.assertEqual(app.undo_box_button.text, "↩️ Undo box")
        self.assertIs(
            app.undo_button.parent_slot.parent,
            app.undo_box_button.parent_slot.parent,
        )
        self.assertEqual(app.remove_irrelevant_button.text, "🗑️ Remove irrelevant")
        self.assertEqual(app.save_label_button.text, "💾 Save label")

    def test_browser_label_buttons_hide_legacy_birds_and_shortcut_all_choices(self):
        with self.csv_path.open("a", newline="", encoding="utf-8") as target:
            csv.writer(target).writerows(
                [
                    ["camera1", "woodpecker.jpg", "woodpecker", "", ""],
                    ["camera1", "buzzard.jpg", "buzzard", "", ""],
                    ["camera1", "falcon.jpg", "falcon", "", ""],
                    ["camera1", "songbird.jpg", "song bird", "", ""],
                    ["camera1", "owl.jpg", "owl", "", ""],
                    ["camera1", "marten.jpg", "marten", "", ""],
                    ["camera1", "generic-bird.jpg", "bird", "", ""],
                ]
            )
        app = review_app_module.ReviewApp(self.csv_path)
        with review_app_module.ui.column() as container:
            with patch.object(app, "image_path_for", return_value=None):
                app.build()
        self.addCleanup(container.delete)

        captions = [button.text for button in app.label_buttons]

        self.assertTrue(
            all(" [" in caption and caption.endswith("]") for caption in captions)
        )
        shortcuts = [caption.rsplit("[", 1)[1][:-1] for caption in captions]
        self.assertIn("🐦 bird [g]", captions)
        self.assertIn("🦦 marten [m]", captions)
        self.assertTrue(
            all(
                not any(
                    legacy in caption
                    for legacy in (
                        "woodpecker",
                        "buzzard",
                        "falcon",
                        "song bird",
                        "owl",
                    )
                )
                for caption in captions
            )
        )
        self.assertEqual(len(shortcuts), len(set(shortcuts)))

        chosen = []
        app.choose_label = chosen.append
        event = SimpleNamespace(
            action=SimpleNamespace(keydown=True, repeat=False),
            key=SimpleNamespace(name="g", space=False, backspace=False, enter=False),
            modifiers=SimpleNamespace(alt=False, ctrl=False, meta=False),
        )
        app.on_key(event)

        self.assertEqual(chosen, ["bird"])

    def test_unseen_image_omits_prediction_prefix_in_overlay_and_sidebar(self):
        app = review_app_module.ReviewApp(self.csv_path)
        with review_app_module.ui.column() as container:
            with (
                patch.object(app, "image_path_for", return_value=None),
                patch.object(app, "prediction_for_row", return_value=None),
            ):
                app.build()
                self.addCleanup(container.delete)
                self.assertEqual(app.prediction.text, "🏷️ 🦊 fox")
                overlay = review_app_module.box_overlay_svg(
                    app.review.rows[0], (4000, 3000)
                )
                root = ET.fromstring(f"<svg>{overlay}</svg>")
                self.assertEqual("".join(root.find("g/text").itertext()), "🏷️ 🦊 fox")

                app.change_mode("Seen")
                self.assertEqual(app.prediction.text, "🏷️ Predicted: 🦊 fox")

    def test_missing_model_prediction_is_not_shown_as_predicted_in_sidebar(self):
        csv_path = Path(self.temp_dir.name) / "missing-prediction.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(
                target,
                fieldnames=["location_id", "image_file", "class", "manual_label"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "location_id": "camera1",
                    "image_file": "missing.jpg",
                    "class": "",
                    "manual_label": "fox",
                }
            )
        app = review_app_module.ReviewApp(csv_path)
        with review_app_module.ui.column() as container:
            with patch.object(app, "image_path_for", return_value=None):
                app.build()
                app.change_mode("Seen")
        self.addCleanup(container.delete)

        self.assertEqual(app.prediction.text, "🏷️ True: 🦊 fox")
        self.assertNotIn("Predicted:", app.prediction.text)
        self.assertNotIn("🐾", app.prediction.text)

    def test_unseen_mode_hides_validation_truth_but_incorrect_mode_shows_it(self):
        app = review_app_module.ReviewApp(self.csv_path)
        filename = "camera1_unchecked.jpg"
        prediction = {
            "true_label": "hare",
            "predicted_label": "bird",
            "status": "incorrect",
        }
        app.review.latest_predictions[filename] = prediction
        app.review.latest_incorrect_filenames.add(filename)
        image_path = Path(self.temp_dir.name) / "preview.png"
        Image.new("RGB", (64, 64)).save(image_path)

        with review_app_module.ui.column() as container:
            with (
                patch.object(app, "image_path_for", return_value=image_path),
                patch.object(
                    review_app_module,
                    "box_overlay_svg",
                    wraps=review_app_module.box_overlay_svg,
                ) as render_overlay,
            ):
                app.build()
                self.addCleanup(container.delete)
                self.assertEqual(app.prediction.text, "🏷️ 🦊 fox")
                self.assertIsNone(render_overlay.call_args.args[2])

                app.change_mode("Latest model incorrect")

                self.assertEqual(
                    app.prediction.text,
                    "🏷️ True: 🐇 hare | Predicted: 🐦 bird",
                )
                self.assertEqual(render_overlay.call_args.args[2], prediction)

    def test_inference_error_shows_current_prediction_when_validation_is_correct(self):
        app = review_app_module.ReviewApp(self.csv_path)
        row = app.review.rows[2]
        app.review.latest_predictions["camera1_incorrect.jpg"] = {
            "true_label": "hare",
            "predicted_label": "hare",
            "status": "correct",
        }
        for mode in ("Latest model incorrect", "Unseen + errors"):
            with self.subTest(mode=mode):
                app.review.set_mode(mode)
                prediction = app.prediction_for_row(row)
                self.assertEqual(prediction["predicted_label"], "fox")
                self.assertEqual(prediction["true_label"], "hare")

    def test_browser_model_name_is_visible_only_in_latest_model_modes(self):
        run_name = "deepfaune_finetuned_20261020-191127_latest"
        with (
            patch.object(
                review_module,
                "latest_incorrect_predictions",
                return_value=(run_name, {"camera1_correct.jpg"}),
            ),
            patch.object(review_module, "load_validation_predictions", return_value={}),
        ):
            app = review_app_module.ReviewApp(self.csv_path)
        with review_app_module.ui.column() as container:
            with patch.object(app, "image_path_for", return_value=None):
                app.build()
        self.addCleanup(container.delete)
        header_items = app.mode_select.parent_slot.children
        run_label = app.run_label

        self.assertIs(header_items[-3], run_label)
        self.assertIs(header_items[-2], app.mode_select)
        self.assertIs(header_items[-1], app.exclude_true_labels_select)
        self.assertEqual(run_label.text, run_name)
        self.assertFalse(run_label.visible)
        with patch.object(app, "image_path_for", return_value=None):
            for mode in (
                "Latest model incorrect",
                "Unseen",
                "Unseen + errors",
                "Seen",
                "All",
            ):
                with self.subTest(mode=mode):
                    app.change_mode(mode)
                    self.assertEqual(run_label.visible, mode.startswith("Latest model"))

    def test_unseen_mode_advances_after_labeling_removes_current_row(self):
        csv_path = Path(self.temp_dir.name) / "unseen.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(
                target,
                fieldnames=["location_id", "image_file", "class", "manual_label"],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "location_id": "camera1",
                        "image_file": "first.jpg",
                        "class": "fox",
                        "manual_label": "",
                    },
                    {
                        "location_id": "camera1",
                        "image_file": "second.jpg",
                        "class": "hare",
                        "manual_label": "",
                    },
                ]
            )
        review = LabelReview(csv_path)

        review.choose("fox")

        self.assertEqual(review.pending, [1])
        self.assertEqual(review.position, 0)

    def test_drag_box_normalizes_reversed_and_out_of_bounds_coordinates(self):
        box = review_module.box_from_drag((120, 60), (-20, -10), (100, 80))

        self.assertEqual(box, (0.5, 0.375, 1.0, 0.75))

    def test_drag_box_rejects_tiny_selections(self):
        self.assertIsNone(review_module.box_from_drag((10, 20), (12, 25), (100, 80)))

    def test_latest_incorrect_predictions_come_from_newest_run(self):
        models_dir = Path(self.temp_dir.name) / "models"
        older = models_dir / "deepfaune_finetuned_20261019-200723_old"
        latest = models_dir / "deepfaune_finetuned_20261020-191127_new"
        for run_dir, filename in (
            (older, "camera1_old.jpg"),
            (latest, "camera1_latest.jpg"),
        ):
            run_dir.mkdir(parents=True)
            incorrect_dir = (
                models_dir.parent
                / "labels"
                / f"labels_{run_dir.name}"
                / "validation_predictions"
                / "incorrect"
            )
            incorrect_dir.mkdir(parents=True)
            (incorrect_dir / filename).touch()

        run_name, filenames = review_module.latest_incorrect_predictions(models_dir)

        self.assertEqual(run_name, latest.name)
        self.assertEqual(filenames, {"camera1_latest.jpg"})

    def test_latest_run_without_errors_does_not_fall_back_to_older_run(self):
        models_dir = Path(self.temp_dir.name) / "models"
        older = models_dir / "deepfaune_finetuned_20261019-200723_old"
        older.mkdir(parents=True)
        incorrect_dir = (
            models_dir.parent
            / "labels"
            / f"labels_{older.name}"
            / "validation_predictions"
            / "incorrect"
        )
        incorrect_dir.mkdir(parents=True)
        (incorrect_dir / "camera1_old.jpg").touch()
        latest = models_dir / "deepfaune_finetuned_20261020-191127_latest"
        latest.mkdir(parents=True)
        validation_dir = (
            models_dir.parent
            / "labels"
            / f"labels_{latest.name}"
            / "validation_predictions"
        )
        validation_dir.mkdir(parents=True)
        with (validation_dir / "predictions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as target:
            writer = csv.DictWriter(
                target,
                fieldnames=[
                    "location_id",
                    "image_file",
                    "true_label",
                    "predicted_label",
                    "status",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "location_id": "camera1",
                    "image_file": "correct.jpg",
                    "true_label": "fox",
                    "predicted_label": "fox",
                    "status": "correct",
                }
            )

        run_name, filenames = review_module.latest_incorrect_predictions(models_dir)

        self.assertEqual(run_name, latest.name)
        self.assertEqual(filenames, set())

    def test_latest_validation_predictions_are_loaded_for_csv_rows(self):
        models_dir = Path(self.temp_dir.name) / "models"
        run_dir = models_dir / "deepfaune_finetuned_20261020-191127_latest"
        run_dir.mkdir(parents=True)
        validation_dir = (
            models_dir.parent
            / "labels"
            / f"labels_{run_dir.name}"
            / "validation_predictions"
        )
        incorrect_dir = validation_dir / "incorrect"
        incorrect_dir.mkdir(parents=True)
        (incorrect_dir / "camera1_incorrect.jpg").touch()
        with (validation_dir / "predictions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as target:
            writer = csv.DictWriter(
                target,
                fieldnames=[
                    "location_id",
                    "image_file",
                    "true_label",
                    "predicted_label",
                    "status",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "location_id": "camera1",
                    "image_file": "incorrect.jpg",
                    "true_label": "crow",
                    "predicted_label": "bird",
                    "status": "incorrect",
                }
            )

        with patch.object(review_module, "MODELS_DIR", models_dir):
            review = LabelReview(self.csv_path)

        self.assertEqual(
            review.latest_predictions["camera1_incorrect.jpg"]["predicted_label"],
            "bird",
        )

    def test_box_overlay_displays_latest_finetuned_prediction(self):
        row = {
            "class": "bird",
            "manual_label": "bird",
            "box": "[0.5, 0.5, 0.4, 0.4]",
        }
        prediction = {
            "true_label": "bird",
            "predicted_label": "pigeon",
            "status": "incorrect",
        }

        overlay = review_app_module.box_overlay_svg(row, (4000, 3000), prediction)

        self.assertIn("🏷️ True: 🐦 bird | Predicted: 🕊️ pigeon", overlay)
        self.assertNotIn("🏷️ True: 🐦 bird | Predicted: 🐦 bird", overlay)
        self.assertIn('stroke-width="4"', overlay)
        self.assertIn('vector-effect="non-scaling-stroke"', overlay)
        self.assertIn('font-size="16"', overlay)

    def test_box_overlay_omits_prediction_when_latest_model_has_none(self):
        overlay = review_app_module.box_overlay_svg(
            {
                "class": "",
                "manual_label": "fox",
                "box": "[0.5, 0.5, 0.4, 0.4]",
            },
            (400, 300),
        )

        self.assertIn("True: 🦊 fox", overlay)
        self.assertNotIn("Predicted:", overlay)
        self.assertNotIn("🐾", overlay)

    def test_box_overlay_draws_additional_boxes_without_labels(self):
        overlay = review_app_module.box_overlay_svg(
            {
                "class": "roe deer",
                "manual_label": "",
                "box": "[0.5, 0.5, 0.2, 0.2]",
                "additional_boxes": "[[0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]]",
            },
            (400, 300),
        )
        root = ET.fromstring(f"<svg>{overlay}</svg>")

        self.assertEqual(len(root.findall("rect")), 3)
        self.assertEqual(len(root.findall("g")), 1)
        self.assertEqual(root.find("rect").get("x"), "160.0")
        self.assertEqual(
            review_app_module.row_boxes({"box": "", "additional_boxes": "[[0.5]]"}),
            [],
        )

    def test_box_overlay_keeps_annotation_size_in_display_pixels(self):
        row = {
            "class": "fox",
            "manual_label": "hare",
            "box": "[0.5, 0.5, 0.4, 0.4]",
        }
        for image_size, display_size in (
            ((640, 480), (800, 600)),
            ((4000, 3000), (800, 600)),
            ((6000, 4000), (400, 300)),
        ):
            with self.subTest(image_size=image_size, display_size=display_size):
                overlay = review_app_module.box_overlay_svg(
                    row, image_size, display_size=display_size
                )
                root = ET.fromstring(f"<svg>{overlay}</svg>")
                outline = root.find("rect")
                annotation = root.find("g")
                self.assertEqual(outline.get("stroke-width"), "4")
                self.assertEqual(outline.get("vector-effect"), "non-scaling-stroke")
                scale_x = image_size[0] / display_size[0]
                scale_y = image_size[1] / display_size[1]
                self.assertIn(
                    f"scale({scale_x} {scale_y})", annotation.get("transform")
                )
                self.assertEqual(annotation.find("text").get("font-size"), "16")
                self.assertEqual(annotation.find("rect").get("height"), "24")

    def test_browser_updates_overlay_when_display_size_changes(self):
        app = review_app_module.ReviewApp(self.csv_path)
        event = SimpleNamespace(args={"width": 800, "height": 600})
        with patch.object(app, "refresh") as refresh:
            app.on_image_resize(event)
            self.assertEqual(app.display_size, (800, 600))
            refresh.assert_called_once()
            app.on_image_resize(event)
            refresh.assert_called_once()

    def test_box_overlay_wraps_long_annotation_without_shrinking_text(self):
        row = {
            "class": "fallow deer",
            "manual_label": "pigeon",
            "box": "[0.5, 0.5, 0.4, 0.4]",
        }
        overlay = review_app_module.box_overlay_svg(
            row, (4000, 3000), display_size=(248, 186)
        )
        root = ET.fromstring(f"<svg>{overlay}</svg>")
        annotation = root.find("g")
        text = annotation.find("text")
        self.assertEqual(text.get("font-size"), "16")
        self.assertEqual(len(text.findall("tspan")), 3)
        self.assertEqual(annotation.find("rect").get("height"), "72")
        self.assertEqual(
            " ".join(text.itertext()),
            "🏷️ True: 🕊️ pigeon | Predicted: 🦌 fallow deer",
        )

    def test_accept_prediction_uses_latest_finetuned_class(self):
        app = review_app_module.ReviewApp(self.csv_path)
        app.current_index = 0
        app.review.latest_predictions = {
            "camera1_unchecked.jpg": {
                "true_label": "fox",
                "predicted_label": "bird",
                "status": "incorrect",
            }
        }
        chosen = []
        app.choose_label = chosen.append

        app.accept()

        self.assertEqual(chosen, ["bird"])

    def test_incorrect_mode_shows_separate_accept_and_keep_manual_actions(self):
        app = review_app_module.ReviewApp(self.csv_path)
        incorrect = {
            "true_label": "hare",
            "predicted_label": "fox",
            "status": "incorrect",
        }
        app.review.latest_predictions = {"camera1_incorrect.jpg": incorrect}
        app.review.latest_incorrect_filenames.add("camera1_incorrect.jpg")
        with review_app_module.ui.column() as container:
            with patch.object(app, "image_path_for", return_value=None):
                app.build()
                self.addCleanup(container.delete)
                self.assertEqual(app.accept_button.text, "✅ Accept prediction")
                app.change_mode("Latest model incorrect")
                self.assertEqual(app.accept_button.text, "✅ Accept prediction")
                self.assertIsNotNone(getattr(app, "keep_manual_button", None))
                self.assertTrue(app.keep_manual_button.visible)
                chosen = []
                app.choose_label = chosen.append
                app.accept()
                app.on_key(
                    SimpleNamespace(
                        action=SimpleNamespace(keydown=True, repeat=False),
                        key=SimpleNamespace(
                            name=" ", space=True, backspace=False, enter=False
                        ),
                        modifiers=SimpleNamespace(alt=False, ctrl=False, meta=False),
                    )
                )

        self.assertEqual(chosen, ["fox", "hare"])

    def test_accept_without_prediction_saves_nothing(self):
        app = review_app_module.ReviewApp(self.csv_path)
        app.current_index = 0
        app.review.rows[0]["class"] = ""
        app.review.latest_predictions = {}
        chosen = []
        app.choose_label = chosen.append

        with patch.object(review_app_module.ui, "notify") as notify:
            app.accept()

        self.assertEqual(chosen, [])
        notify.assert_called_once()

    def test_space_shortcut_accepts_current_prediction(self):
        app = review_app_module.ReviewApp(self.csv_path)
        app.current_index = 0
        app.review.latest_predictions = {
            "camera1_unchecked.jpg": {
                "true_label": "fox",
                "predicted_label": "bird",
                "status": "incorrect",
            }
        }
        chosen = []
        app.choose_label = chosen.append
        event = SimpleNamespace(
            action=SimpleNamespace(keydown=True, repeat=False),
            key=SimpleNamespace(name=" ", space=True, backspace=False, enter=False),
            modifiers=SimpleNamespace(alt=False, ctrl=False, meta=False),
        )

        app.on_key(event)

        self.assertEqual(chosen, ["bird"])

    def test_held_space_reaches_client_guard_without_accepting_again(self):
        app = review_app_module.ReviewApp(self.csv_path)
        with review_app_module.ui.column() as container:
            with patch.object(app, "image_path_for", return_value=None):
                app.build()
        self.addCleanup(container.delete)
        self.assertTrue(app.keyboard._props["repeating"])
        event = SimpleNamespace(
            action=SimpleNamespace(keydown=True, repeat=True),
            key=SimpleNamespace(name=" ", space=True, backspace=False, enter=False),
            modifiers=SimpleNamespace(alt=False, ctrl=False, meta=False),
        )
        with patch.object(app, "accept") as accept:
            app.on_key(event)
        accept.assert_not_called()

    def test_keyboard_shortcuts_leave_focused_buttons_to_native_activation(self):
        app = review_app_module.ReviewApp(self.csv_path)
        with review_app_module.ui.column() as container:
            with patch.object(app, "image_path_for", return_value=None):
                app.build()
        self.addCleanup(container.delete)

        self.assertIn("button", app.keyboard._props["ignore"])

    def test_z_shortcut_toggles_subject_focus_without_changing_labels(self):
        app = review_app_module.ReviewApp(self.csv_path)
        with review_app_module.ui.column() as container:
            with patch.object(app, "image_path_for", return_value=None):
                app.build()
        self.addCleanup(container.delete)
        before = self.csv_path.read_bytes()
        with patch.object(app, "choose_label") as choose_label:
            for key_name, focused in (("z", True), ("Z", False)):
                event = SimpleNamespace(
                    action=SimpleNamespace(keydown=True, repeat=False),
                    key=SimpleNamespace(
                        name=key_name, space=False, backspace=False, enter=False
                    ),
                    modifiers=SimpleNamespace(alt=False, ctrl=False, meta=False),
                )
                app.on_key(event)
                self.assertEqual(app.focus_subject, focused)
                self.assertEqual(app.viewer._props["focusSubject"], focused)
            choose_label.assert_not_called()
        self.assertEqual(self.csv_path.read_bytes(), before)

    def test_backspace_shortcut_undoes_label(self):
        app = review_app_module.ReviewApp(self.csv_path)
        event = SimpleNamespace(
            action=SimpleNamespace(keydown=True, repeat=False),
            key=SimpleNamespace(
                name="Backspace", space=False, backspace=True, enter=False
            ),
            modifiers=SimpleNamespace(alt=False, ctrl=False, meta=False),
        )

        with patch.object(app, "undo_label") as undo_label:
            app.on_key(event)

        undo_label.assert_called_once_with()

    def test_subject_focus_tooltip_includes_z_shortcut(self):
        app = review_app_module.ReviewApp(self.csv_path)
        with review_app_module.ui.column() as container:
            with patch.object(app, "image_path_for", return_value=None):
                app.build()
        self.addCleanup(container.delete)
        tooltips = [
            element.text
            for element in app.subject_zoom_button._client().elements.values()
            if element.tag == "q-tooltip"
            and element.text.startswith("🎯 Focus subject")
        ]

        self.assertEqual(tooltips, ["🎯 Focus subject [Z]"])

    def test_letter_shortcut_selects_matching_label(self):
        app = review_app_module.ReviewApp(self.csv_path)
        chosen = []
        app.choose_label = chosen.append
        event = SimpleNamespace(
            action=SimpleNamespace(keydown=True, repeat=False),
            key=SimpleNamespace(name="r", space=False, backspace=False, enter=False),
            modifiers=SimpleNamespace(alt=False, ctrl=False, meta=False),
        )

        app.on_key(event)

        self.assertEqual(chosen, ["roe deer"])

    def test_modes_select_rows_and_preserve_current_row(self):
        models_dir = Path(self.temp_dir.name) / "models"
        (models_dir / "deepfaune_finetuned_20261020-191127_latest").mkdir(parents=True)
        incorrect_dir = (
            models_dir.parent
            / "labels"
            / "labels_deepfaune_finetuned_20261020-191127_latest"
            / "validation_predictions"
            / "incorrect"
        )
        incorrect_dir.mkdir(parents=True)
        (incorrect_dir / "camera1_correct.jpg").touch()

        with patch.object(review_module, "MODELS_DIR", models_dir):
            review = LabelReview(self.csv_path)

        self.assertEqual(review.pending, [0])
        review.set_mode("Latest model incorrect")
        self.assertEqual(review.pending, [1, 2])
        review.set_mode("Unseen + errors")
        self.assertEqual(review.pending, [0, 1, 2])
        self.assertEqual(review.pending[review.position], 1)
        review.set_mode("Seen")
        self.assertEqual(review.pending, [1, 2])
        self.assertEqual(review.position, 0)
        review.set_mode("All")
        self.assertEqual(review.pending, [0, 1, 2])
        review.position = 2
        review.set_mode("Latest model incorrect")
        self.assertEqual(review.pending, [1, 2])
        self.assertEqual(review.position, 1)

    def test_gui_dropdown_selects_review_mode(self):
        image_dir = self.csv_path.parent / "images"
        image_dir.mkdir()
        for image_name in ("unchecked", "correct", "incorrect"):
            Image.new("RGB", (64, 64), "black").save(
                image_dir / f"camera1_{image_name}.jpg"
            )

        models_dir = Path(self.temp_dir.name) / "models"
        (models_dir / "deepfaune_finetuned_20261020-191127_latest").mkdir(parents=True)
        incorrect_dir = (
            models_dir.parent
            / "labels"
            / "labels_deepfaune_finetuned_20261020-191127_latest"
            / "validation_predictions"
            / "incorrect"
        )
        incorrect_dir.mkdir(parents=True)
        (incorrect_dir / "camera1_correct.jpg").touch()

        with patch.object(review_module, "MODELS_DIR", models_dir):
            review = LabelReview(self.csv_path)
        with patch.object(review_module.plt, "show"):
            review.show()
        self.addCleanup(plt.close, review.figure)

        self.assertEqual(len(review.image_axes.images), 1)
        review.toggle_mode_menu()
        review.select_mode("Seen")

        self.assertEqual(review.pending, [1, 2])
        self.assertEqual(review.mode, "Seen")
        self.assertEqual(review.mode_button.label.get_text(), "Mode: Seen v")
        self.assertEqual(len(review.image_axes.images), 1)

    def test_box_and_annotation_are_green_and_keep_true_label_when_correct(self):
        review = LabelReview(self.csv_path)
        review.set_mode("All")
        figure, axes = plt.subplots()
        self.addCleanup(plt.close, figure)
        review.image_axes = axes

        review.draw_box_label(review.rows[1], (100, 100))

        self.assertEqual(axes.patches[0].get_edgecolor(), to_rgba("#2e7d32"))
        self.assertEqual(axes.texts[0].get_text(), "True: fox | Predicted: fox")

    def test_bounding_box_color_reflects_prediction_and_human_label(self):
        for predicted_label, manual_label, expected in (
            ("fox", "", "#f9a825"),
            ("", "fox", "#1565c0"),
            ("fox", "fox", "#2e7d32"),
            ("fox", "hare", "#c62828"),
        ):
            with self.subTest(
                predicted_label=predicted_label, manual_label=manual_label
            ):
                self.assertEqual(
                    review_module.bounding_box_color(predicted_label, manual_label),
                    expected,
                )

    def test_unchecked_boxes_are_yellow_in_static_and_browser_viewers(self):
        review = LabelReview(self.csv_path)
        row = {**review.rows[0], "box": "[0.5, 0.5, 0.4, 0.4]", "manual_label": ""}
        figure, axes = plt.subplots()
        self.addCleanup(plt.close, figure)
        review.image_axes = axes

        review.draw_box_label(row, (100, 100))

        self.assertEqual(axes.patches[0].get_edgecolor(), to_rgba("#f9a825"))
        for manual, color in (
            ("", "#f9a825"),
            (row["class"], "#2e7d32"),
            ("other species", "#c62828"),
        ):
            with self.subTest(manual=manual):
                overlay = review_app_module.box_overlay_svg(
                    {**row, "manual_label": manual}, (100, 100)
                )
                self.assertIn(f'stroke="{color}"', overlay)
        human_only_overlay = review_app_module.box_overlay_svg(
            {**row, "class": "", "manual_label": "fox"}, (100, 100)
        )
        self.assertIn('stroke="#1565c0"', human_only_overlay)
        overlay = review_app_module.box_overlay_svg(
            {**row, "manual_label": "corrected species"},
            (100, 100),
            prediction={
                "true_label": row["class"],
                "predicted_label": row["class"],
                "status": "correct",
            },
            display_size=(960, 540),
        )
        self.assertIn('stroke="#c62828"', overlay)
        self.assertIn("corrected species", overlay)

    def test_box_and_annotation_are_red_and_keep_true_label_when_incorrect(self):
        review = LabelReview(self.csv_path)
        review.set_mode("All")
        figure, axes = plt.subplots()
        self.addCleanup(plt.close, figure)
        review.image_axes = axes

        review.draw_box_label(review.rows[2], (100, 100))

        self.assertEqual(axes.patches[0].get_edgecolor(), to_rgba("#c62828"))
        self.assertEqual(axes.texts[0].get_text(), "True: hare | Predicted: fox")

    def test_static_box_label_omits_missing_prediction(self):
        review = LabelReview(self.csv_path)
        figure, axes = plt.subplots()
        self.addCleanup(plt.close, figure)
        review.image_axes = axes

        review.draw_box_label(
            {
                "class": "",
                "manual_label": "fox",
                "box": "[0.5, 0.5, 0.4, 0.4]",
            },
            (100, 100),
        )

        self.assertEqual([text.get_text() for text in axes.texts], ["True: fox"])

    def test_undo_restores_manual_label_that_existed_before_review(self):
        review = LabelReview(self.csv_path)
        review.set_mode("All")

        review.choose("fox")
        review.undo()

        self.assertEqual(review.rows[0]["manual_label"], "")
        self.assertEqual(review.rows[1]["manual_label"], "fox")

    def test_validation_results_are_added_to_training_data_without_images(self):
        finetune_classifier = import_module("src.labelling.finetune_classifier")

        data_dir = Path(self.temp_dir.name) / "training_data"
        data_dir.mkdir()
        pd.DataFrame(
            [
                ("camera1", "train.jpg", "fox", "train"),
                ("camera1", "correct.jpg", "fox", "val"),
                ("camera1", "incorrect.jpg", "hare", "val"),
            ],
            columns=["location_id", "image_file", "manual_label", "split"],
        ).to_csv(data_dir / "labels.csv", index=False)

        finetune_classifier.save_validation_predictions(
            data_dir, [0, 0], ["fox", "hare"]
        )

        with (data_dir / "labels.csv").open(encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual([row["predicted_label"] for row in rows], ["", "fox", "fox"])
        self.assertEqual([path.name for path in data_dir.iterdir()], ["labels.csv"])

    def test_validation_results_are_read_from_training_data(self):
        models_dir = Path(self.temp_dir.name) / "models"
        run_dir = models_dir / "deepfaune_finetuned_20261020-191127_latest"
        (run_dir / "training_data").mkdir(parents=True)
        pd.DataFrame(
            [
                ("camera1", "train.jpg", "fox", "train", ""),
                ("camera1", "right.jpg", "fox", "val", "fox"),
                ("camera1", "wrong.jpg", "hare", "val", "fox"),
            ],
            columns=[
                "location_id",
                "image_file",
                "manual_label",
                "split",
                "predicted_label",
            ],
        ).to_csv(run_dir / "training_data" / "labels.csv", index=False)

        run_name, incorrect = review_module.latest_incorrect_predictions(models_dir)
        predictions = review_module.load_validation_predictions(run_name, models_dir)

        self.assertEqual(run_name, run_dir.name)
        self.assertEqual(incorrect, {"camera1_wrong.jpg"})
        self.assertEqual(
            predictions["camera1_right.jpg"],
            {
                "location_id": "camera1",
                "image_file": "right.jpg",
                "true_label": "fox",
                "predicted_label": "fox",
                "status": "correct",
            },
        )
        self.assertNotIn("camera1_train.jpg", predictions)

    def test_latest_model_correct_mode_is_removed(self):
        review = LabelReview(self.csv_path)
        self.assertNotIn("Latest model correct", review_module.MODE_OPTIONS)
        with self.assertRaises(ValueError):
            review.set_mode("Latest model correct")


if __name__ == "__main__":
    unittest.main()
