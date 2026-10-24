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
                empty_images_module.shutil,
                "copy2",
                side_effect=OSError("backup failed"),
            ),
            self.assertRaisesRegex(OSError, "backup failed"),
        ):
            empty_images_module.remove_empty_images(
                csv_path=self.csv_path, annotated_dir=self.annotated_dir
            )
        self.assertTrue((self.original_dir / "empty.jpg").exists())
        self.assertTrue((self.annotated_dir / "camera1_empty.jpg").exists())

    def test_gui_remove_empty_reloads_active_csv_and_clears_undo(self):
        app = self.build_app()
        app.change_mode("All")
        app.review.history.append((0, "none"))
        app.box_history.append((0, ""))
        self.assertEqual(app.remove_empty_button.text, "Remove empty")

        with patch.object(review_app_module.ui, "notify"):
            asyncio.run(app.remove_empty())

        self.assertEqual(app.review.mode, "All")
        self.assertEqual([row["image_file"] for row in app.review.rows], self.names[1:])
        self.assertEqual(app.review.original_bytes, self.csv_path.read_bytes())
        self.assertEqual(app.review.history, [])
        self.assertEqual(app.box_history, [])
        self.assertFalse(app.undo_box_button.enabled)
        self.assertFalse(app.undo_button.enabled)

    def test_gui_remove_empty_requires_confirmation(self):
        app = self.build_app()
        before = self.csv_path.read_bytes()

        for confirmed in (False, True):

            async def answer():
                self.assertTrue(app.cleanup_dialog_open)
                return confirmed

            async def confirm():
                with app.remove_empty_button.parent_slot.parent:
                    await app.confirm_remove_empty()

            with (
                patch.object(
                    review_app_module.ui.dialog,
                    "__await__",
                    lambda dialog: answer().__await__(),
                ),
                patch.object(app, "remove_empty", new_callable=AsyncMock) as remove,
            ):
                asyncio.run(confirm())
                self.assertEqual(remove.await_count, int(confirmed))
                self.assertFalse(app.cleanup_dialog_open)
        self.assertEqual(self.csv_path.read_bytes(), before)
        self.assertTrue((self.original_dir / "empty.jpg").exists())

    def test_gui_remove_empty_reports_stale_csv_without_deleting_images(self):
        app = self.build_app()
        self.csv_path.write_bytes(self.csv_path.read_bytes() + b"\n")

        with patch.object(review_app_module.ui, "notify") as notify:
            asyncio.run(app.remove_empty())

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

        self.assertEqual(captions.count("red deer"), 1)
        self.assertEqual(captions.count("fox [f]"), 1)
        self.assertIn("machine class", captions)
        self.assertTrue(all(caption.strip() for caption in captions))
        self.assertNotIn(" red deer ", captions)

    def test_unseen_image_omits_prediction_prefix_in_overlay_and_sidebar(self):
        app = review_app_module.ReviewApp(self.csv_path)
        with review_app_module.ui.column() as container:
            with (
                patch.object(app, "image_path_for", return_value=None),
                patch.object(app, "prediction_for_row", return_value=None),
            ):
                app.build()
                self.addCleanup(container.delete)
                self.assertEqual(app.prediction.text, "fox")
                overlay = review_app_module.box_overlay_svg(
                    app.review.rows[0], (4000, 3000)
                )
                root = ET.fromstring(f"<svg>{overlay}</svg>")
                self.assertEqual("".join(root.find("g/text").itertext()), "fox")

                app.change_mode("Seen")
                self.assertEqual(app.prediction.text, "Predicted: fox")

    def test_browser_model_name_is_visible_only_in_latest_model_incorrect_mode(self):
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

        self.assertIs(header_items[-2], run_label)
        self.assertIs(header_items[-1], app.mode_select)
        self.assertEqual(run_label.text, run_name)
        self.assertFalse(run_label.visible)
        with patch.object(app, "image_path_for", return_value=None):
            for mode in (
                "Latest model incorrect",
                "Unseen",
                "Latest model incorrect",
                "Unseen + errors",
                "Seen",
                "All",
            ):
                with self.subTest(mode=mode):
                    app.change_mode(mode)
                    self.assertEqual(
                        run_label.visible, mode == "Latest model incorrect"
                    )

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
            incorrect_dir = run_dir / "validation_predictions" / "incorrect"
            incorrect_dir.mkdir(parents=True)
            (incorrect_dir / filename).touch()

        run_name, filenames = review_module.latest_incorrect_predictions(models_dir)

        self.assertEqual(run_name, latest.name)
        self.assertEqual(filenames, {"camera1_latest.jpg"})

    def test_latest_run_without_errors_does_not_fall_back_to_older_run(self):
        models_dir = Path(self.temp_dir.name) / "models"
        older = models_dir / "deepfaune_finetuned_20261019-200723_old"
        incorrect_dir = older / "validation_predictions" / "incorrect"
        incorrect_dir.mkdir(parents=True)
        (incorrect_dir / "camera1_old.jpg").touch()
        latest = models_dir / "deepfaune_finetuned_20261020-191127_latest"
        validation_dir = latest / "validation_predictions"
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
        validation_dir = run_dir / "validation_predictions"
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
                    "true_label": "jay",
                    "predicted_label": "song bird",
                    "status": "incorrect",
                }
            )

        with patch.object(review_module, "MODELS_DIR", models_dir):
            review = LabelReview(self.csv_path)

        self.assertEqual(
            review.latest_predictions["camera1_incorrect.jpg"]["predicted_label"],
            "song bird",
        )

    def test_box_overlay_displays_latest_finetuned_prediction(self):
        row = {
            "class": "jay",
            "manual_label": "jay",
            "box": "[0.5, 0.5, 0.4, 0.4]",
        }
        prediction = {
            "true_label": "jay",
            "predicted_label": "song bird",
            "status": "incorrect",
        }

        overlay = review_app_module.box_overlay_svg(row, (4000, 3000), prediction)

        self.assertIn("True: jay | Predicted: song bird", overlay)
        self.assertNotIn("True: jay | Predicted: jay", overlay)
        self.assertIn('stroke-width="4"', overlay)
        self.assertIn('vector-effect="non-scaling-stroke"', overlay)
        self.assertIn('font-size="16"', overlay)

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
            "class": "jay",
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
        self.assertEqual(len(text.findall("tspan")), 2)
        self.assertEqual(annotation.find("rect").get("height"), "48")
        self.assertEqual(" ".join(text.itertext()), "True: pigeon | Predicted: jay")

    def test_accept_prediction_uses_latest_finetuned_class(self):
        app = review_app_module.ReviewApp(self.csv_path)
        app.current_index = 0
        app.review.latest_predictions = {
            "camera1_unchecked.jpg": {
                "true_label": "fox",
                "predicted_label": "song bird",
                "status": "incorrect",
            }
        }
        chosen = []
        app.choose_label = chosen.append

        app.accept()

        self.assertEqual(chosen, ["song bird"])

    def test_space_shortcut_accepts_current_prediction(self):
        app = review_app_module.ReviewApp(self.csv_path)
        app.current_index = 0
        app.review.latest_predictions = {
            "camera1_unchecked.jpg": {
                "true_label": "fox",
                "predicted_label": "song bird",
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

        self.assertEqual(chosen, ["song bird"])

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
        incorrect_dir = (
            models_dir
            / "deepfaune_finetuned_20261020-191127_latest"
            / "validation_predictions"
            / "incorrect"
        )
        incorrect_dir.mkdir(parents=True)
        (incorrect_dir / "camera1_correct.jpg").touch()

        with patch.object(review_module, "MODELS_DIR", models_dir):
            review = LabelReview(self.csv_path)

        self.assertEqual(review.pending, [0])
        review.set_mode("Latest model incorrect")
        self.assertEqual(review.pending, [1])
        review.set_mode("Unseen + errors")
        self.assertEqual(review.pending, [0, 1])
        self.assertEqual(review.pending[review.position], 1)
        review.set_mode("Seen")
        self.assertEqual(review.pending, [1, 2])
        self.assertEqual(review.position, 0)
        review.set_mode("All")
        self.assertEqual(review.pending, [0, 1, 2])
        review.position = 2
        review.set_mode("Latest model incorrect")
        self.assertEqual(review.pending, [1])
        self.assertEqual(review.position, 0)

    def test_gui_dropdown_selects_review_mode(self):
        image_dir = self.csv_path.parent / "images"
        image_dir.mkdir()
        for image_name in ("unchecked", "correct", "incorrect"):
            Image.new("RGB", (64, 64), "black").save(
                image_dir / f"camera1_{image_name}.jpg"
            )

        models_dir = Path(self.temp_dir.name) / "models"
        incorrect_dir = (
            models_dir
            / "deepfaune_finetuned_20261020-191127_latest"
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

    def test_box_and_annotation_are_red_and_keep_true_label_when_incorrect(self):
        review = LabelReview(self.csv_path)
        review.set_mode("All")
        figure, axes = plt.subplots()
        self.addCleanup(plt.close, figure)
        review.image_axes = axes

        review.draw_box_label(review.rows[2], (100, 100))

        self.assertEqual(axes.patches[0].get_edgecolor(), to_rgba("#c62828"))
        self.assertEqual(axes.texts[0].get_text(), "True: hare | Predicted: fox")

    def test_undo_restores_manual_label_that_existed_before_review(self):
        review = LabelReview(self.csv_path)
        review.set_mode("All")

        review.choose("fox")
        review.undo()

        self.assertEqual(review.rows[0]["manual_label"], "")
        self.assertEqual(review.rows[1]["manual_label"], "fox")

    def test_validation_images_use_reviewer_box_style_and_status_colors(self):
        finetune_classifier = import_module("src.labelling.finetune_classifier")

        image_dir = Path(self.temp_dir.name) / "images" / "2024"
        (image_dir / "camera1").mkdir(parents=True)
        Image.new("RGB", (100, 100), "black").save(
            image_dir / "camera1" / "correct.jpg"
        )
        Image.new("RGB", (100, 100), "black").save(
            image_dir / "camera1" / "incorrect.jpg"
        )
        rows = pd.DataFrame(
            [
                {
                    "location_id": "camera1",
                    "image_file": "correct.jpg",
                    "image_path": str(image_dir / "camera1" / "correct.jpg"),
                    "box": "[0.5, 0.5, 0.4, 0.4]",
                },
                {
                    "location_id": "camera1",
                    "image_file": "incorrect.jpg",
                    "image_path": str(image_dir / "camera1" / "incorrect.jpg"),
                    "box": "[0.5, 0.5, 0.4, 0.4]",
                },
            ]
        )
        output_dir = Path(self.temp_dir.name) / "validation"

        finetune_classifier.save_validation_predictions(
            rows, [0, 1], [0, 0], ["fox", "hare"], output_dir
        )

        with (output_dir / "predictions.csv").open(encoding="utf-8") as source:
            predictions = list(csv.DictReader(source))
        self.assertEqual(predictions[1]["true_label"], "hare")
        self.assertEqual(predictions[1]["predicted_label"], "fox")
        self.assertEqual(predictions[1]["status"], "incorrect")

        correct = Image.open(output_dir / "correct" / "camera1_correct.jpg")
        incorrect = Image.open(output_dir / "incorrect" / "camera1_incorrect.jpg")
        self.addCleanup(correct.close)
        self.addCleanup(incorrect.close)
        self.assertEqual(correct.size, (100, 100))
        self.assertEqual(incorrect.size, (100, 100))
        for image, expected in (
            (correct, (46, 125, 50)),
            (incorrect, (198, 40, 40)),
        ):
            actual = image.getpixel((50, 30))
            self.assertLessEqual(max(abs(a - b) for a, b in zip(actual, expected)), 15)


if __name__ == "__main__":
    unittest.main()
