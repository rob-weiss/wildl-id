import csv
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgba
from PIL import Image

review_module = import_module("src.labelling.review_labels")
LabelReview = review_module.LabelReview


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

    def test_gui_dropdown_selects_review_mode(self):
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

        review.toggle_mode_menu()
        review.select_mode("Seen")

        self.assertEqual(review.pending, [1, 2])
        self.assertEqual(review.mode, "Seen")
        self.assertEqual(review.mode_button.label.get_text(), "Mode: Seen v")

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
        from unittest.mock import patch

        finetune_classifier = import_module("src.labelling.finetune_classifier")

        image_dir = Path(self.temp_dir.name) / "images"
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
                    "box": "[0.5, 0.5, 0.4, 0.4]",
                },
                {
                    "location_id": "camera1",
                    "image_file": "incorrect.jpg",
                    "box": "[0.5, 0.5, 0.4, 0.4]",
                },
            ]
        )
        output_dir = Path(self.temp_dir.name) / "validation"

        with patch.object(finetune_classifier, "IMAGE_DIR", image_dir):
            finetune_classifier.save_validation_predictions(
                rows, [0, 1], [0, 0], ["fox", "hare"], output_dir
            )

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
