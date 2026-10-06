"""Review machine labels with Matplotlib; save each choice to manual_label."""

import argparse
import ast
import csv
import math
import os
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, TextBox
from PIL import Image

DEFAULT_CSV = (
    Path(__file__).resolve().parents[2]
    / "data/labels_MegaDetectorV6_MDV6-yolov10-e_classified"
    / "detection_results_MegaDetectorV6_MDV6-yolov10-e_classified.csv"
)
SHORTCUTS = {
    "r": "roe deer",
    "w": "wild boar",
    "p": "pigeon",
    "b": "badger",
    "m": "marten",
    "d": "dog",
    "f": "fox",
    "h": "hare",
    "s": "squirrel",
    "j": "jay",
    "o": "racoon",
    "c": "crow",
    "a": "human",
    "n": "none",
    "u": "unknown",
    "l": "owl",
    "t": "cat",
    "v": "vehicle",
}


class LabelReview:
    def __init__(self, csv_path, images_dir=None):
        self.csv_path = Path(csv_path)
        self.images_dir = (
            Path(images_dir) if images_dir else self.csv_path.parent / "images"
        )
        with self.csv_path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            self.columns = list(reader.fieldnames or [])
            if not {"location_id", "image_file", "class"}.issubset(self.columns):
                raise ValueError("CSV needs location_id, image_file, and class columns")
            self.rows = list(reader)
        if "manual_label" not in self.columns:
            self.columns.append("manual_label")
        for row in self.rows:
            row.setdefault("manual_label", "")
        self.pending = [
            index for index, row in enumerate(self.rows) if not row["manual_label"]
        ]
        self.position = 0
        self.history = []
        self.original_bytes = self.csv_path.read_bytes()

    def save(self):
        if self.csv_path.read_bytes() != self.original_bytes:
            raise RuntimeError(
                "CSV changed outside this reviewer. Close and reopen it before saving."
            )
        backup = self.csv_path.with_suffix(self.csv_path.suffix + ".before-review.bak")
        if not backup.exists():
            shutil.copy2(self.csv_path, backup)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                newline="",
                encoding="utf-8",
                dir=self.csv_path.parent,
                delete=False,
            ) as target:
                temporary_path = Path(target.name)
                writer = csv.DictWriter(target, fieldnames=self.columns)
                writer.writeheader()
                writer.writerows(self.rows)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary_path, self.csv_path)
            self.original_bytes = self.csv_path.read_bytes()
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def choose(self, label):
        label = label.strip()
        if not label or self.position >= len(self.pending):
            return
        index = self.pending[self.position]
        previous = self.rows[index]["manual_label"]
        self.rows[index]["manual_label"] = label
        try:
            self.save()
        except Exception:
            self.rows[index]["manual_label"] = previous
            raise
        self.history.append(self.position)
        self.position += 1

    def undo(self):
        if not self.history:
            return
        position = self.history[-1]
        row = self.rows[self.pending[position]]
        previous = row["manual_label"]
        row["manual_label"] = ""
        try:
            self.save()
        except Exception:
            row["manual_label"] = previous
            raise
        self.position = self.history.pop()

    def show(self):
        for setting in plt.rcParams:
            if setting.startswith("keymap."):
                plt.rcParams[setting] = []
        self.figure = plt.figure(figsize=(14, 9))
        self.figure.canvas.manager.set_window_title("Wildlife label review")
        self.image_axes = self.figure.add_axes((0.02, 0.14, 0.72, 0.78))
        self.buttons = []
        self.add_button((0.02, 0.035, 0.19, 0.055), "Correct [Space]", self.accept)
        self.add_button((0.23, 0.035, 0.19, 0.055), "Other label [Enter]", self.edit)
        self.add_button((0.44, 0.035, 0.12, 0.055), "Undo", self.back)
        self.add_button(
            (0.58, 0.035, 0.12, 0.055), "Close", lambda: plt.close(self.figure)
        )
        labels = list(
            dict.fromkeys([*SHORTCUTS.values(), *(row["class"] for row in self.rows)])
        )
        keys = {label: key for key, label in SHORTCUTS.items()}
        height = min(0.055, 0.74 / ((len(labels) + 1) // 2))
        for index, label in enumerate(labels):
            caption = f"{label} [{keys[label]}]" if label in keys else label
            self.add_button(
                (
                    0.76 + (index % 2) * 0.115,
                    0.86 - (index // 2) * height,
                    0.11,
                    height * 0.85,
                ),
                caption,
                lambda label=label: self.record(label),
            )
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.textbox = TextBox(self.figure.add_axes((0.79, 0.10, 0.18, 0.05)), "Label ")
        self.textbox.on_submit(self.submit)
        self.draw()
        plt.show()

    def add_button(self, bounds, caption, callback):
        button = Button(self.figure.add_axes(bounds), caption)
        button.on_clicked(lambda event: callback())
        self.buttons.append(button)

    def draw(self):
        self.image_axes.clear()
        self.image_axes.axis("off")
        if self.position >= len(self.pending):
            self.image_axes.set_title("All labels checked")
        else:
            row = self.rows[self.pending[self.position]]
            image_path = self.images_dir / f"{row['location_id']}_{row['image_file']}"
            original_path = (
                self.csv_path.parent.parent / row["location_id"] / row["image_file"]
            )
            use_original = original_path.is_file()
            if use_original:
                image_path = original_path
            try:
                with Image.open(image_path) as image:
                    self.image_axes.imshow(image.convert("RGB"))
                    if use_original:
                        self.draw_box_label(row, image.size)
            except (OSError, ValueError) as error:
                self.image_axes.text(
                    0.5, 0.5, f"Cannot open image:\n{error}", ha="center", wrap=True
                )
            self.image_axes.set_title(
                f"{self.position + 1}/{len(self.pending)} unchecked images | "
                f"{row['location_id']} / {row['image_file']}\nMachine label: {row['class']}",
                fontsize=12,
            )
        self.figure.canvas.draw_idle()

    def draw_box_label(self, row, image_size):
        try:
            box = ast.literal_eval(row.get("box", ""))
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                return
            if not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in box
            ):
                return
            x_center, y_center, width, height = box
            if not all(0 <= value <= 1 for value in box) or width == 0 or height == 0:
                return
        except (ValueError, SyntaxError, TypeError):
            return
        image_width, image_height = image_size
        left = max(0, (x_center - width / 2) * image_width)
        top = max(0, (y_center - height / 2) * image_height)
        self.image_axes.add_patch(
            Rectangle(
                (left, top),
                width * image_width,
                height * image_height,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
        )
        self.image_axes.annotate(
            row["class"],
            (left, top),
            xytext=(0, -3 if top < 30 else 3),
            textcoords="offset points",
            ha="left",
            va="top" if top < 30 else "bottom",
            color="white",
            fontsize=12,
            fontweight="bold",
            bbox={"facecolor": "red", "edgecolor": "none", "pad": 3},
        )

    def record(self, label):
        try:
            self.choose(label)
        except (OSError, RuntimeError) as error:
            self.image_axes.set_title(f"Not saved: {error}", color="red", wrap=True)
            self.figure.canvas.draw_idle()
            return
        self.draw()

    def accept(self):
        if self.position < len(self.pending):
            self.record(self.rows[self.pending[self.position]]["class"])

    def edit(self):
        if self.position < len(self.pending):
            self.textbox.begin_typing()
            self.figure.canvas.draw_idle()

    def submit(self, text):
        if not text.strip():
            return
        self.textbox.eventson = False
        self.textbox.stop_typing()
        self.textbox.set_val("")
        self.textbox.eventson = True
        self.record(text)

    def back(self):
        try:
            self.undo()
        except (OSError, RuntimeError) as error:
            self.image_axes.set_title(f"Not saved: {error}", color="red", wrap=True)
            self.figure.canvas.draw_idle()
            return
        self.draw()

    def on_key(self, event):
        if self.textbox.capturekeystrokes:
            return
        if event.key == " ":
            self.accept()
        elif event.key in SHORTCUTS:
            self.record(SHORTCUTS[event.key])
        elif event.key == "enter":
            self.edit()
        elif event.key == "backspace":
            self.back()
        elif event.key == "escape":
            plt.close(self.figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--images", type=Path, help="Annotated images directory (default: beside CSV)"
    )
    arguments = parser.parse_args()
    LabelReview(arguments.csv, arguments.images).show()
