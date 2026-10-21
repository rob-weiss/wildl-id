"""Review machine labels in a local browser app and save choices to manual_label."""

import argparse
import ast
import csv
import math
import os
import shutil
import sys
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
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODE_OPTIONS = (
    "Unseen",
    "Latest model incorrect",
    "Unseen + errors",
    "Seen",
    "All",
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
    "o": "raccoon",
    "c": "crow",
    "a": "human",
    "n": "none",
    "u": "unknown",
    "l": "owl",
    "t": "cat",
    "v": "vehicle",
}


def latest_incorrect_predictions(models_dir=None):
    models_dir = Path(models_dir or MODELS_DIR)
    run_dirs = sorted(
        (path for path in models_dir.glob("deepfaune_finetuned_*") if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    for run_dir in run_dirs:
        incorrect_dir = run_dir / "validation_predictions" / "incorrect"
        if incorrect_dir.is_dir():
            return run_dir.name, {
                path.name for path in incorrect_dir.iterdir() if path.is_file()
            }
    return None, set()


def box_from_drag(start, end, image_size):
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        return None
    coordinates = (*start, *end)
    if not all(math.isfinite(value) for value in coordinates):
        return None
    x_start, y_start, x_end, y_end = coordinates
    left, right = sorted(
        (max(0, min(image_width, x_start)), max(0, min(image_width, x_end)))
    )
    top, bottom = sorted(
        (max(0, min(image_height, y_start)), max(0, min(image_height, y_end)))
    )
    box_width = right - left
    box_height = bottom - top
    if box_width < 4 or box_height < 4:
        return None
    return (
        (left + right) / (2 * image_width),
        (top + bottom) / (2 * image_height),
        box_width / image_width,
        box_height / image_height,
    )


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
        self.latest_model_name, self.latest_incorrect_filenames = (
            latest_incorrect_predictions()
        )
        self.mode = "Unseen"
        self.pending = self.filtered_indices()
        self.position = 0
        self.history = []
        self.original_bytes = self.csv_path.read_bytes()

    def filtered_indices(self):
        indices = []
        for index, row in enumerate(self.rows):
            manual_label = str(row.get("manual_label") or "").strip()
            image_name = f"{row['location_id']}_{row['image_file']}"
            is_unseen = not manual_label
            is_latest_error = image_name in self.latest_incorrect_filenames
            if (
                self.mode == "Unseen"
                and is_unseen
                or self.mode == "Latest model incorrect"
                and is_latest_error
                or self.mode == "Unseen + errors"
                and (is_unseen or is_latest_error)
                or self.mode == "Seen"
                and not is_unseen
                or self.mode == "All"
            ):
                indices.append(index)
        return indices

    def set_mode(self, mode):
        if mode not in MODE_OPTIONS:
            raise ValueError(f"Unknown review mode: {mode}")
        current_index = (
            self.pending[self.position] if self.position < len(self.pending) else None
        )
        self.mode = mode
        self.pending = self.filtered_indices()
        if current_index is None:
            self.position = 0
        elif current_index in self.pending:
            self.position = self.pending.index(current_index)
        else:
            self.position = next(
                (
                    position
                    for position, index in enumerate(self.pending)
                    if index > current_index
                ),
                0,
            )

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
        old_position = self.position
        previous = self.rows[index]["manual_label"]
        self.rows[index]["manual_label"] = label
        try:
            self.save()
        except Exception:
            self.rows[index]["manual_label"] = previous
            raise
        self.history.append((index, previous))
        self.pending = self.filtered_indices()
        if index in self.pending:
            self.position = self.pending.index(index) + 1
        else:
            self.position = next(
                (
                    position
                    for position, pending_index in enumerate(self.pending)
                    if pending_index > index
                ),
                min(old_position, len(self.pending)),
            )

    def undo(self):
        if not self.history:
            return
        index, previous = self.history[-1]
        row = self.rows[index]
        current = row["manual_label"]
        row["manual_label"] = previous
        try:
            self.save()
        except Exception:
            row["manual_label"] = current
            raise
        self.history.pop()
        self.pending = self.filtered_indices()
        if index not in self.pending:
            self.pending.append(index)
            self.pending.sort()
        self.position = self.pending.index(index)

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
        latest_run = (
            self.latest_model_name.removeprefix("deepfaune_finetuned_").split("_")[0]
            if self.latest_model_name
            else "none found"
        )
        self.mode_axes = self.figure.add_axes((0.76, 0.86, 0.22, 0.07))
        self.mode_axes.set_title(f"Latest run: {latest_run}", fontsize=9)
        self.mode_button = Button(self.mode_axes, f"Mode: {self.mode} v")
        self.mode_button.on_clicked(lambda event: self.toggle_mode_menu())
        self.mode_menu_axes = []
        self.mode_menu_buttons = []
        self.mode_menu_open = False
        labels = list(
            dict.fromkeys([*SHORTCUTS.values(), *(row["class"] for row in self.rows)])
        )
        keys = {label: key for key, label in SHORTCUTS.items()}
        height = min(0.055, 0.55 / ((len(labels) + 1) // 2))
        for index, label in enumerate(labels):
            caption = f"{label} [{keys[label]}]" if label in keys else label
            self.add_button(
                (
                    0.76 + (index % 2) * 0.115,
                    0.73 - (index // 2) * height,
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

    def toggle_mode_menu(self):
        if not self.mode_menu_axes:
            for index, mode in enumerate(MODE_OPTIONS):
                axes = self.figure.add_axes(
                    (0.76, 0.80 - index * 0.05, 0.22, 0.045), zorder=10
                )
                button = Button(axes, mode)
                button.on_clicked(lambda event, mode=mode: self.select_mode(mode))
                self.mode_menu_axes.append(axes)
                self.mode_menu_buttons.append(button)
        self.mode_menu_open = not self.mode_menu_open
        for axes in self.mode_menu_axes:
            axes.set_visible(self.mode_menu_open)
        self.figure.canvas.draw_idle()

    def select_mode(self, mode):
        self.set_mode(mode)
        self.mode_button.label.set_text(f"Mode: {mode} v")
        self.mode_menu_open = False
        for axes in self.mode_menu_axes:
            axes.set_visible(False)
        self.draw()

    def draw(self):
        self.image_axes.clear()
        self.image_axes.axis("off")
        if not self.pending:
            self.image_axes.set_title("No images match selected filters")
        elif self.position >= len(self.pending):
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
            title = (
                f"{self.position + 1}/{len(self.pending)} selected images | "
                f"{row['location_id']} / {row['image_file']}\n"
                f"Machine label: {row['class']}"
            )
            manual_label = str(row.get("manual_label") or "").strip()
            if manual_label:
                title += f"\nManual label: {manual_label}"
            self.image_axes.set_title(title, fontsize=12)
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
        manual_label = str(row.get("manual_label") or "").strip()
        is_correct = bool(manual_label) and manual_label == row["class"]
        color = "#2e7d32" if is_correct else "#c62828"
        label = (
            f"True: {manual_label} | Predicted: {row['class']}"
            if manual_label
            else f"Predicted: {row['class']}"
        )
        self.image_axes.add_patch(
            Rectangle(
                (left, top),
                width * image_width,
                height * image_height,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
        )
        self.image_axes.annotate(
            label,
            (left, top),
            xytext=(0, -3 if top < 30 else 3),
            textcoords="offset points",
            ha="left",
            va="top" if top < 30 else "bottom",
            color="white",
            fontsize=12,
            fontweight="bold",
            bbox={"facecolor": color, "edgecolor": "none", "pad": 3},
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
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.labelling.review_app import run_review_app

    run_review_app(arguments.csv, arguments.images)
