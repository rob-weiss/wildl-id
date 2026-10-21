"""Browser-based wildlife image label reviewer."""

import argparse
import ast
import html
import math
from pathlib import Path

from nicegui import events, ui
from PIL import Image

from .review_labels import (
    DEFAULT_CSV,
    MODE_OPTIONS,
    SHORTCUTS,
    LabelReview,
    box_from_drag,
)


def parse_box(value):
    try:
        box = ast.literal_eval(value or "")
    except (ValueError, SyntaxError, TypeError):
        return None
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    if not all(isinstance(item, (int, float)) and math.isfinite(item) for item in box):
        return None
    if not all(0 <= item <= 1 for item in box) or box[2] <= 0 or box[3] <= 0:
        return None
    return box


def box_overlay_svg(row, image_size, prediction=None):
    image_width, image_height = image_size
    box = parse_box(row.get("box", ""))
    if box is None:
        return ""
    x_center, y_center, width, height = box
    left = (x_center - width / 2) * image_width
    top = (y_center - height / 2) * image_height
    box_width = width * image_width
    box_height = height * image_height
    manual_label = str(row.get("manual_label") or "").strip()
    if prediction:
        true_label = prediction["true_label"]
        predicted_label = prediction["predicted_label"]
        is_correct = prediction["status"] == "correct"
    else:
        true_label = manual_label
        predicted_label = row["class"]
        is_correct = bool(manual_label) and manual_label == row["class"]
    color = "#2e7d32" if is_correct else "#c62828"
    label = (
        f"True: {true_label} | Predicted: {predicted_label}"
        if true_label
        else f"Predicted: {predicted_label}"
    )
    font_size = min(112, max(24, round(image_width * 0.028)))
    padding_x = round(font_size * 0.45)
    label_height = round(font_size * 1.5)
    label_width = min(
        image_width,
        max(font_size * 6, len(label) * font_size * 0.58 + padding_x * 2),
    )
    label_left = min(max(0, left), image_width - label_width)
    label_top = max(0, top - label_height - round(font_size * 0.25))
    escaped_label = html.escape(label, quote=False)
    return (
        f'<rect x="{left:.1f}" y="{top:.1f}" width="{box_width:.1f}" '
        f'height="{box_height:.1f}" fill="none" stroke="{color}" '
        'stroke-width="8" vector-effect="non-scaling-stroke" />'
        f'<rect x="{label_left:.1f}" y="{label_top:.1f}" '
        f'width="{label_width:.1f}" height="{label_height}" rx="2" '
        f'fill="{color}" />'
        f'<text x="{label_left + padding_x:.1f}" '
        f'y="{label_top + round(font_size * 1.12):.1f}" '
        f'fill="white" font-size="{font_size}" font-weight="bold">'
        f"{escaped_label}</text>"
    )


class ReviewApp:
    def __init__(self, csv_path, images_dir=None):
        self.review = LabelReview(csv_path, images_dir)
        self.current_index = None
        self.current_image_path = None
        self.image_size = None
        self.drag_start = None
        self.box_history = []
        self.label_buttons = []

    def build(self):
        ui.colors(primary="#2e7d32", secondary="#c62828")
        self.keyboard = ui.keyboard(on_key=self.on_key, repeating=False)
        ui.add_head_html(
            """<style>
            body { background: #f4f6f4; }
            .review-image { max-height: calc(100vh - 150px); }
            </style>"""
        )
        with ui.column().classes("w-full min-h-screen p-4 gap-3"):
            with ui.row().classes("w-full items-center justify-between gap-4"):
                ui.label("Wildlife label review").classes("text-xl font-semibold")
                self.mode_select = (
                    ui.select(
                        list(MODE_OPTIONS),
                        value=self.review.mode,
                        label="Review mode",
                        on_change=lambda event: self.change_mode(event.value),
                    )
                    .props("outlined dense")
                    .classes("w-64")
                )
                run_name = self.review.latest_model_name or "No validation run found"
                ui.label(run_name).classes("text-sm text-gray-600")

            with ui.row().classes("w-full items-start gap-4 flex-wrap"):
                with ui.column().classes("grow min-w-0"):
                    self.image = (
                        ui.interactive_image(
                            size=(640, 480),
                            events=["mousedown", "mouseup"],
                            on_mouse=self.on_image_mouse,
                        )
                        .classes("w-full review-image")
                        .style("cursor: crosshair")
                    )
                    self.image.tooltip("Drag to add or replace the bounding box")
                    self.empty_message = ui.label().classes("text-gray-700")
                with ui.column().classes("w-80 max-w-full shrink-0 gap-2"):
                    self.counter = ui.label()
                    self.image_name = ui.label().classes("font-medium break-all")
                    self.prediction = ui.label().classes("text-gray-700")
                    with ui.row().classes("w-full gap-2"):
                        self.accept_button = ui.button(
                            "Accept prediction", icon="check", on_click=self.accept
                        ).props("no-caps")
                        self.undo_button = ui.button(
                            "Undo label", icon="undo", on_click=self.undo_label
                        ).props("no-caps")
                    with ui.row().classes("w-full gap-2"):
                        self.undo_box_button = ui.button(
                            "Undo box", icon="crop_free", on_click=self.undo_box
                        ).props("no-caps")
                    ui.label("Labels").classes("font-medium mt-2")
                    with ui.grid(columns=2).classes("w-full gap-2"):
                        labels = list(
                            dict.fromkeys(
                                [
                                    *SHORTCUTS.values(),
                                    *(row["class"] for row in self.review.rows),
                                ]
                            )
                        )
                        for label in labels:
                            key = next(
                                (
                                    key
                                    for key, value in SHORTCUTS.items()
                                    if value == label
                                ),
                                None,
                            )
                            caption = f"{label} [{key}]" if key else label
                            button = ui.button(
                                caption,
                                on_click=lambda label=label: self.choose_label(label),
                            ).props("outline no-caps dense")
                            self.label_buttons.append(button)
                    self.custom_label = ui.input(label="Other label").classes("w-full")
                    self.custom_label.on(
                        "keydown.enter", lambda event: self.save_custom_label()
                    )
                    ui.button(
                        "Save label", icon="save", on_click=self.save_custom_label
                    ).props("no-caps")
        self.undo_box_button.disable()
        self.refresh()

    def image_path_for(self, row):
        original = (
            self.review.csv_path.parent.parent / row["location_id"] / row["image_file"]
        )
        if original.is_file():
            return original
        annotated = self.review.images_dir / f"{row['location_id']}_{row['image_file']}"
        return annotated if annotated.is_file() else None

    def refresh(self):
        if not self.review.pending:
            self.current_index = None
            self.current_image_path = None
            self.image.set_visibility(False)
            self.empty_message.set_text("No images match this review mode")
            self.empty_message.set_visibility(True)
            self.counter.set_text("0 images")
            self.image_name.set_text("")
            self.prediction.set_text("")
            self.accept_button.disable()
            for button in self.label_buttons:
                button.disable()
            return
        if self.review.position >= len(self.review.pending):
            self.current_index = None
            self.current_image_path = None
            self.image.set_visibility(False)
            self.empty_message.set_text("All matching labels checked")
            self.empty_message.set_visibility(True)
            self.counter.set_text(f"{len(self.review.pending)} images")
            self.image_name.set_text("")
            self.prediction.set_text("")
            self.accept_button.disable()
            for button in self.label_buttons:
                button.disable()
            return

        self.current_index = self.review.pending[self.review.position]
        row = self.review.rows[self.current_index]
        self.current_image_path = self.image_path_for(row)
        self.counter.set_text(
            f"{self.review.position + 1} / {len(self.review.pending)}"
        )
        self.image_name.set_text(f"{row['location_id']} / {row['image_file']}")
        prediction = self.prediction_for_row(row)
        if prediction:
            self.prediction.set_text(
                f"True: {prediction['true_label']} | "
                f"Predicted: {prediction['predicted_label']}"
            )
        else:
            self.prediction.set_text(f"Predicted: {row['class']}")
        self.empty_message.set_visibility(self.current_image_path is None)
        if self.current_image_path is None:
            self.empty_message.set_text("Image file not found")
            self.image.set_visibility(False)
        else:
            try:
                with Image.open(self.current_image_path) as source:
                    self.image_size = source.size
                self.image.set_source(self.current_image_path)
                self.image.set_content(
                    box_overlay_svg(row, self.image_size, prediction)
                )
                self.image.set_visibility(True)
            except (OSError, ValueError) as error:
                self.empty_message.set_text(f"Cannot open image: {error}")
                self.empty_message.set_visibility(True)
                self.image.set_visibility(False)
        self.accept_button.enable()
        for button in self.label_buttons:
            button.enable()

    def prediction_for_row(self, row):
        image_name = f"{row['location_id']}_{row['image_file']}"
        return self.review.latest_predictions.get(image_name)

    def on_image_mouse(self, event: events.MouseEventArguments):
        if self.current_index is None or self.image_size is None:
            return
        point = (event.image_x, event.image_y)
        if event.type == "mousedown":
            self.drag_start = point
            return
        if event.type != "mouseup" or self.drag_start is None:
            return
        box = box_from_drag(self.drag_start, point, self.image_size)
        self.drag_start = None
        if box is None:
            self.refresh()
            return
        row = self.review.rows[self.current_index]
        previous_box = row.get("box", "")
        row["box"] = repr(box)
        try:
            self.review.save()
        except (OSError, RuntimeError) as error:
            row["box"] = previous_box
            ui.notify(f"Box not saved: {error}", type="negative")
            self.refresh()
            return
        self.box_history.append((self.current_index, previous_box))
        self.undo_box_button.enable()
        self.refresh()

    def change_mode(self, mode):
        self.review.set_mode(mode)
        self.refresh()

    def on_key(self, event: events.KeyEventArguments):
        if not event.action.keydown or event.action.repeat:
            return
        if event.modifiers.alt or event.modifiers.ctrl or event.modifiers.meta:
            return
        if event.key.space:
            self.accept()
        elif event.key.backspace:
            self.undo_label()
        elif event.key.enter:
            self.custom_label.run_method("focus")
        else:
            label = SHORTCUTS.get(event.key.name.lower())
            if label:
                self.choose_label(label)

    def choose_label(self, label):
        try:
            self.review.choose(label)
        except (OSError, RuntimeError) as error:
            ui.notify(f"Label not saved: {error}", type="negative")
        self.refresh()

    def accept(self):
        if self.current_index is not None:
            row = self.review.rows[self.current_index]
            prediction = self.prediction_for_row(row)
            label = prediction["predicted_label"] if prediction else row["class"]
            self.choose_label(label)

    def save_custom_label(self):
        self.choose_label(self.custom_label.value or "")
        self.custom_label.value = ""

    def undo_label(self):
        try:
            self.review.undo()
        except (OSError, RuntimeError) as error:
            ui.notify(f"Label undo failed: {error}", type="negative")
        self.refresh()

    def undo_box(self):
        if not self.box_history:
            return
        index, previous_box = self.box_history[-1]
        row = self.review.rows[index]
        current_box = row.get("box", "")
        row["box"] = previous_box
        try:
            self.review.save()
        except (OSError, RuntimeError) as error:
            row["box"] = current_box
            ui.notify(f"Box undo failed: {error}", type="negative")
            return
        self.box_history.pop()
        if not self.box_history:
            self.undo_box_button.disable()
        self.refresh()


def run_review_app(csv_path=DEFAULT_CSV, images_dir=None):
    reviewer = ReviewApp(csv_path, images_dir)
    reviewer.build()
    ui.run(
        host="127.0.0.1",
        port=8080,
        title="Wildlife label review",
        reload=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--images", type=Path, help="Annotated images directory (default: beside CSV)"
    )
    arguments = parser.parse_args()
    run_review_app(arguments.csv, arguments.images)


if __name__ == "__main__":
    main()
