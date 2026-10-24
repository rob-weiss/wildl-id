"""Browser-based wildlife image label reviewer."""

import argparse
import ast
import html
import math
import textwrap
from pathlib import Path

from nicegui import events, run, ui
from PIL import Image

from .remove_empty_images import is_confirmed_empty, remove_empty_images
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


def box_overlay_svg(row, image_size, prediction=None, display_size=None, zoom_scale=1):
    image_width, image_height = image_size
    display_width, display_height = display_size or image_size
    scale_x = image_width / display_width
    scale_y = image_height / display_height
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
        else predicted_label
    )
    font_size = 16
    padding_x = round(font_size * 0.45)
    line_height = round(font_size * 1.5)
    characters_per_line = max(
        1, int((display_width - padding_x * 2) / (font_size * 0.65))
    )
    lines = textwrap.wrap(label, width=characters_per_line)
    label_height = line_height * len(lines)
    label_width = min(
        display_width,
        max(font_size * 6, max(map(len, lines)) * font_size * 0.65 + padding_x * 2),
    )
    label_left = min(max(0, left / scale_x), display_width - label_width)
    label_top = max(0, top / scale_y - label_height - round(font_size * 0.25))
    escaped_label = "".join(
        f'<tspan x="{padding_x}" dy="{line_height if index else 0}">'
        f"{html.escape(line, quote=False)}</tspan>"
        for index, line in enumerate(lines)
    )
    return (
        f'<rect x="{left:.1f}" y="{top:.1f}" width="{box_width:.1f}" '
        f'height="{box_height:.1f}" fill="none" stroke="{color}" '
        f'stroke-width="{4 / zoom_scale:g}" vector-effect="non-scaling-stroke" />'
        f'<g transform="translate({label_left * scale_x:.1f} '
        f'{label_top * scale_y:.1f}) scale({scale_x} {scale_y})">'
        '<rect x="0" y="0" '
        f'width="{label_width:.1f}" height="{label_height}" rx="2" '
        f'fill="{color}" />'
        f'<text x="{padding_x}" y="{round(font_size * 1.12)}" '
        f'fill="white" font-size="{font_size}" font-family="sans-serif" font-weight="bold">'
        f"{escaped_label}</text></g>"
    )


class ZoomViewer(
    ui.element,
    component="review_zoom.js",
    esm={"@panzoom/panzoom": "vendor"},
):
    pass


class ReviewApp:
    def __init__(self, csv_path, images_dir=None):
        self.review = LabelReview(csv_path, images_dir)
        self.current_index = None
        self.current_image_path = None
        self.image_size = None
        self.display_size = None
        self.zoom_scale = 1
        self.drag_start = None
        self.box_history = []
        self.label_buttons = []
        self.cleanup_dialog_open = False
        self.focus_subject = False
        self.image_mode = "box"

    def build(self):
        ui.colors(primary="#2e7d32", secondary="#c62828")
        self.keyboard = ui.keyboard(on_key=self.on_key, repeating=True)
        self.keyboard.on(
            "key",
            js_handler="""payload => {
                if (payload.action === 'keydown' && payload.code === 'Space' &&
                    !payload.altKey && !payload.ctrlKey && !payload.metaKey) {
                    payload.event.preventDefault();
                }
            }""",
        )
        ui.add_head_html(
            """<style>
            body { background: #f4f6f4; }
            .review-viewport { background: #e5e7e5; }
            </style>"""
        )
        with ui.column().classes("w-full min-h-screen p-4 gap-3"):
            with ui.row().classes("w-full items-center justify-between gap-4"):
                ui.label("Wildlife label review").classes("text-xl font-semibold")
                run_name = self.review.latest_model_name or "No validation run found"
                self.run_label = ui.label(run_name).classes("text-sm text-gray-600")
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

            with ui.row().classes("w-full items-start gap-4 flex-wrap"):
                with ui.column().classes("grow min-w-0").style("flex-basis: 600px"):
                    with ui.row().classes("w-full items-center gap-1"):
                        ui.button(
                            icon="zoom_in",
                            on_click=lambda: self.viewer.run_method("zoomIn"),
                        ).props("flat round dense aria-label='Zoom in'").tooltip(
                            "Zoom in"
                        )
                        ui.button(
                            icon="zoom_out",
                            on_click=lambda: self.viewer.run_method("zoomOut"),
                        ).props("flat round dense aria-label='Zoom out'").tooltip(
                            "Zoom out"
                        )
                        ui.button(icon="fit_screen", on_click=self.fit_image).props(
                            "flat round dense aria-label='Fit image'"
                        ).tooltip("Fit image")
                        self.subject_zoom_button = (
                            ui.button(
                                icon="center_focus_strong",
                                on_click=self.toggle_subject_focus,
                            )
                            .props(
                                "flat round dense aria-label='Focus subject' aria-pressed=false"
                            )
                            .tooltip("Focus subject")
                        )
                        ui.toggle(
                            {"box": "Box", "pan": "Pan"},
                            value="box",
                            on_change=lambda event: self.set_image_mode(event.value),
                        ).props("dense no-caps").classes("ml-auto")
                    self.viewer = (
                        ZoomViewer()
                        .classes("w-full")
                        .style("height: min(70vh, 680px); cursor: crosshair")
                    )
                    self.viewer.on("viewport", self.on_image_resize)
                    with self.viewer:
                        self.image = (
                            ui.interactive_image(
                                size=None,
                                events=["mousedown", "mouseup"],
                                on_mouse=self.on_image_mouse,
                            )
                            .classes("review-image")
                            .style("width: 100%; height: 100%; cursor: inherit")
                        )
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
                        self.remove_empty_button = (
                            ui.button(
                                "Remove empty",
                                icon="delete_sweep",
                                on_click=self.confirm_remove_empty,
                            )
                            .props("outline no-caps")
                            .classes("text-red-700")
                        )
                    ui.label("Labels").classes("font-medium mt-2")
                    with ui.grid(columns=2).classes("w-full gap-2"):
                        manual_labels = (
                            str(row.get("manual_label") or "").strip()
                            for row in self.review.rows
                        )
                        labels = list(
                            dict.fromkeys(
                                [
                                    *SHORTCUTS.values(),
                                    *(row["class"] for row in self.review.rows),
                                    *(label for label in manual_labels if label),
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
        original = self.review.original_path(row)
        if original is not None:
            return original
        annotated = self.review.images_dir / f"{row['location_id']}_{row['image_file']}"
        return annotated if annotated.is_file() else None

    def refresh(self):
        self.run_label.set_visibility(self.review.mode == "Latest model incorrect")
        self.undo_button.set_enabled(bool(self.review.history))
        if not self.review.pending:
            self.current_index = None
            self.current_image_path = None
            self.image.set_visibility(False)
            self.viewer.set_visibility(False)
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
            self.viewer.set_visibility(False)
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
        self.viewer._props.update(
            {
                "imageKey": f"{row['location_id']}/{row['image_file']}",
                "subject": parse_box(row.get("box", "")),
                "focusSubject": self.focus_subject,
                "interactionMode": self.image_mode,
            }
        )
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
            self.prediction.set_text(
                f"Predicted: {row['class']}"
                if str(row.get("manual_label") or "").strip()
                else row["class"]
            )
        self.empty_message.set_visibility(self.current_image_path is None)
        if self.current_image_path is None:
            self.empty_message.set_text("Image file not found")
            self.image.set_visibility(False)
            self.viewer.set_visibility(False)
        else:
            try:
                with Image.open(self.current_image_path) as source:
                    self.image_size = source.size
                self.viewer._props["imageSize"] = list(self.image_size)
                self.image.set_source(self.current_image_path)
                self.image.set_content(
                    box_overlay_svg(
                        row,
                        self.image_size,
                        prediction,
                        self.display_size,
                        self.zoom_scale,
                    )
                )
                self.image.set_visibility(True)
                self.viewer.set_visibility(True)
            except (OSError, ValueError) as error:
                self.empty_message.set_text(f"Cannot open image: {error}")
                self.empty_message.set_visibility(True)
                self.image.set_visibility(False)
                self.viewer.set_visibility(False)
            self.viewer.update()
        self.accept_button.enable()
        for button in self.label_buttons:
            button.enable()

    def prediction_for_row(self, row):
        image_name = f"{row['location_id']}_{row['image_file']}"
        return self.review.latest_predictions.get(image_name)

    def on_image_resize(self, event):
        display_size = (event.args["width"], event.args["height"])
        zoom_scale = event.args.get("scale", 1)
        if min(display_size) <= 0 or zoom_scale <= 0:
            return
        if display_size == self.display_size and zoom_scale == self.zoom_scale:
            return
        self.display_size = display_size
        self.zoom_scale = zoom_scale
        self.refresh()

    def toggle_subject_focus(self):
        self.focus_subject = not self.focus_subject
        self.subject_zoom_button.props(
            f"aria-pressed={str(self.focus_subject).lower()}"
        )
        self.subject_zoom_button.classes(
            add="bg-green-100" if self.focus_subject else "",
            remove="" if self.focus_subject else "bg-green-100",
        )
        self.viewer._props["focusSubject"] = self.focus_subject
        self.viewer.update()

    def fit_image(self):
        if self.focus_subject:
            self.toggle_subject_focus()
        self.viewer.run_method("fit")

    def set_image_mode(self, mode):
        if mode not in ("box", "pan"):
            raise ValueError(f"Unknown image mode: {mode}")
        self.image_mode = mode
        self.drag_start = None
        self.viewer._props["interactionMode"] = mode
        self.viewer.style(f"cursor: {'grab' if mode == 'pan' else 'crosshair'}")
        self.viewer.update()

    def on_image_mouse(self, event: events.MouseEventArguments):
        if self.image_mode != "box" or self.cleanup_dialog_open:
            return
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
        if self.cleanup_dialog_open:
            return
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
        elif event.key.name.lower() == "z":
            self.toggle_subject_focus()
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

    async def confirm_remove_empty(self):
        if self.cleanup_dialog_open:
            return
        count = sum(is_confirmed_empty(row) for row in self.review.rows)
        if not count:
            ui.notify("No confirmed empty images to remove")
            return
        with ui.dialog().props("persistent") as dialog, ui.card():
            ui.label(f"Remove {count} confirmed empty images?").classes(
                "text-lg font-semibold"
            )
            ui.label(
                "Original and annotated images will be permanently deleted. "
                "The CSV will be backed up and deleted paths logged."
            ).classes("max-w-md")
            with ui.row().classes("w-full justify-end"):
                ui.button("Cancel", on_click=lambda: dialog.submit(False)).props(
                    "flat no-caps"
                )
                ui.button(
                    "Remove empty",
                    icon="delete_sweep",
                    color="negative",
                    on_click=lambda: dialog.submit(True),
                ).props("no-caps")
        self.cleanup_dialog_open = True
        try:
            if await dialog:
                dialog.clear()
                with dialog, ui.card():
                    ui.spinner()
                    ui.label("Removing empty images...")
                dialog.open()
                await self.remove_empty()
        finally:
            self.cleanup_dialog_open = False
            dialog.delete()

    async def remove_empty(self):
        try:
            count = await run.io_bound(
                remove_empty_images,
                csv_path=self.review.csv_path,
                annotated_dir=self.review.images_dir,
                expected_bytes=self.review.original_bytes,
            )
            updated_review = LabelReview(self.review.csv_path, self.review.images_dir)
            updated_review.set_mode(self.review.mode)
        except (OSError, RuntimeError, ValueError) as error:
            ui.notify(f"Cleanup failed: {error}", type="negative")
            return
        self.review = updated_review
        self.box_history.clear()
        self.drag_start = None
        self.image_size = None
        self.undo_box_button.disable()
        self.refresh()
        ui.notify(f"Removed {count} confirmed empty images", type="positive")

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
