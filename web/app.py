import csv
from pathlib import Path

from fastapi import FastAPI
from nicegui import ui
from src.labelling.review_app import ReviewApp
from src.labelling.review_labels import DEFAULT_CSV

app = FastAPI()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def build_review_page(csv_path: Path = DEFAULT_CSV) -> None:
    try:
        reviewer = ReviewApp(csv_path)
        reviewer.build()
    except (OSError, ValueError, csv.Error) as error:
        ui.label("Review data unavailable").classes("text-lg font-semibold")
        ui.label(str(error))
        return

    reviewer.remove_empty_button.set_visibility(False)


def build_default_review_page() -> None:
    build_review_page(DEFAULT_CSV)


@ui.page("/")
def review_page() -> None:
    build_default_review_page()


ui.run_with(app, mount_path="/")
