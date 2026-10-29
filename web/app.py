import csv
import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from nicegui import ui
from src.labelling.review_app import ReviewApp
from src.labelling.review_labels import DEFAULT_CSV
from web.visualisations import VisualisationRunStore

app = FastAPI()
visualisation_runs = VisualisationRunStore()
logger = logging.getLogger(__name__)
ASSETS_DIRECTORY = Path(__file__).resolve().parent.parent / "docs" / "assets"
app.mount("/assets", StaticFiles(directory=ASSETS_DIRECTORY), name="brand-assets")


def add_brand_logo() -> None:
    ui.image("/assets/wildl-id-logo.svg").props('alt="wildl-id logo"').classes(
        "w-64 max-w-full"
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def build_review_page(csv_path: Path = DEFAULT_CSV) -> None:
    add_brand_logo()
    try:
        reviewer = ReviewApp(
            csv_path,
            image_base_url=os.environ.get("WILDL_ID_IMAGE_BASE_URL"),
        )
        reviewer.build()
    except (OSError, ValueError, csv.Error) as error:
        ui.label("Review data unavailable").classes("text-lg font-semibold")
        ui.label(str(error))
        return

    reviewer.remove_empty_button.set_visibility(False)


def build_default_review_page() -> None:
    build_review_page(DEFAULT_CSV)


@ui.page("/")
def home_page() -> None:
    with ui.column().classes("w-full max-w-3xl mx-auto p-6 gap-4"):
        add_brand_logo()
        ui.label("Wildlife analysis").classes("text-2xl font-semibold")
        ui.link("Manual review", "/review").classes("text-lg")
        ui.link("Visualisations", "/visualisations").classes("text-lg")


@ui.page("/review")
def review_page() -> None:
    build_default_review_page()


@ui.page("/visualisations")
def visualisations_page() -> None:
    add_brand_logo()
    ui.label("Visualisations").classes("text-xl font-semibold")
    status = ui.label()
    spinner = ui.spinner(size="lg")
    spinner.set_visibility(False)
    results = ui.column().classes("w-full gap-3")

    def show_results(output_directory: Path) -> None:
        summary_path = output_directory / "summary_statistics.txt"
        chart_paths = sorted(output_directory.glob("*.svg"))
        if not summary_path.is_file() or not chart_paths:
            status.set_text("No visualisation results are available.")
            return

        summary_text = summary_path.read_text(encoding="utf-8")
        results.clear()
        with results:
            chart_image = {}
            ui.select(
                {str(path): path.name for path in chart_paths},
                value=str(chart_paths[0]),
                label="Chart",
                on_change=lambda event: chart_image["element"].set_source(event.value),
            ).classes("w-full")
            chart_image["element"] = ui.image(chart_paths[0]).classes("w-full")
            with ui.expansion("Summary statistics").classes("w-full"):
                ui.label(summary_text).classes("whitespace-pre-wrap")

    if visualisation_runs.current_directory is not None:
        show_results(visualisation_runs.current_directory)

    async def generate() -> None:
        button.disable()
        spinner.set_visibility(True)
        status.set_text("Generating visualisations...")
        try:
            output_directory = await visualisation_runs.generate(DEFAULT_CSV)
            if output_directory is None:
                status.set_text("Visualisations could not be generated.")
                return
            show_results(output_directory)
            status.set_text("Visualisations ready")
        except Exception:
            logger.exception("Visualisation generation failed")
            status.set_text("Visualisations failed; see container logs for details.")
            ui.notify(
                "Visualisations failed; see container logs for details.",
                type="negative",
            )
        finally:
            spinner.set_visibility(False)
            button.enable()

    button = ui.button("Generate visualisations", on_click=generate).props("no-caps")


ui.run_with(app, favicon=ASSETS_DIRECTORY / "wildl-id-icon.svg", mount_path="/")
