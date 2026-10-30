import csv
import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from nicegui import ui
from src.labelling.review_app import IMAGES_DIR, ReviewApp
from src.labelling.review_labels import DEFAULT_CSV
from web.gallery import LOCAL_IMAGE_ROUTE, build_gallery_page, build_recent_gallery
from web.visualisations import (
    FILTERED_PLOT_TYPES,
    FilteredPlotRunStore,
    VisualisationRunStore,
)

app = FastAPI()
visualisation_runs = VisualisationRunStore()
filtered_plot_runs = FilteredPlotRunStore()
logger = logging.getLogger(__name__)
ASSETS_DIRECTORY = Path(__file__).resolve().parent.parent / "docs" / "assets"
app.mount("/assets", StaticFiles(directory=ASSETS_DIRECTORY), name="brand-assets")
app.mount(
    LOCAL_IMAGE_ROUTE,
    StaticFiles(directory=IMAGES_DIR, check_dir=False),
    name="local-images",
)


def add_brand_logo() -> None:
    ui.image("/assets/wildl-id-logo.svg").props('alt="wildl-id logo"').classes(
        "w-64 max-w-full"
    )


def add_navigation(current_path: str) -> None:
    with (
        ui.left_drawer(value=True, bordered=True)
        .props("mini mini-width=64 width=240 behavior=desktop")
        .classes("bg-gray-50 p-0 gap-2") as drawer
    ):

        def toggle_navigation() -> None:
            collapsed = not drawer.props.get("mini", False)
            drawer.props["mini"] = collapsed
            drawer.update()
            toggle.props(
                f"icon={'chevron_right' if collapsed else 'chevron_left'} "
                f'aria-label="{"Expand" if collapsed else "Collapse"} navigation" '
                f"aria-expanded={str(not collapsed).lower()}"
            )

        with ui.row().classes("w-full items-center gap-0 flex-nowrap"):
            with ui.link(target="/").classes("shrink-0 p-2"):
                ui.image("/assets/wildl-id-icon.svg").props(
                    'alt="wildl-id home"'
                ).classes("w-8 h-8")
            ui.image("/assets/wildl-id-logo.svg").props('alt="wildl-id logo"').classes(
                "q-mini-drawer-hide w-40 max-w-full"
            )

        toggle = (
            ui.button(icon="chevron_right", on_click=toggle_navigation)
            .props('flat round aria-label="Expand navigation" aria-expanded=false')
            .classes("self-start ml-1")
            .mark("navigation-toggle")
        )
        toggle.tooltip("Toggle navigation")
        with ui.list().classes("w-full"):
            for title, target, icon in (
                ("Home", "/", "home"),
                ("Manual review", "/review", "fact_check"),
                ("Gallery", "/gallery", "photo_library"),
                ("Visualisations", "/visualisations", "bar_chart"),
            ):
                with (
                    ui.item()
                    .props(f'href="{target}" clickable aria-label="{title}"')
                    .classes("min-h-12") as item
                ):
                    if current_path == target:
                        item.props('active aria-current="page"')
                    with ui.item_section().props("avatar"):
                        ui.icon(icon).classes("text-2xl")
                    with ui.item_section():
                        ui.item_label(title)
                    ui.tooltip(title)


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
    add_navigation("/")
    with ui.column().classes("w-full max-w-3xl mx-auto p-4 sm:p-6 gap-6"):
        add_brand_logo()
        ui.label("Wildlife analysis").classes("text-2xl font-semibold")
        ui.label(
            "Explore camera-trap images, review wildlife labels, "
            "and visualise animal activity."
        ).classes("text-base text-gray-700 max-w-xl")
        build_recent_gallery(
            DEFAULT_CSV, image_base_url=os.environ.get("WILDL_ID_IMAGE_BASE_URL")
        )
        with ui.column().classes("w-full gap-0"):
            for title, target, icon, description in (
                (
                    "Manual review",
                    "/review",
                    "fact_check",
                    "Check detections and correct species labels.",
                ),
                (
                    "Gallery",
                    "/gallery",
                    "photo_library",
                    "Browse wildlife images and filter your sightings.",
                ),
                (
                    "Visualisations",
                    "/visualisations",
                    "bar_chart",
                    "Generate charts and summary statistics.",
                ),
            ):
                with ui.row().classes(
                    "w-full items-start flex-nowrap gap-4 py-5 border-t border-gray-200"
                ):
                    ui.icon(icon).classes("text-2xl text-gray-600 shrink-0 mt-1")
                    with ui.column().classes("min-w-0 gap-1"):
                        ui.link(title, target).classes("text-lg font-medium")
                        ui.label(description).classes("text-gray-600")


@ui.page("/review")
def review_page() -> None:
    add_navigation("/review")
    build_default_review_page()


@ui.page("/gallery")
def gallery_page() -> None:
    add_navigation("/gallery")
    add_brand_logo()
    build_gallery_page(
        DEFAULT_CSV, image_base_url=os.environ.get("WILDL_ID_IMAGE_BASE_URL")
    )


@ui.page("/visualisations")
def visualisations_page() -> None:
    add_navigation("/visualisations")
    add_brand_logo()
    ui.label("Visualisations").classes("text-xl font-semibold")
    ui.label("Filter-based plots").classes("text-lg font-semibold mt-4")
    ui.label(
        "Uses the gallery filter syntax. Daily yearly plots create one SVG per "
        "matching species; timelines include all matching species. Sort order "
        "and result limits do not affect plots."
    ).classes("text-sm text-gray-600")
    with ui.row().classes("w-full items-center gap-2"):
        plot_filter = (
            ui.input("Plot filter", placeholder='e.g. "foxes in 2024"')
            .props("clearable")
            .classes("flex-grow")
        )
        plot_type = ui.select(
            FILTERED_PLOT_TYPES,
            value="daily-yearly",
            label="Plot type",
        ).classes("min-w-64")
        dynamic_button = ui.button("Generate filtered plot", icon="show_chart").props(
            "no-caps"
        )
    dynamic_status = ui.label().classes("text-sm text-gray-600")
    dynamic_spinner = ui.spinner(size="sm")
    dynamic_spinner.set_visibility(False)
    filtered_results = ui.column().classes("w-full gap-3")

    def show_filtered_results(paths: list[Path]) -> None:
        filtered_results.clear()
        with filtered_results:
            chart_image = {}
            ui.select(
                {str(path): path.name for path in paths},
                value=str(paths[0]),
                label="Filtered SVG",
                on_change=lambda event: chart_image["element"].set_source(event.value),
            ).classes("w-full")
            chart_image["element"] = ui.image(paths[0]).classes("w-full max-w-6xl")

    if filtered_plot_runs.current_paths:
        show_filtered_results(filtered_plot_runs.current_paths)

    async def generate_filtered_plot() -> None:
        dynamic_button.disable()
        dynamic_spinner.set_visibility(True)
        dynamic_status.set_text("Generating filtered plot...")
        try:
            paths = await filtered_plot_runs.generate(
                DEFAULT_CSV,
                plot_filter.value or "",
                plot_type.value,
            )
            show_filtered_results(paths)
            dynamic_status.set_text(f"{len(paths)} filtered SVG ready")
        except ValueError as error:
            dynamic_status.set_text(str(error))
        except Exception:
            logger.exception("Filtered plot generation failed")
            dynamic_status.set_text(
                "Filtered plot generation failed; see container logs for details."
            )
        finally:
            dynamic_spinner.set_visibility(False)
            dynamic_button.enable()

    dynamic_button.on_click(generate_filtered_plot)
    plot_filter.on("keydown.enter", generate_filtered_plot)

    ui.separator()
    ui.label("Existing plots").classes("text-lg font-semibold")
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
