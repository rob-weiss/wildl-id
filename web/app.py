import csv
import logging
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from nicegui import ui
from src.alerting import notification_log
from src.labelling.label_data import DATE_FORMAT
from src.labelling.review_app import IMAGES_DIR, ReviewApp
from src.labelling.review_labels import DEFAULT_CSV
from web.camera_map import build_camera_map_page
from web.gallery import LOCAL_IMAGE_ROUTE, build_gallery_page, build_recent_gallery
from web.predictions import build_predictions_page
from web.visualisations import (
    FILTERED_PLOT_TYPES,
    FilteredPlotRunStore,
    VisualisationRunStore,
)

app = FastAPI()
visualisation_runs = VisualisationRunStore()
filtered_plot_runs = FilteredPlotRunStore()
logger = logging.getLogger(__name__)
LOCAL_TIMEZONE = ZoneInfo("Europe/Berlin")
ASSETS_DIRECTORY = Path(__file__).resolve().parent.parent / "docs" / "assets"
app.mount("/assets", StaticFiles(directory=ASSETS_DIRECTORY), name="brand-assets")
app.mount(
    LOCAL_IMAGE_ROUTE,
    StaticFiles(directory=IMAGES_DIR, check_dir=False),
    name="local-images",
)


def add_brand_logo() -> None:
    ui.image("/assets/wildl-id-logo.svg").props('alt="wildl-id logo"').classes(
        "w-64 max-w-full mx-auto"
    )


def add_navigation(current_path: str) -> None:
    ui.colors(primary="#214a3a")
    ui.add_head_html(
        """<style>
        body { background: #f4f6f4; }
        a:not(.q-item):not(.q-btn) { color: #53675d; }
        a:not(.q-item):not(.q-btn):hover { color: #214a3a; }
        .q-drawer--mini .navigation-header {
            flex-direction: column;
            justify-content: center;
        }
        </style>"""
    )
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

        # Fixed height keeps the navigation items in place when toggling.
        with (
            ui.element("div")
            .classes(
                "navigation-header w-full h-24 shrink-0 flex flex-nowrap "
                "items-center justify-between gap-1 px-2"
            )
            .mark("navigation-header")
        ):
            with ui.link(target="/").classes("q-mini-drawer-only"):
                ui.image("/assets/wildl-id-icon.svg").props(
                    'alt="wildl-id home"'
                ).classes("w-8 h-8")
            with ui.link(target="/").classes("q-mini-drawer-hide"):
                ui.image("/assets/wildl-id-logo.svg").props(
                    'alt="wildl-id home"'
                ).classes("w-40")
            toggle = (
                ui.button(icon="chevron_right", on_click=toggle_navigation)
                .props('flat round aria-label="Expand navigation" aria-expanded=false')
                .mark("navigation-toggle")
            )
            toggle.tooltip("Toggle navigation")
        with ui.list().classes("w-full"):
            for title, target, icon, description in (
                ("Home", "/", "home", "Recent wildlife sightings and camera activity."),
                (
                    "Review",
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
                    "Activity charts and summary statistics.",
                ),
                (
                    "Predictions",
                    "/predictions",
                    "show_chart",
                    "Expected activity over the next 24 hours.",
                ),
                (
                    "Camera map",
                    "/map",
                    "map",
                    "Camera locations and wind for the next three hours.",
                ),
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
                    ui.tooltip(f"{title}: {description}")


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


def build_default_review_page() -> None:
    build_review_page(DEFAULT_CSV)


def parse_summary_statistics(
    summary_text: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    metrics = []
    species_counts = []
    location_counts = []
    section = "metrics"
    for line in summary_text.splitlines():
        value = line.strip()
        if value == "SPECIES COUNTS:":
            section = "species"
            continue
        if value == "LOCATION COUNTS:":
            section = "locations"
            continue
        if section == "metrics" and ":" in value:
            metric, result = value.split(":", 1)
            if metric.strip() and result.strip():
                metrics.append({"metric": metric.strip(), "value": result.strip()})
            continue
        if section in {"species", "locations"}:
            fields = value.rsplit(maxsplit=1)
            if len(fields) == 2 and fields[1].isdigit():
                row = {"count": fields[1]}
                row["species" if section == "species" else "location"] = fields[0]
                (species_counts if section == "species" else location_counts).append(
                    row
                )
    if not metrics:
        raise ValueError("Summary statistics contain no metric rows")
    return metrics, species_counts, location_counts


def plot_display_name(path: Path) -> str:
    return Path(path).stem.replace("_", " ")


@ui.page("/")
def home_page() -> None:
    add_navigation("/")
    with ui.column().classes("w-full max-w-3xl mx-auto p-4 sm:p-6 gap-6"):
        with ui.column().classes("w-full items-center gap-2 text-center"):
            add_brand_logo()
            ui.label("Wildlife analysis").classes("text-2xl font-semibold")
            ui.label(
                "Recent sightings, image review, and activity across your cameras."
            ).classes("text-base text-gray-700 max-w-xl")
        build_recent_gallery(
            DEFAULT_CSV, image_base_url=os.environ.get("WILDL_ID_IMAGE_BASE_URL")
        )
        build_recent_notifications()


def build_recent_notifications() -> None:
    ui.label("Latest notifications").classes("text-xl font-semibold")

    @ui.refreshable
    def notification_list() -> None:
        notifications = notification_log.recent_notifications(5)
        if not notifications:
            ui.label("No notifications sent yet").classes("text-sm text-gray-600")
            return
        with ui.column().classes("w-full gap-2"):
            for notification in notifications:
                sent = notification["sent"].astimezone(LOCAL_TIMEZONE)
                with ui.card().classes("w-full gap-1"):
                    ui.label(notification["title"]).classes("font-medium")
                    ui.label(notification["message"]).classes(
                        "text-sm whitespace-pre-line"
                    )
                    ui.label(f"{sent:{DATE_FORMAT} %H:%M}").classes(
                        "text-xs text-gray-600"
                    )

    notification_list()
    ui.timer(60, notification_list.refresh, immediate=False)


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


@ui.page("/predictions")
async def predictions_page() -> None:
    add_navigation("/predictions")
    add_brand_logo()
    await build_predictions_page(DEFAULT_CSV)


@ui.page("/map")
async def camera_map_page() -> None:
    add_navigation("/map")
    add_brand_logo()
    await build_camera_map_page()


@ui.page("/visualisations")
def visualisations_page() -> None:
    add_navigation("/visualisations")
    add_brand_logo()
    ui.label("Visualisations").classes("text-xl font-semibold")
    ui.label("Filter-based plots").classes("text-lg font-semibold mt-4")
    ui.label(
        "Enter filter terms as in the gallery to select the images. Daily/yearly "
        "charts create one plot per matching species; individual timelines "
        "compare all matching species. Sort order and result limits are ignored."
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
    selected_plot = {"path": None}

    def show_filtered_results(preferred_path: Path | None = None) -> None:
        filtered_results.clear()
        available_plots = [
            (path, f"{plot_run.prompt or 'All images'} · {plot_display_name(path)}")
            for plot_run in filtered_plot_runs.runs
            for path in plot_run.paths
        ]
        if not available_plots:
            selected_plot["path"] = None
            return
        options = {str(path): label for path, label in available_plots}
        selected_path = str(preferred_path or selected_plot["path"] or "")
        if selected_path not in options:
            selected_path = str(available_plots[-1][0])
        selected_plot["path"] = selected_path
        with filtered_results:
            chart_image = {}
            ui.select(
                options,
                value=selected_path,
                label="Filtered SVGs",
                on_change=lambda event: show_selected_plot(event, chart_image),
            ).classes("w-full")
            chart_image["element"] = ui.image(selected_path).classes("w-full max-w-6xl")
            ui.button(
                "Remove selected SVG",
                icon="delete",
                on_click=remove_selected_plot,
            ).props("outline no-caps")

    def show_selected_plot(event, chart_image) -> None:
        selected_plot["path"] = event.value
        chart_image["element"].set_source(event.value)

    async def remove_selected_plot() -> None:
        selected_path = selected_plot["path"]
        if selected_path is None:
            return
        path = Path(selected_path)
        plot_run = next(
            (run for run in filtered_plot_runs.runs if path in run.paths), None
        )
        if plot_run is None:
            dynamic_status.set_text("Selected SVG is no longer available")
            show_filtered_results()
            return
        await filtered_plot_runs.remove_plot(plot_run, path)
        dynamic_status.set_text("Filtered SVG removed")
        show_filtered_results()

    if filtered_plot_runs.runs:
        show_filtered_results()

    async def generate_filtered_plot() -> None:
        dynamic_button.disable()
        dynamic_spinner.set_visibility(True)
        dynamic_status.set_text("Generating filtered plot...")
        try:
            plot_run = await filtered_plot_runs.generate(
                DEFAULT_CSV,
                plot_filter.value or "",
                plot_type.value,
            )
            show_filtered_results(plot_run.paths[-1])
            dynamic_status.set_text(
                f"{len(plot_run.paths)} filtered SVG ready; "
                f"{len(filtered_plot_runs.runs)} runs retained"
            )
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
    ui.label("Activity charts").classes("text-lg font-semibold")
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
        try:
            metrics, species_counts, location_counts = parse_summary_statistics(
                summary_text
            )
        except ValueError:
            logger.exception("Could not parse visualisation summary statistics")
            status.set_text("Summary statistics unavailable; see container logs.")
            metrics, species_counts, location_counts = [], [], []
        results.clear()
        with results:
            chart_image = {}
            ui.select(
                {str(path): plot_display_name(path) for path in chart_paths},
                value=str(chart_paths[0]),
                label="Chart",
                on_change=lambda event: chart_image["element"].set_source(event.value),
            ).classes("w-full")
            chart_image["element"] = ui.image(chart_paths[0]).classes("w-full")
            with ui.expansion("Summary statistics").classes("w-full"):
                ui.table(
                    columns=[
                        {
                            "name": "metric",
                            "label": "Statistic",
                            "field": "metric",
                            "align": "left",
                        },
                        {
                            "name": "value",
                            "label": "Value",
                            "field": "value",
                            "align": "left",
                        },
                    ],
                    rows=metrics,
                    row_key="metric",
                ).props("flat bordered").classes("w-full")
                ui.label("Species counts").classes("font-medium mt-3")
                ui.table(
                    columns=[
                        {
                            "name": "species",
                            "label": "Species",
                            "field": "species",
                            "align": "left",
                        },
                        {
                            "name": "count",
                            "label": "Images",
                            "field": "count",
                            "align": "right",
                        },
                    ],
                    rows=species_counts,
                    row_key="species",
                ).props("flat bordered").classes("w-full")
                ui.label("Location counts").classes("font-medium mt-3")
                ui.table(
                    columns=[
                        {
                            "name": "location",
                            "label": "Location",
                            "field": "location",
                            "align": "left",
                        },
                        {
                            "name": "count",
                            "label": "Images",
                            "field": "count",
                            "align": "right",
                        },
                    ],
                    rows=location_counts,
                    row_key="location",
                ).props("flat bordered").classes("w-full")

    if visualisation_runs.current_directory is not None:
        show_results(visualisation_runs.current_directory)

    async def generate() -> None:
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

    ui.timer(0.1, generate, once=True)


ui.run_with(
    app,
    title="wildl-id | Wildlife Identification",
    favicon=ASSETS_DIRECTORY / "wildl-id-icon.svg",
    mount_path="/",
)
