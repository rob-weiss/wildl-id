import asyncio
import base64
import calendar
import io
import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd
from nicegui import run
from src.visualisation import evaluate_labels
from web.gallery import (
    filter_rows,
    load_rows,
    parse_filter_prompt,
    row_datetime,
    row_label,
)

DEFAULT_OUTPUT_ROOT = Path("/tmp/wildl-id-visualisations")
FILTERED_PLOT_TYPES = {
    "yearly-seasonal": "Year-over-year seasonal activity",
    "month-hour-heatmap": "Month-by-hour activity heatmap",
}
MAX_DYNAMIC_PLOT_SPECIES = 6


def filtered_activity_frame(csv_path: Path, prompt: str) -> pd.DataFrame:
    """Return timestamped rows matching the gallery's local filter syntax."""
    rows = load_rows(csv_path)
    labels = {
        str(value).strip()
        for row in rows
        for value in (row.get("class"), row.get("manual_label"))
        if str(value or "").strip()
    }
    locations = {row["location_id"] for row in rows}
    gallery_filter = parse_filter_prompt(prompt, labels, locations)
    if prompt.strip() and gallery_filter.is_empty():
        raise ValueError("No supported filter terms found")

    gallery_filter.oldest_first = False
    gallery_filter.limit = None
    records = []
    for row in filter_rows(rows, gallery_filter):
        timestamp = row_datetime(row)
        if timestamp is not None:
            records.append({"timestamp": timestamp, "species": row_label(row)})
    if not records:
        raise ValueError("No timestamped rows match these filters")

    frame = pd.DataFrame(records)
    frame["year"] = frame["timestamp"].dt.year
    frame["month"] = frame["timestamp"].dt.month
    frame["hour"] = frame["timestamp"].dt.hour
    return frame


def _yearly_seasonal_figure(frame: pd.DataFrame):
    species_names = frame["species"].value_counts().head(MAX_DYNAMIC_PLOT_SPECIES).index
    figure, axes = plt.subplots(
        len(species_names),
        1,
        figsize=(12, max(3.2 * len(species_names), 4)),
        sharex=True,
        squeeze=False,
    )
    month_numbers = range(1, 13)
    for axis, species in zip(axes[:, 0], species_names):
        species_frame = frame[frame["species"] == species]
        for year, year_frame in species_frame.groupby("year", sort=True):
            monthly_counts = (
                year_frame.groupby("month").size().reindex(month_numbers, fill_value=0)
            )
            annual_total = monthly_counts.sum()
            if annual_total:
                axis.plot(
                    month_numbers,
                    monthly_counts / annual_total * 100,
                    marker="o",
                    linewidth=1.8,
                    label=str(year),
                )
        axis.set_title(str(species).capitalize(), loc="left", fontweight="bold")
        axis.set_ylabel("Share (%)")
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.3)
        axis.legend(title="Year", ncols=min(frame["year"].nunique(), 6))

    axes[-1, 0].set_xticks(
        list(month_numbers), [calendar.month_abbr[month] for month in month_numbers]
    )
    axes[-1, 0].set_xlabel("Month")
    figure.suptitle("Year-over-year seasonal activity", fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    return figure


def _month_hour_heatmap_figure(frame: pd.DataFrame):
    species_names = frame["species"].value_counts().head(MAX_DYNAMIC_PLOT_SPECIES).index
    matrices = []
    maximum = 0.0
    for species in species_names:
        counts = (
            frame.loc[frame["species"] == species]
            .groupby(["month", "hour"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=range(1, 13), columns=range(24), fill_value=0)
        )
        values = counts.to_numpy(dtype=float)
        total = values.sum()
        if total:
            values = values / total * 100
        maximum = max(maximum, float(values.max()))
        matrices.append(values)

    figure, axes = plt.subplots(
        len(species_names),
        1,
        figsize=(13, max(3.2 * len(species_names), 4)),
        sharex=True,
        squeeze=False,
    )
    image = None
    for axis, species, values in zip(axes[:, 0], species_names, matrices):
        image = axis.imshow(
            values,
            aspect="auto",
            origin="lower",
            cmap="YlGnBu",
            vmin=0,
            vmax=maximum or 1,
        )
        axis.set_title(str(species).capitalize(), loc="left", fontweight="bold")
        axis.set_yticks(
            range(12), [calendar.month_abbr[month] for month in range(1, 13)]
        )
    axes[-1, 0].set_xticks(
        range(0, 24, 3), [f"{hour:02d}:00" for hour in range(0, 24, 3)]
    )
    axes[-1, 0].set_xlabel("Hour of day")
    figure.suptitle("Month-by-hour activity", fontweight="bold")
    if image is not None:
        figure.colorbar(
            image,
            ax=axes[:, 0].tolist(),
            label="Share of species detections (%)",
            pad=0.02,
        )
    figure.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.12, hspace=0.4)
    return figure


def generate_filtered_plot_svg(csv_path: Path, prompt: str, plot_type: str) -> str:
    """Generate a data-URI SVG for a supported filtered activity plot."""
    if plot_type not in FILTERED_PLOT_TYPES:
        raise ValueError(f"Unknown filtered plot type: {plot_type}")
    frame = filtered_activity_frame(csv_path, prompt)
    if plot_type == "yearly-seasonal":
        figure = _yearly_seasonal_figure(frame)
    else:
        figure = _month_hour_heatmap_figure(frame)

    output = io.BytesIO()
    try:
        figure.savefig(output, format="svg", bbox_inches="tight")
    finally:
        plt.close(figure)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


class VisualisationRunStore:
    def __init__(self, output_root: Path = DEFAULT_OUTPUT_ROOT):
        self.output_root = Path(output_root)
        self.current_directory: Path | None = None
        self._lock = asyncio.Lock()
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    async def generate(self, csv_path: Path) -> Path | None:
        if self._lock.locked():
            raise RuntimeError("A visualisation run is already running")

        async with self._lock:
            self._is_running = True
            run_directory = None
            try:
                self.output_root.mkdir(parents=True, exist_ok=True)
                run_directory = Path(
                    tempfile.mkdtemp(prefix="run-", dir=self.output_root)
                )
                generated = await run.io_bound(
                    evaluate_labels.main,
                    csv_path=Path(csv_path),
                    output_dir=run_directory,
                )
                if not generated:
                    return None
                if not (run_directory / "summary_statistics.txt").is_file():
                    raise RuntimeError("Visualisation summary was not generated")
                if not list(run_directory.glob("*.svg")):
                    raise RuntimeError("No visualisation charts were generated")

                previous_directory = self.current_directory
                self.current_directory = run_directory
                run_directory = None
                if previous_directory is not None:
                    shutil.rmtree(previous_directory, ignore_errors=True)
                return self.current_directory
            finally:
                if run_directory is not None:
                    shutil.rmtree(run_directory, ignore_errors=True)
                self._is_running = False
