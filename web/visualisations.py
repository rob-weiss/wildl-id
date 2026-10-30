import asyncio
import math
import re
import shutil
import tempfile
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from astral import LocationInfo
from astral.sun import sun
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
FILTERED_PLOT_ROOT = Path("/tmp/wildl-id-filtered-plots")
FILTERED_PLOT_TYPES = {
    "daily-yearly": "Daily yearly activity by species",
    "individual-timelines": "Individual species timelines",
}
SUN_LOCATION = LocationInfo(
    "Camera Location", "Germany", "Europe/Berlin", 48.7667, 8.9333
)
STANDARD_TIMEZONE = pytz.timezone("Etc/GMT-1")


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


def _sunrise_sunset(day: date):
    try:
        solar_times = sun(SUN_LOCATION.observer, date=day)
        return tuple(
            solar_times[key].astimezone(STANDARD_TIMEZONE).replace(tzinfo=None)
            for key in ("sunrise", "sunset")
        )
    except Exception:
        return None, None


def _add_solar_activity(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["date"] = frame["timestamp"].dt.date
    solar_by_date = {day: _sunrise_sunset(day) for day in frame["date"].unique()}
    frame["sunrise"] = frame["date"].map(lambda day: solar_by_date[day][0])
    frame["sunset"] = frame["date"].map(lambda day: solar_by_date[day][1])
    frame = frame[frame["sunrise"].notna() & frame["sunset"].notna()].copy()
    frame["hours_from_sunrise"] = (
        frame["timestamp"] - frame["sunrise"]
    ).dt.total_seconds() / 3600
    frame["hours_from_sunset"] = (
        frame["timestamp"] - frame["sunset"]
    ).dt.total_seconds() / 3600
    return frame


def matplotlib_date_values(values) -> np.ndarray:
    dates = pd.DatetimeIndex(pd.to_datetime(values))
    return np.asarray(mdates.date2num(dates.to_pydatetime()))


def _create_daily_yearly_figure(frame: pd.DataFrame):
    species = frame["species"].iloc[0]
    start_date = frame["date"].min()
    end_date = frame["date"].max()
    figure = plt.figure(figsize=(18, 10), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=[4, 1, 0.2],
        height_ratios=[1, 4],
        hspace=0.05,
        wspace=0.2,
    )
    ax_top = figure.add_subplot(grid[0, 0])
    ax_main = figure.add_subplot(grid[1, 0], sharex=ax_top)
    ax_right = figure.add_subplot(grid[1, 1], sharey=ax_main)
    ax_colorbar = figure.add_subplot(grid[1, 2])

    frame = frame.copy()
    frame["hour_decimal"] = (
        frame["timestamp"].dt.hour + frame["timestamp"].dt.minute / 60
    )
    date_values = matplotlib_date_values(frame["date"])
    scatter = ax_main.scatter(
        frame["hour_decimal"],
        date_values,
        c=frame["hours_from_sunset"],
        cmap="RdYlBu_r",
        alpha=0.6,
        s=30,
        edgecolors="black",
        linewidth=0.3,
        vmin=-3,
        vmax=3,
    )

    interval_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    solar_rows = [(_sunrise_sunset(day.date()), day.date()) for day in interval_dates]
    valid_solar = [
        (sunrise, sunset, day)
        for (sunrise, sunset), day in solar_rows
        if sunrise is not None and sunset is not None
    ]
    if valid_solar:
        sunrise_hours = [value.hour + value.minute / 60 for value, _, _ in valid_solar]
        sunset_hours = [value.hour + value.minute / 60 for _, value, _ in valid_solar]
        solar_date_values = matplotlib_date_values([day for _, _, day in valid_solar])
        ax_main.fill_betweenx(
            solar_date_values,
            [value - 0.5 for value in sunrise_hours],
            sunrise_hours,
            alpha=0.3,
            color="gold",
            label="0.5 hours before Sunrise",
            zorder=5,
        )
        ax_main.fill_betweenx(
            solar_date_values,
            sunrise_hours,
            [value + 1.5 for value in sunrise_hours],
            alpha=0.2,
            color="gold",
            label="1.5 hours after Sunrise",
            zorder=5,
        )
        ax_main.fill_betweenx(
            solar_date_values,
            [value - 1.5 for value in sunset_hours],
            sunset_hours,
            alpha=0.2,
            color="orange",
            label="1.5 hours before Sunset",
            zorder=5,
        )
        ax_main.fill_betweenx(
            solar_date_values,
            sunset_hours,
            [value + 0.5 for value in sunset_hours],
            alpha=0.3,
            color="orange",
            label="0.5 hours after Sunset",
            zorder=5,
        )
        ax_main.plot(
            sunset_hours,
            solar_date_values,
            color="orange",
            linewidth=3,
            label="Sunset Time",
            zorder=10,
        )
        ax_main.plot(
            sunrise_hours,
            solar_date_values,
            color="gold",
            linewidth=3,
            label="Sunrise Time",
            zorder=10,
        )

    ax_top.hist(
        frame["hour_decimal"],
        bins=48,
        color="steelblue",
        alpha=0.5,
        density=True,
    )
    if frame["hour_decimal"].nunique() > 1:
        from scipy import stats

        kde = stats.gaussian_kde(frame["hour_decimal"])
        x_range = np.linspace(0, 24, 200)
        ax_top.plot(x_range, kde(x_range), "r-", linewidth=2, label="Density")
    ax_top.set_ylabel("Density")
    ax_top.tick_params(labelbottom=False)
    ax_top.grid(True, alpha=0.3, axis="y")
    ax_top.set_xlim(0, 24)

    ax_right.hist(
        date_values,
        bins=min(52, max(1, len(np.unique(date_values)))),
        orientation="horizontal",
        color="steelblue",
        alpha=0.5,
        edgecolor="black",
        density=True,
    )
    if len(np.unique(date_values)) > 1:
        from scipy import stats

        kde = stats.gaussian_kde(date_values)
        y_range = np.linspace(date_values.min(), date_values.max(), 100)
        ax_right.plot(kde(y_range), y_range, "r-", linewidth=2)
    ax_right.set_xlabel("Density")
    ax_right.tick_params(labelleft=False)
    ax_right.grid(True, alpha=0.3, axis="x")

    ax_main.set_xlabel("Hour of Day")
    ax_main.set_ylabel("Date")
    ax_main.set_xlim(0, 24)
    ax_main.set_xticks(range(0, 25, 2), [f"{hour:02d}:00" for hour in range(0, 25, 2)])
    ax_main.yaxis_date()
    date_locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax_main.yaxis.set_major_locator(date_locator)
    ax_main.yaxis.set_major_formatter(mdates.ConciseDateFormatter(date_locator))
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc="upper right", fontsize=8)
    ax_main.invert_yaxis()

    colorbar = figure.colorbar(scatter, cax=ax_colorbar)
    colorbar.set_label("Hours from Sunset", rotation=270, labelpad=20)
    sunset_hot_zone = frame["hours_from_sunset"].between(-1.5, 0.5).sum()
    sunrise_hot_zone = frame["hours_from_sunrise"].between(-0.5, 1.5).sum()
    sunset_hot_zone_pct = 100 * sunset_hot_zone / len(frame)
    sunrise_hot_zone_pct = 100 * sunrise_hot_zone / len(frame)
    ax_main.text(
        0.02,
        0.98,
        f"Sunset hot zone: {sunset_hot_zone} ({sunset_hot_zone_pct:.1f}%)\n"
        f"Sunrise hot zone: {sunrise_hot_zone} ({sunrise_hot_zone_pct:.1f}%)",
        transform=ax_main.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
        fontsize=9,
        fontweight="bold",
    )
    figure.suptitle(
        f"{species.capitalize()} Activity Throughout Year and Day "
        f"({start_date} to {end_date}, n={len(frame)})",
        fontsize=14,
        fontweight="bold",
    )
    return figure


def _create_individual_timelines_figure(frame: pd.DataFrame):
    species_names = sorted(frame["species"].unique())
    columns = 2
    rows = math.ceil(len(species_names) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(16, max(4 * rows, 5)),
        sharex=True,
        squeeze=False,
    )
    colors = plt.get_cmap("tab20", max(len(species_names), 1))
    for index, species in enumerate(species_names):
        axis = axes.flat[index]
        daily = (
            frame.loc[frame["species"] == species]
            .groupby(frame.loc[frame["species"] == species]["timestamp"].dt.date)
            .size()
            .rename_axis("date")
            .reset_index(name="count")
            .sort_values("date")
        )
        daily["moving_avg"] = (
            daily["count"].rolling(7, center=True, min_periods=1).mean()
        )
        color = colors(index)
        axis.stem(
            pd.to_datetime(daily["date"]),
            daily["count"],
            linefmt=color,
            markerfmt="o",
            basefmt=" ",
        )
        axis.plot(
            pd.to_datetime(daily["date"]),
            daily["moving_avg"],
            color="darkred",
            linewidth=2,
            label="7-observation moving avg",
        )
        species_count = (frame["species"] == species).sum()
        axis.set_title(f"{species.capitalize()} Activity (n={species_count})")
        axis.set_ylabel("Detections")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", fontsize=8)
        axis.tick_params(axis="x", rotation=45)

    for axis in axes.flat[len(species_names) :]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        axis.set_xlabel("Date")
    start_date = frame["timestamp"].min().date()
    end_date = frame["timestamp"].max().date()
    figure.suptitle(
        f"Individual Species Activity Timelines ({start_date} to {end_date})",
        fontsize=16,
        fontweight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.98])
    return figure


def _species_filename(species: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", species.casefold()).strip("_")
    return f"{14 + index:02d}_{slug}_daily_yearly_activity_pattern.svg"


def _save_svg(figure, path: Path) -> None:
    with plt.rc_context({"svg.fonttype": "none"}):
        figure.savefig(path, format="svg", bbox_inches="tight")


def generate_filtered_plot_svgs(
    csv_path: Path, prompt: str, plot_type: str, output_dir: Path
) -> list[Path]:
    """Generate filter-scoped SVG files in a dedicated output directory."""
    if plot_type not in FILTERED_PLOT_TYPES:
        raise ValueError(f"Unknown filtered plot type: {plot_type}")
    frame = filtered_activity_frame(csv_path, prompt)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_plot in output_dir.glob("*.svg"):
        old_plot.unlink()

    if plot_type == "daily-yearly":
        frame = _add_solar_activity(frame)
        if frame.empty:
            raise ValueError("No rows have valid sunrise and sunset times")
        paths = []
        for index, species in enumerate(sorted(frame["species"].unique())):
            species_frame = frame[frame["species"] == species]
            figure = _create_daily_yearly_figure(species_frame)
            path = output_dir / _species_filename(species, index)
            try:
                _save_svg(figure, path)
            finally:
                plt.close(figure)
            paths.append(path)
    else:
        figure = _create_individual_timelines_figure(frame)
        path = output_dir / "09_individual_species_timelines.svg"
        try:
            _save_svg(figure, path)
        finally:
            plt.close(figure)
        paths = [path]
    return paths


class FilteredPlotRunStore:
    def __init__(self, output_root: Path = FILTERED_PLOT_ROOT):
        self.output_root = Path(output_root)
        self.current_directory: Path | None = None
        self.current_paths: list[Path] = []
        self._lock = asyncio.Lock()
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    async def generate(self, csv_path: Path, prompt: str, plot_type: str) -> list[Path]:
        if self._lock.locked():
            raise RuntimeError("A filtered plot run is already running")

        async with self._lock:
            self._is_running = True
            run_directory = None
            try:
                self.output_root.mkdir(parents=True, exist_ok=True)
                run_directory = Path(
                    tempfile.mkdtemp(prefix="run-", dir=self.output_root)
                )
                generated = await run.io_bound(
                    generate_filtered_plot_svgs,
                    Path(csv_path),
                    prompt,
                    plot_type,
                    run_directory,
                )
                if not generated or not all(path.is_file() for path in generated):
                    raise RuntimeError("Filtered plot SVGs were not generated")

                previous_directory = self.current_directory
                self.current_directory = run_directory
                self.current_paths = generated
                run_directory = None
                if previous_directory is not None:
                    shutil.rmtree(previous_directory, ignore_errors=True)
                return self.current_paths
            finally:
                if run_directory is not None:
                    shutil.rmtree(run_directory, ignore_errors=True)
                self._is_running = False


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
