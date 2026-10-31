"""Read-only inference and display for the fitted count-only Poisson model."""

import json
import os
import unicodedata
from dataclasses import dataclass
from datetime import timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from astral import Observer
from astral.sun import elevation, sun
from src.labelling import label_data
from web.gallery import RECENT_SPECIES, row_label

MODEL_COLUMNS = (
    "horizon",
    "hour_sin",
    "hour_cos",
    "year_sin",
    "year_cos",
    "solar_elevation",
    "daylight_hours",
)
DEFAULT_MODEL = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "activity_forecast"
    / "poisson"
    / "metadata.json"
)


class ForecastUnavailable(ValueError):
    pass


@dataclass(frozen=True)
class ForecastResult:
    frame: pd.DataFrame
    origin: pd.Timestamp
    latest_record: pd.Timestamp
    training_cutoff: pd.Timestamp
    local_timezone: str
    assumed_gap: bool


def configured_model_path() -> Path:
    return Path(os.environ.get("WILDL_ID_FORECAST_MODEL", str(DEFAULT_MODEL)))


def read_model(path: Path) -> dict:
    try:
        bundle = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            bundle["schema"] != 1
            or bundle["kind"] != "poisson"
            or bundle.get("include_weather")
            or bundle.get("include_moon")
            or tuple(bundle["columns"]) != MODEL_COLUMNS
        ):
            raise ValueError("Unsupported fitted model")
        species, sites = bundle["species"], bundle["sites"]
        if (
            not RECENT_SPECIES.issubset(species)
            or len(set(species)) != len(species)
            or len(set(sites)) != len(sites)
            or not sites
            or len(bundle["models"]) != len(species)
        ):
            raise ValueError("Invalid model species or sites")
        for model in bundle["models"]:
            if model is not None:
                for name, length in (
                    ("mean", 10),
                    ("scale", 10),
                    ("coef", 10 + len(sites)),
                ):
                    values = np.asarray(model[name], dtype=float)
                    if values.shape != (length,) or not np.isfinite(values).all():
                        raise ValueError("Invalid model dimensions")
                if (np.asarray(model["scale"]) <= 0).any() or not np.isfinite(
                    model["intercept"]
                ):
                    raise ValueError("Invalid model parameters")
        for site in bundle["config"]["sites"]:
            ZoneInfo(site["local_timezone"])
            if (
                not -90 <= site["latitude"] <= 90
                or not -180 <= site["longitude"] <= 180
            ):
                raise ValueError("Invalid site coordinates")
            if (
                type(site["ingestion_delay_hours"]) is not int
                or site["ingestion_delay_hours"] < 0
            ):
                raise ValueError("Invalid ingestion delay")
        cutoff = pd.Timestamp(bundle["training_cutoff"])
        if pd.isna(cutoff) or cutoff.tzinfo is None:
            raise ValueError("Invalid training cutoff")
        for values in bundle["references"].values():
            values = np.asarray(values, dtype=float)
            if (
                values.shape != (len(species),)
                or not np.isfinite(values).all()
                or (values < 0).any()
            ):
                raise ValueError("Invalid model reference")
        return bundle
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ForecastUnavailable(
            "Trained Poisson model unavailable or incompatible"
        ) from error


def forecast_next24(
    csv_path: Path,
    model_path: Path,
    location_id: str,
    assume_continuous: bool = False,
    now=None,
) -> ForecastResult:
    bundle = read_model(model_path)
    now = (
        pd.Timestamp.now(tz="UTC")
        if now is None
        else pd.Timestamp(now).tz_convert("UTC")
    )
    origin = now.floor("h") + pd.Timedelta(hours=1)
    cutoff = pd.Timestamp(bundle["training_cutoff"]).tz_convert("UTC")
    if cutoff > origin:
        raise ForecastUnavailable("Model uses observations after the forecast origin")
    site = next(
        (
            site
            for site in bundle["config"]["sites"]
            if site["location_id"] == location_id
        ),
        None,
    )
    if site is None:
        raise ForecastUnavailable("Camera is not configured in the trained model")
    history_end = origin - pd.Timedelta(hours=site["ingestion_delay_hours"])
    if history_end > now.floor("h"):
        raise ForecastUnavailable(
            "Recent observation coverage includes an unfinished hour"
        )
    try:
        data = (
            pd.DataFrame(label_data.load_rows(csv_path))
            if Path(csv_path).name == "human_labels.csv"
            else pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        )
        if not {"location_id", "image_file", "class"}.issubset(data.columns):
            raise ValueError("Missing label columns")
        data["location_id"] = data.location_id.map(
            lambda value: unicodedata.normalize("NFC", value.strip())
        )
        data = data.loc[data.location_id == location_id].copy()
        clock = (
            timezone(timedelta(hours=1))
            if site["timestamp_timezone"] == "UTC+01:00"
            else ZoneInfo(site["timestamp_timezone"])
        )
        records = []
        for row in data.to_dict("records"):
            raw = (
                row.get("timestamp_exif", "").strip()
                or row.get("timestamp", "").strip()
            )
            stamp = pd.Timestamp(raw)
            if pd.isna(stamp) or not row["image_file"].strip():
                raise ValueError("Missing capture time or image identity")
            if stamp.tzinfo is None:
                stamp = stamp.tz_localize(clock, ambiguous="raise", nonexistent="raise")
            stamp = stamp.tz_convert("UTC")
            if stamp <= now:
                records.append(
                    {
                        "timestamp": stamp,
                        "image_file": unicodedata.normalize(
                            "NFC", row["image_file"].strip()
                        ),
                        "species": row_label(row),
                    }
                )
        observations = pd.DataFrame(records)
        if observations.empty:
            raise ValueError("No camera observations")
        if (observations.groupby("image_file").timestamp.nunique() > 1).any():
            raise ValueError("Conflicting image timestamps")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ForecastUnavailable(
            "Camera observations are missing or invalid"
        ) from error
    latest = observations.timestamp.max()
    hours = pd.date_range(history_end - pd.Timedelta(hours=168), periods=168, freq="h")
    if observations.timestamp.min().floor("h") + pd.Timedelta(hours=1) > hours[0]:
        raise ForecastUnavailable(
            "Seven days of camera observation coverage are unavailable"
        )
    assumed_gap = latest.floor("h") < history_end - pd.Timedelta(hours=1)
    if assumed_gap and not assume_continuous:
        raise ForecastUnavailable("Recent observation coverage is incomplete")
    observations["hour"] = observations.timestamp.dt.floor("h")
    unique = observations.drop_duplicates(["image_file", "species"])
    index = pd.MultiIndex.from_product(
        [hours, bundle["species"]], names=["hour", "species"]
    )
    counts = (
        unique.groupby(["hour", "species"])
        .size()
        .reindex(index, fill_value=0)
        .unstack()
        .reindex(columns=bundle["species"])
    )
    past = counts.to_numpy(dtype=float)
    future_hours = pd.date_range(origin, periods=24, freq="h")
    local_future = future_hours.tz_convert(site["local_timezone"])
    local_past = hours.tz_convert(site["local_timezone"])
    observer = Observer(site["latitude"], site["longitude"])
    features = []
    for horizon, (stamp, local) in enumerate(zip(future_hours, local_future), 1):
        solar = sun(observer, date=local.date(), tzinfo=site["local_timezone"])
        features.append(
            [
                horizon,
                np.sin(2 * np.pi * local.hour / 24),
                np.cos(2 * np.pi * local.hour / 24),
                np.sin(2 * np.pi * (local.dayofyear - 1) / 365.25),
                np.cos(2 * np.pi * (local.dayofyear - 1) / 365.25),
                elevation(observer, stamp.to_pydatetime()),
                (solar["sunset"] - solar["sunrise"]).total_seconds() / 3600,
            ]
        )
    feature_matrix = np.asarray(features)
    encoded_site = np.tile(
        [float(location_id == name) for name in bundle["sites"]], (24, 1)
    )
    rows = []
    for species in sorted(RECENT_SPECIES):
        species_index = bundle["species"].index(species)
        baseline = np.array(
            [
                past[local_past.hour == stamp.hour, species_index].mean()
                if (local_past.hour == stamp.hour).any()
                else past[:, species_index].mean()
                for stamp in local_future
            ]
        )
        model = bundle["models"][species_index]
        means = baseline
        method = (
            "baseline"
            if model is None or location_id not in bundle["sites"]
            else "poisson"
        )
        if method == "poisson":
            values = np.column_stack(
                [
                    feature_matrix,
                    baseline,
                    np.full(24, past[-24:, species_index].mean()),
                    np.full(24, past[:, species_index].mean()),
                ]
            )
            design = np.column_stack(
                [
                    (values - np.asarray(model["mean"])) / np.asarray(model["scale"]),
                    encoded_site,
                ]
            )
            means = np.exp(design @ np.asarray(model["coef"]) + model["intercept"])
        if not np.isfinite(means).all():
            raise ForecastUnavailable("Model produced invalid activity values")
        reference = bundle["references"].get(
            location_id, [0.0] * len(bundle["species"])
        )[species_index]
        for stamp, value in zip(future_hours, means):
            rows.append(
                {
                    "species": species,
                    "valid_time": stamp,
                    "expected_count": value,
                    "activity_index": min(1.0, value / reference)
                    if reference > 0
                    else np.nan,
                    "clipped": bool(reference > 0 and value > reference),
                    "method": method,
                }
            )
    return ForecastResult(
        pd.DataFrame(rows), origin, latest, cutoff, site["local_timezone"], assumed_gap
    )


def chart_options(species, result=None, relative=True):
    colours = {
        "roe deer": "#157f72",
        "wild boar": "#6d5b87",
        "badger": "#53657d",
        "marten": "#a0445e",
        "fox": "#b85c32",
        "hare": "#287b9a",
        "raccoon": "#8a741b",
    }
    labels, values = [], []
    if result is not None:
        frame = result.frame.loc[result.frame.species == species].sort_values(
            "valid_time"
        )
        labels = [
            stamp.tz_convert(result.local_timezone).strftime("%H:%M %z")
            for stamp in frame.valid_time
        ]
        column = "activity_index" if relative else "expected_count"
        values = [
            round(float(value), 4) if pd.notna(value) else None
            for value in frame[column]
        ]
    return {
        "animation": False,
        "color": [colours[species]],
        "grid": {"left": 58, "right": 12, "top": 32, "bottom": 34},
        "tooltip": {"trigger": "axis", "confine": True},
        "xAxis": {
            "type": "category",
            "data": labels,
            "boundaryGap": False,
            "axisLabel": {"interval": 6, "fontSize": 10},
        },
        "yAxis": {
            "type": "value",
            "name": "Relative activity" if relative else "Images/hour",
            "min": 0,
            "max": 1 if relative else None,
            "nameTextStyle": {"fontSize": 11},
        },
        "series": [
            {
                "name": species.capitalize(),
                "type": "line",
                "showSymbol": False,
                "connectNulls": False,
                "data": values,
                "lineStyle": {"width": 2},
                "areaStyle": {"opacity": 0.06},
            }
        ],
    }


async def build_predictions_page(csv_path: Path) -> None:
    from nicegui import run, ui

    state = {"result": None, "loading": False}
    with ui.column().classes("w-full max-w-6xl mx-auto p-4 sm:p-6 gap-4"):
        ui.label("Next 24 hours").classes("text-2xl font-semibold")
        ui.label("Exploratory Poisson model").classes("text-sm text-gray-600")
        with ui.row().classes("w-full items-center gap-3"):
            camera = (
                ui.select({}, label="Camera")
                .classes("w-64 max-w-full")
                .mark("forecast-camera")
            )
            units = (
                ui.toggle(
                    {"relative": "Relative activity", "count": "Images/hour"},
                    value="relative",
                )
                .props("no-caps")
                .mark("forecast-units")
            )
            refresh_button = (
                ui.button(icon="refresh")
                .props('flat round aria-label="Refresh forecasts"')
                .mark("forecast-refresh")
            )
            refresh_button.tooltip("Refresh forecasts")
        assumption = ui.checkbox("Assume continuous observation", value=False).mark(
            "forecast-assumption"
        )
        ui.label("Observation coverage is unverified").classes("text-sm text-amber-800")
        status = ui.label().classes("text-sm font-medium").mark("forecast-status")
        dates = ui.label().classes("text-sm text-gray-600")
        provenance = ui.label().classes("text-sm text-gray-600")
        plots, captions = {}, {}
        with ui.element("div").classes(
            "w-full grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-5"
        ):
            for species in sorted(RECENT_SPECIES):
                with ui.column().classes(
                    "w-full min-w-0 gap-1 border-t border-gray-200 pt-4"
                ):
                    ui.label(species.capitalize()).classes("text-lg font-medium")
                    captions[species] = ui.label().classes("text-xs text-gray-600")
                    plots[species] = (
                        ui.echart(chart_options(species))
                        .classes("w-full h-64")
                        .mark(f"forecast-{species.replace(' ', '-')}")
                    )

    def render():
        result = state["result"]
        relative = units.value == "relative"
        for species, chart in plots.items():
            chart.options.clear()
            chart.options.update(chart_options(species, result, relative))
            chart.update()
            caption = ""
            if result is not None:
                frame = result.frame.loc[result.frame.species == species]
                caption = f"{frame.expected_count.sum():.2f} expected images"
                if (frame.method == "baseline").any():
                    caption += " | Recent-history fallback"
                if relative and frame.activity_index.isna().all():
                    caption += " | Relative index unavailable"
                elif relative and frame.clipped.any():
                    caption += " | Index clipped at 1"
            captions[species].set_text(caption)

    async def refresh():
        if state["loading"]:
            return
        state["loading"] = True
        refresh_button.disable()
        camera.disable()
        assumption.disable()
        submitted_inputs = None
        try:
            model_path = configured_model_path()
            bundle = read_model(model_path)
            options = {
                site["location_id"]: site["location_id"]
                for site in bundle["config"]["sites"]
            }
            selected = camera.value if camera.value in options else next(iter(options))
            camera.set_options(options, value=selected)
            submitted_inputs = (selected, bool(assumption.value))
            state["result"] = None
            render()
            status.set_text("Updating forecasts...")
            result = await run.io_bound(
                forecast_next24,
                csv_path,
                model_path,
                selected,
                assume_continuous=assumption.value,
            )
            if (camera.value, bool(assumption.value)) != submitted_inputs:
                return
            state["result"] = result
            first = result.origin.tz_convert(result.local_timezone)
            last = (result.origin + pd.Timedelta(hours=24)).tz_convert(
                result.local_timezone
            )
            dates.set_text(f"{first:%d %b %H:%M %Z} to {last:%d %b %H:%M %Z}")
            latest = result.latest_record.tz_convert(result.local_timezone)
            provenance.set_text(
                f"Last record {latest:%d %b %Y %H:%M %Z} | Model cutoff {result.training_cutoff:%d %b %Y}"
            )
            status.set_text(
                "Forecast ready: continuous operation assumed"
                if result.assumed_gap
                else "Forecast ready"
            )
        except (ForecastUnavailable, OSError, ValueError, KeyError) as error:
            state["result"] = None
            status.set_text(str(error))
            dates.set_text("")
            provenance.set_text("")
        finally:
            render()
            refresh_button.enable()
            camera.enable()
            assumption.enable()
            state["loading"] = False
            if (
                submitted_inputs is not None
                and (camera.value, bool(assumption.value)) != submitted_inputs
            ):
                await refresh()

    units.on_value_change(lambda _: render())
    camera.on_value_change(lambda _: refresh())
    assumption.on_value_change(lambda _: refresh())
    refresh_button.on_click(refresh)
    await refresh()
    ui.timer(60, refresh)
