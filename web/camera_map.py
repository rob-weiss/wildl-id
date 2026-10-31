"""Satellite map of camera locations with the Open-Meteo wind forecast."""

import html
import json
import math
import os
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_LOCATIONS = (
    Path(__file__).resolve().parents[1] / "config" / "camera_locations.json"
)
DEFAULT_CENTER = (48.637, 9.011)
LOCAL_TIMEZONE = "Europe/Berlin"
SATELLITE_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
SATELLITE_ATTRIBUTION = (
    "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, "
    "and the GIS User Community"
)
WIND_API = "https://api.open-meteo.com/v1/forecast"
WIND_HOURS = 24
WIND_CACHE_SECONDS = 900
COMPASS = (
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
)

_wind_cache: dict[tuple, tuple[float, dict]] = {}
_wind_cache_lock = threading.Lock()


@dataclass(frozen=True)
class Camera:
    location_id: str
    latitude: float | None
    longitude: float | None

    @property
    def located(self) -> bool:
        return self.latitude is not None and self.longitude is not None


@dataclass(frozen=True)
class WindHour:
    time: pd.Timestamp
    speed: float
    direction: float
    gusts: float


def configured_locations_path() -> Path:
    return Path(os.environ.get("WILDL_ID_CAMERA_LOCATIONS", str(DEFAULT_LOCATIONS)))


def _coordinate(value, limit: float) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Coordinates must be numbers or null")
    if not -limit <= value <= limit:
        raise ValueError("Coordinate out of range")
    return float(value)


def read_camera_locations(path: Path) -> list[Camera]:
    cameras = []
    for entry in json.loads(Path(path).read_text(encoding="utf-8"))["cameras"]:
        location_id = entry["location_id"]
        if not isinstance(location_id, str) or not location_id.strip():
            raise ValueError("Camera location_id must be a non-empty string")
        latitude = _coordinate(entry.get("latitude"), 90)
        longitude = _coordinate(entry.get("longitude"), 180)
        if (latitude is None) != (longitude is None):
            raise ValueError(f"{location_id}: set both latitude and longitude")
        cameras.append(Camera(location_id.strip(), latitude, longitude))
    if len({camera.location_id for camera in cameras}) != len(cameras):
        raise ValueError("Duplicate camera location_id")
    return cameras


def _request_wind(coordinates: tuple[tuple[float, float], ...]):
    query = urllib.parse.urlencode(
        {
            "latitude": ",".join(f"{lat:.6f}" for lat, _ in coordinates),
            "longitude": ",".join(f"{lon:.6f}" for _, lon in coordinates),
            "hourly": "wind_speed_10m,wind_direction_10m,wind_gusts_10m",
            "forecast_hours": WIND_HOURS,
            "wind_speed_unit": "kmh",
            "timezone": "GMT",
            "timeformat": "unixtime",
        }
    )
    with urllib.request.urlopen(f"{WIND_API}?{query}", timeout=10) as response:
        payload = json.load(response)
    return payload if isinstance(payload, list) else [payload]


def fetch_wind(cameras: list[Camera]) -> dict[str, list[WindHour]]:
    located = [camera for camera in cameras if camera.located]
    if not located:
        return {}
    coordinates = tuple((camera.latitude, camera.longitude) for camera in located)
    with _wind_cache_lock:
        cached = _wind_cache.get(coordinates)
    if cached and time.monotonic() - cached[0] < WIND_CACHE_SECONDS:
        return cached[1]
    payloads = _request_wind(coordinates)
    if len(payloads) != len(located):
        raise ValueError("Wind forecast does not match camera locations")
    result = {}
    for camera, payload in zip(located, payloads):
        hourly = payload["hourly"]
        hours = [
            WindHour(
                pd.Timestamp(stamp, unit="s", tz="UTC"),
                float(speed),
                float(direction),
                float(gusts),
            )
            for stamp, speed, direction, gusts in zip(
                hourly["time"],
                hourly["wind_speed_10m"],
                hourly["wind_direction_10m"],
                hourly["wind_gusts_10m"],
            )
        ]
        if len(hours) != WIND_HOURS:
            raise ValueError("Incomplete wind forecast")
        result[camera.location_id] = hours
    with _wind_cache_lock:
        _wind_cache[coordinates] = (time.monotonic(), result)
    return result


def _offset(point: list[float], metres: float, bearing: float) -> list[float]:
    latitude, longitude = point
    return [
        latitude + metres * math.cos(bearing) / 111_320,
        longitude
        + metres * math.sin(bearing) / (111_320 * math.cos(math.radians(latitude))),
    ]


def arrow_points(
    latitude: float, longitude: float, speed: float, direction: float
) -> list[list[float]]:
    # Meteorological direction is where wind comes from; the arrow points downwind.
    bearing = math.radians((direction + 180) % 360)
    length = max(40.0, speed * 15)
    start = [latitude, longitude]
    tip = _offset(start, length, bearing)
    head = length * 0.3
    return [
        start,
        tip,
        _offset(tip, head, bearing + math.pi - 0.45),
        tip,
        _offset(tip, head, bearing + math.pi + 0.45),
    ]


def wind_colour(speed: float) -> str:
    if speed < 10:
        return "#7dd3fc"
    if speed < 20:
        return "#facc15"
    if speed < 35:
        return "#fb923c"
    return "#ef4444"


def compass(direction: float) -> str:
    return COMPASS[round(direction / 22.5) % 16]


async def build_camera_map_page() -> None:
    from nicegui import run, ui

    state: dict = {"wind": {}, "arrows": []}
    with ui.column().classes("w-full max-w-6xl mx-auto p-4 sm:p-6 gap-4"):
        ui.label("Camera map").classes("text-2xl font-semibold")
        status = ui.label().classes("text-sm text-gray-600").mark("map-status")
        try:
            cameras = read_camera_locations(configured_locations_path())
        except (OSError, ValueError, KeyError, TypeError) as error:
            cameras = []
            status.set_text(f"Camera locations unavailable: {error}")
        located = [camera for camera in cameras if camera.located]
        center = (
            (
                sum(camera.latitude for camera in located) / len(located),
                sum(camera.longitude for camera in located) / len(located),
            )
            if located
            else DEFAULT_CENTER
        )
        leaflet = (
            ui.leaflet(center=center, zoom=15 if located else 13)
            .classes("w-full h-[60vh] min-h-80")
            .mark("camera-map")
        )
        leaflet.clear_layers()
        leaflet.tile_layer(
            url_template=SATELLITE_TILES,
            options={"attribution": SATELLITE_ATTRIBUTION, "maxZoom": 19},
        )
        markers = [
            (
                camera,
                leaflet.generic_layer(
                    name="circleMarker",
                    args=[
                        [camera.latitude, camera.longitude],
                        {
                            "radius": 7,
                            "color": "#ffffff",
                            "weight": 2,
                            "fillColor": "#157f72",
                            "fillOpacity": 1,
                        },
                    ],
                ),
            )
            for camera in located
        ]
        with ui.row().classes("w-full items-center gap-4 flex-nowrap"):
            ui.label("Wind").classes("text-sm font-medium")
            slider = (
                ui.slider(min=0, max=WIND_HOURS - 1, step=1, value=0)
                .props("markers snap")
                .classes("flex-1 min-w-0")
                .mark("wind-hour")
            )
            hour_label = (
                ui.label().classes("text-sm font-medium w-36").mark("wind-time")
            )
        wind_list = ui.column().classes("w-full gap-1 text-sm").mark("wind-list")
        missing = [camera.location_id for camera in cameras if not camera.located]
        if missing:
            ui.label(
                f"Missing coordinates in {configured_locations_path().name}: "
                + ", ".join(missing)
            ).classes("text-sm text-amber-800").mark("map-missing")

    def label_markers() -> None:
        for camera, marker in markers:
            marker.run_method(
                "bindTooltip",
                html.escape(camera.location_id),
                {"permanent": True, "direction": "top", "offset": [0, -8]},
            )

    def render() -> None:
        for arrow in state["arrows"]:
            leaflet.remove_layer(arrow)
        state["arrows"] = []
        wind_list.clear()
        index = int(slider.value)
        hour_label.set_text("")
        for camera in located:
            hours = state["wind"].get(camera.location_id)
            if not hours:
                continue
            hour = hours[index]
            hour_label.set_text(
                hour.time.tz_convert(LOCAL_TIMEZONE).strftime("%a %d %b %H:%M %Z")
            )
            state["arrows"].append(
                leaflet.generic_layer(
                    name="polyline",
                    args=[
                        arrow_points(
                            camera.latitude,
                            camera.longitude,
                            hour.speed,
                            hour.direction,
                        ),
                        {
                            "color": wind_colour(hour.speed),
                            "weight": 3,
                            "opacity": 0.95,
                        },
                    ],
                )
            )
            with wind_list:
                ui.label(
                    f"{camera.location_id}: {hour.speed:.0f} km/h from "
                    f"{compass(hour.direction)} ({hour.direction:.0f}°), "
                    f"gusts {hour.gusts:.0f} km/h"
                )

    async def refresh() -> None:
        if not located:
            if cameras:
                status.set_text("Add GPS coordinates to show cameras and wind")
            return
        status.set_text("Loading wind forecast...")
        try:
            state["wind"] = await run.io_bound(fetch_wind, located)
            status.set_text("Wind at 10 m from Open-Meteo; arrows point downwind")
        except (OSError, ValueError, KeyError, TypeError):
            state["wind"] = {}
            status.set_text("Wind forecast unavailable")
        render()

    leaflet.on("init", label_markers)
    slider.on_value_change(lambda _: render())
    await refresh()
    ui.timer(WIND_CACHE_SECONDS, refresh)
