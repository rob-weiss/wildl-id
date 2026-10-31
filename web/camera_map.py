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
RANKING_HOURS = 3
WIND_SPREAD = 45.0
VIEW_RADIUS = 120.0


@dataclass(frozen=True)
class Camera:
    location_id: str
    latitude: float | None
    longitude: float | None
    view_direction: float | None = None
    view_angle: float | None = None

    @property
    def located(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    @property
    def has_view(self) -> bool:
        return self.view_direction is not None and self.view_angle is not None


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
        view_direction = _coordinate(entry.get("view_direction"), 360)
        view_angle = _coordinate(entry.get("view_angle"), 360)
        if (view_direction is None) != (view_angle is None):
            raise ValueError(f"{location_id}: set both view_direction and view_angle")
        if view_direction is not None and (view_direction < 0 or view_angle <= 0):
            raise ValueError(f"{location_id}: view angles out of range")
        cameras.append(
            Camera(location_id.strip(), latitude, longitude, view_direction, view_angle)
        )
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


def initial_view(
    cameras: list[Camera], size: tuple[int, int] = (560, 300)
) -> tuple[tuple[float, float], int]:
    """Centre and zoom showing every camera with its view radius in a small map."""
    if not cameras:
        return DEFAULT_CENTER, 13
    latitudes = [camera.latitude for camera in cameras]
    longitudes = [camera.longitude for camera in cameras]
    center = (
        (min(latitudes) + max(latitudes)) / 2,
        (min(longitudes) + max(longitudes)) / 2,
    )
    margin = 2 * VIEW_RADIUS
    height = (max(latitudes) - min(latitudes)) * 111_320 + margin
    width = (max(longitudes) - min(longitudes)) * 111_320 * math.cos(
        math.radians(center[0])
    ) + margin
    # Web Mercator metres per pixel at zoom 0, scaled to this latitude.
    scale = 156_543.03 * math.cos(math.radians(center[0]))
    zoom = min(
        math.floor(math.log2(scale * size[0] / width)),
        math.floor(math.log2(scale * size[1] / height)),
    )
    return center, max(3, min(18, zoom))


def sector_overlap(
    center_a: float, width_a: float, center_b: float, width_b: float
) -> float:
    """Degrees shared by two compass sectors, sampled at one-degree steps."""

    def inside(bearing, center, width):
        return abs((bearing - center + 180) % 360 - 180) <= width / 2

    return float(
        sum(
            inside(bearing + 0.5, center_a, width_a)
            and inside(bearing + 0.5, center_b, width_b)
            for bearing in range(360)
        )
    )


def wind_match(camera: Camera, hours: list[WindHour]) -> float:
    # Wind blowing from the viewed area carries the hunter's scent away from game.
    window = min(camera.view_angle, 2 * WIND_SPREAD)
    return sum(
        sector_overlap(
            camera.view_direction, camera.view_angle, hour.direction, 2 * WIND_SPREAD
        )
        / window
        for hour in hours
    ) / len(hours)


def rank_seats(
    cameras: list[Camera], wind: dict[str, list[WindHour]], index: int
) -> list[tuple[Camera, float]]:
    ranked = [
        (camera, wind_match(camera, hours[index : index + RANKING_HOURS]))
        for camera in cameras
        if camera.has_view and (hours := wind.get(camera.location_id))
    ]
    return sorted(ranked, key=lambda item: (-item[1], item[0].location_id))


def view_polygon(camera: Camera) -> list[list[float]]:
    start = [camera.latitude, camera.longitude]
    steps = max(2, math.ceil(camera.view_angle / 10))
    first = camera.view_direction - camera.view_angle / 2
    return [
        start,
        *(
            _offset(
                start,
                VIEW_RADIUS,
                math.radians(first + camera.view_angle * step / steps),
            )
            for step in range(steps + 1)
        ),
    ]


def match_colour(score: float | None) -> str:
    if score is None:
        return "#e5e7eb"
    if score >= 0.67:
        return "#22c55e"
    if score >= 0.33:
        return "#facc15"
    return "#ef4444"


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
        center, zoom = initial_view(located)
        leaflet = (
            ui.leaflet(center=center, zoom=zoom)
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
        ranking = rank_seats(located, state["wind"], index)
        scores = {camera.location_id: score for camera, score in ranking}
        for camera in located:
            if camera.has_view:
                colour = match_colour(scores.get(camera.location_id))
                state["arrows"].append(
                    leaflet.generic_layer(
                        name="polygon",
                        args=[
                            view_polygon(camera),
                            {
                                "color": colour,
                                "weight": 1,
                                "fillColor": colour,
                                "fillOpacity": 0.3,
                                "interactive": False,
                            },
                        ],
                    )
                )
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

        def wind_text(camera):
            hour = state["wind"][camera.location_id][index]
            return (
                f"{camera.location_id}: {hour.speed:.0f} km/h from "
                f"{compass(hour.direction)} ({hour.direction:.0f}°), "
                f"gusts {hour.gusts:.0f} km/h"
            )

        with wind_list:
            if ranking:
                window = state["wind"][ranking[0][0].location_id][
                    index : index + RANKING_HOURS
                ]
                first, last = (
                    stamp.tz_convert(LOCAL_TIMEZONE)
                    for stamp in (
                        window[0].time,
                        window[-1].time + pd.Timedelta(hours=1),
                    )
                )
                ui.label(
                    f"Best seats {first:%H:%M}-{last:%H:%M}: wind from the viewed area"
                ).classes("font-medium")
                for rank, (camera, score) in enumerate(ranking, 1):
                    ui.label(
                        f"{rank}. {score:.0%} · {wind_text(camera)} · view "
                        f"{compass(camera.view_direction)} {camera.view_angle:.0f}°"
                    ).mark("seat-rank")
            for camera in located:
                if not camera.has_view and camera.location_id in state["wind"]:
                    ui.label(f"{wind_text(camera)} · no view angle")

    async def refresh() -> None:
        if not located:
            if cameras:
                status.set_text("Add GPS coordinates to show cameras and wind")
            return
        status.set_text("Loading wind forecast...")
        try:
            state["wind"] = await run.io_bound(fetch_wind, located)
            status.set_text("Wind at 10 m from Open-Meteo; arrows point downwind")
        except OSError, ValueError, KeyError, TypeError:
            state["wind"] = {}
            status.set_text("Wind forecast unavailable")
        render()

    leaflet.on("init", label_markers)
    slider.on_value_change(lambda _: render())
    await refresh()
    ui.timer(WIND_CACHE_SECONDS, refresh)
