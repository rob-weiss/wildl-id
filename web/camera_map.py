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
from src.labelling.label_data import DATE_FORMAT

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
RANKING_HOUR_OPTIONS = range(1, 11)
WIND_ARC_PADDING = 10.0
ARC_CLEARANCE = 30.0
WIND_ARC_COLOUR = "#9ca3af"
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


def downwind_bearing(direction: float) -> float:
    # Meteorological direction is where wind comes from.
    return (direction + 180) % 360


def downwind_sector(hours: list[WindHour]) -> tuple[float, float]:
    """Centre and width of the narrowest sector holding every downwind bearing."""
    bearings = sorted({downwind_bearing(hour.direction) for hour in hours})
    gaps = [
        ((bearings[(i + 1) % len(bearings)] - bearing) % 360 or 360, i)
        for i, bearing in enumerate(bearings)
    ]
    largest, before = max(gaps)
    width = 360 - largest
    start = bearings[(before + 1) % len(bearings)]
    return (start + width / 2) % 360, min(360.0, width + 2 * WIND_ARC_PADDING)


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


def arc_distance(
    view_direction: float, view_angle: float, center: float, width: float
) -> float:
    """Smallest angular gap between two sectors in degrees; 0 when they overlap."""
    offset = (center - view_direction) % 360
    half_sum = (view_angle + width) / 2
    return max(0.0, min(offset, 360 - offset) - half_sum)


def arc_colour(distance: float | None) -> str:
    if distance is None:
        return "#e5e7eb"
    if distance <= 0:
        return "#ef4444"
    return "#22c55e" if distance > ARC_CLEARANCE else "#facc15"


def rank_seats(
    cameras: list[Camera],
    wind: dict[str, list[WindHour]],
    index: int,
    hours: int = RANKING_HOURS,
) -> list[tuple[Camera, float]]:
    ranked = []
    for camera in cameras:
        forecast = wind.get(camera.location_id, [])[index : index + hours]
        if camera.has_view and forecast:
            center, width = downwind_sector(forecast)
            distance = arc_distance(
                camera.view_direction, camera.view_angle, center, width
            )
            ranked.append((camera, distance))
    return sorted(ranked, key=lambda item: (item[1], item[0].location_id))


def sector_polygon(
    latitude: float, longitude: float, center: float, width: float, radius: float
) -> list[list[float]]:
    start = [latitude, longitude]
    steps = max(2, math.ceil(width / 10))
    first = center - width / 2
    return [
        start,
        *(
            _offset(start, radius, math.radians(first + width * step / steps))
            for step in range(steps + 1)
        ),
    ]


def view_polygon(camera: Camera) -> list[list[float]]:
    return sector_polygon(
        camera.latitude,
        camera.longitude,
        camera.view_direction,
        camera.view_angle,
        VIEW_RADIUS,
    )


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
                ui.slider(
                    min=0,
                    max=WIND_HOURS - RANKING_HOURS,
                    step=1,
                    value=0,
                )
                .props("markers snap")
                .classes("flex-1 min-w-0")
                .mark("wind-hour")
            )
            hour_label = (
                ui.label().classes("text-sm font-medium w-36").mark("wind-time")
            )
            hours_select = (
                ui.select(
                    {hours: f"{hours} h" for hours in RANKING_HOUR_OPTIONS},
                    value=RANKING_HOURS,
                    label="Hours",
                )
                .classes("w-24")
                .mark("wind-hours")
            )
        ui.label(
            "The grey arc shows where the wind blows over the selected hours. "
            "The view sector is red when the arc overlaps it, yellow when "
            "within 30°, green when further away."
        ).classes("text-sm text-gray-600")
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
        window = int(hours_select.value)
        hour_label.set_text("")
        ranking = rank_seats(located, state["wind"], index, window)
        distances = {camera.location_id: gap for camera, gap in ranking}
        for camera in located:
            if camera.has_view:
                colour = arc_colour(distances.get(camera.location_id))
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
            forecast = hours[index : index + window]
            if forecast:
                hour_label.set_text(
                    forecast[0]
                    .time.tz_convert(LOCAL_TIMEZONE)
                    .strftime(f"%a {DATE_FORMAT} %H:%M %Z")
                )
            if not forecast:
                continue
            center, width = downwind_sector(forecast)
            max_speed = max(hour.speed for hour in forecast)
            state["arrows"].append(
                leaflet.generic_layer(
                    name="polygon",
                    args=[
                        sector_polygon(
                            camera.latitude,
                            camera.longitude,
                            center,
                            width,
                            max(40.0, max_speed * 15),
                        ),
                        {
                            "color": WIND_ARC_COLOUR,
                            "weight": 2,
                            "dashArray": "6 4",
                            "fillColor": WIND_ARC_COLOUR,
                            "fillOpacity": 0.35,
                            "interactive": False,
                        },
                    ],
                )
            )

        def wind_text(camera, hour):
            time_label = hour.time.tz_convert(LOCAL_TIMEZONE).strftime("%H:%M")
            return (
                f"{camera.location_id} · {time_label}: {hour.speed:.0f} km/h from "
                f"{compass(hour.direction)} ({hour.direction:.0f}°), "
                f"gusts {hour.gusts:.0f} km/h"
            )

        with wind_list:
            if ranking:
                window_hours = state["wind"][ranking[0][0].location_id][
                    index : index + window
                ]
                first, last = (
                    stamp.tz_convert(LOCAL_TIMEZONE)
                    for stamp in (
                        window_hours[0].time,
                        window_hours[-1].time + pd.Timedelta(hours=1),
                    )
                )
                ui.label(
                    f"Seats {first:%H:%M}-{last:%H:%M}: gap between view and wind arc"
                ).classes("font-medium")
                for rank, (camera, gap) in enumerate(ranking, 1):
                    selected_hour = state["wind"][camera.location_id][index]
                    ui.label(
                        f"{rank}. {gap:.0f}° · "
                        f"{wind_text(camera, selected_hour)} "
                        f"· view "
                        f"{compass(camera.view_direction)} {camera.view_angle:.0f}°"
                    ).mark("seat-rank")
            for camera in located:
                hours = state["wind"].get(camera.location_id)
                if not hours:
                    continue
                for hour in hours[index : index + window]:
                    details = wind_text(camera, hour)
                    if not camera.has_view:
                        details += " · no view angle"
                    ui.label(details).mark("wind-forecast")

    async def refresh() -> None:
        if not located:
            if cameras:
                status.set_text("Add GPS coordinates to show cameras and wind")
            return
        status.set_text("Loading wind forecast...")
        try:
            state["wind"] = await run.io_bound(fetch_wind, located)
            status.set_text(
                "Wind at 10 m from Open-Meteo; arcs show where the wind blows "
                "over the selected hours"
            )
        except OSError, ValueError, KeyError, TypeError:
            state["wind"] = {}
            status.set_text("Wind forecast unavailable")
        render()

    def change_hours() -> None:
        slider.props["max"] = WIND_HOURS - int(hours_select.value)
        slider.update()
        if slider.value > slider.props["max"]:
            slider.set_value(slider.props["max"])
        render()

    leaflet.on("init", label_markers)
    slider.on_value_change(lambda _: render())
    hours_select.on_value_change(lambda _: change_hours())
    await refresh()
    ui.timer(WIND_CACHE_SECONDS, refresh)
