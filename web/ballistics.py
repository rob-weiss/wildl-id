"""Ballistics computation and plotting for common hunting calibers with local seat wind."""

from __future__ import annotations

import io
import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import py_ballisticcalc as pbc
from nicegui import run, ui
from py_ballisticcalc.unit import Angular, Distance, Energy, Velocity, Weight
from web.camera_map import configured_locations_path, fetch_wind, read_camera_locations

logger = logging.getLogger(__name__)

DEFAULT_BALLISTICS_CONFIG = (
    Path(__file__).resolve().parents[1] / "config" / "ballistics.json"
)


@dataclass(frozen=True)
class Cartridge:
    id: str
    name: str
    caliber: str
    bullet_weight_g: float
    muzzle_velocity_mps: float
    ballistic_coefficient: float
    bullet_diameter_mm: float
    drag_table: str = "G1"
    sight_height_cm: float = 5.0
    zero_distance_m: float = 100.0
    zero_height_cm: float = 4.0
    description: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> Cartridge:
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            caliber=str(data.get("caliber", "")),
            bullet_weight_g=float(data["bullet_weight_g"]),
            muzzle_velocity_mps=float(data["muzzle_velocity_mps"]),
            ballistic_coefficient=float(data["ballistic_coefficient"]),
            bullet_diameter_mm=float(data.get("bullet_diameter_mm", 7.82)),
            drag_table=str(data.get("drag_table", "G1")),
            sight_height_cm=float(data.get("sight_height_cm", 5.0)),
            zero_distance_m=float(data.get("zero_distance_m", 100.0)),
            zero_height_cm=float(data.get("zero_height_cm", 4.0)),
            description=str(data.get("description", "")),
        )


DEFAULT_CARTRIDGES: tuple[Cartridge, ...] = (
    Cartridge(
        id="rws-308-win-hit-short-rifle",
        name="RWS .308 Win. HIT Short Rifle (9.7g / 150gr)",
        caliber=".308 Win",
        bullet_weight_g=9.7,
        muzzle_velocity_mps=870.0,
        ballistic_coefficient=0.386,
        bullet_diameter_mm=7.82,
        sight_height_cm=5.0,
        zero_distance_m=100.0,
        zero_height_cm=4.0,
        description="Lead-free copper bullet optimized for short barrels.",
    ),
    Cartridge(
        id="rws-30-06-spring-hit",
        name="RWS .30-06 Spring. HIT (10.7g / 165gr)",
        caliber=".30-06 Spring.",
        bullet_weight_g=10.7,
        muzzle_velocity_mps=840.0,
        ballistic_coefficient=0.400,
        bullet_diameter_mm=7.82,
        sight_height_cm=5.0,
        zero_distance_m=100.0,
        zero_height_cm=4.0,
        description="Versatile all-around hunting cartridge for medium and large game.",
    ),
    Cartridge(
        id="rws-8x57-is-hit",
        name="RWS 8x57 IS HIT (10.4g / 160gr)",
        caliber="8x57 IS",
        bullet_weight_g=10.4,
        muzzle_velocity_mps=820.0,
        ballistic_coefficient=0.334,
        bullet_diameter_mm=8.22,
        sight_height_cm=5.0,
        zero_distance_m=100.0,
        zero_height_cm=4.0,
        description="Traditional German driven-hunt and woodland caliber.",
    ),
    Cartridge(
        id="brenneke-7x64-tag",
        name="Brenneke 7x64 TAG (10.5g / 162gr)",
        caliber="7x64",
        bullet_weight_g=10.5,
        muzzle_velocity_mps=860.0,
        ballistic_coefficient=0.410,
        bullet_diameter_mm=7.25,
        sight_height_cm=5.0,
        zero_distance_m=100.0,
        zero_height_cm=4.0,
        description="Classic flat-shooting German universal cartridge.",
    ),
    Cartridge(
        id="rws-9-3x62-hit",
        name="RWS 9.3x62 HIT (16.2g / 250gr)",
        caliber="9.3x62",
        bullet_weight_g=16.2,
        muzzle_velocity_mps=740.0,
        ballistic_coefficient=0.360,
        bullet_diameter_mm=9.30,
        sight_height_cm=5.0,
        zero_distance_m=100.0,
        zero_height_cm=4.0,
        description="Heavy stopping caliber for wild boar and driven hunts.",
    ),
    Cartridge(
        id="rws-300-win-mag-hit",
        name="RWS .300 Win. Mag. HIT (10.7g / 165gr)",
        caliber=".300 Win. Mag.",
        bullet_weight_g=10.7,
        muzzle_velocity_mps=925.0,
        ballistic_coefficient=0.400,
        bullet_diameter_mm=7.82,
        sight_height_cm=5.0,
        zero_distance_m=100.0,
        zero_height_cm=4.0,
        description="High-velocity magnum caliber for extended hunting distances.",
    ),
    Cartridge(
        id="rws-6-5-creedmoor-hit",
        name="RWS 6.5 Creedmoor HIT (7.8g / 120gr)",
        caliber="6.5 Creedmoor",
        bullet_weight_g=7.8,
        muzzle_velocity_mps=870.0,
        ballistic_coefficient=0.380,
        bullet_diameter_mm=6.71,
        sight_height_cm=5.0,
        zero_distance_m=100.0,
        zero_height_cm=4.0,
        description="Modern low-recoil caliber with high precision.",
    ),
    Cartridge(
        id="rws-223-rem-target-elite",
        name="RWS .223 Rem. Target Elite Plus (3.4g / 52gr)",
        caliber=".223 Rem.",
        bullet_weight_g=3.4,
        muzzle_velocity_mps=1010.0,
        ballistic_coefficient=0.230,
        bullet_diameter_mm=5.69,
        sight_height_cm=5.0,
        zero_distance_m=100.0,
        zero_height_cm=4.0,
        description="Small-caliber predator and roe deer cartridge.",
    ),
)


@dataclass(frozen=True)
class BallisticResult:
    distances: np.ndarray
    heights_cm: np.ndarray
    velocities_mps: np.ndarray
    energies_j: np.ndarray
    windages_cm: np.ndarray
    gee_distance_m: float
    e0_j: float
    e100_j: float
    v0_mps: float
    v100_mps: float
    wind_drift_100m_cm: float
    wind_drift_200m_cm: float
    wind_drift_300m_cm: float


def configured_ballistics_path() -> Path:
    return Path(
        os.environ.get("WILDL_ID_BALLISTICS_CONFIG", str(DEFAULT_BALLISTICS_CONFIG))
    )


def load_cartridges(config_path: Path | None = None) -> list[Cartridge]:
    """Load cartridges from config file if available, otherwise return defaults."""
    path = config_path or configured_ballistics_path()
    if path.is_file():
        try:
            content = path.read_text(encoding="utf-8").strip()
            if content:
                data = json.loads(content)
                entries = data.get("cartridges", []) if isinstance(data, dict) else data
                parsed = [
                    Cartridge.from_dict(item)
                    for item in entries
                    if isinstance(item, dict)
                ]
                if parsed:
                    return parsed
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            logger.warning("Failed to load ballistics config from %s: %s", path, error)
    return list(DEFAULT_CARTRIDGES)


def max_seat_wind_speed(
    wind_data: dict[str, list],
    hours_ahead: int,
    now: datetime | None = None,
) -> float:
    """Compute maximum wind speed in km/h across seats for the next n hours."""
    if not wind_data or hours_ahead <= 0:
        return 0.0
    now_ts = pd.Timestamp(now or datetime.now(UTC)).tz_convert("UTC")
    speeds: list[float] = []
    for seat_hours in wind_data.values():
        upcoming = [h for h in seat_hours if h.time >= now_ts][:hours_ahead]
        if not upcoming and seat_hours:
            upcoming = seat_hours[:hours_ahead]
        for hour in upcoming:
            speeds.append(float(hour.speed))
    return max(speeds) if speeds else 0.0


def get_max_seat_wind(
    hours_ahead: int,
    locations_path: Path | None = None,
    now: datetime | None = None,
) -> tuple[float, int]:
    """Fetch wind and return (max_wind_kmh, located_seat_count)."""
    try:
        path = locations_path or configured_locations_path()
        if not path.is_file():
            return 0.0, 0
        cameras = read_camera_locations(path)
        located = [c for c in cameras if c.located]
        if not located:
            return 0.0, 0
        wind_data = fetch_wind(located)
        speed = max_seat_wind_speed(wind_data, hours_ahead, now=now)
        return speed, len(located)
    except Exception as error:
        logger.warning("Could not obtain seat wind: %s", error)
        return 0.0, 0


def compute_trajectory(
    cartridge: Cartridge,
    wind_speed_kmh: float = 0.0,
    max_distance_m: float = 300.0,
    step_m: float = 1.0,
) -> BallisticResult:
    """Calculate ballistic trajectory using py-ballisticcalc."""
    calc = pbc.Calculator()
    table = getattr(pbc, f"Table{cartridge.drag_table}", pbc.TableG1)
    dm = pbc.DragModel(
        bc=cartridge.ballistic_coefficient,
        drag_table=table,
        weight=Weight.Gram(cartridge.bullet_weight_g),
        diameter=Distance.Millimeter(cartridge.bullet_diameter_mm),
    )
    ammo = pbc.Ammo(dm, Velocity.MPS(cartridge.muzzle_velocity_mps))

    weapon_base = pbc.Weapon(
        sight_height=Distance.Centimeter(cartridge.sight_height_cm)
    )
    base_zero = calc.zero_angle(
        pbc.Shot(weapon=weapon_base, ammo=ammo),
        Distance.Meter(cartridge.zero_distance_m),
    )
    zero_offset_rad = math.atan(
        (cartridge.zero_height_cm / 100.0) / cartridge.zero_distance_m
    )
    target_elevation = base_zero + Angular.Radian(zero_offset_rad)
    weapon = pbc.Weapon(
        sight_height=Distance.Centimeter(cartridge.sight_height_cm),
        zero_elevation=target_elevation,
    )

    winds = []
    if wind_speed_kmh > 0:
        # Full 90 degree crosswind from the side
        winds.append(
            pbc.Wind(
                velocity=Velocity.KMH(wind_speed_kmh),
                direction_from=Angular.Degree(90),
            )
        )

    shot = pbc.Shot(weapon=weapon, ammo=ammo, winds=winds if winds else None)
    hit_result = calc.fire(
        shot,
        trajectory_range=Distance.Meter(max_distance_m),
        trajectory_step=Distance.Meter(step_m),
    )

    records = hit_result.records
    distances = np.array([r.distance >> Distance.Meter for r in records])
    heights_cm = np.array([r.height >> Distance.Centimeter for r in records])
    velocities_mps = np.array([r.velocity >> Velocity.MPS for r in records])
    energies_j = np.array([r.energy >> Energy.Joule for r in records])
    windages_cm = np.array([r.windage >> Distance.Centimeter for r in records])

    # GEE (Günstigste Einschießentfernung / Most Recommended Distance):
    # Second zero crossing where trajectory drops back to 0 cm after 110 m
    mask = distances > 110.0
    if np.any(mask):
        gee_idx = np.abs(heights_cm[mask]).argmin()
        gee_distance_m = float(distances[mask][gee_idx])
    else:
        gee_distance_m = float(cartridge.zero_distance_m)

    idx_100 = int(np.abs(distances - 100.0).argmin())
    idx_200 = int(np.abs(distances - 200.0).argmin())
    idx_300 = int(np.abs(distances - 300.0).argmin())

    return BallisticResult(
        distances=distances,
        heights_cm=heights_cm,
        velocities_mps=velocities_mps,
        energies_j=energies_j,
        windages_cm=windages_cm,
        gee_distance_m=gee_distance_m,
        e0_j=float(energies_j[0]),
        e100_j=float(energies_j[idx_100]),
        v0_mps=float(velocities_mps[0]),
        v100_mps=float(velocities_mps[idx_100]),
        wind_drift_100m_cm=float(windages_cm[idx_100]),
        wind_drift_200m_cm=float(windages_cm[idx_200]),
        wind_drift_300m_cm=float(windages_cm[idx_300]),
    )


def generate_ballistics_plot_svg(
    cartridge: Cartridge,
    result: BallisticResult,
    wind_speed_kmh: float = 0.0,
) -> str:
    """Generate multi-panel ballistic curves matching the reference design."""
    with plt.rc_context({"svg.fonttype": "none"}):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 13), sharex=True)

        for ax in (ax1, ax2, ax3, ax4):
            ax.set_xlim(0, 300)
            ax.axvline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axvline(200, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

        # Plot 1: Trajectory (Height deviation from line of sight)
        ax1.plot(
            result.distances,
            result.heights_cm,
            color="darkgreen",
            lw=2,
            label=f"Trajectory (+{cartridge.zero_height_cm:.1f}cm @ {cartridge.zero_distance_m:.0f}m)",
        )
        ax1.axhline(
            0, color="black", linestyle="--", linewidth=0.8, label="Line of sight"
        )
        ax1.plot(
            100,
            cartridge.zero_height_cm,
            "ro",
            label=f"Target point (+{cartridge.zero_height_cm:.1f} cm)",
        )
        ax1.axvline(
            result.gee_distance_m,
            color="orange",
            linestyle="--",
            label=f"GEE / MRD: ca. {result.gee_distance_m:.0f}m",
        )
        ax1.set_ylabel("Trajectory (cm)", fontsize=11)
        ax1.set_title(
            f"Ballistic Curves: {cartridge.name}",
            fontsize=13,
            fontweight="bold",
        )
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="lower left")

        # Plot 2: Side wind drift (horizontal deflection)
        wind_label = (
            f"Wind drift ({wind_speed_kmh:.1f} km/h side wind)"
            if wind_speed_kmh > 0
            else "Wind drift (no wind)"
        )
        ax2.plot(
            result.distances,
            result.windages_cm,
            color="darkcyan",
            lw=2,
            label=wind_label,
        )
        ax2.set_ylabel("Wind Drift (cm)", fontsize=11)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper left")

        # Plot 3: Bullet velocity
        ax3.plot(
            result.distances,
            result.velocities_mps,
            color="blue",
            lw=2,
            label="Velocity",
        )
        ax3.set_ylabel("Velocity (m/s)", fontsize=11)
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.legend(loc="upper right")

        # Plot 4: Kinetic energy
        ax4.plot(
            result.distances,
            result.energies_j,
            color="crimson",
            lw=2,
            label="Kinetic Energy",
        )
        ax4.axhline(
            2000,
            color="purple",
            linestyle="--",
            linewidth=1.2,
            label="2000 J (German hoofed game min. at 100m)",
        )
        ax4.set_xlabel("Distance (m)", fontsize=12)
        ax4.set_ylabel("Energy (J)", fontsize=11)
        ax4.grid(True, linestyle=":", alpha=0.6)
        ax4.legend(loc="upper right")

        plt.tight_layout()
        buffer = io.StringIO()
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        plt.close(fig)
        return buffer.getvalue()


async def build_ballistics_page() -> None:
    """Build the ballistics web page in NiceGUI."""
    cartridges = load_cartridges()
    cartridge_by_id = {c.id: c for c in cartridges}
    default_cartridge = cartridges[0]

    state = {
        "selected_cartridge_id": default_cartridge.id,
        "forecast_hours": 5,
        "wind_speed_kmh": 0.0,
        "seat_count": 0,
        "svg": "",
        "result": None,
    }

    # Fetch initial wind in background
    wind_speed, seats = await run.io_bound(get_max_seat_wind, state["forecast_hours"])
    state["wind_speed_kmh"] = wind_speed
    state["seat_count"] = seats

    with ui.column().classes("w-full max-w-4xl mx-auto p-4 sm:p-6 gap-6"):
        with ui.column().classes("w-full gap-1"):
            ui.label("Ballistics & Wind Drift").classes("text-2xl font-semibold")
            ui.label(
                "Trajectory, velocity, energy, and crosswind deflection based on seats on the map."
            ).classes("text-sm text-gray-600")

        # Controls
        with ui.card().classes("w-full gap-4"):
            with ui.row().classes("w-full items-center gap-4 flex-wrap"):
                cartridge_options = {c.id: c.name for c in cartridges}
                cartridge_select = (
                    ui.select(
                        options=cartridge_options,
                        value=state["selected_cartridge_id"],
                        label="Hunting Caliber / Cartridge",
                    )
                    .classes("flex-grow min-w-72")
                    .mark("ballistics-cartridge-select")
                )

                hours_input = (
                    ui.number(
                        label="Wind Horizon (hours)",
                        value=state["forecast_hours"],
                        min=1,
                        max=24,
                        step=1,
                    )
                    .classes("w-44")
                    .mark("ballistics-hours-input")
                )

        metrics_container = ui.row().classes("w-full gap-4 flex-wrap")
        plot_container = ui.column().classes("w-full items-center")
        table_container = ui.column().classes("w-full")

        async def update_view() -> None:
            cartridge = cartridge_by_id.get(
                state["selected_cartridge_id"], default_cartridge
            )
            wind_speed = state["wind_speed_kmh"]

            result = await run.cpu_bound(compute_trajectory, cartridge, wind_speed)
            state["result"] = result
            svg_content = await run.cpu_bound(
                generate_ballistics_plot_svg, cartridge, result, wind_speed
            )
            state["svg"] = svg_content

            # Render metrics
            metrics_container.clear()
            with metrics_container:
                with ui.card().classes("flex-1 min-w-44 p-3"):
                    ui.label("Muzzle Energy (E0)").classes(
                        "text-xs text-gray-500 uppercase"
                    )
                    ui.label(f"{result.e0_j:.0f} J").classes("text-xl font-bold")
                    ui.label(
                        f"v0: {result.v0_mps:.0f} m/s · {cartridge.bullet_weight_g:.1f} g"
                    ).classes("text-xs text-gray-600")

                with ui.card().classes("flex-1 min-w-44 p-3"):
                    ui.label("Energy @ 100m (E100)").classes(
                        "text-xs text-gray-500 uppercase"
                    )
                    ui.label(f"{result.e100_j:.0f} J").classes("text-xl font-bold")
                    ui.label(f"v100: {result.v100_mps:.0f} m/s").classes(
                        "text-xs text-gray-600"
                    )

                with ui.card().classes("flex-1 min-w-44 p-3"):
                    ui.label("GEE / MRD").classes("text-xs text-gray-500 uppercase")
                    ui.label(f"~{result.gee_distance_m:.0f} m").classes(
                        "text-xl font-bold text-orange-600"
                    )
                    ui.label(f"+{cartridge.zero_height_cm:.1f} cm @ 100 m").classes(
                        "text-xs text-gray-600"
                    )

                with ui.card().classes("flex-1 min-w-44 p-3"):
                    ui.label(f"Max Side Wind ({state['forecast_hours']}h)").classes(
                        "text-xs text-gray-500 uppercase"
                    )
                    ui.label(f"{wind_speed:.1f} km/h").classes(
                        "text-xl font-bold text-teal-700"
                    )
                    ui.label(f"Seats: {state['seat_count']} · 90° crosswind").classes(
                        "text-xs text-gray-600"
                    )

                with ui.card().classes("flex-1 min-w-44 p-3"):
                    ui.label("Wind Drift @ 100m / 200m").classes(
                        "text-xs text-gray-500 uppercase"
                    )
                    ui.label(
                        f"{result.wind_drift_100m_cm:+.1f} / {result.wind_drift_200m_cm:+.1f} cm"
                    ).classes("text-xl font-bold text-teal-800")
                    ui.label(f"@ 300m: {result.wind_drift_300m_cm:+.1f} cm").classes(
                        "text-xs text-gray-600"
                    )

            # Render Plot
            plot_container.clear()
            with plot_container:
                ui.html(svg_content).classes("w-full max-w-4xl").mark("ballistics-svg")

            # Render Table
            table_container.clear()
            with table_container:
                ui.label("Ballistic Drop Table").classes("text-lg font-semibold mt-2")
                table_distances = [0, 50, 100, 150, 200, 250, 300]
                rows = []
                for dist in table_distances:
                    idx = int(np.abs(result.distances - dist).argmin())
                    rows.append(
                        {
                            "distance": f"{dist} m",
                            "trajectory": f"{result.heights_cm[idx]:+.1f} cm",
                            "velocity": f"{result.velocities_mps[idx]:.0f} m/s",
                            "energy": f"{result.energies_j[idx]:.0f} J",
                            "wind_drift": f"{result.windages_cm[idx]:+.1f} cm",
                        }
                    )
                columns = [
                    {
                        "name": "distance",
                        "label": "Distance",
                        "field": "distance",
                        "align": "center",
                    },
                    {
                        "name": "trajectory",
                        "label": "Trajectory",
                        "field": "trajectory",
                        "align": "center",
                    },
                    {
                        "name": "velocity",
                        "label": "Velocity",
                        "field": "velocity",
                        "align": "center",
                    },
                    {
                        "name": "energy",
                        "label": "Energy",
                        "field": "energy",
                        "align": "center",
                    },
                    {
                        "name": "wind_drift",
                        "label": "Wind Drift (90°)",
                        "field": "wind_drift",
                        "align": "center",
                    },
                ]
                ui.table(columns=columns, rows=rows, row_key="distance").classes(
                    "w-full"
                ).props("flat bordered dense").mark("ballistics-table")

        async def on_cartridge_change(event) -> None:
            state["selected_cartridge_id"] = str(event.value)
            await update_view()

        async def on_hours_change(event) -> None:
            try:
                hours = int(event.value)
                hours = max(hours, 1)
                state["forecast_hours"] = hours
            except (ValueError, TypeError):
                return
            wind_speed, seats = await run.io_bound(
                get_max_seat_wind, state["forecast_hours"]
            )
            state["wind_speed_kmh"] = wind_speed
            state["seat_count"] = seats
            await update_view()

        cartridge_select.on_value_change(on_cartridge_change)
        hours_input.on_value_change(on_hours_change)

        await update_view()
