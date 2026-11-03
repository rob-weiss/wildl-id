"""Tests for ballistics calculations, cartridge configuration, and web interface."""

import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
import pandas as pd
from nicegui.testing import User
from web import app as web_app
from web import ballistics
from web.camera_map import Camera, WindHour


class TestBallisticsConfig(unittest.TestCase):
    def test_default_cartridges_include_rws_308_short_rifle(self):
        cartridges = ballistics.load_cartridges()
        self.assertGreaterEqual(len(cartridges), 8)

        rws_308 = cartridges[0]
        self.assertEqual(rws_308.id, "rws-308-win-hit-short-rifle")
        self.assertIn("RWS .308 Win. HIT Short Rifle", rws_308.name)
        self.assertEqual(rws_308.caliber, ".308 Win.")
        self.assertAlmostEqual(rws_308.bullet_weight_g, 9.7)
        self.assertAlmostEqual(rws_308.muzzle_velocity_mps, 870.0)
        self.assertAlmostEqual(rws_308.ballistic_coefficient, 0.386)
        self.assertAlmostEqual(rws_308.bullet_diameter_mm, 7.82)
        self.assertAlmostEqual(rws_308.sight_height_cm, 5.0)
        self.assertAlmostEqual(rws_308.zero_distance_m, 100.0)
        self.assertAlmostEqual(rws_308.zero_height_cm, 4.0)

    def test_default_cartridges_include_common_calibers(self):
        cartridges = ballistics.load_cartridges()
        names = {c.name for c in cartridges}
        self.assertTrue(any("Hornady .308 Win. ECX" in name for name in names))
        self.assertTrue(any("Sellier & Bellot .308 Win." in name for name in names))
        self.assertTrue(any("Hornady .223 Rem. NTX" in name for name in names))
        self.assertTrue(any("Hornady .223 Rem. ECX" in name for name in names))
        calibers = {c.caliber for c in cartridges}
        for expected in (
            ".308 Win.",
            ".30-06 Spring.",
            "8x57 IS",
            "7x64",
            "9.3x62",
            ".300 Win. Mag.",
            "6.5 Creedmoor",
            ".223 Rem.",
        ):
            self.assertIn(expected, calibers)

    def test_load_cartridges_uses_config_file_when_present(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "custom_cartridges.json"
            config_path.write_text(
                json.dumps(
                    {
                        "cartridges": [
                            {
                                "id": "custom-8x57",
                                "name": "Custom 8x57 IS Load",
                                "caliber": "8x57 IS",
                                "bullet_weight_g": 12.7,
                                "muzzle_velocity_mps": 770.0,
                                "ballistic_coefficient": 0.330,
                                "bullet_diameter_mm": 8.22,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            loaded = ballistics.load_cartridges(config_path=config_path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].id, "custom-8x57")
            self.assertEqual(loaded[0].name, "Custom 8x57 IS Load")
            self.assertAlmostEqual(loaded[0].bullet_weight_g, 12.7)
            self.assertAlmostEqual(loaded[0].sight_height_cm, 5.0)
            self.assertAlmostEqual(loaded[0].zero_distance_m, 100.0)

    def test_load_cartridges_falls_back_to_defaults_on_corrupt_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "invalid.json"
            config_path.write_text("invalid json content", encoding="utf-8")
            loaded = ballistics.load_cartridges(config_path=config_path)
            self.assertEqual(len(loaded), len(ballistics.DEFAULT_CARTRIDGES))


class TestBallisticsWind(unittest.TestCase):
    def test_max_seat_wind_speed_calculates_max_over_n_hours(self):
        now = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)
        wind_data = {
            "Seat1": [
                WindHour(pd.Timestamp("2026-11-03 12:00:00+00:00"), 10.0, 90.0, 15.0),
                WindHour(pd.Timestamp("2026-11-03 13:00:00+00:00"), 22.0, 90.0, 28.0),
                WindHour(pd.Timestamp("2026-11-03 14:00:00+00:00"), 15.0, 90.0, 20.0),
                WindHour(pd.Timestamp("2026-11-03 15:00:00+00:00"), 35.0, 90.0, 42.0),
            ],
            "Seat2": [
                WindHour(pd.Timestamp("2026-11-03 12:00:00+00:00"), 8.0, 90.0, 12.0),
                WindHour(pd.Timestamp("2026-11-03 13:00:00+00:00"), 18.0, 90.0, 25.0),
                WindHour(pd.Timestamp("2026-11-03 14:00:00+00:00"), 25.0, 90.0, 30.0),
                WindHour(pd.Timestamp("2026-11-03 15:00:00+00:00"), 20.0, 90.0, 25.0),
            ],
        }

        # 2 hours horizon: 12:00 and 13:00 -> max speed is 22.0
        self.assertAlmostEqual(
            ballistics.max_seat_wind_speed(wind_data, hours_ahead=2, now=now), 22.0
        )
        # 3 hours horizon: 12:00, 13:00, 14:00 -> max speed is 25.0 (from Seat2 at 14:00)
        self.assertAlmostEqual(
            ballistics.max_seat_wind_speed(wind_data, hours_ahead=3, now=now), 25.0
        )
        # 4 hours horizon: includes 15:00 -> max speed is 35.0 (from Seat1 at 15:00)
        self.assertAlmostEqual(
            ballistics.max_seat_wind_speed(wind_data, hours_ahead=4, now=now), 35.0
        )

    def test_max_seat_wind_speed_empty(self):
        self.assertEqual(ballistics.max_seat_wind_speed({}, 5), 0.0)
        self.assertEqual(ballistics.max_seat_wind_speed({"Seat": []}, 0), 0.0)

    def test_get_max_seat_wind_handles_missing_file(self):
        speed, seats = ballistics.get_max_seat_wind(
            hours_ahead=5, locations_path=Path("/nonexistent/cameras.json")
        )
        self.assertEqual(speed, 0.0)
        self.assertEqual(seats, 0)


class TestBallisticsTrajectory(unittest.TestCase):
    def test_compute_trajectory_matches_reference_ballistics(self):
        cartridge = ballistics.DEFAULT_CARTRIDGES[0]
        result = ballistics.compute_trajectory(cartridge, wind_speed_kmh=0.0)

        # Muzzle energy E0 ~ 3671 J
        self.assertAlmostEqual(result.e0_j, 3671.0, delta=20.0)
        # Energy at 100m E100 ~ 3026 J
        self.assertAlmostEqual(result.e100_j, 3026.0, delta=20.0)
        # GEE distance ~ 180 m
        self.assertAlmostEqual(result.gee_distance_m, 180.0, delta=5.0)
        # Sighting at 100m is +4.0 cm
        idx_100 = int(np.abs(result.distances - 100.0).argmin())
        self.assertAlmostEqual(result.heights_cm[idx_100], 4.0, delta=0.2)
        # Without wind, windage is 0 everywhere
        self.assertAlmostEqual(result.wind_drift_100m_cm, 0.0, places=3)
        self.assertAlmostEqual(result.wind_drift_200m_cm, 0.0, places=3)

    def test_compute_trajectory_with_side_wind(self):
        cartridge = ballistics.DEFAULT_CARTRIDGES[0]
        result = ballistics.compute_trajectory(cartridge, wind_speed_kmh=20.0)

        # Side wind causes positive deflection
        self.assertGreater(result.wind_drift_100m_cm, 2.0)
        self.assertGreater(result.wind_drift_200m_cm, result.wind_drift_100m_cm)
        self.assertGreater(result.wind_drift_300m_cm, result.wind_drift_200m_cm)

    def test_generate_ballistics_plot_svg(self):
        cartridge = ballistics.DEFAULT_CARTRIDGES[0]
        result = ballistics.compute_trajectory(cartridge, wind_speed_kmh=15.0)
        svg = ballistics.generate_ballistics_plot_svg(
            cartridge, result, wind_speed_kmh=15.0
        )

        self.assertIn("<svg", svg)
        self.assertIn("Ballistic Curves: RWS .308 Win. HIT Short Rifle", svg)
        self.assertIn("Trajectory (cm)", svg)
        self.assertIn("Wind Drift (cm)", svg)
        self.assertIn("Velocity (m/s)", svg)
        self.assertIn("Energy (J)", svg)
        self.assertIn("2000 J", svg)
        self.assertIn("+4 cm limit", svg)


class TestBallisticsWebPage(unittest.IsolatedAsyncioTestCase):
    async def test_ballistics_page_elements(self):
        cameras = [
            Camera("Seat1", 48.65, 9.00),
            Camera("Seat2", 48.66, 9.01),
        ]
        now = datetime.now(UTC)
        fake_wind = {
            "Seat1": [
                WindHour(pd.Timestamp(now + timedelta(hours=i)), 12.0 + i, 90.0, 18.0)
                for i in range(10)
            ],
            "Seat2": [
                WindHour(pd.Timestamp(now + timedelta(hours=i)), 10.0 + i, 90.0, 15.0)
                for i in range(10)
            ],
        }

        with (
            patch("web.ballistics.read_camera_locations", return_value=cameras),
            patch("web.ballistics.fetch_wind", return_value=fake_wind),
        ):
            async with web_app.app.router.lifespan_context(web_app.app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=web_app.app),
                    base_url="http://test",
                ) as client:
                    user = User(client)
                    await user.open("/ballistics")
                    await user.should_see("Ballistics & Wind Drift")
                    await user.should_see("Hunting Caliber / Cartridge")
                    await user.should_see("Wind Horizon (hours)")
                    await user.should_see("Muzzle Energy (E0)")
                    await user.should_see("GEE / MRD")
                    await user.should_see("Ballistic Drop Table")

                    # Verify SVG and table exist
                    svgs = user.find(marker="ballistics-svg").elements
                    self.assertEqual(len(svgs), 1)

                    tables = user.find(marker="ballistics-table").elements
                    self.assertEqual(len(tables), 1)
