import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from web import camera_map


class CameraMapTests(unittest.TestCase):
    def write_locations(self, cameras):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "locations.json"
        path.write_text(json.dumps({"cameras": cameras}), encoding="utf-8")
        return path

    def test_reads_located_and_missing_cameras(self):
        path = self.write_locations(
            [
                {"location_id": "A", "latitude": 48.6, "longitude": 9},
                {"location_id": "B", "latitude": None, "longitude": None},
            ]
        )
        cameras = camera_map.read_camera_locations(path)
        self.assertEqual([camera.located for camera in cameras], [True, False])
        self.assertEqual(cameras[0].longitude, 9.0)

    def test_rejects_invalid_coordinates(self):
        for entry in (
            {"location_id": "A", "latitude": 91, "longitude": 9},
            {"location_id": "A", "latitude": 48.6, "longitude": None},
            {"location_id": "A", "latitude": "48.6", "longitude": 9},
            {"location_id": "A", "latitude": True, "longitude": 9},
            {"location_id": "A", "view_direction": 90},
            {"location_id": "A", "view_direction": -10, "view_angle": 60},
            {"location_id": "A", "view_direction": 10, "view_angle": 0},
        ):
            with self.subTest(entry=entry), self.assertRaises((TypeError, ValueError)):
                camera_map.read_camera_locations(self.write_locations([entry]))

    def test_arrow_points_downwind(self):
        start, tip, *_ = camera_map.arrow_points(48.6, 9.0, 10, 270)
        self.assertEqual(start, [48.6, 9.0])
        self.assertAlmostEqual(tip[0], 48.6)
        self.assertGreater(tip[1], 9.0)
        start, tip, *_ = camera_map.arrow_points(48.6, 9.0, 10, 0)
        self.assertLess(tip[0], 48.6)

    def test_fetch_wind_parses_open_meteo_hours(self):
        cameras = [
            camera_map.Camera("A", 48.6, 9.0),
            camera_map.Camera("B", 48.7, 9.1),
            camera_map.Camera("C", None, None),
        ]
        hourly = {
            "time": [1_790_000_000 + 3600 * hour for hour in range(24)],
            "wind_speed_10m": [5.0] * 24,
            "wind_direction_10m": [180.0] * 24,
            "wind_gusts_10m": [12.0] * 24,
        }
        camera_map._wind_cache.clear()
        with patch.object(
            camera_map, "_request_wind", return_value=[{"hourly": hourly}] * 2
        ) as request:
            wind = camera_map.fetch_wind(cameras)
            camera_map.fetch_wind(cameras)
        request.assert_called_once_with(((48.6, 9.0), (48.7, 9.1)))
        self.assertEqual(set(wind), {"A", "B"})
        self.assertEqual(len(wind["A"]), 24)
        self.assertEqual(wind["A"][0].time.tzname(), "UTC")

    def test_seats_rank_by_wind_from_viewed_area(self):
        import pandas as pd

        def hours(direction):
            start = pd.Timestamp("2026-10-31T12:00", tz="UTC")
            return [
                camera_map.WindHour(start + pd.Timedelta(hours=h), 10, direction, 15)
                for h in range(24)
            ]

        upwind = camera_map.Camera("Upwind", 48.6, 9.0, 270, 60)
        side = camera_map.Camera("Side", 48.6, 9.0, 0, 90)
        downwind = camera_map.Camera("Downwind", 48.6, 9.0, 90, 60)
        blind = camera_map.Camera("Blind", 48.6, 9.0)
        wind = {name: hours(270) for name in ("Upwind", "Side", "Downwind", "Blind")}
        ranking = camera_map.rank_seats([downwind, side, blind, upwind], wind, 0)
        self.assertEqual(
            [camera.location_id for camera, _ in ranking],
            ["Upwind", "Downwind", "Side"],
        )
        self.assertEqual([round(score, 2) for _, score in ranking], [1.0, 0.0, 0.0])

        wind["Side"] = hours(0)
        wind["Side"][2] = camera_map.WindHour(wind["Side"][2].time, 10, 180, 15)
        self.assertAlmostEqual(
            dict(camera_map.rank_seats([side], wind, 0))[side], 2 / 3
        )
        self.assertEqual(dict(camera_map.rank_seats([side], wind, 3))[side], 1.0)

    def test_view_polygon_spans_opening_angle(self):
        camera = camera_map.Camera("A", 48.6, 9.0, 90, 60)
        start, first, *_, last = camera_map.view_polygon(camera)
        self.assertEqual(start, [48.6, 9.0])
        self.assertGreater(first[0], 48.6)
        self.assertLess(last[0], 48.6)
        self.assertGreater(first[1], 9.0)

    def test_initial_view_centres_bounding_box(self):
        cameras = [
            camera_map.Camera("A", 48.65, 8.98),
            camera_map.Camera("B", 48.66, 9.02),
            camera_map.Camera("C", 48.651, 8.99),
        ]
        center, zoom = camera_map.initial_view(cameras)
        self.assertAlmostEqual(center[0], 48.655)
        self.assertAlmostEqual(center[1], 9.0)
        self.assertEqual(zoom, 14)
        self.assertEqual(camera_map.initial_view([cameras[0]])[1], 16)
        self.assertEqual(camera_map.initial_view([]), (camera_map.DEFAULT_CENTER, 13))

    def test_fetch_wind_rejects_incomplete_forecast(self):
        camera_map._wind_cache.clear()
        hourly = {
            "time": [0],
            "wind_speed_10m": [None],
            "wind_direction_10m": [0],
            "wind_gusts_10m": [0],
        }
        with (
            patch.object(
                camera_map, "_request_wind", return_value=[{"hourly": hourly}]
            ),
            self.assertRaises((TypeError, ValueError)),
        ):
            camera_map.fetch_wind([camera_map.Camera("A", 48.6, 9.0)])


if __name__ == "__main__":
    unittest.main()
