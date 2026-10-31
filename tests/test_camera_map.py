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
