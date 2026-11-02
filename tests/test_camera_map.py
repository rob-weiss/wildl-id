import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from web import camera_map


class CameraMapTests(unittest.TestCase):
    def test_zero_days_hides_even_current_species(self):
        now = datetime(2026, 11, 2, 12, tzinfo=UTC)
        rows = [{"location_id": "A", "class": "fox", "timestamp_exif": now.isoformat()}]

        self.assertEqual(camera_map.recent_species_by_location(rows, 0, now), {})

    def test_recent_species_use_exif_effective_labels_and_rolling_days(self):
        now = datetime(2026, 11, 2, 12, tzinfo=UTC)
        rows = [
            {
                "location_id": location,
                "class": species,
                "manual_label": manual,
                "timestamp_exif": taken,
                "timestamp": now.isoformat(),
            }
            for location, species, manual, taken in (
                ("A", "fox", "", now.isoformat()),
                ("A", "fox", "", (now - timedelta(hours=1)).isoformat()),
                ("A", "human", "Badger", (now - timedelta(days=2)).isoformat()),
                ("B", "wild boar", "", (now - timedelta(days=7)).isoformat()),
                ("B", "roe deer", "", (now - timedelta(days=7, seconds=1)).isoformat()),
                ("A", "fox", "human", now.isoformat()),
                ("A", "fox", "", (now + timedelta(seconds=1)).isoformat()),
                ("A", "marten", "", "bad"),
                ("A", "hare", "", ""),
                ("A", "squirrel", "", now.isoformat()),
            )
        ]

        self.assertEqual(
            camera_map.recent_species_by_location(rows, 7, now),
            {"A": ("badger", "fox"), "B": ("wild boar",)},
        )
        self.assertEqual(
            camera_map.recent_species_by_location(rows, 1, now), {"A": ("fox",)}
        )

    def test_sighting_marker_label_escapes_sites_and_names_species(self):
        camera = camera_map.Camera("A <script>", 48.6, 9.0)
        label = camera_map.sighting_marker_label(camera, ("badger", "fox"))

        self.assertIn("A &lt;script&gt;", label)
        self.assertNotIn("<script>", label)
        self.assertIn('data-species="fox"', label)
        self.assertIn('aria-label="badger"', label)
        self.assertEqual(label.count('data-species="fox"'), 1)

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

    def test_downwind_bearing_is_opposite_wind_source(self):
        self.assertEqual(camera_map.downwind_bearing(270), 90)
        self.assertEqual(camera_map.downwind_bearing(0), 180)

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

    def test_seats_rank_by_gap_between_view_and_wind_arc(self):
        import pandas as pd

        def hours(direction):
            start = pd.Timestamp("2026-10-31T12:00", tz="UTC")
            return [
                camera_map.WindHour(start + pd.Timedelta(hours=h), 10, direction, 15)
                for h in range(24)
            ]

        # Wind from 270 blows towards 90; the arc is exactly 90.
        upwind = camera_map.Camera("Upwind", 48.6, 9.0, 270, 60)
        side = camera_map.Camera("Side", 48.6, 9.0, 0, 90)
        downwind = camera_map.Camera("Downwind", 48.6, 9.0, 90, 60)
        blind = camera_map.Camera("Blind", 48.6, 9.0)
        wind = {name: hours(270) for name in ("Upwind", "Side", "Downwind", "Blind")}
        ranking = camera_map.rank_seats([upwind, side, blind, downwind], wind, 0)
        self.assertEqual(
            [camera.location_id for camera, _ in ranking],
            ["Upwind", "Side", "Downwind"],
        )
        self.assertEqual([gap for _, gap in ranking], [150.0, 45.0, 0.0])

    def test_arc_colour_by_distance(self):
        red, yellow, green, grey = "#ef4444", "#facc15", "#22c55e", "#e5e7eb"
        self.assertEqual(camera_map.arc_distance(0, 60, 40, 40), 0.0)
        self.assertEqual(camera_map.arc_distance(0, 60, 320, 40), 0.0)
        self.assertEqual(camera_map.arc_distance(0, 60, 70, 40), 20.0)
        self.assertEqual(camera_map.arc_distance(0, 60, 180, 40), 130.0)
        self.assertEqual(camera_map.arc_colour(0), red)
        self.assertEqual(camera_map.arc_colour(30), yellow)
        self.assertEqual(camera_map.arc_colour(30.5), green)
        self.assertEqual(camera_map.arc_colour(None), grey)

    def test_downwind_sector_covers_all_selected_hours(self):
        import pandas as pd

        def hours(*directions):
            start = pd.Timestamp("2026-10-31T12:00", tz="UTC")
            return [
                camera_map.WindHour(start + pd.Timedelta(hours=h), 10, d, 15)
                for h, d in enumerate(directions)
            ]

        padding = 2 * camera_map.WIND_ARC_PADDING
        self.assertEqual(camera_map.downwind_sector(hours(270)), (90.0, padding))
        self.assertEqual(
            camera_map.downwind_sector(hours(270, 0, 270)), (135.0, 90 + padding)
        )
        self.assertEqual(
            camera_map.downwind_sector(hours(90, 180)), (315.0, 90 + padding)
        )

    def test_wind_arc_is_a_sector_pointing_downwind(self):
        start, first, *_, last = camera_map.sector_polygon(
            48.6, 9.0, center=90, width=60, radius=100
        )
        self.assertEqual(start, [48.6, 9.0])
        self.assertGreater(first[1], 9.0)
        self.assertGreater(last[1], 9.0)
        self.assertGreater(first[0], last[0])

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
        self.assertEqual(zoom, 15)
        self.assertEqual(camera_map.initial_view([cameras[0]])[1], 17)
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
