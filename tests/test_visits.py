import csv
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src import visits


def detection(name, time, species="fox", location="A", **extra):
    return {
        "location_id": location,
        "image_file": name,
        "timestamp_exif": time,
        "class": species,
        **extra,
    }


class LongestRecentVisitsTests(unittest.TestCase):
    def test_exact_window_edges_and_ties_choose_latest_visit(self):
        now = datetime(2026, 11, 2, 12, tzinfo=UTC)
        rows = [
            detection("boundary.jpg", "2026-11-01T12:00:00Z", location="A"),
            detection("early.jpg", "2026-11-01T12:20:00Z", location="A"),
            detection("recent.jpg", "2026-11-02T11:40:00Z", location="B"),
            detection("now.jpg", "2026-11-02T12:00:00Z", location="B"),
            detection("future.jpg", "2026-11-02T12:10:00Z", location="B"),
        ]

        result = visits.longest_recent_visits(rows, now=now)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].location_id, "B")
        self.assertEqual(result[0].observed_span_seconds, 1200)
        self.assertEqual(
            result, visits.longest_recent_visits(list(reversed(rows)), now=now)
        )

    def test_selects_longest_positive_visit_per_animal_species(self):
        now = datetime(2026, 11, 2, 12, tzinfo=UTC)
        rows = [
            detection(name, time, species, location, **extra)
            for name, time, species, location, extra in (
                ("a1.jpg", "2026-11-02T10:00:00Z", "fox", "A", {}),
                ("a2.jpg", "2026-11-02T10:20:00Z", "fox", "A", {}),
                ("a3.jpg", "2026-11-02T10:40:00Z", "fox", "A", {}),
                ("b1.jpg", "2026-11-02T11:00:00Z", "fox", "B", {}),
                ("b2.jpg", "2026-11-02T11:25:00Z", "fox", "B", {}),
                ("s1.jpg", "2026-11-02T11:00:00Z", "squirrel", "A", {}),
                ("s2.jpg", "2026-11-02T11:05:00Z", "squirrel", "A", {}),
                ("h1.jpg", "2026-11-02T11:00:00Z", "human", "A", {}),
                ("h2.jpg", "2026-11-02T11:10:00Z", "human", "A", {}),
                (
                    "deer1.jpg",
                    "2026-11-02T11:00:00Z",
                    "human",
                    "B",
                    {"manual_label": "roe deer"},
                ),
                (
                    "deer2.jpg",
                    "2026-11-02T11:15:00Z",
                    "human",
                    "B",
                    {"manual_label": "roe deer"},
                ),
                ("single.jpg", "2026-11-02T11:30:00Z", "badger", "A", {}),
                ("same1.jpg", "2026-11-02T11:00:00Z", "wild boar", "A", {}),
                ("same2.jpg", "2026-11-02T11:00:00Z", "wild boar", "A", {}),
                ("future1.jpg", "2026-11-02T12:05:00Z", "badger", "B", {}),
                ("future2.jpg", "2026-11-02T12:20:00Z", "badger", "B", {}),
            )
        ]

        result = visits.longest_recent_visits(rows, now=now)

        self.assertEqual(
            [
                (visit.species, visit.location_id, visit.observed_span_seconds)
                for visit in result
            ],
            [("fox", "A", 2400), ("roe deer", "B", 900), ("squirrel", "A", 300)],
        )
        self.assertEqual(
            result, visits.longest_recent_visits(list(reversed(rows)), now=now)
        )

    def test_only_uses_observations_inside_last_day_without_splitting_visit_rules(self):
        now = datetime(2026, 11, 2, 12, tzinfo=UTC)
        rows = [
            detection("before.jpg", "2026-11-01T11:50:00Z"),
            detection("inside1.jpg", "2026-11-01T12:10:00Z"),
            detection("inside2.jpg", "2026-11-01T12:20:00Z"),
            detection("old1.jpg", "2026-11-01T10:00:00Z", species="badger"),
            detection("old2.jpg", "2026-11-01T10:20:00Z", species="badger"),
        ]

        result = visits.longest_recent_visits(rows, now=now)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].observed_span_seconds, 600)
        self.assertEqual(result[0].image_count, 2)


class VisitGroupingTests(unittest.TestCase):
    def test_conflicting_image_rows_are_excluded_independent_of_order(self):
        rows = [
            detection("species.jpg", "2026-11-01T18:00:00"),
            detection("species.jpg", "2026-11-01T18:00:00", species="wild boar"),
            detection("time.jpg", "2026-11-01T18:00:00"),
            detection("time.jpg", "2026-11-02T18:00:00"),
            detection("valid.jpg", "2026-11-01T18:05:00"),
        ]
        result = visits.group_visits(rows)

        self.assertEqual(len(result.visits), 1)
        self.assertEqual(result.visits[0].image_keys, (("A", "valid.jpg"),))
        self.assertEqual(result.excluded_count, 4)
        self.assertEqual(result, visits.group_visits(list(reversed(rows))))

    def test_exports_visit_summary_and_member_images(self):
        result = visits.group_visits(
            [
                detection("first.jpg", "2026-11-01T18:00:00"),
                detection("next.jpg", "2026-11-01T18:20:00"),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            summary_path, members_path = visits.export_visits(result, Path(directory))
            with summary_path.open(newline="", encoding="utf-8") as source:
                summary = list(csv.DictReader(source))
            with members_path.open(newline="", encoding="utf-8") as source:
                members = list(csv.DictReader(source))

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["image_count"], "2")
        self.assertEqual(float(summary[0]["observed_span_seconds"]), 1200)
        self.assertEqual(summary[0]["independence_minutes"], "30")
        self.assertEqual(
            [member["image_file"] for member in members], ["first.jpg", "next.jpg"]
        )
        self.assertTrue(
            all(member["visit_id"] == summary[0]["visit_id"] for member in members)
        )

    def test_rolling_gaps_and_exact_boundary_with_unsorted_input(self):
        rows = [
            detection("last.jpg", "2026-11-01T19:20:00"),
            detection("middle.jpg", "2026-11-01T18:25:00"),
            detection("first.jpg", "2026-11-01T18:00:00"),
            detection("late.jpg", "2026-11-01T18:50:00"),
        ]
        original = [dict(row) for row in rows]

        result = visits.group_visits(rows)

        self.assertEqual([visit.image_count for visit in result.visits], [3, 1])
        self.assertEqual(result.visits[0].observed_span_seconds, 50 * 60)
        self.assertEqual(result.visits[1].observed_span_seconds, 0)
        self.assertEqual(rows, original)
        self.assertEqual(result, visits.group_visits(list(reversed(rows))))

    def test_groups_by_site_and_effective_species_across_midnight(self):
        rows = [
            detection("first.jpg", "2026-11-01T23:50:00"),
            detection("other-site.jpg", "2026-11-01T23:55:00", location="B"),
            detection("boar.jpg", "2026-11-02T00:00:00", species="wild boar"),
            detection(
                "corrected.jpg",
                "2026-11-02T00:10:00",
                species="human",
                manual_label=" Fox ",
            ),
        ]

        result = visits.group_visits(rows)

        self.assertEqual(len(result.visits), 3)
        self.assertEqual(
            result.image_visits[("A", "first.jpg")],
            result.image_visits[("A", "corrected.jpg")],
        )
        self.assertNotEqual(
            result.image_visits[("A", "first.jpg")],
            result.image_visits[("B", "other-site.jpg")],
        )

    def test_uses_only_exif_and_reports_unusable_rows(self):
        rows = [
            detection(
                "valid.jpg", "2026-11-01T18:00:00", timestamp="1999-01-01T00:00:00"
            ),
            detection("missing.jpg", "", timestamp="2026-11-01T18:01:00"),
            detection("bad.jpg", "invalid", timestamp="2026-11-01T18:02:00"),
            detection("unlabelled.jpg", "2026-11-01T18:03:00", species="unknown"),
            detection("no-site.jpg", "2026-11-01T18:04:00", location=""),
        ]

        result = visits.group_visits(rows)

        self.assertEqual(len(result.visits), 1)
        self.assertEqual(result.excluded_count, 4)
        self.assertEqual(result.visits[0].started_at.year, 2026)
        self.assertIsNone(visits.exif_datetime(rows[1]))
        self.assertIsNone(visits.exif_datetime(rows[2]))

    def test_deduplicates_image_identity_and_keeps_distinct_photos_at_same_time(self):
        first = detection("first.jpg", "2026-11-01T18:00:00")
        result = visits.group_visits(
            [first, dict(first), detection("second.jpg", "2026-11-01T18:00:00")]
        )

        self.assertEqual(result.visits[0].image_count, 2)
        self.assertEqual(len(result.image_visits), 2)

    def test_offsets_use_elapsed_time_and_naive_times_use_berlin(self):
        rows = [
            detection("first.jpg", "2026-10-25T02:50:00+02:00"),
            detection("next.jpg", "2026-10-25T02:10:00+01:00"),
        ]
        result = visits.group_visits(rows)

        self.assertEqual(len(result.visits), 1)
        self.assertEqual(result.visits[0].observed_span_seconds, 20 * 60)
        self.assertEqual(
            visits.exif_datetime(
                detection("winter.jpg", "2026-11-01T18:00:00")
            ).utcoffset(),
            timedelta(hours=1),
        )
        self.assertIsNone(
            visits.exif_datetime(detection("ambiguous.jpg", "2026-10-25T02:10:00"))
        )
        self.assertIsNone(
            visits.exif_datetime(detection("nonexistent.jpg", "2026-03-29T02:10:00"))
        )


if __name__ == "__main__":
    unittest.main()
