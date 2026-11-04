import tempfile
import unittest
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from web import gallery
from src.labelling import label_data
from web.gallery import (
    FILTER_EXAMPLES,
    LOCAL_IMAGE_ROUTE,
    describe_filter,
    filter_rows,
    image_source,
    parse_filter_prompt,
)

TODAY = date(2026, 10, 30)
LABELS = {"wild boar", "fox", "badger", "roe deer", "red deer", "bird"}
LOCATIONS = {"Douglasien", "Madenhäule", "Sauloch hinten", "Sauloch vorne"}


def make_row(
    image_file,
    label,
    timestamp,
    location="Douglasien",
    manual_label="",
    lighting="bright",
    temperature="10.0",
):
    return {
        "location_id": location,
        "image_file": image_file,
        "timestamp": timestamp,
        "timestamp_exif": timestamp,
        "class": label,
        "lighting": lighting,
        "temperature_celsius": temperature,
        "manual_label": manual_label,
    }


def parse(prompt):
    return parse_filter_prompt(prompt, LABELS, LOCATIONS, TODAY)


def names(rows):
    return [row["image_file"] for row in rows]


class VisitFrameTests(unittest.TestCase):
    def test_calendar_filters_use_berlin_dates_for_utc_exif(self):
        row = make_row("new-year.jpg", "fox", "2023-12-31T23:30:00Z")

        self.assertEqual(names(filter_rows([row], parse("in 2024"))), ["new-year.jpg"])
        self.assertEqual(filter_rows([row], parse("in 2023")), [])
        self.assertEqual(gallery.row_datetime(row).year, 2024)

    def test_grouping_precedes_filters_and_preserves_input_order(self):
        rows = [
            make_row("last.jpg", "fox", "2026-10-30T12:50:00"),
            make_row("first.jpg", "fox", "2026-10-30T12:00:00"),
            make_row("middle.jpg", "fox", "2026-10-30T12:25:00"),
            make_row("boar.jpg", "wild boar", "2026-10-30T12:30:00"),
            make_row("next.jpg", "fox", "2026-10-30T13:20:00"),
        ]
        grouped = gallery.with_visit_groups(rows)

        self.assertEqual(names(grouped), names(rows))
        self.assertNotIn("_visit", rows[0])
        self.assertEqual(grouped[0]["_visit"], grouped[1]["_visit"])
        self.assertEqual(grouped[0]["_visit_colour"], grouped[2]["_visit_colour"])
        self.assertNotEqual(grouped[0]["_visit_colour"], grouped[3]["_visit_colour"])
        self.assertNotEqual(grouped[0]["_visit_colour"], grouped[4]["_visit_colour"])
        limited = filter_rows(grouped, parse("only foxes latest 2"))
        self.assertEqual(names(limited), ["next.jpg", "last.jpg"])
        self.assertEqual(limited[1]["_visit"].image_count, 3)

    def test_dates_and_sorting_never_fall_back_to_timestamp(self):
        first = make_row("first.jpg", "fox", "2099-01-01T00:00:00")
        first["timestamp_exif"] = "2026-10-30T12:00:00"
        second = make_row("second.jpg", "fox", "1999-01-01T00:00:00")
        second["timestamp_exif"] = "2026-10-30T12:10:00"
        missing = make_row("missing.jpg", "fox", "2026-10-30T12:20:00")
        missing["timestamp_exif"] = ""
        invalid = make_row("invalid.jpg", "fox", "2026-10-30T12:30:00")
        invalid["timestamp_exif"] = "bad"

        rows = gallery.with_visit_groups([first, missing, second, invalid])

        self.assertIsNone(gallery.row_datetime(missing))
        self.assertIsNone(gallery.row_datetime(invalid))
        self.assertNotIn("_visit", rows[1])
        self.assertEqual(
            names(filter_rows(rows, parse("only foxes"))),
            ["second.jpg", "first.jpg", "missing.jpg", "invalid.jpg"],
        )


class RecentSightingsTests(unittest.TestCase):
    def test_bird_queries_include_birds_pigeons_and_crows(self):
        rows = [
            make_row(f"{label}.jpg", label, "2026-10-30T12:00:00")
            for label in ("bird", "pigeon", "crow", "fox")
        ]
        for prompt in ("birds", "vogel"):
            with self.subTest(prompt=prompt):
                self.assertEqual(
                    set(names(filter_rows(rows, parse(prompt)))),
                    {"bird.jpg", "pigeon.jpg", "crow.jpg"},
                )

    def test_rolling_window_species_and_manual_labels(self):
        now = datetime(2026, 10, 30, 12)
        rows = [
            make_row(f"{label}.jpg", label, now.isoformat())
            for label in (
                "roe deer",
                "wild boar",
                "badger",
                "marten",
                "fox",
                "hare",
                "raccoon",
            )
        ]
        rows += [
            make_row("boundary.jpg", "fox", (now - timedelta(hours=24)).isoformat()),
            make_row(
                "old.jpg", "fox", (now - timedelta(hours=24, seconds=1)).isoformat()
            ),
            make_row("future.jpg", "fox", (now + timedelta(seconds=1)).isoformat()),
            make_row("undated.jpg", "fox", "invalid"),
            make_row("human.jpg", "human", now.isoformat()),
            make_row("red-deer.jpg", "red deer", now.isoformat()),
            make_row("corrected.jpg", "human", now.isoformat(), manual_label="Fox"),
            make_row("excluded.jpg", "fox", now.isoformat(), manual_label="human"),
        ]

        self.assertTrue(callable(getattr(gallery, "recent_rows", None)))
        result = names(gallery.recent_rows(rows, now))

        self.assertEqual(
            set(result),
            {
                "roe deer.jpg",
                "wild boar.jpg",
                "badger.jpg",
                "marten.jpg",
                "fox.jpg",
                "hare.jpg",
                "raccoon.jpg",
                "boundary.jpg",
                "corrected.jpg",
            },
        )
        self.assertEqual(result[-1], "boundary.jpg")

    def test_timezone_offsets_and_exif_fallback(self):
        now = datetime(2026, 10, 30, 12, tzinfo=UTC)
        rows = [
            make_row("newest.jpg", "fox", "2026-10-30T12:30:00+01:00"),
            make_row("old.jpg", "fox", "2026-10-29T12:30:00+01:00"),
            make_row("exif.jpg", "fox", "invalid"),
        ]
        rows[-1]["timestamp_exif"] = "2026-10-30T11:00:00Z"

        self.assertTrue(callable(getattr(gallery, "recent_rows", None)))
        self.assertEqual(
            names(gallery.recent_rows(rows, now)), ["newest.jpg", "exif.jpg"]
        )


class SmartFilterTests(unittest.TestCase):
    def test_relevant_species_prompt_uses_effective_manual_labels(self):
        rows = [
            make_row(f"{label}.jpg", label, "2026-10-29T12:00:00")
            for label in sorted(
                set(gallery.RECENT_SPECIES) | {"squirrel", "human", "none"}
            )
        ]
        rows.extend(
            [
                make_row(
                    "corrected.jpg", "human", "2026-10-29T12:00:00", manual_label="Fox"
                ),
                make_row(
                    "excluded.jpg", "fox", "2026-10-29T12:00:00", manual_label="human"
                ),
            ]
        )

        gallery_filter = parse("Relevant species only")

        self.assertTrue(gallery_filter.relevant_only)
        self.assertFalse(gallery_filter.is_empty())
        self.assertEqual(describe_filter(gallery_filter), "relevant species")
        self.assertEqual(
            set(names(filter_rows(rows, gallery_filter))),
            {f"{label}.jpg" for label in gallery.RECENT_SPECIES} | {"corrected.jpg"},
        )
        self.assertFalse(parse("").relevant_only)

    def test_relevant_species_composes_with_dates_lighting_and_exclusions(self):
        rows = [
            make_row("badger.jpg", "badger", "2026-10-29T12:00:00", lighting="dark"),
            make_row("fox.jpg", "fox", "2026-10-29T12:00:00", lighting="dark"),
            make_row(
                "squirrel.jpg", "squirrel", "2026-10-29T12:00:00", lighting="dark"
            ),
            make_row("old.jpg", "badger", "2026-10-01T12:00:00", lighting="dark"),
            make_row("day.jpg", "badger", "2026-10-29T12:00:00"),
        ]

        gallery_filter = parse(
            "Relevant species only at night last 7 days except foxes"
        )

        self.assertEqual(names(filter_rows(rows, gallery_filter)), ["badger.jpg"])

    def test_hour_prompts_are_time_windows_not_image_limits(self):
        for prompt, hours in (
            ("last 24h", 24),
            ("last 24 hours", 24),
            ("past 24 hrs", 24),
            ("previous 2h", 2),
            ("in the last hour", 1),
            ("past an hour", 1),
        ):
            with self.subTest(prompt=prompt):
                gallery_filter = parse(prompt)
                self.assertFalse(gallery_filter.is_empty())
                self.assertIsNone(gallery_filter.limit)
                self.assertEqual(gallery_filter.hours, hours)
                self.assertIn(f"last {hours} hours", describe_filter(gallery_filter))

    def test_hour_window_uses_timestamps_offsets_and_has_no_count_limit(self):
        now = datetime.fromisoformat("2026-10-30T12:00:00+00:00")
        rows = [
            make_row(
                f"recent-{index}.jpg",
                "fox",
                (now - timedelta(minutes=index)).isoformat(),
            )
            for index in range(30)
        ]
        rows += [
            make_row("boundary.jpg", "fox", "2026-10-29T13:00:00+01:00"),
            make_row("old.jpg", "fox", "2026-10-29T12:59:59+01:00"),
            make_row("future.jpg", "fox", (now + timedelta(seconds=1)).isoformat()),
            make_row("invalid.jpg", "fox", "invalid"),
            make_row("undated.jpg", "fox", ""),
        ]
        for prompt in ("last 24h", "last 24 hours"):
            with self.subTest(prompt=prompt):
                gallery_filter = parse(prompt)
                self.assertFalse(gallery_filter.is_empty())
                self.assertIsNone(gallery_filter.limit)
                self.assertEqual(
                    names(filter_rows(rows, gallery_filter, now=now)),
                    [f"recent-{index}.jpg" for index in range(30)] + ["boundary.jpg"],
                )

    def test_hour_window_composes_with_filters_and_matches_home(self):
        now = datetime(2026, 10, 30, 12)
        rows = [
            make_row("recent.jpg", "fox", "2026-10-30T11:00:00"),
            make_row("boundary.jpg", "fox", "2026-10-29T12:00:00"),
            make_row("old.jpg", "fox", "2026-10-29T11:59:59"),
            make_row("human.jpg", "human", "2026-10-30T11:00:00"),
        ]
        gallery_filter = parse("only foxes last 24h")
        self.assertIn("last 24 hours", describe_filter(gallery_filter))
        self.assertEqual(
            names(filter_rows(rows, gallery_filter, now=now)),
            names(gallery.recent_rows(rows, now)),
        )
        self.assertEqual(parse("latest 24").limit, 24)
        self.assertEqual(parse("last 24").limit, 24)
        self.assertEqual(parse("foxes last 24h latest 2").limit, 2)

    def test_all_dropdown_examples_produce_supported_filters(self):
        for category, examples in FILTER_EXAMPLES.items():
            for prompt in examples:
                with self.subTest(category=category, prompt=prompt):
                    self.assertFalse(parse(prompt).is_empty())

    def test_wild_boar_of_last_two_weeks(self):
        rows = [
            make_row("old.jpg", "wild boar", "2026-10-01T22:00:00"),
            make_row("fox.jpg", "fox", "2026-10-20T22:00:00"),
            make_row("recent.jpg", "wild boar", "2026-10-20T22:00:00"),
            make_row(
                "newest.jpg", "fox", "2026-10-29T01:00:00", manual_label="wild boar"
            ),
        ]

        gallery_filter = parse("Show only wild boar of the last two weeks")

        self.assertEqual(gallery_filter.labels, {"wild boar"})
        self.assertEqual(gallery_filter.start, date(2026, 10, 16))
        self.assertEqual(gallery_filter.end, TODAY)
        self.assertEqual(
            names(filter_rows(rows, gallery_filter)), ["newest.jpg", "recent.jpg"]
        )

    def test_aliases_and_plurals(self):
        self.assertEqual(parse("boars").labels, {"wild boar"})
        self.assertEqual(parse("foxes").labels, {"fox"})
        self.assertEqual(parse("all deer").labels, {"roe deer", "red deer"})
        self.assertEqual(
            parse("birds").labels,
            {"bird", "pigeon", "crow"},
        )

    def test_negated_labels_and_locations(self):
        gallery_filter = parse("everything except foxes and badgers, not at Douglasien")

        self.assertEqual(gallery_filter.excluded_labels, {"fox", "badger"})
        self.assertEqual(gallery_filter.excluded_locations, {"Douglasien"})
        self.assertFalse(gallery_filter.labels)

    def test_location_without_umlaut_and_first_word(self):
        self.assertEqual(parse("foxes at Madenhaeule").locations, {"Madenhäule"})
        self.assertEqual(
            parse("Sauloch").locations, {"Sauloch hinten", "Sauloch vorne"}
        )

    def test_absolute_dates(self):
        gallery_filter = parse("badgers in March 2025")
        self.assertEqual(
            (gallery_filter.start, gallery_filter.end),
            (date(2025, 3, 1), date(2025, 3, 31)),
        )
        gallery_filter = parse("between 2025-01-05 and 10.02.2025")
        self.assertEqual(
            (gallery_filter.start, gallery_filter.end),
            (date(2025, 1, 5), date(2025, 2, 10)),
        )
        self.assertEqual(parse("since 2025").start, date(2025, 1, 1))
        self.assertEqual(parse("before 2025-06-01").end, date(2025, 5, 31))

    def test_relative_and_seasonal_dates(self):
        self.assertEqual(parse("past 3 months").start, date(2026, 7, 30))
        self.assertEqual(parse("this year").start, date(2026, 1, 1))
        self.assertEqual(parse("yesterday").end, date(2026, 10, 29))
        self.assertEqual(parse("foxes in winter").months, {12, 1, 2})
        self.assertEqual(parse("jays in may").months, {5})
        self.assertFalse(parse("may show foxes").months)

    def test_lighting_temperature_and_status(self):
        rows = [
            make_row(
                "cold-night.jpg",
                "fox",
                "2025-01-01T01:00:00",
                lighting="dark",
                temperature="-3",
            ),
            make_row(
                "warm-night.jpg",
                "fox",
                "2025-01-02T01:00:00",
                lighting="dark",
                temperature="5",
            ),
            make_row("cold-day.jpg", "fox", "2025-01-03T12:00:00", temperature="-1"),
        ]

        self.assertEqual(
            names(filter_rows(rows, parse("foxes at night below 0 °C"))),
            ["cold-night.jpg"],
        )
        self.assertEqual(parse("not at night").lighting, "bright")
        self.assertEqual(parse("warmer than 4").min_temperature, 4)
        self.assertIsNone(parse("over 5 foxes").min_temperature)
        self.assertIs(parse("unlabelled foxes").reviewed, False)
        self.assertEqual(parse("not reviewed foxes").labels, {"fox"})
        self.assertIs(parse("manually labelled").reviewed, True)

    def test_automatic_label_status_is_distinct_from_manual_review(self):
        self.assertIs(parse("labelled").automatically_labelled, True)
        self.assertIsNone(parse("manually labelled").automatically_labelled)
        rows = [
            make_row("detected.jpg", "fox", "2025-01-01T00:00:00"),
            make_row("empty.jpg", "none", "2025-01-02T00:00:00"),
            make_row("manual.jpg", "", "2025-01-03T00:00:00", manual_label="fox"),
            make_row("new.jpg", "", "2025-01-04T00:00:00"),
        ]
        self.assertEqual(
            names(filter_rows(rows, parse("automatically labelled"))),
            ["empty.jpg", "detected.jpg"],
        )
        self.assertEqual(
            names(filter_rows(rows, parse("not automatically labelled"))),
            ["new.jpg", "manual.jpg"],
        )
        self.assertIn("automatically labelled", describe_filter(parse("labelled")))

    def test_misclassified_order_and_limit(self):
        rows = [
            make_row("a.jpg", "fox", "2025-01-01T00:00:00", manual_label="badger"),
            make_row("b.jpg", "fox", "2025-01-02T00:00:00", manual_label="fox"),
            make_row("c.jpg", "fox", "2025-01-03T00:00:00", manual_label="bird"),
        ]

        self.assertEqual(
            names(filter_rows(rows, parse("misclassified oldest first"))),
            ["a.jpg", "c.jpg"],
        )
        self.assertEqual(names(filter_rows(rows, parse("latest 1"))), ["c.jpg"])

    def test_misclassified_needs_a_prediction(self):
        rows = [
            make_row("a.jpg", "", "2025-01-01T00:00:00", manual_label="badger"),
            make_row("b.jpg", "unknown", "2025-01-02T00:00:00", manual_label="fox"),
            make_row("c.jpg", "fox", "2025-01-03T00:00:00", manual_label="bird"),
        ]

        self.assertEqual(names(filter_rows(rows, parse("misclassified"))), ["c.jpg"])

    def test_unknown_prompt_is_empty_filter(self):
        gallery_filter = parse("show me something nice")

        self.assertTrue(gallery_filter.is_empty())
        self.assertEqual(describe_filter(gallery_filter), "all images")

    def test_description_lists_interpretation(self):
        description = describe_filter(parse("wild boar of the last two weeks"))

        self.assertIn("wild boar", description)
        self.assertIn("16-10-2026 – 30-10-2026", description)


class ImageSourceTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.images_dir = Path(temporary_directory.name)

    def test_prefers_local_image(self):
        local_path = self.images_dir / "2025" / "Madenhäule" / "fox 1.jpg"
        row = make_row("fox 1.jpg", "fox", "2025-03-01T00:00:00", location="Madenhäule")

        source = image_source(
            row,
            {("Madenhäule", "fox 1.jpg"): local_path},
            self.images_dir,
            "https://cdn.example.test/images",
        )

        self.assertEqual(
            source, f"{LOCAL_IMAGE_ROUTE}/2025/Madenh%C3%A4ule/fox%201.jpg"
        )

    def test_falls_back_to_cloud_by_year(self):
        row = make_row("fox.jpg", "fox", "2025-03-01T00:00:00")

        self.assertEqual(
            image_source(row, {}, self.images_dir, "https://cdn.example.test/images/"),
            "https://cdn.example.test/images/2025/Douglasien/fox.jpg",
        )
        self.assertIsNone(image_source(row, {}, self.images_dir, None))


class CanonicalMetadataTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.labels_dir = Path(temporary_directory.name) / "labels"
        self.predictions_csv = (
            self.labels_dir / "labels_model" / "predictions.csv"
        )
        self.metadata_csv = self.labels_dir / "image_metadata.csv"

    def test_prediction_gallery_uses_canonical_metadata(self):
        prediction = {
            "location_id": "Madenha\u0308ule",
            "image_file": "fox.jpg",
            "class": "fox",
            "box": "[0.1, 0.2, 0.3, 0.4]",
            "confidence": "0.95",
            "lighting": "bright",
            "temperature_celsius": "20",
            "reviewed": "true",
        }
        label_data.write_records(
            self.predictions_csv, [prediction], label_data.PREDICTION_COLUMNS
        )
        label_data.write_records(
            self.metadata_csv,
            [
                {
                    "location_id": "Madenhäule",
                    "image_file": "fox.jpg",
                    "lighting": "dark",
                    "temperature_celsius": "7",
                }
            ],
            label_data.METADATA_COLUMNS,
        )

        [row] = gallery.load_rows(self.predictions_csv)

        self.assertEqual(row["lighting"], "dark")
        self.assertEqual(row["temperature_celsius"], "7")
        self.assertEqual(row["class"], "fox")
        self.assertEqual(row["box"], prediction["box"])
        self.assertEqual(row["confidence"], "0.95")
        self.assertEqual(row["reviewed"], "true")

    def test_blank_canonical_metadata_overrides_prediction_values(self):
        prediction = {
            "location_id": "camera1",
            "image_file": "fox.jpg",
            "class": "fox",
            "lighting": "bright",
            "temperature_celsius": "20",
        }
        label_data.write_records(
            self.predictions_csv, [prediction], label_data.PREDICTION_COLUMNS
        )
        label_data.write_records(
            self.metadata_csv,
            [
                {
                    "location_id": "camera1",
                    "image_file": "fox.jpg",
                    "lighting": "",
                    "temperature_celsius": "",
                }
            ],
            label_data.METADATA_COLUMNS,
        )

        [row] = gallery.load_rows(self.predictions_csv)

        self.assertEqual(row["lighting"], "")
        self.assertEqual(row["temperature_celsius"], "")

    def test_prediction_without_metadata_has_blank_lighting(self):
        prediction = {
            "location_id": "camera1",
            "image_file": "fox.jpg",
            "class": "fox",
            "box": "[0.1, 0.2, 0.3, 0.4]",
            "confidence": "0.95",
            "lighting": "bright",
            "reviewed": "true",
        }
        label_data.write_records(
            self.predictions_csv, [prediction], label_data.PREDICTION_COLUMNS
        )
        label_data.write_records(
            self.metadata_csv, [], label_data.METADATA_COLUMNS
        )

        [row] = gallery.load_rows(self.predictions_csv)

        self.assertEqual(row["lighting"], "")
        self.assertEqual(row["class"], "fox")
        self.assertEqual(row["box"], prediction["box"])
        self.assertEqual(row["confidence"], "0.95")
        self.assertEqual(row["reviewed"], "true")

    def test_human_label_gallery_uses_canonical_metadata(self):
        repository = self.predictions_csv.parents[2]
        human_csv = self.labels_dir / "human_labels.csv"
        label_data.write_records(
            human_csv,
            [
                {
                    "location_id": "camera1",
                    "image_file": "fox.jpg",
                    "box": "",
                    "manual_label": "fox",
                }
            ],
            label_data.HUMAN_COLUMNS,
        )
        label_data.write_records(
            self.metadata_csv,
            [
                {
                    "location_id": "camera1",
                    "image_file": "fox.jpg",
                    "lighting": "dark",
                    "temperature_celsius": "7",
                }
            ],
            label_data.METADATA_COLUMNS,
        )
        prediction_path = label_data.predictions_path(repository)
        label_data.write_records(
            prediction_path,
            [
                {
                    "location_id": "camera1",
                    "image_file": "fox.jpg",
                    "class": "fox",
                    "lighting": "bright",
                    "temperature_celsius": "20",
                }
            ],
            label_data.PREDICTION_COLUMNS,
        )

        [row] = gallery.load_rows(human_csv)

        self.assertEqual(row["lighting"], "dark")
        self.assertEqual(row["temperature_celsius"], "7")


if __name__ == "__main__":
    unittest.main()
