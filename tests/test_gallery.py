import tempfile
import unittest
from datetime import date
from pathlib import Path

from web.gallery import (
    LOCAL_IMAGE_ROUTE,
    describe_filter,
    filter_rows,
    image_source,
    parse_filter_prompt,
)

TODAY = date(2026, 10, 30)
LABELS = {"wild boar", "fox", "badger", "roe deer", "red deer", "jay"}
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
        "timestamp_exif": "",
        "class": label,
        "lighting": lighting,
        "temperature_celsius": temperature,
        "manual_label": manual_label,
    }


def parse(prompt):
    return parse_filter_prompt(prompt, LABELS, LOCATIONS, TODAY)


def names(rows):
    return [row["image_file"] for row in rows]


class SmartFilterTests(unittest.TestCase):
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

    def test_misclassified_order_and_limit(self):
        rows = [
            make_row("a.jpg", "fox", "2025-01-01T00:00:00", manual_label="badger"),
            make_row("b.jpg", "fox", "2025-01-02T00:00:00", manual_label="fox"),
            make_row("c.jpg", "fox", "2025-01-03T00:00:00", manual_label="jay"),
        ]

        self.assertEqual(
            names(filter_rows(rows, parse("misclassified oldest first"))),
            ["a.jpg", "c.jpg"],
        )
        self.assertEqual(names(filter_rows(rows, parse("latest 1"))), ["c.jpg"])

    def test_unknown_prompt_is_empty_filter(self):
        gallery_filter = parse("show me something nice")

        self.assertTrue(gallery_filter.is_empty())
        self.assertEqual(describe_filter(gallery_filter), "all images")

    def test_description_lists_interpretation(self):
        description = describe_filter(parse("wild boar of the last two weeks"))

        self.assertIn("wild boar", description)
        self.assertIn("2026-10-16 – 2026-10-30", description)


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


if __name__ == "__main__":
    unittest.main()
