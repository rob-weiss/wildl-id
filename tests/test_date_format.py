import unittest
from datetime import date, datetime

from src.labelling.label_data import format_date, format_datetime


class DateFormatTests(unittest.TestCase):
    def test_formats_dates_without_leading_zero(self):
        self.assertEqual(format_date(date(2026, 11, 2)), "2. Nov 26")

    def test_formats_datetime_with_clock_emoji(self):
        self.assertEqual(
            format_datetime(datetime(2026, 11, 2, 20, 10)),
            "🕒 2. Nov 26 20:10",
        )


if __name__ == "__main__":
    unittest.main()
