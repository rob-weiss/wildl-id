import sys
import unittest
from contextlib import redirect_stdout
from importlib import import_module
from io import StringIO
from unittest.mock import patch

wildl_id = import_module("wildl_id")


class WildlifeCommandLineTests(unittest.TestCase):
    def test_download_option_is_rejected(self):
        with patch.object(sys, "argv", ["wildl_id.py", "--download"]):
            with self.assertRaises(SystemExit) as error:
                wildl_id.parse_arguments()

        self.assertEqual(error.exception.code, 2)

    def test_help_examples_use_consistent_indentation(self):
        output = StringIO()
        with (
            patch.object(sys, "argv", ["wildl_id.py", "--help"]),
            redirect_stdout(output),
            self.assertRaises(SystemExit) as error,
        ):
            wildl_id.parse_arguments()

        self.assertEqual(error.exception.code, 0)
        help_text = output.getvalue()
        self.assertIn(
            "\nExamples:\n    wildl_id.py                    Run interactive menu\n",
            help_text,
        )
        self.assertIn("\n    wildl_id.py --organize", help_text)
        self.assertNotIn("\n  wildl_id.py --organize", help_text)
        self.assertIn("Generate visualisations and analysis reports", help_text)

    def test_menu_uses_british_english(self):
        output = StringIO()
        with redirect_stdout(output):
            wildl_id.print_menu()

        menu = output.getvalue()
        self.assertIn("Organise new images", menu)
        self.assertIn("visualisations", menu.lower())
        self.assertNotIn("Organize", menu)
        self.assertNotIn("Visualizations", menu)

    def test_all_runs_supported_actions_in_order(self):
        calls = []
        with (
            patch.object(
                wildl_id,
                "organize_images",
                side_effect=lambda: calls.append("organize") or True,
            ),
            patch.object(
                wildl_id,
                "label_images",
                side_effect=lambda **kwargs: calls.append("label") or True,
            ),
            patch.object(
                wildl_id,
                "generate_visualizations",
                side_effect=lambda: calls.append("visualise") or True,
            ),
        ):
            result = wildl_id.run_all()

        self.assertTrue(result)
        self.assertEqual(calls, ["organize", "label", "visualise"])

    def test_all_stops_and_returns_failure(self):
        calls = []
        with (
            patch.object(
                wildl_id,
                "organize_images",
                side_effect=lambda: calls.append("organize") or False,
            ),
            patch.object(
                wildl_id,
                "label_images",
                side_effect=lambda **kwargs: calls.append("label") or True,
            ),
            patch.object(
                wildl_id,
                "generate_visualizations",
                side_effect=lambda: calls.append("visualise") or True,
            ),
        ):
            result = wildl_id.run_all()

        self.assertFalse(result)
        self.assertEqual(calls, ["organize"])

    def test_main_returns_nonzero_when_requested_action_fails(self):
        with (
            patch.object(sys, "argv", ["wildl_id.py", "--organize"]),
            patch.object(wildl_id, "organize_images", return_value=False),
        ):
            result = wildl_id.main()

        self.assertEqual(result, 1)

    def test_main_returns_nonzero_when_visualisation_rejects_input(self):
        with (
            patch.object(sys, "argv", ["wildl_id.py", "--visualize"]),
            patch.object(wildl_id, "visualisation_main", return_value=False),
        ):
            result = wildl_id.main()

        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
