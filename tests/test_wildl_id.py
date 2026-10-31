import csv
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from importlib import import_module
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

wildl_id = import_module("wildl_id")


class WildlifeCommandLineTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(
            wildl_id, "label_new_images", return_value=True, create=True
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_download_option_is_accepted(self):
        with patch.object(sys, "argv", ["wildl_id.py", "--download"]):
            args = wildl_id.parse_arguments()

        self.assertTrue(args.download)

    def test_force_cli_reaches_label_and_pipeline_actions(self):
        for flag, action in (("--label", "label_images"), ("--all", "run_all")):
            with (
                self.subTest(flag=flag),
                patch.object(sys, "argv", ["wildl_id.py", flag, "--force"]),
                patch.object(wildl_id, action, return_value=True) as invoked,
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(wildl_id.main(), 0)
                invoked.assert_called_once_with(
                    camera="seissiger", reprocess_missing=False, force=True
                )

    def test_label_action_forwards_resume_and_force_options(self):
        for force in (False, True):
            with (
                self.subTest(force=force),
                patch.object(
                    wildl_id, "process_images_with_pytorch_wildlife"
                ) as process,
                redirect_stdout(StringIO()),
            ):
                self.assertTrue(wildl_id.label_images(camera="zeiss", force=force))
                process.assert_called_once_with(
                    camera="zeiss", reprocess_missing=False, force=force
                )

    def test_pipeline_forwards_force_to_labelling(self):
        with (
            patch.object(wildl_id, "organize_images", return_value=True),
            patch.object(wildl_id, "label_images", return_value=True) as label,
            patch.object(wildl_id, "generate_visualizations", return_value=True),
            redirect_stdout(StringIO()),
        ):
            self.assertTrue(wildl_id.run_all(camera="zeiss", force=True))

        label.assert_called_once_with(
            camera="zeiss", reprocess_missing=False, force=True
        )

    def test_menu_has_download_choice_five_and_exit_choice_six(self):
        output = StringIO()
        with redirect_stdout(output):
            wildl_id.print_menu()

        menu = output.getvalue()
        self.assertIn("5. Download new images from email", menu)
        self.assertIn("6. Exit", menu)

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

    def test_download_action_organizes_when_result_has_pending_images(self):
        result = SimpleNamespace(
            saved_images=2,
            duplicates=0,
            pending_organization=True,
            failures=(),
        )
        with (
            patch.object(wildl_id, "download_main", return_value=result),
            patch.object(wildl_id, "organize_images", return_value=True) as organize,
        ):
            self.assertTrue(wildl_id.download_images())

        organize.assert_called_once_with()

    def test_download_action_skips_organizer_without_pending_images(self):
        result = SimpleNamespace(
            saved_images=0,
            duplicates=0,
            pending_organization=False,
            failures=(),
        )
        with (
            patch.object(wildl_id, "download_main", return_value=result),
            patch.object(wildl_id, "organize_images") as organize,
        ):
            self.assertTrue(wildl_id.download_images())

        organize.assert_not_called()

    def test_download_action_returns_failure_when_organizer_fails(self):
        result = SimpleNamespace(
            saved_images=1,
            duplicates=0,
            pending_organization=True,
            failures=(),
        )
        with (
            patch.object(wildl_id, "download_main", return_value=result),
            patch.object(wildl_id, "organize_images", return_value=False),
        ):
            self.assertFalse(wildl_id.download_images())

    def test_download_action_organizes_partial_success_and_returns_failure(self):
        result = SimpleNamespace(
            saved_images=1,
            duplicates=0,
            pending_organization=True,
            failures=("INBOX UID 4: transient failure",),
        )
        with (
            patch.object(wildl_id, "download_main", return_value=result),
            patch.object(wildl_id, "organize_images", return_value=True) as organize,
        ):
            self.assertFalse(wildl_id.download_images())

        organize.assert_called_once_with()

    def test_download_cli_calls_download_action(self):
        with (
            patch.object(sys, "argv", ["wildl_id.py", "--download"]),
            patch.object(wildl_id, "download_images", return_value=True) as download,
        ):
            self.assertEqual(wildl_id.main(), 0)

        download.assert_called_once_with(organize=True)

    def test_download_cli_returns_nonzero_on_download_failure(self):
        with (
            patch.object(sys, "argv", ["wildl_id.py", "--download"]),
            patch.object(wildl_id, "download_images", return_value=False),
        ):
            self.assertEqual(wildl_id.main(), 1)

    def test_download_and_notify_cli_runs_alerts_after_download(self):
        calls = []
        with (
            patch.object(sys, "argv", ["wildl_id.py", "--download", "--notify"]),
            patch.object(
                wildl_id,
                "download_images",
                side_effect=lambda **kwargs: calls.append("download") or True,
            ),
            patch.object(
                wildl_id,
                "label_new_images",
                side_effect=lambda **kwargs: calls.append("label") or True,
            ),
            patch.object(
                wildl_id,
                "send_alerts",
                side_effect=lambda: calls.append("notify") or True,
            ),
        ):
            self.assertEqual(wildl_id.main(), 0)

        self.assertEqual(calls, ["download", "label", "notify"])

    def test_send_alerts_skips_when_not_configured(self):
        result = SimpleNamespace(enabled=False, sent=0, failures=())
        with (
            patch.object(wildl_id, "alerts_main", return_value=result),
            redirect_stdout(StringIO()),
        ):
            self.assertTrue(wildl_id.send_alerts())

    def test_send_alerts_returns_failure_on_alert_errors(self):
        result = SimpleNamespace(enabled=True, sent=1, failures=("offline",))
        with (
            patch.object(wildl_id, "alerts_main", return_value=result),
            redirect_stdout(StringIO()),
        ):
            self.assertFalse(wildl_id.send_alerts())

    def test_download_and_organize_cli_only_runs_organizer_once(self):
        with (
            patch.object(sys, "argv", ["wildl_id.py", "--download", "--organize"]),
            patch.object(wildl_id, "download_images", return_value=True) as download,
            patch.object(wildl_id, "organize_images", return_value=True) as organize,
        ):
            self.assertEqual(wildl_id.main(), 0)

        download.assert_called_once_with(organize=False)
        organize.assert_called_once_with()

    def test_download_and_all_cli_only_runs_organizer_once(self):
        with (
            patch.object(sys, "argv", ["wildl_id.py", "--download", "--all"]),
            patch.object(wildl_id, "download_images", return_value=True) as download,
            patch.object(wildl_id, "run_all", return_value=True) as run_all,
            patch.object(wildl_id, "organize_images") as organize,
        ):
            self.assertEqual(wildl_id.main(), 0)

        download.assert_called_once_with(organize=False)
        run_all.assert_called_once_with(
            camera="seissiger", reprocess_missing=False, force=False
        )
        organize.assert_not_called()

    def test_menu_download_choice_runs_action_and_six_exits(self):
        output = StringIO()
        with (
            patch.object(sys, "argv", ["wildl_id.py"]),
            patch("builtins.input", side_effect=["5", "6"]),
            patch.object(wildl_id, "download_images", return_value=True) as download,
            redirect_stdout(output),
        ):
            self.assertEqual(wildl_id.main(), 0)

        download.assert_called_once_with()
        self.assertIn("5. Download new images from email", output.getvalue())

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


class IncrementalMailLabellingTests(unittest.TestCase):
    def test_only_new_unclassified_photos_are_written_to_active_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "labels.csv"
            with csv_path.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["location_id", "image_file", "class", "manual_label"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "location_id": "camera1",
                            "image_file": "classified.jpg",
                            "class": "fox",
                        },
                        {
                            "location_id": "camera1",
                            "image_file": "reviewed.jpg",
                            "class": "",
                            "manual_label": "badger",
                        },
                        {
                            "location_id": "camera1",
                            "image_file": "unknown.jpg",
                            "class": "unknown",
                        },
                    ]
                )
            before = csv_path.read_bytes()
            paths = {
                ("camera1", name): Path(temporary) / name
                for name in ("classified.jpg", "reviewed.jpg", "unknown.jpg", "new.jpg")
            }
            with (
                patch.object(wildl_id, "DEFAULT_CSV", csv_path, create=True),
                patch.object(wildl_id, "image_paths", return_value=paths, create=True),
                patch.object(
                    wildl_id, "process_images_with_pytorch_wildlife"
                ) as process,
            ):
                self.assertTrue(callable(getattr(wildl_id, "label_new_images", None)))
                self.assertTrue(wildl_id.label_new_images(camera="seissiger"))
            process.assert_called_once_with(
                camera="seissiger",
                output_csv_path=csv_path,
                image_keys={("camera1", "unknown.jpg"), ("camera1", "new.jpg")},
            )
            self.assertEqual(csv_path.read_bytes(), before)

    def test_empty_scan_does_not_load_models(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(
                wildl_id, "DEFAULT_CSV", Path(temporary) / "absent.csv", create=True
            ),
            patch.object(wildl_id, "image_paths", return_value={}, create=True),
            patch.object(wildl_id, "process_images_with_pytorch_wildlife") as process,
        ):
            self.assertTrue(callable(getattr(wildl_id, "label_new_images", None)))
            self.assertTrue(wildl_id.label_new_images())
            process.assert_not_called()

    def test_model_failure_returns_failure_without_modifying_labels(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(
                wildl_id, "DEFAULT_CSV", Path(temporary) / "absent.csv", create=True
            ),
            patch.object(
                wildl_id,
                "image_paths",
                return_value={("camera1", "new.jpg"): Path(temporary) / "new.jpg"},
                create=True,
            ),
            patch.object(
                wildl_id,
                "process_images_with_pytorch_wildlife",
                side_effect=RuntimeError("fixture failure"),
            ),
        ):
            self.assertTrue(callable(getattr(wildl_id, "label_new_images", None)))
            self.assertFalse(wildl_id.label_new_images())


if __name__ == "__main__":
    unittest.main()
