import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from web.visualisations import VisualisationRunStore


class VisualisationRunStoreTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.output_root = self.root / "runs"
        self.output_root.mkdir()
        self.csv_path = self.root / "labels.csv"
        self.csv_path.write_text("fixture", encoding="utf-8")
        self.store = VisualisationRunStore(self.output_root)
        self.previous_run = self.output_root / "previous"
        self.previous_run.mkdir()
        (self.previous_run / "summary_statistics.txt").write_text(
            "previous", encoding="utf-8"
        )
        self.store.current_directory = self.previous_run

    @staticmethod
    def write_results(output_dir):
        (output_dir / "summary_statistics.txt").write_text(
            "generated", encoding="utf-8"
        )
        (output_dir / "01_chart.svg").write_text("<svg/>", encoding="utf-8")

    async def test_success_replaces_previous_run(self):
        def generate(csv_path, output_dir):
            self.write_results(output_dir)
            return True

        with patch("web.visualisations.evaluate_labels.main", side_effect=generate):
            result = await self.store.generate(self.csv_path)

        self.assertIsNotNone(result)
        self.assertEqual(self.store.current_directory, result)
        self.assertTrue((result / "summary_statistics.txt").is_file())
        self.assertTrue((result / "01_chart.svg").is_file())
        self.assertFalse(self.previous_run.exists())
        self.assertFalse(self.store.is_running)

    async def test_false_result_removes_partial_output_and_keeps_previous(self):
        def generate(csv_path, output_dir):
            (output_dir / "partial.svg").write_text("partial", encoding="utf-8")
            return False

        with patch("web.visualisations.evaluate_labels.main", side_effect=generate):
            result = await self.store.generate(self.csv_path)

        self.assertIsNone(result)
        self.assertEqual(self.store.current_directory, self.previous_run)
        self.assertTrue(self.previous_run.exists())
        self.assertEqual(list(self.output_root.iterdir()), [self.previous_run])
        self.assertFalse(self.store.is_running)

    async def test_exception_removes_failed_output_and_keeps_previous(self):
        def generate(csv_path, output_dir):
            (output_dir / "partial.svg").write_text("partial", encoding="utf-8")
            raise RuntimeError("visualisation failed")

        with patch("web.visualisations.evaluate_labels.main", side_effect=generate):
            with self.assertRaisesRegex(RuntimeError, "visualisation failed"):
                await self.store.generate(self.csv_path)

        self.assertEqual(self.store.current_directory, self.previous_run)
        self.assertTrue(self.previous_run.exists())
        self.assertEqual(list(self.output_root.iterdir()), [self.previous_run])
        self.assertFalse(self.store.is_running)

    async def test_second_run_is_rejected_while_first_is_active(self):
        started = threading.Event()
        release = threading.Event()

        def generate(csv_path, output_dir):
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test did not release visualization worker")
            self.write_results(output_dir)
            return True

        with patch("web.visualisations.evaluate_labels.main", side_effect=generate):
            first_run = asyncio.create_task(self.store.generate(self.csv_path))
            self.assertTrue(await asyncio.to_thread(started.wait, 2))
            self.assertTrue(self.store.is_running)
            with self.assertRaisesRegex(RuntimeError, "already running"):
                await self.store.generate(self.csv_path)
            release.set()
            result = await first_run

        self.assertIsNotNone(result)
        self.assertFalse(self.store.is_running)

    async def test_run_directory_creation_failure_keeps_previous_run(self):
        with patch("web.visualisations.tempfile.mkdtemp", side_effect=OSError("full")):
            with self.assertRaisesRegex(OSError, "full"):
                await self.store.generate(self.csv_path)

        self.assertEqual(self.store.current_directory, self.previous_run)
        self.assertTrue(self.previous_run.exists())
        self.assertFalse(self.store.is_running)


if __name__ == "__main__":
    unittest.main()
