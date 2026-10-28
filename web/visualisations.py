import asyncio
import shutil
import tempfile
from pathlib import Path

from nicegui import run
from src.visualisation import evaluate_labels

DEFAULT_OUTPUT_ROOT = Path("/tmp/wildl-id-visualisations")


class VisualisationRunStore:
    def __init__(self, output_root: Path = DEFAULT_OUTPUT_ROOT):
        self.output_root = Path(output_root)
        self.current_directory: Path | None = None
        self._lock = asyncio.Lock()
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    async def generate(self, csv_path: Path) -> Path | None:
        if self._lock.locked():
            raise RuntimeError("A visualisation run is already running")

        async with self._lock:
            self._is_running = True
            run_directory = None
            try:
                self.output_root.mkdir(parents=True, exist_ok=True)
                run_directory = Path(
                    tempfile.mkdtemp(prefix="run-", dir=self.output_root)
                )
                generated = await run.io_bound(
                    evaluate_labels.main,
                    csv_path=Path(csv_path),
                    output_dir=run_directory,
                )
                if not generated:
                    return None
                if not (run_directory / "summary_statistics.txt").is_file():
                    raise RuntimeError("Visualisation summary was not generated")
                if not list(run_directory.glob("*.svg")):
                    raise RuntimeError("No visualisation charts were generated")

                previous_directory = self.current_directory
                self.current_directory = run_directory
                run_directory = None
                if previous_directory is not None:
                    shutil.rmtree(previous_directory, ignore_errors=True)
                return self.current_directory
            finally:
                if run_directory is not None:
                    shutil.rmtree(run_directory, ignore_errors=True)
                self._is_running = False
