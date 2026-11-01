import plistlib
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

from web import background

job_control = import_module("src.background.job_control")
installer = import_module("macos.launchd.install")

STDERR = (
    "Processing images:   7%|▋         | 7/100 [00:14<03:10,  2.05s/it]\r"
    "Processing images:  24%|██▍       | 24/100 [00:47<02:26,  1.93s/it]\r"
)
STDOUT = """
======================================================================
🦌  WILDLIFE CAMERA IMAGE ANALYSIS SYSTEM  🦌
======================================================================
Downloaded 1 images; skipped 0 duplicates
Sent 0 push alerts

======================================================================
🦌  WILDLIFE CAMERA IMAGE ANALYSIS SYSTEM  🦌
======================================================================
Downloaded 2 images; skipped 1 duplicates
Moving 2 images to images/ (image_0030251.jpg .. image_0030252.jpg)
Automatically labelling 100 of 17360 pending images (newest first) into x.csv
Found 100 total images in 5 locations, processing 100 images
"""


class BackgroundStatusTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.lock = self.root / "mail-job.lock"
        self.pause = self.root / "mail-job.paused"
        self.stdout = self.root / "stdout.log"
        self.stderr = self.root / "stderr.log"
        self.stdout.write_text(STDOUT, encoding="utf-8")
        self.stderr.write_text(STDERR, encoding="utf-8")
        self.plist = self.root / "agent.plist"
        self.plist.write_bytes(
            plistlib.dumps(
                {
                    "ProgramArguments": [
                        "/env/python",
                        "/repo/wildl_id.py",
                        "--download",
                    ],
                    "WorkingDirectory": str(self.root),
                    "StandardOutPath": str(self.stdout),
                    "StandardErrorPath": str(self.stderr),
                }
            )
        )

    def status(self):
        return background.mail_job_status(
            lock_path=self.lock,
            pause_path=self.pause,
            plist_path=self.plist,
        )

    def test_idle_status_lists_latest_run_events_without_progress(self):
        status = self.status()

        self.assertIsNone(status.running_pid)
        self.assertFalse(status.paused)
        self.assertIsNone(status.progress)
        self.assertEqual(
            status.events,
            [
                "Downloaded 2 images; skipped 1 duplicates",
                "Moving 2 images to images/ (image_0030251.jpg .. image_0030252.jpg)",
                "Automatically labelling 100 of 17360 pending images (newest first)",
            ],
        )
        self.assertIsNotNone(status.last_activity)

    def test_running_status_reports_labelling_progress(self):
        job_control.set_paused(True, self.pause)
        with job_control.exclusive_run(self.lock):
            status = self.status()

        self.assertIsNotNone(status.running_pid)
        self.assertTrue(status.paused)
        self.assertEqual(status.progress, (24, 100, "02:26"))

    def test_previous_runs_finished_bar_is_not_shown_while_loading(self):
        self.stderr.write_text(
            STDERR + "Processing images: 100%|███| 100/100 [03:10<00:00,  1.9s/it]\n",
            encoding="utf-8",
        )
        with job_control.exclusive_run(self.lock):
            self.assertIsNone(self.status().progress)

    def test_run_now_starts_installed_command_as_manual_run(self):
        calls = []

        def popen(arguments, **kwargs):
            calls.append((arguments, kwargs))
            return SimpleNamespace(poll=lambda: None)

        message = background.run_now(self.plist, self.lock, popen=popen)

        self.assertEqual(message, "Mail job started")
        arguments, kwargs = calls[0]
        self.assertEqual(
            arguments, ["/env/python", "/repo/wildl_id.py", "--download", "--manual"]
        )
        self.assertEqual(kwargs["cwd"], str(self.root))
        self.assertTrue(kwargs["start_new_session"])

    def test_run_now_refuses_while_job_runs(self):
        def popen(*args, **kwargs):
            raise AssertionError("must not start a second run")

        with job_control.exclusive_run(self.lock):
            message = background.run_now(self.plist, self.lock, popen=popen)

        self.assertEqual(message, "Mail job is already running")

    def test_agent_status_reads_launchd_state(self):
        output = "gui/501/x = {\n\tstate = running\n\truns = 3\n\tpid = 4242\n}\n"

        def runner(arguments, **kwargs):
            return SimpleNamespace(returncode=0, stdout=output)

        self.assertEqual(
            background.agent_status("x", runner=runner), "running (pid 4242)"
        )
        self.assertEqual(
            background.agent_status(
                "x", runner=lambda *a, **k: SimpleNamespace(returncode=113, stdout="")
            ),
            "not installed",
        )

        def missing(*args, **kwargs):
            raise FileNotFoundError("launchctl")

        self.assertEqual(background.agent_status("x", runner=missing), "unavailable")

    def test_agent_labels_match_installer(self):
        self.assertEqual(background.MAIL_AGENT, installer.LABEL)
        self.assertEqual(background.WEB_AGENT, installer.WEB_LABEL)


if __name__ == "__main__":
    unittest.main()
