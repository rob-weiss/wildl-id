import os
import tempfile
import unittest
from importlib import import_module
from pathlib import Path

job_control = import_module("src.background.job_control")


class JobControlTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.lock = self.root / "logs" / "mail-job.lock"
        self.pause = self.root / "logs" / "mail-job.paused"

    def test_second_run_cannot_acquire_lock_and_pid_is_recorded(self):
        with job_control.exclusive_run(self.lock) as first:
            self.assertTrue(first)
            self.assertEqual(job_control.running_pid(self.lock), os.getpid())
            with job_control.exclusive_run(self.lock) as second:
                self.assertFalse(second)
        self.assertIsNone(job_control.running_pid(self.lock))
        with job_control.exclusive_run(self.lock) as again:
            self.assertTrue(again)

    def test_stale_pid_is_not_reported_as_running(self):
        self.lock.parent.mkdir(parents=True)
        self.lock.write_text("999999999", encoding="utf-8")

        self.assertIsNone(job_control.running_pid(self.lock))

    def test_pause_flag_persists_until_cleared(self):
        self.assertFalse(job_control.is_paused(self.pause))
        job_control.set_paused(True, self.pause)
        self.assertTrue(job_control.is_paused(self.pause))
        job_control.set_paused(False, self.pause)
        self.assertFalse(job_control.is_paused(self.pause))


if __name__ == "__main__":
    unittest.main()
