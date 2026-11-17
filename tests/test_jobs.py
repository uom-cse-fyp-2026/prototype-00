"""VM-side control-plane tests; fake tmux and signals, no compiler downloads."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import jobs


class JobControlTests(unittest.TestCase):
    def test_rejects_path_traversal(self):
        for job in ("../other", "/tmp/other", "not-a-job"):
            with self.assertRaises(ValueError):
                jobs.read_state(job)

    def test_guard_blocks_other_command_but_allows_worker(self):
        state = {"id": "owned-job", "task": "setup"}
        with mock.patch.object(jobs, "read_state", return_value=(None, state)), \
                mock.patch.object(jobs, "is_running", return_value=True), \
                mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                jobs.guard()
            os.environ["NM_JOB_ID"] = "owned-job"
            jobs.guard()

    def test_reused_worker_pid_is_not_signalled(self):
        state = {"status": "running", "worker_pid": 123, "worker_start_ticks": "old", "session": "owned"}
        with mock.patch.object(jobs, "read_state", return_value=(None, state)), \
                mock.patch.object(jobs, "process_identity", return_value="new"), \
                mock.patch.object(jobs, "session_exists", return_value=False), \
                mock.patch.object(jobs.os, "kill") as kill, \
                contextlib.redirect_stdout(io.StringIO()):
            jobs.stop()
            kill.assert_not_called()

    def test_orphaned_child_keeps_job_active(self):
        state = {"status": "running", "worker_pid": 123, "worker_start_ticks": "worker",
                 "child_pid": 124, "child_start_ticks": "child", "session": "owned"}
        with mock.patch.object(jobs, "process_identity", side_effect=lambda pid: "child" if pid == 124 else None):
            self.assertTrue(jobs.is_running(state))

    def test_new_job_preserves_arguments_without_shell_interpolation(self):
        with tempfile.TemporaryDirectory(prefix="nm job ") as tmp:
            root = Path(tmp)
            control = root / ".nm"
            directory = root / "results/jobs"
            with mock.patch.object(jobs, "ROOT", root), mock.patch.object(jobs, "CONTROL", control), \
                    mock.patch.object(jobs, "JOBS", directory), mock.patch.object(jobs, "LATEST", control / "latest-job.json"), \
                    mock.patch.object(jobs.shutil, "which", return_value="/usr/bin/tmux"), \
                    mock.patch.object(jobs, "tmux") as tmux, contextlib.redirect_stdout(io.StringIO()):
                args = ["--results", "two words; $(touch sentinel)"]
                jobs.start("all", args, True)
                _, state = jobs.read_state()
                self.assertEqual(state["arguments"], args)
                self.assertEqual(state["status"], "starting")
                self.assertFalse((root / "sentinel").exists())
                self.assertEqual(tmux.call_args.args[0], "new-session")

    def test_duplicate_start_does_not_launch_tmux(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(jobs, "CONTROL", root / ".nm"), mock.patch.object(jobs, "JOBS", root / "jobs"), \
                    mock.patch.object(jobs.shutil, "which", return_value="/usr/bin/tmux"), \
                    mock.patch.object(jobs, "guard", side_effect=RuntimeError("busy")), \
                    mock.patch.object(jobs, "tmux") as tmux:
                with self.assertRaises(RuntimeError):
                    jobs.start("all", [], True)
                tmux.assert_not_called()


if __name__ == "__main__":
    unittest.main()
