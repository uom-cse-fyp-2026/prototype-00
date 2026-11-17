"""VM-side control-plane tests; fake tmux and signals, no compiler downloads."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
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
                    mock.patch.object(jobs, "tmux") as tmux, \
                    mock.patch.dict(os.environ, {"JOBS": "16", "LINK_JOBS": "2"}), contextlib.redirect_stdout(io.StringIO()):
                args = ["--results", "two words; $(touch sentinel)"]
                jobs.start("all", args, True)
                _, state = jobs.read_state()
                self.assertEqual(state["arguments"], args)
                self.assertEqual(state["status"], "starting")
                self.assertEqual(state["build_environment"], {"JOBS": "16", "LINK_JOBS": "2"})
                self.assertFalse((root / "sentinel").exists())
                commands = [call.args[0] for call in tmux.call_args_list]
                self.assertEqual(commands, ["new-session", "set-option", "respawn-pane"])
                self.assertEqual(tmux.call_args_list[0].args[-1], "exec /bin/sleep 86400")

    def test_replacement_is_launched_before_old_session_is_killed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = {"id": "previous", "status": "succeeded", "session": "old-session"}
            with mock.patch.object(jobs, "ROOT", root), mock.patch.object(jobs, "CONTROL", root / ".nm"), \
                    mock.patch.object(jobs, "JOBS", root / "jobs"), mock.patch.object(jobs, "LATEST", root / ".nm/latest-job.json"), \
                    mock.patch.object(jobs.shutil, "which", return_value="/usr/bin/tmux"), \
                    mock.patch.object(jobs, "guard"), mock.patch.object(jobs, "read_state", return_value=(None, old)), \
                    mock.patch.object(jobs, "session_exists", return_value=True), \
                    mock.patch.object(jobs, "is_running", return_value=False), \
                    mock.patch.object(jobs, "tmux") as tmux, contextlib.redirect_stdout(io.StringIO()):
                jobs.start("all", [], True)
                commands = [call.args[0] for call in tmux.call_args_list]
                self.assertEqual(commands, ["new-session", "set-option", "respawn-pane", "kill-session"])
                self.assertEqual(tmux.call_args.args[-1], "=old-session")

    def test_dead_startup_pane_does_not_block_retry(self):
        with mock.patch.object(jobs, "session_exists", return_value=True), \
                mock.patch.object(jobs, "tmux", return_value=SimpleNamespace(returncode=0, stdout="1\n")):
            self.assertFalse(jobs.is_running({"status": "starting", "session": "owned"}))

    def test_launch_error_is_saved_and_old_session_is_not_killed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            error = jobs.subprocess.CalledProcessError(1, ["tmux"], stderr="server exited unexpectedly")
            with mock.patch.object(jobs, "ROOT", root), mock.patch.object(jobs, "CONTROL", root / ".nm"), \
                    mock.patch.object(jobs, "JOBS", root / "jobs"), mock.patch.object(jobs, "LATEST", root / ".nm/latest-job.json"), \
                    mock.patch.object(jobs.shutil, "which", return_value="/usr/bin/tmux"), \
                    mock.patch.object(jobs, "session_exists", return_value=False), \
                    mock.patch.object(jobs, "tmux", side_effect=error) as tmux:
                with self.assertRaisesRegex(RuntimeError, "server exited unexpectedly"):
                    jobs.start("all", [], True)
                directory, state = jobs.read_state()
                self.assertEqual(state["status"], "failed_to_start")
                self.assertEqual(state["exit_code"], 1)
                self.assertIn("server exited unexpectedly", (directory / "console.log").read_text())
                self.assertEqual([call.args[0] for call in tmux.call_args_list], ["new-session"])

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
