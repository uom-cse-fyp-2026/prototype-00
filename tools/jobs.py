#!/usr/bin/env python3
"""Project-specific tmux jobs. Uses Ubuntu's Python 3.8 even before setup."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTROL = ROOT / ".nm"
JOBS = ROOT / "results" / "jobs"
LATEST = CONTROL / "latest-job.json"
SOCKET = "neurominds-" + hashlib.sha256(str(ROOT).encode()).hexdigest()[:12]
ALLOWED = {"setup", "all", "sweep", "build", "test", "calibrate", "demo", "bench"}
ACTIVE = {"starting", "running", "stopping"}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(str(tmp), str(path))


def tmux(*args, check=True, capture=False):
    if not shutil.which("tmux"):
        raise RuntimeError("Install tmux on the VM: sudo apt-get install tmux")
    return subprocess.run(["tmux", "-L", SOCKET, "-f", "/dev/null", *map(str, args)], check=check,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None, text=True)


def session_exists(name):
    if not shutil.which("tmux"):
        return False
    return tmux("has-session", "-t", "=" + name, check=False, capture=True).returncode == 0


def read_state(job=None):
    if job is None:
        if not LATEST.exists():
            return None, None
        job = json.loads(LATEST.read_text())["id"]
    if not re.fullmatch(r"[0-9]{8}T[0-9]{12}Z-[a-z]+", job):
        raise ValueError("invalid job ID")
    directory = JOBS / job
    state = directory / "state.json"
    if not state.exists():
        raise RuntimeError("Job state is missing: " + str(state))
    data = json.loads(state.read_text())
    if data.get("id") != job or data.get("root") != str(ROOT) or data.get("socket") != SOCKET:
        raise RuntimeError("Job record belongs to a different checkout; refusing to control its processes.")
    return directory, data


def process_identity(pid):
    try:
        text = Path("/proc/%s/stat" % pid).read_text()
        # comm may contain spaces/parentheses; fields after its final ')' start at 3.
        return text[text.rfind(")") + 2:].split()[19]
    except (OSError, IndexError):
        return None


def child_alive(state):
    return bool(state.get("child_pid") and state.get("child_start_ticks") and
                process_identity(state["child_pid"]) == state["child_start_ticks"])


def worker_alive(state):
    return bool(state.get("worker_pid") and state.get("worker_start_ticks") and
                process_identity(state["worker_pid"]) == state["worker_start_ticks"])


def is_running(state):
    if state["status"] not in ACTIVE:
        return False
    if state.get("worker_pid"):
        return worker_alive(state) or child_alive(state)
    return session_exists(state["session"])


def guard():
    _, state = read_state()
    if state and is_running(state) and os.environ.get("NM_JOB_ID") != state["id"]:
        raise RuntimeError("Job %s is %s. Use just attach/status/stop before updating code or starting another task." %
                           (state["id"], state["task"]))


def start(task, arguments, detached):
    if task not in ALLOWED:
        raise ValueError("unsupported background task: " + task)
    if not shutil.which("tmux"):
        raise RuntimeError("Install tmux on the VM: sudo apt-get install tmux")
    CONTROL.mkdir(exist_ok=True)
    JOBS.mkdir(parents=True, exist_ok=True)
    with (CONTROL / "job-start.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        guard()
        old_directory, old_state = read_state()
        if old_state and session_exists(old_state["session"]) and not is_running(old_state):
            # The previous pane is this runner's completed session. Logs remain on disk.
            tmux("kill-session", "-t", "=" + old_state["session"], check=False, capture=True)
        job_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + task
        directory = JOBS / job_id
        directory.mkdir()
        state = {"id": job_id, "task": task, "arguments": arguments, "status": "starting",
                 "created_at_utc": now(), "session": "nm-" + job_id, "socket": SOCKET,
                 "root": str(ROOT), "log": str(directory / "console.log"), "exit_code": None}
        write_json(directory / "state.json", state)
        write_json(LATEST, {"id": job_id})
        worker = "exec " + " ".join(shlex.quote(str(arg)) for arg in
                                    (sys.executable, Path(__file__).resolve(), "worker", job_id))
        try:
            tmux("new-session", "-d", "-s", state["session"], "-n", "task", "-c", ROOT, worker)
        except subprocess.CalledProcessError:
            state.update(status="failed_to_start", ended_at_utc=now())
            write_json(directory / "state.json", state)
            raise
    print("Started %s: %s" % (job_id, task), flush=True)
    print("Log: %s\nDetach: Ctrl+B, then D. Reconnect: just attach. Status: just status." % state["log"], flush=True)
    if not detached:
        if os.environ.get("TMUX"):
            print("Already inside tmux. The new job is detached; use just status or attach from another SSH terminal.")
        elif sys.stdin.isatty() and sys.stdout.isatty():
            attach(job_id)
        else:
            print("No interactive terminal; job continues detached.")


def worker(job):
    directory, state = read_state(job)
    tmux("set-option", "-w", "-t", state["session"] + ":task", "remain-on-exit", "on")
    state.update(status="running", started_at_utc=now(), worker_pid=os.getpid(),
                 worker_start_ticks=process_identity(os.getpid()))
    write_json(directory / "state.json", state)
    command = ["bash", str(ROOT / "nm"), state["task"], *state["arguments"]]
    environment = os.environ.copy()
    environment["NM_JOB_ID"] = job
    environment["PYTHONUNBUFFERED"] = "1"
    child = None
    stopping = [None]

    def request_stop(signum, frame):
        if stopping[0] is None:
            stopping[0] = time.monotonic()
        if child and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, request_stop)
    try:
        with (directory / "console.log").open("ab", buffering=0) as log:
            header = ("[%s] %s\n" % (now(), " ".join(shlex.quote(str(x)) for x in command))).encode()
            log.write(header)
            sys.stdout.buffer.write(header)
            sys.stdout.buffer.flush()
            if stopping[0] is not None:
                raise InterruptedError("stopped before command started")
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, bufsize=0)
            if stopping[0] is not None:
                request_stop(signal.SIGTERM, None)
            state["child_pid"] = child.pid
            state["child_start_ticks"] = process_identity(child.pid)
            write_json(directory / "state.json", state)
            selector = selectors.DefaultSelector()
            selector.register(child.stdout, selectors.EVENT_READ)
            while True:
                if stopping[0] is not None and state["status"] != "stopping":
                    state["status"] = "stopping"
                    write_json(directory / "state.json", state)
                if stopping[0] is not None and child.poll() is None and time.monotonic() - stopping[0] > 10:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                events = selector.select(.5)
                eof = False
                for key, _ in events:
                    data = os.read(key.fd, 65536)
                    if not data:
                        eof = True
                        break
                    log.write(data)
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
                if eof:
                    break
                if child.poll() is not None and not events:
                    break
            selector.close()
            code = child.wait()
            state.update(status="stopped" if stopping[0] is not None else "succeeded" if code == 0 else "failed",
                         exit_code=code, ended_at_utc=now())
            footer = ("\n[%s] %s; exit status %s. Logs retained. Use just status.\n" % (now(), state["status"], code)).encode()
            log.write(footer)
            sys.stdout.buffer.write(footer)
            sys.stdout.buffer.flush()
    except BaseException as error:
        if child and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        state.update(status="stopped" if stopping[0] is not None else "failed", exit_code=130 if stopping[0] is not None else 1,
                     ended_at_utc=now(), error=str(error))
        with (directory / "console.log").open("a") as log:
            log.write("\nERROR: %s\n" % error)
        print("ERROR:", error, file=sys.stderr)
    finally:
        write_json(directory / "state.json", state)
    return state["exit_code"]


def attach(job=None):
    _, state = read_state(job)
    if not state:
        raise RuntimeError("No job yet. Run just setup or just run.")
    if not session_exists(state["session"]):
        raise RuntimeError("tmux session is gone. Logs remain available with just status and just logs.")
    if os.environ.get("TMUX"):
        raise RuntimeError("Detach from the current tmux client first (Ctrl+B, D), then run just attach.")
    tmux("attach-session", "-t", "=" + state["session"])


def status():
    directory, state = read_state()
    if not state:
        print("No project job has been started.")
        return
    label = state["status"]
    if label in ACTIVE and child_alive(state) and not worker_alive(state):
        label = "orphaned subprocess still active (use just stop)"
    elif label in ACTIVE and not is_running(state):
        label = "interrupted (worker/session is gone; not a successful completion)"
    print("Job: %s\nTask: %s\nStatus: %s\nExit status: %s\nLog: %s" %
          (state["id"], state["task"], label, state["exit_code"], state["log"]))
    log = directory / "console.log"
    if log.exists():
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 8192))
            print("\n" + "\n".join(stream.read().decode(errors="replace").splitlines()[-15:]))


def stop():
    _, state = read_state()
    if not state:
        raise RuntimeError("No project job exists.")
    if worker_alive(state) and state["status"] in ACTIVE:
        os.kill(state["worker_pid"], signal.SIGTERM)
        print("Stop requested. The worker terminates its child process group; check just status.")
    elif child_alive(state):
        os.killpg(state["child_pid"], signal.SIGTERM)
        deadline = time.monotonic() + 10
        while child_alive(state) and time.monotonic() < deadline:
            time.sleep(.2)
        if child_alive(state):
            os.killpg(state["child_pid"], signal.SIGKILL)
        directory, _ = read_state(state["id"])
        state.update(status="stopped", exit_code=143, ended_at_utc=now())
        write_json(directory / "state.json", state)
        print("Stopped the orphaned project's subprocess group. Logs retained.")
    elif state["status"] == "starting" and session_exists(state["session"]):
        tmux("send-keys", "-t", state["session"] + ":task", "C-c")
        print("Interrupted the starting job; check just status.")
    elif session_exists(state["session"]):
        tmux("kill-session", "-t", "=" + state["session"])
        print("Closed the completed project's tmux pane. Logs remain on disk.")
    else:
        print("No running project job or pane.")


def main():
    if sys.platform != "linux":
        raise RuntimeError("The job runner is intended for the Ubuntu VM only.")
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    start_parser = subs.add_parser("start")
    start_parser.add_argument("--detach", action="store_true")
    start_parser.add_argument("task", choices=sorted(ALLOWED))
    start_parser.add_argument("arguments", nargs=argparse.REMAINDER)
    for cmd in ("status", "stop", "logs", "guard", "list"):
        subs.add_parser(cmd)
    for cmd in ("attach", "worker"):
        part = subs.add_parser(cmd)
        part.add_argument("job", nargs="?" if cmd == "attach" else None)
    args = parser.parse_args()
    if args.command == "start":
        start(args.task, args.arguments, args.detach)
    elif args.command == "worker":
        return worker(args.job)
    elif args.command == "attach":
        attach(args.job)
    elif args.command == "status":
        status()
    elif args.command == "stop":
        stop()
    elif args.command == "guard":
        guard()
    elif args.command == "logs":
        directory, state = read_state()
        if not state:
            raise RuntimeError("No job logs yet.")
        run = subprocess.run(["tail", "-n", "50", "-f", str(directory / "console.log")])
        return run.returncode
    elif args.command == "list":
        for path in sorted(JOBS.glob("*/state.json"), reverse=True):
            state = json.loads(path.read_text())
            print("%s  %-10s  %s" % (state["id"], state["status"], state["task"]))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        sys.exit(1)
