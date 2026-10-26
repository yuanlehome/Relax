# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real Linux subprocess regressions for CLI completion with background
tools."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from examples.claude_code_harbor.cli_lifecycle import Process, TerminalObserver, same_process


def event(kind, **fields):
    return (json.dumps({"type": kind, "session_id": "main", **fields}) + "\n").encode()


def test_terminal_requires_main_session_and_resets_on_later_activity():
    observer = TerminalObserver()
    result = event("result", subtype="success", is_error=False, stop_reason="end_turn")
    observer.feed(result, 0)
    assert observer.candidate is None
    observer.feed(event("system", subtype="init"), 1)
    observer.feed(result[:20], 2)
    assert observer.candidate is None
    observer.feed(result[20:], 3)
    assert observer.candidate[1] == 3
    observer.feed(event("assistant"), 4)
    assert observer.candidate is None
    observer.feed(result.replace(b'"main"', b'"subagent"'), 5)
    assert observer.candidate is None
    observer.feed(result, 6)
    assert observer.candidate[1] == 6
    observer.feed(event("result", subtype="error", is_error=True, stop_reason="end_turn"), 7)
    assert observer.candidate is None


def test_oversized_or_partial_lines_do_not_create_terminal():
    observer = TerminalObserver()
    observer.feed(event("system", subtype="init"), 0)
    observer.feed(b"x" * (1024 * 1024 + 1), 1)
    observer.feed(event("result", subtype="success", is_error=False, stop_reason="end_turn"), 2)
    assert observer.candidate is None and not observer.buffer
    observer.feed(event("result", subtype="success", is_error=False, stop_reason="end_turn"), 3)
    assert observer.candidate[1] == 3
    observer.feed(b'{"type":"assistant",', 4)
    assert observer.candidate is None


def test_reused_pid_or_changed_executable_is_not_owned():
    record = Process(11, 1, "original", (1, 2))
    assert same_process(record, {11: Process(11, 1, "original", (1, 2))})
    assert not same_process(record, {11: Process(11, 1, "new", (1, 2))})
    assert not same_process(record, {11: Process(11, 1, "original", (1, 3))})


@pytest.mark.skipif(sys.platform != "linux" or not hasattr(os, "pidfd_open"), reason="Linux pidfds required")
@pytest.mark.parametrize(
    "mode", ["hung", "nonzero", "continued", "stderr", "no_permit", "pipe_orphan", "resume_during_cleanup"]
)
def test_real_cli_and_bash_lifecycle(tmp_path: Path, mode: str):
    # Run the subreaper in its own process: cleanup cannot affect pytest's
    # other fixtures or subprocesses. The fake CLI owns an actual Bash tool.
    source = tmp_path / "cli.py"
    source.write_text("""
import json, os, subprocess, sys, time
from pathlib import Path
mode, root = sys.argv[1:]
def emit(kind, **fields):
    print(json.dumps(dict(type=kind, session_id="main", **fields)), flush=True)
tool = subprocess.Popen(["/bin/bash", "-c", 'trap "" TERM; sleep 60 & wait; echo unsafe > "$1"', "bash", root + "/unsafe"])
emit("system", subtype="init")
time.sleep(.6)
if mode == "pipe_orphan":
    sys.exit(0)
result = dict(type="result", session_id="main", subtype="success", is_error=False, stop_reason="end_turn")
print(json.dumps(result), file=sys.stderr if mode == "stderr" else sys.stdout, flush=True)
if mode == "resume_during_cleanup":
    while not Path(root + "/draining").exists():
        time.sleep(.02)
    emit("assistant")
    time.sleep(.4)
    Path(root + "/stop-agent").touch()
if mode == "continued":
    time.sleep(.15)
    emit("assistant")
    time.sleep(.7)
    assert tool.poll() is None
    Path(root + "/continued").touch()
    print(json.dumps(result), flush=True)
tool.wait()
sys.exit(7 if mode == "nonzero" else 0)
""")
    harness = """
import ctypes, json, subprocess, sys, threading, time
from pathlib import Path
from examples.claude_code_harbor.cli_lifecycle import monitor_cli
from examples.claude_code_harbor import cli_lifecycle
from examples.claude_code_harbor.sandbox_worker import cleanup_descendants
ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)
source, mode, directory = sys.argv[1:]
root = Path(directory)
events = []
signals = []
original_signal = cli_lifecycle.send_signal
def record_signal(record, sig):
    signals.append([record.pid, sig])
    original_signal(record, sig)
cli_lifecycle.send_signal = record_signal
def progress(event):
    events.append(event)
    if event["phase"] == "cleanup_started":
        (root / "draining").touch()
    if event["phase"] == "cleanup_ready" and mode != "no_permit":
        (root / "tool-cleanup-permit.json").write_text(json.dumps(dict(generation=event["generation"], deadline=time.monotonic()+10)))
if mode in ("stderr", "no_permit"):
    timer = threading.Timer(2.2, lambda: (root / "stop-agent").touch())
    timer.start()
started = time.monotonic()
child = subprocess.Popen([sys.executable, source, mode, directory], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
try:
    with (root / "output").open("wb") as output:
        code, evidence = monitor_cli(child, output, cli_executable=Path(sys.executable), control=root, progress=progress, grace_s=.4)
finally:
    cleanup_descendants()
print(json.dumps(dict(code=code, elapsed=time.monotonic()-started, evidence=evidence, events=events, signals=signals)))
"""
    result = subprocess.run(
        [sys.executable, "-c", harness, str(source), mode, str(tmp_path)], capture_output=True, text=True, timeout=12
    )
    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout)
    assert evidence["elapsed"] < 10
    assert not (tmp_path / "unsafe").exists()
    if mode in {"hung", "nonzero", "continued"}:
        assert evidence["evidence"]["tool_cleanup_attempts"] == 1
        assert evidence["code"] == (7 if mode == "nonzero" else 0)
    elif mode != "resume_during_cleanup":
        assert evidence["evidence"]["tool_cleanup_attempts"] == 0
    if mode == "continued":
        assert (tmp_path / "continued").exists()
        assert any(row["phase"] == "candidate_cancelled" for row in evidence["events"])
    if mode == "pipe_orphan":
        assert evidence["code"] == 0 and evidence["elapsed"] < 3
    if mode == "resume_during_cleanup":
        assert evidence["evidence"]["tool_cleanup_attempts"] == 1
        assert len(evidence["signals"]) == 1 and evidence["signals"][0][1] == 15
        assert any(row.get("result") == "cli_activity_resumed" for row in evidence["events"])


@pytest.mark.skipif(sys.platform != "linux" or not hasattr(os, "pidfd_open"), reason="Linux pidfds required")
@pytest.mark.parametrize("stream_name", ["stdout", "stderr"])
def test_cli_survives_competing_pipe_reader(tmp_path: Path, stream_name: str) -> None:
    # A tool may open its inherited stdout pipe for reading through /proc.
    # Deplete that pipe after readiness, then require the real CLI to finish.
    source = tmp_path / "cli.py"
    source.write_text("""
import json, sys, time
from pathlib import Path
root, stream_name = sys.argv[1:]
print("competing-reader", file=getattr(sys, stream_name), flush=True)
while not (Path(root) / "release").exists():
    time.sleep(.01)
print(json.dumps(dict(type="system", subtype="init", session_id="main")), flush=True)
print(json.dumps(dict(type="result", session_id="main", subtype="success", is_error=False,
                     stop_reason="end_turn")), flush=True)
print("stderr survived", file=sys.stderr, flush=True)
time.sleep(.2)
""")
    harness = """
import json, os, subprocess, sys
from pathlib import Path
from examples.claude_code_harbor import cli_lifecycle
source, directory, stream_name = sys.argv[1:]
root = Path(directory)
child = subprocess.Popen([sys.executable, source, directory, stream_name],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
watched = getattr(child, stream_name).fileno()
child_fd = 1 if stream_name == "stdout" else 2
other = os.open(f"/proc/{child.pid}/fd/{child_fd}", os.O_RDONLY | os.O_NONBLOCK)
original_read = os.read
raced = False
stolen = b""
events = []
def competing_read(fd, size):
    global raced, stolen
    if fd == watched and not raced:
        raced = True
        stolen = original_read(other, size)
        try:
            return original_read(fd, size)
        except BlockingIOError:
            (root / "release").touch()
            raise
    return original_read(fd, size)
cli_lifecycle.os.read = competing_read
try:
    with (root / "output").open("wb") as output:
        code, evidence = cli_lifecycle.monitor_cli(child, output, cli_executable=Path(sys.executable),
                                                  control=root, progress=events.append)
finally:
    cli_lifecycle.os.read = original_read
    os.close(other)
    if child.poll() is None:
        child.kill()
    child.wait(timeout=2)
print(json.dumps(dict(code=code, evidence=evidence, events=events, raced=raced, stolen=stolen.decode())))
"""
    result = subprocess.run(
        [sys.executable, "-c", harness, str(source), str(tmp_path), stream_name],
        capture_output=True,
        text=True,
        timeout=6,
    )
    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout)
    assert evidence["raced"] and evidence["stolen"] == "competing-reader\n"
    assert evidence["code"] == 0
    assert evidence["evidence"]["cli_result"]["subtype"] == "success"
    assert evidence["evidence"]["tool_cleanup_attempts"] == 0
    assert evidence["events"][-1]["phase"] == "cli_exited"
    assert b"stderr survived" in (tmp_path / "output").read_bytes()
