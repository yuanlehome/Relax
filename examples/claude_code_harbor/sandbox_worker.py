# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Private stdio controller inside one persistent task container."""

from __future__ import annotations

import ctypes
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def execute(argv: list[str], *, log: str, timeout: float | None, **kwargs: Any) -> int:
    with Path(log).open("ab") as output:
        return subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT, timeout=timeout, **kwargs).returncode


def cleanup_descendants() -> None:
    """Reap only this controller's children, including adopted background
    jobs."""
    own = os.getpid()
    for sig in (signal.SIGTERM, signal.SIGKILL):
        parents: dict[int, int] = {}
        states: dict[int, str] = {}
        for path in Path("/proc").iterdir():
            if path.name.isdigit():
                try:
                    fields = (path / "stat").read_text().split(") ", 1)[1].split()
                    parents[int(path.name)] = int(fields[1])
                    states[int(path.name)] = fields[0]
                except (ProcessLookupError, FileNotFoundError):
                    pass
        descendants = {own}
        while True:
            expanded = descendants | {pid for pid, parent in parents.items() if parent in descendants}
            if expanded == descendants:
                break
            descendants = expanded
        for pid in descendants - {own}:
            try:
                if states[pid] != "Z":
                    os.kill(pid, sig)
            except ProcessLookupError:
                pass
        time.sleep(0.1)
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
            except ChildProcessError:
                break


def run_command(command: dict[str, Any]) -> dict[str, Any]:
    operation = command["operation"]
    if operation == "initialize":
        code = execute(
            ["/bin/bash", "/run/relax-control/install_task_runtime.sh"], log="/logs/agent/setup.log", timeout=900
        )
        if code:
            raise RuntimeError(f"Runtime installation exited {code}")
        os.environ["PATH"] = "/opt/srt-bins/bin:/usr/local/bin:" + os.environ.get("PATH", "")
        workdir = Path(command["workdir"])
        if not workdir.is_dir():
            raise RuntimeError("Task workdir is missing from its image")
        code = execute(
            [
                sys.executable,
                "/tmp/mimo_git_runtime.py",
                "prepare",
                "--config",
                "/tmp/mimo_git_targets.json",
                "--config-sha256",
                command["git_config_sha256"],
                "--state",
                "/opt/mimo-verifier-state",
            ],
            log="/logs/agent/setup.log",
            timeout=1800,
        )
        if code:
            raise RuntimeError(f"Required anti-leak initialization exited {code}")
        return {"initialized": True}
    if operation in ("probe", "agent"):
        environment = dict(os.environ)
        environment.update(command["env"])
        environment.pop("DISABLE_COMPACT", None)
        if operation == "probe":
            argv = ["/usr/local/bin/claude", "--version"]
            timeout: float | None = 120
            options: dict[str, Any] = {}
        else:
            argv = [
                "/usr/local/bin/claude",
                "--verbose",
                "--output-format=stream-json",
                "--settings",
                "/run/relax-control/settings.json",
                "--disallowedTools",
                command["disallowed_tools"],
                "--permission-mode=bypassPermissions",
                "--print",
            ]
            # Runtime owns the active-time deadline. A separate wall deadline
            # would also charge cross-step holds and weight synchronization.
            timeout = None
            options = {"input": command["instruction"].encode("utf-8")}
        try:
            if operation == "agent":
                with Path("/logs/agent/claude-code.txt").open("ab") as output:
                    # File-backed stdin avoids blocking the controller on a
                    # large instruction before it can observe cancellation.
                    with tempfile.TemporaryFile() as instruction:
                        instruction.write(options["input"])
                        instruction.seek(0)
                        child = subprocess.Popen(
                            argv,
                            stdin=instruction,
                            stdout=output,
                            stderr=output,
                            cwd=command["workdir"],
                            env=environment,
                        )
                        while child.poll() is None:
                            if Path("/run/relax-control/stop-agent").exists():
                                child.terminate()
                                break
                            time.sleep(0.2)
                        try:
                            code = child.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            code = child.wait(timeout=2)
            else:
                code = execute(
                    argv,
                    log="/logs/agent/claude-code.txt",
                    timeout=timeout,
                    cwd=command["workdir"],
                    env=environment,
                    **options,
                )
        finally:
            cleanup_descendants()
        return {"returncode": code}
    if operation == "verify":
        # The host publishes this directory only after the agent has exited.
        source = Path("/run/relax-verifier")
        destination = Path("/tests")
        # Follow source Harbor: discard agent-created verifier files and
        # symlinks before injecting the trusted tests and reward directory.
        for path in (destination, Path("/logs/verifier")):
            if path.is_symlink() or (path.exists() and not path.is_dir()):
                path.unlink()
            elif path.exists():
                shutil.rmtree(path)
            path.mkdir()
        for path in source.iterdir():
            if not path.is_file() or path.is_symlink():
                raise RuntimeError("Verifier staging requires regular files")
            shutil.copy2(path, destination / path.name)
        reward_file = Path("/logs/verifier/reward.txt")
        reward_file.unlink(missing_ok=True)
        timed_out = False
        try:
            code = execute(
                ["/bin/bash", "/tests/test.sh"],
                log="/logs/verifier/test-output.txt",
                timeout=float(command["timeout"]),
                cwd="/tests",
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            code = None
        finally:
            cleanup_descendants()
        if timed_out or not reward_file.is_file():
            raise RuntimeError("Verifier timed out or produced no reward")
        reward = float(reward_file.read_text().strip())
        if not math.isfinite(reward) or reward < 0:
            raise RuntimeError("Verifier infrastructure failure: invalid or negative reward")
        return {"reward": reward, "returncode": code}
    raise ValueError("Unknown private controller operation")


def main() -> None:
    # This file runs only inside an isolated --pid namespace. Refuse a direct
    # invocation on the host, where descendant cleanup would be unsafe.
    if os.environ.get("RELAX_HARBOR_PRIVATE_CONTAINER") != "1" or not Path("/run/relax-control").is_dir():
        raise RuntimeError("sandbox_worker must run inside the private task container")
    # Adopt detached tool jobs, so cleanup never needs to signal arbitrary PIDs.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot enable child subreaper")
    Path("/logs/agent").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    sys.stdout.write(json.dumps({"ok": True, "ready": True}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        try:
            result = {"ok": True, **run_command(json.loads(line))}
        except Exception as error:
            result = {"ok": False, "error_type": type(error).__name__, "error": str(error)[:1000]}
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
