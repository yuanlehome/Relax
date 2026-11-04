# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Manage worker-local Apptainer SIF leases and container-held locks."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import hmac
import json
import math
import os
import runpy
import shutil
import signal
import socket
import subprocess
import sys
import time
import warnings
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any


def _open_lock(path: Path) -> IO:
    return path.open("a+b")


def _compression_level() -> str:
    value = os.environ.get("HARBOR_SIF_COMPRESSION_LEVEL", "")
    if value not in ("", "1", "9"):
        raise ValueError("HARBOR_SIF_COMPRESSION_LEVEL must be unset, 1, or 9")
    return value


def _compression_algorithm(level: str) -> str:
    value = os.environ.get("HARBOR_SIF_COMPRESSION_ALGORITHM", "gzip")
    if value not in ("gzip", "zstd"):
        raise ValueError("HARBOR_SIF_COMPRESSION_ALGORITHM must be gzip or zstd")
    if value == "zstd" and level != "1":
        raise ValueError("SIF zstd compression requires HARBOR_SIF_COMPRESSION_LEVEL=1")
    return value


async def _lock(path: Path, mode: int) -> IO:
    handle = _open_lock(path)
    try:
        while True:
            try:
                fcntl.flock(handle, mode | fcntl.LOCK_NB)
                return handle
            except BlockingIOError:
                await asyncio.sleep(0.1)
    except BaseException:
        handle.close()
        raise


@dataclass
class ImageLease:
    path: Path

    @property
    def lock_path(self) -> Path:
        return self.path.parent.parent / "locks" / f"{self.path.stem}.lock"


class SifCache:
    def __init__(self, root: Path, executable: str = "apptainer") -> None:
        self.compression_level = _compression_level()
        self.compression_algorithm = _compression_algorithm(self.compression_level)
        self.root = root.resolve()
        self.executable = executable
        for directory in (self.root, self.root / "images", self.root / "locks"):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        (self.root / "work").mkdir(mode=0o700, exist_ok=True)

    @staticmethod
    def key(reference: str) -> str:
        return hashlib.sha256(reference.encode()).hexdigest()

    def _image_key(self, reference: str) -> str:
        if not self.compression_level:
            return self.key(reference)
        return self.key(f"apptainer-build-{self.compression_algorithm}{self.compression_level}\0{reference}")

    def _ready(self, key: str) -> Path | None:
        path = self.root / "images" / f"{key}.sif"
        return path if path.is_file() else None

    def _discard_if_idle(self, key: str) -> None:
        with _open_lock(self.root / "locks" / f"{key}.lock") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            path = self.root / "images" / f"{key}.sif"
            path.unlink(missing_ok=True)

    async def _remove_work(self, work: Path, lock_fds: tuple[int, ...]) -> None:
        """Keep large directory deletion off the agent's signal-handling loop.

        The helper stays in the Managed agent process group and inherits the
        original locks. Cancellation joins it before releasing those locks;
        partial directories stay unready and can be removed by the next owner.
        """
        if not lock_fds:
            raise ValueError("SIF work cleanup must retain its original lock")
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            str(Path(__file__)),
            "cleanup",
            str(work),
            pass_fds=lock_fds,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            return_code = await process.wait()
            if return_code:
                raise RuntimeError(f"SIF work cleanup failed ({return_code}): {work}")
        finally:
            if process.returncode is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
            cleanup = asyncio.create_task(process.wait())
            cancelled_during_join = False
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    cancelled_during_join = True
                    continue
            cleanup.result()
            if cancelled_during_join:
                raise asyncio.CancelledError

    async def _sweep_idle(self) -> None:
        for path in (self.root / "images").glob("*.sif"):
            self._discard_if_idle(path.stem)
        for work in (self.root / "work").iterdir():
            with _open_lock(self.root / "locks" / f"{work.name}.build.lock") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if not work.exists():
                    continue  # Another sweeper may have retired this old entry.
                await self._remove_work(work, (lock.fileno(),))

    async def _build(self, work: Path, reference: str, lock_fds: tuple[int, ...]) -> Path:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__)),
            "build",
            str(work),
            self.executable,
            reference,
            str(os.getpid()),
            env={
                **os.environ,
                "HARBOR_SIF_COMPRESSION_LEVEL": self.compression_level,
                "HARBOR_SIF_COMPRESSION_ALGORITHM": self.compression_algorithm,
            },
            pass_fds=lock_fds,
            start_new_session=True,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        async def stop() -> None:
            if process.returncode is None:
                process.terminate()
            await process.wait()

        try:
            output, _ = await process.communicate()
            if process.returncode:
                raise RuntimeError(
                    f"SIF builder failed ({process.returncode}): {output[-4096:].decode(errors='replace')}"
                )
        finally:
            cleanup = asyncio.create_task(stop())
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise
        return work / "image.sif"

    @asynccontextmanager
    async def acquire(self, reference: str) -> AsyncIterator[ImageLease]:
        key = self._image_key(reference)
        lock_path = self.root / "locks" / f"{key}.lock"
        use_lock = await _lock(lock_path, fcntl.LOCK_SH)
        build_lock = None
        try:
            path = self._ready(key)
            if path is None:
                use_lock.close()
                build_lock = await _lock(self.root / "locks" / f"{key}.build.lock", fcntl.LOCK_EX)
                use_lock = await _lock(lock_path, fcntl.LOCK_SH)
                path = self._ready(key)
                if path is None:
                    use_lock.close()
                    use_lock = await _lock(lock_path, fcntl.LOCK_EX)
                    work = self.root / "work" / key
                    if work.exists():
                        await self._remove_work(work, (build_lock.fileno(), use_lock.fileno()))
                    incomplete = self.root / "images" / f"{key}.sif"
                    incomplete.unlink(missing_ok=True)
                    await self._sweep_idle()
                    work.mkdir(mode=0o700)
                    cancelled = False
                    try:
                        generated = await self._build(work, reference, (build_lock.fileno(), use_lock.fileno()))
                        generated.chmod(0o444)
                        path = self.root / "images" / f"{key}.sif"
                        generated.replace(path)
                    except asyncio.CancelledError:
                        cancelled = True
                        raise
                    finally:
                        # The owned builder has been joined by _build. Disk GC
                        # must not consume the short TERM budget on this path.
                        if not cancelled:
                            await self._remove_work(work, (build_lock.fileno(), use_lock.fileno()))
                    # The image build lock keeps the SIF published while EX is downgraded to SH.
                    fcntl.flock(use_lock, fcntl.LOCK_SH)
                build_lock.close()
                build_lock = None
            yield ImageLease(path)
        finally:
            use_lock.close()
            if build_lock is not None:
                build_lock.close()
            try:
                self._discard_if_idle(key)
            except OSError as error:
                warnings.warn(f"Idle SIF cleanup deferred: {error}", RuntimeWarning, stacklevel=2)


def build(work: Path, executable: str, reference: str, parent_pid: int) -> None:
    """Own one image conversion process and its locks until cleanup
    completes."""
    if os.getppid() != parent_pid:
        raise RuntimeError("SIF builder lost its requesting adapter")
    stopped = False

    def request_stop(_sig, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    (work / "tmp").mkdir()
    env = {
        **os.environ,
        "APPTAINER_TMPDIR": str(work / "tmp"),
        "TMPDIR": str(work / "tmp"),
    }
    level = _compression_level()
    algorithm = _compression_algorithm(level)
    if level:
        command = [
            executable,
            "build",
            "--disable-cache",
            "--notest",
            "--mksquashfs-args",
            f"-comp {algorithm} -Xcompression-level {level}",
            str(work / "image.sif"),
            f"docker://{reference}",
        ]
    else:
        command = [
            executable,
            "pull",
            "--disable-cache",
            str(work / "image.sif"),
            f"docker://{reference}",
        ]
    with (work / "pull.log").open("wb") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=log, start_new_session=True)
        try:
            while process.poll() is None:
                if stopped or os.getppid() != parent_pid:
                    raise RuntimeError("Rootfs build cancelled or requesting adapter exited")
                time.sleep(0.5)
            if process.returncode != 0:
                with (work / "pull.log").open("rb") as diagnostic:
                    diagnostic.seek(max(0, diagnostic.seek(0, os.SEEK_END) - 4096))
                    tail = diagnostic.read().decode(errors="replace")
                raise RuntimeError(f"Apptainer build exited {process.returncode}: {tail}")
        finally:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    break
                time.sleep(0.2)
            process.wait()
            if os.getppid() != parent_pid:
                shutil.rmtree(work)


def _visible_cpus(cpu_limit: float, allowed: set[int], identity: str) -> set[int]:
    """Spread task affinity within the caller's mask; CPU quota still owns
    usage."""
    if not math.isfinite(cpu_limit) or cpu_limit <= 0 or not allowed:
        raise ValueError("Sandbox CPU limit and inherited affinity must be nonempty and positive")
    ordered = sorted(allowed)
    count = min(math.ceil(cpu_limit), len(ordered))
    offset = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big") % len(ordered)
    return {ordered[(offset + index) % len(ordered)] for index in range(count)}


def _enable_child_subreaper() -> None:
    # This process owns one sandbox. The shared launcher cannot reap children
    # that Apptainer moves into a different process group.
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _owned_child_pids() -> list[int]:
    # Only this single-threaded sandbox supervisor's immediate children. Bound
    # each pass; newly adopted descendants are handled by the next reap loop.
    with Path(f"/proc/self/task/{os.getpid()}/children").open() as source:
        value = source.read(4096)
    fields = value.split()
    if value and not value[-1].isspace():
        fields = fields[:-1]  # Never interpret a PID cut in half by the read limit.
    return [int(pid) for pid in fields[:64]]


def _kill_adopted_children(deadline: float) -> None:
    for pid in _owned_child_pids():
        if time.monotonic() >= deadline:
            return
        descriptor = None
        try:
            descriptor = os.pidfd_open(pid)
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[1]) != os.getpid():
                continue
            if time.monotonic() >= deadline:
                return
            # The original process group has already received TERM/KILL. A
            # setsid descendant can escape that group, but cannot escape this
            # dedicated subreaper. A pidfd prevents signalling a recycled PID.
            signal.pidfd_send_signal(descriptor, signal.SIGKILL)
        except (ProcessLookupError, FileNotFoundError):
            pass  # Still require waitpid/ECHILD; disappearance is not a proof.
        finally:
            if descriptor is not None:
                os.close(descriptor)


def _diagnose_adopted_children(last_wait_pid: int, elapsed: float) -> None:
    # Failure-only evidence; no commands, environment, task text or credentials.
    try:
        rows = []
        for pid in _owned_child_pids()[:4]:
            row: dict[str, Any] = {"pid": pid}
            try:
                fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                row.update(state=fields[0], ppid=int(fields[1]), starttime=fields[19])
                row["threads"] = []
                for index, task in enumerate(Path(f"/proc/{pid}/task").iterdir()):
                    if index >= 4:
                        break
                    state = (task / "stat").read_text().rsplit(")", 1)[1].split()[0]
                    row["threads"].append(
                        {"tid": int(task.name), "state": state, "wchan": (task / "wchan").read_text()[:48]}
                    )
            except OSError as error:
                row["errno"] = error.errno
            rows.append(row)
        data = {"last_wait_pid": last_wait_pid, "elapsed_s": elapsed, "owned_children": rows}
        os.write(2, ("[harbor-supervisor] adopted-child-drain=" + json.dumps(data) + "\n").encode())
    except Exception:
        pass  # Diagnostics cannot turn an unconfirmed cleanup into a proof.


def _join_adopted_children(timeout: float = 1.5) -> bool:
    """Call only after Popen.wait(), in the dedicated sandbox supervisor."""
    started = time.monotonic()
    deadline = started + timeout
    checked_at = started
    final_probe = False
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return True  # ECHILD, not WNOHANG=0, proves all children reaped.
        if final_probe:
            _diagnose_adopted_children(pid, checked_at - started)
            return False
        checked_at = time.monotonic()
        if checked_at >= deadline:
            # The preceding wait may have reaped the final child at the deadline.
            # Permit one ECHILD probe, never an unbounded drain of new children.
            final_probe = True
            continue
        if pid == 0:
            try:
                _kill_adopted_children(deadline)
            except Exception:
                _diagnose_adopted_children(pid, time.monotonic() - started)
                raise
            time.sleep(0.01)


def _publish_supervisor_cleanup() -> None:
    if os.environ.get("RELAX_REQUIRE_RESOURCE_RELEASE") != "1":
        return
    path = Path(os.environ["RELAX_RESOURCE_RELEASE_PATH"]).with_suffix(".supervisor.json")
    data = {
        "schema": 1,
        "session_id": os.environ["RELAX_SESSION_ID"],
        "nonce": os.environ["RELAX_RESOURCE_RELEASE_NONCE"],
        "attempt": os.environ["RELAX_RESOURCE_RELEASE_ATTEMPT"],
        "pid": os.getpid(),
        "children_reaped": True,
    }
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as target:
        os.chmod(temporary, 0o600)
        json.dump(data, target)
    temporary.replace(path)


def _supervisor_shutdown_stage(stage: str) -> None:
    # This helper is a dedicated process with file-backed stderr. These bounded
    # lifecycle records contain no commands, environment, task data or tokens.
    try:
        os.write(2, f"[harbor-supervisor] stage={stage} monotonic={time.monotonic():.6f}\n".encode())
    except OSError:
        pass  # Diagnostics must not prevent cleanup or change the proof.


def _supervisor_signal_result(pid: int, sig: int, outcome: str, error_number: int | None = None) -> None:
    # Lifecycle evidence only; syscall success does not prove delivery or reap.
    data = {
        "target_pgid": pid,
        "signal": int(sig),
        "outcome": outcome,
        "errno": error_number,
        "monotonic": time.monotonic(),
    }
    try:
        os.write(2, ("[harbor-supervisor] group-signal=" + json.dumps(data) + "\n").encode())
    except OSError:
        pass


def _resource_release_termination_profile() -> bool:
    """Private profile propagated by ManagedAgentProcess, also used by the
    adapter."""
    return (
        os.environ.get("RELAX_MANAGED_TERMINATION_PROFILE") == "resource_release_v1"
        and os.environ.get("RELAX_REQUIRE_RESOURCE_RELEASE") == "1"
    )


def run_supervised(parent_pid: int, command: list[str]) -> None:
    """Own one sandbox process group until the adapter exits."""
    stopped = False
    release_profile = _resource_release_termination_profile()

    def request_stop(_sig, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    _enable_child_subreaper()
    if "--cpus" in command:
        # This is the dedicated supervisor, never the Ray actor or agent process.
        # A quota alone leaves Node/Jest seeing all host CPUs and spawning hundreds of workers.
        cpu_limit = float(command[command.index("--cpus") + 1])
        identity = os.environ.get("RELAX_SESSION_ID", str(os.getpid()))
        os.sched_setaffinity(0, _visible_cpus(cpu_limit, os.sched_getaffinity(0), identity))
    process = subprocess.Popen(command, start_new_session=True)
    parent_exited = False
    try:
        while process.poll() is None:
            parent_exited = os.getppid() != parent_pid
            if stopped or parent_exited:
                break
            time.sleep(0.2)
    finally:
        _supervisor_shutdown_stage("terminate-child-group")

        def signal_owned_group(sig: int) -> None:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError as error:
                _supervisor_signal_result(process.pid, sig, "not_found", error.errno)
            except OSError as error:
                _supervisor_signal_result(process.pid, sig, "error", error.errno)
                raise
            else:
                _supervisor_signal_result(process.pid, sig, "syscall_returned")

        if process.returncode is None:
            signal_owned_group(signal.SIGTERM)
            try:
                # Apptainer performs host cgroup and mount cleanup after its
                # payload exits. The release profile reserves 11s in the adapter
                # and 15s in the launcher for this 5 + 3 + 1.5s inner cleanup.
                _supervisor_shutdown_stage("term-grace-direct-child")
                returncode = process.wait(timeout=5.0 if release_profile else 1.0)
            except subprocess.TimeoutExpired:
                signal_owned_group(signal.SIGKILL)
                _supervisor_shutdown_stage("wait-direct-child")
                if release_profile:
                    # A kernel-blocked task may remain after KILL. Do not hang
                    # the supervisor forever or publish a proof without wait().
                    try:
                        returncode = process.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        _supervisor_shutdown_stage("direct-child-reap-timeout")
                        # TimeoutExpired includes argv, which contains the
                        # private cgroup handshake nonce. Keep only the owned ID.
                        raise RuntimeError(f"Apptainer direct child {process.pid} was not reaped after KILL") from None
                else:
                    returncode = process.wait()
        else:
            # poll() already reaped this child; never signal a recycled PGID.
            returncode = process.returncode
        _supervisor_shutdown_stage("join-adopted-children")
        if not _join_adopted_children():
            raise RuntimeError("Apptainer supervisor still owns live descendants after cleanup")
        _supervisor_shutdown_stage("publish-proof")
        _publish_supervisor_cleanup()
        _supervisor_shutdown_stage("complete")
    raise SystemExit(143 if stopped or parent_exited else returncode)


class AuthenticatedApp:
    def __init__(self, app, token: str) -> None:
        self.app = app
        self.credential = f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            if not hmac.compare_digest(headers.get(b"authorization", b""), self.credential):
                await send({"type": "http.response.start", "status": 401, "headers": []})
                await send({"type": "http.response.body", "body": b"Unauthorized"})
                return
            if scope.get("path") == "/health" and scope.get("method") == "GET":

                async def send_health(message) -> None:
                    if message["type"] == "http.response.start" and message["status"] == 200:
                        identity = {
                            b"x-relax-exec-uid": str(os.getuid()).encode(),
                            b"x-relax-exec-euid": str(os.geteuid()).encode(),
                            b"x-relax-exec-gid": str(os.getgid()).encode(),
                            b"x-relax-exec-egid": str(os.getegid()).encode(),
                            b"x-relax-exec-groups": ",".join(map(str, os.getgroups())).encode(),
                        }
                        headers = [(k, v) for k, v in message.get("headers", []) if k.lower() not in identity]
                        message = {**message, "headers": [*headers, *identity.items()]}
                    await send(message)

                await self.app(scope, receive, send_health)
                return
        await self.app(scope, receive, send)


def _run_bound_server(run_server: Callable[..., None], port_file: Path, app: Any, **kwargs: Any) -> None:
    """Publish a listening socket without a close/rebind window during
    bootstrap."""
    address = (kwargs.get("host", "127.0.0.1"), kwargs.get("port", 0))
    with socket.create_server(address, backlog=kwargs.get("backlog", 2048)) as listener:
        pending = port_file.with_suffix(".tmp")
        pending.write_text(f"{listener.getsockname()[1]}\n")
        pending.replace(port_file)
        run_server(app, **{**kwargs, "fd": listener.fileno()})


def serve() -> None:
    """Wrap Harbor's execution service while holding the active image lock."""
    directory = Path(__file__).parent
    sys.path.insert(0, "/run/agent-runtime/server-packages")
    namespace = runpy.run_path(str(directory / "_hbexec.py"))
    entrypoint = namespace["main"]
    entrypoint.__globals__["app"] = AuthenticatedApp(
        namespace["app"], (directory / "server-token").read_text().strip()
    )
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--port-file", type=Path)
    args, remaining = parser.parse_known_args()
    sys.argv[1:] = remaining
    with Path("/run/harbor-image.lock").open("rb") as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        if args.port_file is None:
            entrypoint()
            return
        uvicorn = entrypoint.__globals__["uvicorn"]
        original_run = uvicorn.run

        def run_with_port_file(app: Any, **kwargs: Any) -> None:
            _run_bound_server(original_run, args.port_file, app, **kwargs)

        uvicorn.run = run_with_port_file
        try:
            entrypoint()
        finally:
            uvicorn.run = original_run


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(Path(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]))
    elif sys.argv[1] == "run":
        run_supervised(int(sys.argv[2]), sys.argv[3:])
    elif sys.argv[1] == "cleanup":
        work = Path(sys.argv[2])
        try:
            shutil.rmtree(work)
        except FileNotFoundError:
            if work.exists():
                raise
    else:
        del sys.argv[1]
        serve()
