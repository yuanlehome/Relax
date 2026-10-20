# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Manage worker-local Apptainer rootfs leases and container-held locks."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import hmac
import os
import runpy
import shutil
import signal
import subprocess
import sys
import time
import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO


def _open_lock(path: Path) -> IO:
    return path.open("a+b")


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
        self.root = root.resolve()
        self.executable = executable
        for directory in (self.root, self.root / "images", self.root / "locks"):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        (self.root / "work").mkdir(mode=0o700, exist_ok=True)

    @staticmethod
    def key(reference: str) -> str:
        return hashlib.sha256(reference.encode()).hexdigest()

    def _ready(self, key: str) -> Path | None:
        path = self.root / "images" / f"{key}.rootfs"
        return path if path.is_dir() else None

    def _discard_if_idle(self, key: str) -> None:
        with _open_lock(self.root / "locks" / f"{key}.lock") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            path = self.root / "images" / f"{key}.rootfs"
            if path.exists():
                shutil.rmtree(path)

    def _sweep_idle(self) -> None:
        for path in (self.root / "images").glob("*.rootfs"):
            self._discard_if_idle(path.stem)
        for work in (self.root / "work").iterdir():
            with _open_lock(self.root / "locks" / f"{work.name}.build.lock") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                shutil.rmtree(work)

    async def _build(self, work: Path, reference: str, lock_fds: tuple[int, ...]) -> Path:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__)),
            "build",
            str(work),
            self.executable,
            reference,
            str(os.getpid()),
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
                    f"Rootfs builder failed ({process.returncode}): {output[-4096:].decode(errors='replace')}"
                )
        finally:
            cleanup = asyncio.create_task(stop())
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise
        return work / "rootfs"

    @asynccontextmanager
    async def acquire(self, reference: str) -> AsyncIterator[ImageLease]:
        key = self.key(reference)
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
                        shutil.rmtree(work)
                    incomplete = self.root / "images" / f"{key}.rootfs"
                    if incomplete.exists():
                        shutil.rmtree(incomplete)
                    self._sweep_idle()
                    work.mkdir(mode=0o700)
                    try:
                        generated = await self._build(work, reference, (build_lock.fileno(), use_lock.fileno()))
                        generated.chmod(0o555)
                        path = self.root / "images" / f"{key}.rootfs"
                        generated.replace(path)
                    finally:
                        shutil.rmtree(work)
                    # The image build lock keeps the rootfs published while EX is downgraded to SH.
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
                warnings.warn(f"Idle rootfs cleanup deferred: {error}", RuntimeWarning, stacklevel=2)


def build(work: Path, executable: str, reference: str, parent_pid: int) -> None:
    """Own one pull process and its locks until conversion cleanup
    completes."""
    if os.getppid() != parent_pid:
        raise RuntimeError("Rootfs builder lost its requesting adapter")
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
    command = [
        executable,
        "build",
        "--disable-cache",
        "--sandbox",
        str(work / "rootfs"),
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


def run_supervised(parent_pid: int, command: list[str]) -> None:
    """Own one sandbox process group until the adapter exits."""
    stopped = False

    def request_stop(_sig, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    process = subprocess.Popen(command, start_new_session=True)
    parent_exited = False
    try:
        while process.poll() is None:
            parent_exited = os.getppid() != parent_pid
            if stopped or parent_exited:
                break
            time.sleep(0.2)
    finally:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break
            time.sleep(0.2)
        returncode = process.wait()
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
        await self.app(scope, receive, send)


def serve() -> None:
    """Wrap Harbor's execution service while holding the active image lock."""
    directory = Path(__file__).parent
    sys.path.insert(0, "/run/agent-runtime/server-packages")
    namespace = runpy.run_path(str(directory / "_hbexec.py"))
    entrypoint = namespace["main"]
    entrypoint.__globals__["app"] = AuthenticatedApp(
        namespace["app"], (directory / "server-token").read_text().strip()
    )
    with Path("/run/harbor-image.lock").open("rb") as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        entrypoint()


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(Path(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]))
    elif sys.argv[1] == "run":
        run_supervised(int(sys.argv[2]), sys.argv[3:])
    else:
        del sys.argv[1]
        serve()
