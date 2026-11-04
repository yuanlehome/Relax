# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exact task-cgroup release proof for opt-in Session capacity reuse."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


def cgroup_v2_path(text: str) -> str:
    entries = [line[3:] for line in text.splitlines() if line.startswith("0::")]
    if len(entries) != 1:
        raise ValueError("Expected one cgroup v2 identity")
    return entries[0]


@dataclass(frozen=True)
class TaskCgroup:
    path: Path
    device: int
    inode: int

    @classmethod
    def capture(
        cls,
        raw: str,
        *,
        cpus: float,
        memory_bytes: int,
        pids: int,
        root: Path = Path("/sys/fs/cgroup"),
        own: str | None = None,
    ) -> "TaskCgroup":
        relative = PurePosixPath(raw)
        if not raw.startswith("/") or raw == "/" or ".." in relative.parts or str(relative) != raw:
            raise ValueError("Invalid task cgroup identity")
        own_path = PurePosixPath(own if own is not None else cgroup_v2_path(Path("/proc/self/cgroup").read_text()))
        if relative == own_path or relative in own_path.parents:
            raise ValueError("Task cgroup must not be the outer container or an ancestor")
        path = root / str(relative).lstrip("/")
        if path.resolve() != path or root.resolve() not in path.parents:
            raise ValueError("Task cgroup must be an exact directory under the cgroup mount")
        stat = path.stat()
        quota, period = (path / "cpu.max").read_text().split()
        if quota == "max" or int(period) <= 0 or abs(int(quota) / int(period) - cpus) > 1e-6:
            raise ValueError("Task cgroup CPU limit differs")
        limits = {"memory.max": memory_bytes, "memory.swap.max": 0, "pids.max": pids}
        if any((path / name).read_text().strip() != str(value) for name, value in limits.items()):
            raise ValueError("Task cgroup memory/pids limits differ")
        if not (path / "cgroup.procs").read_text().strip():
            raise ValueError("Task cgroup identity must be captured while populated")
        if (path.stat().st_dev, path.stat().st_ino) != (stat.st_dev, stat.st_ino):
            raise ValueError("Task cgroup changed during identity capture")
        return cls(path, stat.st_dev, stat.st_ino)

    def released(self) -> bool:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return True  # Only identities captured while live reach this method.
        if (stat.st_dev, stat.st_ino) != (self.device, self.inode):
            return False
        try:
            empty = (
                not (self.path / "cgroup.procs").read_text().strip()
                and (self.path / "pids.current").read_text().strip() == "0"
                and dict(line.split() for line in (self.path / "cgroup.events").read_text().splitlines()).get(
                    "populated"
                )
                == "0"
                and not any(child.is_dir() for child in self.path.iterdir())
            )
            stat_after = self.path.stat()
            return empty and (stat_after.st_dev, stat_after.st_ino) == (self.device, self.inode)
        except OSError:
            return False

    def diagnostic(self) -> dict:
        """Bounded failure evidence for this captured task, never a host
        scan."""
        result = {"path": str(self.path), "expected_device": self.device, "expected_inode": self.inode}
        try:
            stat = self.path.stat()
            result.update(device=stat.st_dev, inode=stat.st_ino)
            for name in ("cgroup.procs", "pids.current", "cgroup.events"):
                result[name] = (self.path / name).read_text()[:4096]
            result["children"] = [p.name for p in self.path.iterdir() if p.is_dir()][:16]
            processes = []
            for pid in result["cgroup.procs"].splitlines()[:16]:
                try:
                    fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
                    processes.append(
                        {
                            "pid": int(pid),
                            "comm": Path(f"/proc/{int(pid)}/comm").read_text().strip(),
                            "state": fields[0],
                            "ppid": fields[1],
                            "pgrp": fields[2],
                            "session": fields[3],
                            "starttime": fields[19],
                        }
                    )
                except OSError:
                    continue
            result["processes"] = processes
        except OSError as error:
            result["read_error"] = f"{type(error).__name__}: {error}"
        return result


class ResourceReleaseGuard:
    """Publish outside container mounts; never infer release from agent
    output."""

    def __init__(self) -> None:
        self.required = os.environ.get("RELAX_REQUIRE_RESOURCE_RELEASE") == "1"
        self.identity: TaskCgroup | None = None
        self.started = False
        self.attempt_closed = True
        self.supervisor_exit_confirmed = False
        self.supervisor_pid: int | None = None
        self.supervisor_cleanup: dict | None = None
        self.attempt_nonce: str | None = None
        self.finished = False
        if self.required:
            # One permanent owner only: a second environment must never replace
            # another live environment's unknown receipt with its own success.
            path = Path(os.environ["RELAX_RESOURCE_RELEASE_PATH"])
            try:
                with path.open("x"):
                    os.chmod(path, 0o600)
            except FileExistsError as error:
                raise RuntimeError("Session resource proof requires exactly one Environment owner") from error
            self._publish("unknown")

    def before_start(self) -> None:
        if not self.required:
            return
        if self.finished or not self.attempt_closed:
            raise RuntimeError("Previous Apptainer attempt has unconfirmed task resources; retry prohibited")
        self.identity = None
        self.started = True
        self.attempt_closed = False
        self.supervisor_exit_confirmed = False
        self.supervisor_pid = None
        self.supervisor_cleanup = None
        self.attempt_nonce = secrets.token_hex(24)
        self._publish("unknown")

    def supervisor_environment(self) -> dict[str, str] | None:
        if not self.required:
            return None
        if self.attempt_nonce is None:
            raise RuntimeError("Resource attempt must begin before spawning its supervisor")
        return {**os.environ, "RELAX_RESOURCE_RELEASE_ATTEMPT": self.attempt_nonce}

    def supervisor_reaped(self) -> bool:
        if self.supervisor_pid is None or self.attempt_nonce is None:
            return False
        try:
            path = Path(os.environ["RELAX_RESOURCE_RELEASE_PATH"]).with_suffix(".supervisor.json")
            return json.loads(path.read_text()) == {
                "schema": 1,
                "session_id": os.environ["RELAX_SESSION_ID"],
                "nonce": os.environ["RELAX_RESOURCE_RELEASE_NONCE"],
                "attempt": self.attempt_nonce,
                "pid": self.supervisor_pid,
                "children_reaped": True,
            }
        except (OSError, KeyError, ValueError):
            return False

    def capture(self, raw: str, **limits) -> None:
        if self.required:
            self.identity = TaskCgroup.capture(raw, **limits)

    def after_stop(self, *, supervisor_exited: bool) -> None:
        if self.required and self.started and not self.attempt_closed:
            self.supervisor_exit_confirmed |= supervisor_exited is True
            self.attempt_closed = bool(
                self.supervisor_exit_confirmed
                and self.identity is not None
                and self.supervisor_reaped()
                and self.identity.released()
            )
            if not self.attempt_closed:
                path = str(self.identity.path) if self.identity is not None else "unobserved"
                raise RuntimeError(
                    f"Apptainer task resource release remains unconfirmed: {path}; "
                    f"supervisor_exited={self.supervisor_exit_confirmed}"
                )

    async def wait_after_stop(self, *, supervisor_exited: bool) -> None:
        # Kernel cgroup removal can finish after the supervisor is reaped. Join
        # only the captured task identity, within the existing outer TERM grace.
        deadline = asyncio.get_running_loop().time() + 0.5
        while True:
            try:
                self.after_stop(supervisor_exited=supervisor_exited)
                return
            except RuntimeError:
                if (
                    not self.supervisor_exit_confirmed
                    or self.identity is None
                    or asyncio.get_running_loop().time() >= deadline
                ):
                    self._record_unknown(self.supervisor_exit_confirmed)
                    raise
                await asyncio.sleep(0.025)

    def _record_unknown(self, supervisor_exited: bool) -> None:
        try:
            path = Path(os.environ["RELAX_RESOURCE_RELEASE_PATH"]).with_suffix(".diagnostic.json")
            data = {
                "supervisor_exited": supervisor_exited,
                "started": self.started,
                "supervisor_pid": self.supervisor_pid,
                "children_reaped": self.supervisor_reaped(),
                "cgroup": self.identity.diagnostic() if self.identity is not None else None,
                "supervisor_process": dict(self.supervisor_cleanup) if self.supervisor_cleanup else None,
            }
            encoded = json.dumps(data, ensure_ascii=False)
            detail = data["supervisor_process"]
            # IPC reads at most 16 Ki characters. Escaped control characters in
            # a bounded log tail can otherwise exceed that limit by themselves.
            while len(encoded) > 16000 and detail and detail.get("server_log_tail"):
                tail = detail["server_log_tail"]
                detail["server_log_tail"] = tail[len(tail) // 2 + 1 :]
                detail["server_log_tail_truncated"] = True
                encoded = json.dumps(data, ensure_ascii=False)
            if len(encoded) > 16000:
                data["cgroup"] = {
                    "path": str(self.identity.path) if self.identity is not None else None,
                    "diagnostic_truncated": True,
                }
                encoded = json.dumps(data, ensure_ascii=False)
            path.write_text(encoded)
        except (OSError, KeyError):
            pass  # Diagnostics cannot turn an unknown release into success.

    def finish(self) -> None:
        if self.required:
            # Called only from permanent Environment.stop, never between retries.
            self.finished = True
            if not self.attempt_closed:
                self._record_unknown(self.supervisor_exit_confirmed)
            self._publish("released" if self.attempt_closed else "unknown")

    def _publish(self, status: str) -> None:
        path = Path(os.environ["RELAX_RESOURCE_RELEASE_PATH"])
        data = {
            "schema": 1,
            "session_id": os.environ["RELAX_SESSION_ID"],
            "nonce": os.environ["RELAX_RESOURCE_RELEASE_NONCE"],
            "status": status,
        }
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as target:
            os.chmod(temporary, 0o600)
            json.dump(data, target)
        temporary.replace(path)


def identify_and_exec(identity_path: Path, accepted_path: Path, nonce: str, command: list[str]) -> None:
    """Bind task identity before Git/tools run; join the host startup
    handshake."""
    temporary = identity_path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"nonce": nonce, "cgroup": cgroup_v2_path(Path("/proc/self/cgroup").read_text())}))
    temporary.replace(identity_path)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            if accepted_path.read_text() == nonce:
                os.execvp(command[0], command)
        except FileNotFoundError:
            pass
        time.sleep(0.05)
    raise RuntimeError("Host did not accept the task cgroup identity before startup deadline")


if __name__ == "__main__":
    identify_and_exec(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4:])
