# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Observe CLI completion and drain only its registered Bash tool branches.

Runs inside the private task PID namespace and preserves the actual exit
status. Success is a cleanup candidate; a validated turn limit also notifies
the host so it can fence model requests before requesting a controlled stop.
"""

from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable
from uuid import uuid4


@dataclass
class Process:
    pid: int
    parent: int
    start: str
    executable: tuple[int, int]
    fd: int = -1
    command: tuple[str, ...] = ()


def process_table() -> dict[int, Process]:
    result = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] == "Z":
                continue
            exe = (path / "exe").stat()
            command = tuple(os.fsdecode(part) for part in (path / "cmdline").read_bytes().split(b"\0") if part)
            result[int(path.name)] = Process(
                int(path.name), int(fields[1]), fields[19], (exe.st_dev, exe.st_ino), command=command
            )
        except (OSError, ValueError, IndexError):
            continue
    return result


def descendants(root: int, table: dict[int, Process]) -> set[int]:
    owned = {root}
    while True:
        expanded = owned | {p.pid for p in table.values() if p.parent in owned}
        if expanded == owned:
            return owned
        owned = expanded


def same_process(record: Process, table: dict[int, Process]) -> bool:
    current = table.get(record.pid)
    return current is not None and (current.start, current.executable) == (record.start, record.executable)


def pin_process(record: Process) -> Process | None:
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        return None
    try:
        fd = os.pidfd_open(record.pid)
        if not same_process(record, process_table()):
            os.close(fd)
            return None
        record.fd = fd
        return record
    except OSError:
        return None


def send_signal(record: Process, sig: int) -> bool:
    try:
        signal.pidfd_send_signal(record.fd, sig)
        return True
    except ProcessLookupError:
        return False


def is_expected_turn_limit(result: Any, max_turns: int | None) -> bool:
    """Accept only the configured native CLI boundary, with strict JSON
    types."""
    if not isinstance(result, dict) or not isinstance(max_turns, int) or isinstance(max_turns, bool):
        return False
    turns = result.get("num_turns")
    return (
        max_turns > 0
        and result.get("subtype") == "error_max_turns"
        and result.get("is_error") is True
        and isinstance(turns, int)
        and not isinstance(turns, bool)
        and turns in {max_turns, max_turns + 1}
    )


class TerminalObserver:
    """Parse bounded stdout lines, never the agent-writable log or stderr."""

    def __init__(self, max_turns: int | None = None) -> None:
        self.buffer = bytearray()
        self.discard_line = False
        self.session: str | None = None
        self.candidate: tuple[str, float] | None = None
        self.result: dict[str, Any] | None = None
        self.max_turns = max_turns
        self.first_limit_result: dict[str, Any] | None = None

    def feed(self, data: bytes, now: float) -> bool:
        previous = self.candidate
        pieces = data.split(b"\n")
        for index, piece in enumerate(pieces):
            complete = index < len(pieces) - 1
            if piece.strip():
                self.candidate = None
                self.result = None
            if not self.discard_line:
                self.buffer.extend(piece)
                if len(self.buffer) > 1024 * 1024:
                    self.buffer.clear()
                    self.discard_line = True
            if not complete:
                continue
            if not self.discard_line:
                try:
                    event = json.loads(self.buffer)
                except (ValueError, UnicodeError):
                    event = None
                if isinstance(event, dict):
                    if event.get("type") == "system" and event.get("subtype") == "init":
                        self.session = event.get("session_id") if isinstance(event.get("session_id"), str) else None
                        self.candidate = None
                    elif self.session and event.get("session_id") == self.session and event.get("type") == "result":
                        # Only direct main-CLI stdout can authorize an expected
                        # limit exit. Never infer it from the writable log.
                        self.result = {
                            key: event.get(key) for key in ("subtype", "is_error", "stop_reason", "num_turns")
                        }
                        if self.first_limit_result is None and is_expected_turn_limit(self.result, self.max_turns):
                            # The final result remains revocable; reaching the
                            # configured limit is a separate, immutable fact.
                            self.first_limit_result = dict(self.result)
                        if (
                            event.get("subtype") == "success"
                            and event.get("is_error") is False
                            and event.get("stop_reason") == "end_turn"
                        ) or (event.get("subtype") == "error_max_turns" and event.get("is_error") is True):
                            self.candidate = (uuid4().hex, now)
                    elif event.get("type") in ("assistant", "user", "system", "result"):
                        self.candidate = None
            self.buffer.clear()
            self.discard_line = False
        return previous != self.candidate


def monitor_cli(
    child: subprocess.Popen[bytes],
    output: BinaryIO,
    *,
    cli_executable: Path,
    control: Path,
    progress: Callable[[dict[str, Any]], None],
    grace_s: float = 30.0,
    max_turns: int | None = None,
) -> tuple[int, dict[str, Any]]:
    """Drain streams, negotiate a short relay pause, then stop known tools.

    A host-written, generation-specific permit is required for every signal.
    Its lease holds *new* model requests; already accepted requests must be
    finished before the host grants it. No capture sealing or fake exit code.
    """
    assert child.stdout is not None and child.stderr is not None
    exe = cli_executable.stat()
    cli_inode = (exe.st_dev, exe.st_ino)
    bash = Path("/bin/bash").stat()
    bash_inode = (bash.st_dev, bash.st_ino)
    observer = TerminalObserver(max_turns=max_turns)
    limit_reported = False
    host_stop_signal: int | None = None
    cli: Process | None = None
    registered: dict[int, Process] = {}
    roots: dict[int, Process] = {}
    ready: str | None = None
    draining: dict[str, Any] | None = None
    cleanup_count = 0
    cleanup_attempts = 0
    protected = {cli_inode}
    # The pinned SRT is a Node script; its bwrap/socat launchers exec the
    # musl loader. Match those exact argv paths as well as native inodes.
    protected_commands = {str(Path("/usr/local/bin/srt").resolve())}
    protected_commands.update(f"/opt/srt-bins/real/{name}" for name in ("bwrap", "socat"))
    for name in ("/usr/local/bin/srt", "/usr/bin/bwrap", "/usr/bin/socat"):
        try:
            info = Path(name).stat()
            protected.add((info.st_dev, info.st_ino))
        except OSError:
            pass
    next_scan = 0.0
    selector = selectors.DefaultSelector()
    for stream, kind in ((child.stdout, "stdout"), (child.stderr, "stderr")):
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, kind)

    def emit(phase: str, generation: str | None, **fields: Any) -> None:
        progress({"phase": phase, "generation": generation, "monotonic": time.monotonic(), **fields})

    def permitted(generation: str) -> bool:
        try:
            value = json.loads((control / "tool-cleanup-permit.json").read_text())
            return value["generation"] == generation and time.monotonic() < float(value["deadline"])
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def signal_owned(record: Process, sig: int, generation: str) -> None:
        if permitted(generation) and same_process(record, process_table()):
            if send_signal(record, sig):
                emit("tool_signal", generation, pid=record.pid, start=record.start, signal=int(sig))

    def safe_branch(root: Process, table: dict[int, Process]) -> bool:
        return same_process(root, table) and not any(
            table[pid].executable in protected or bool(protected_commands.intersection(table[pid].command[:5]))
            for pid in descendants(root.pid, table)
            if pid in table
        )

    try:
        while child.poll() is None:
            now = time.monotonic()
            for key, _ in selector.select(0.1):
                try:
                    data = os.read(key.fileobj.fileno(), 65536)
                except BlockingIOError:
                    # A tool may consume pipe data after readiness was observed.
                    continue
                if not data:
                    selector.unregister(key.fileobj)
                    continue
                output.write(data)
                output.flush()
                if key.data == "stdout" and observer.feed(data, now):
                    if draining is not None:
                        emit("cleanup_finished", draining["generation"], result="cli_activity_resumed")
                        draining = None
                    ready = None
                    candidate = observer.candidate
                    emit(
                        "terminal_candidate" if candidate else "candidate_cancelled",
                        candidate[0] if candidate else None,
                    )
                if observer.first_limit_result is not None and not limit_reported:
                    emit("max_turns_reached", None, cli_result=observer.first_limit_result)
                    limit_reported = True
            if (control / "stop-agent").exists():
                child.terminate()
                host_stop_signal = int(signal.SIGTERM)
                break
            if now >= next_scan:
                next_scan = now + 0.5
                table = process_table()
                if cli is None:
                    subtree = descendants(child.pid, table)
                    matches = {pid for pid in subtree if pid in table and table[pid].executable == cli_inode}
                    # The main CLI may itself be the direct child in tests;
                    # nested Claude agents are never selected as the owner.
                    top = [
                        pid
                        for pid in matches
                        if not any(pid in descendants(other, table) for other in matches - {pid})
                    ]
                    if len(top) == 1:
                        cli = pin_process(table[top[0]])
                        if cli is not None:
                            emit("cli_registered", None, pid=cli.pid, start=cli.start, executable=cli.executable)
                if cli is not None and same_process(cli, table) and draining is None:
                    for record in table.values():
                        if record.parent == cli.pid and record.executable == bash_inode and record.pid not in roots:
                            pinned = pin_process(record)
                            if pinned is not None:
                                roots[record.pid] = pinned
                                registered[record.pid] = pinned
                                emit("tool_registered", None, pid=pinned.pid, start=pinned.start, parent=cli.pid)
                    for root in list(roots.values()):
                        branch = descendants(root.pid, table)
                        # A Bash launching another Claude is not an ordinary
                        # background tool branch; leave it to normal lifecycle.
                        if not safe_branch(root, table):
                            continue
                        for pid in branch:
                            if pid in table and pid not in registered:
                                pinned = pin_process(table[pid])
                                if pinned is not None:
                                    registered[pid] = pinned
                if draining is None:
                    for pid, record in list(registered.items()):
                        if not same_process(record, table):
                            os.close(record.fd)
                            registered.pop(pid)
                            roots.pop(pid, None)

            candidate = observer.candidate
            if candidate is not None and now - candidate[1] >= grace_s and draining is None and cli is not None:
                if ready != candidate[0]:
                    ready = candidate[0]
                    emit("cleanup_ready", ready)
                if permitted(ready):
                    table = process_table()
                    active = {pid: p for pid, p in registered.items() if same_process(p, table)}
                    # Only branches with a still-identical Bash root are
                    # eligible; unknown/adopted descendants are not guessed.
                    active_roots = {pid for pid, root in roots.items() if pid in active and safe_branch(root, table)}
                    members = (
                        set().union(*(descendants(pid, table) for pid in active_roots)) if active_roots else set()
                    )
                    active = {pid: p for pid, p in active.items() if pid in members}
                    if same_process(cli, table) and active_roots and active:
                        cleanup_attempts += 1
                        draining = {
                            "generation": ready,
                            "members": active,
                            "roots": active_roots,
                            "stage": 0,
                            "until": now + 2.0,
                        }
                        emit("cleanup_started", ready, roots=sorted(active_roots), pids=sorted(active))
                        for pid in active_roots:
                            signal_owned(active[pid], signal.SIGTERM, ready)
                    else:
                        emit("cleanup_finished", ready, result="no_proven_tool_branch")
                        observer.candidate = None

            if draining is not None:
                generation = draining["generation"]
                table = process_table()
                alive = {pid: p for pid, p in draining["members"].items() if same_process(p, table)}
                root_alive = draining["roots"] & alive.keys()
                if not permitted(generation):
                    emit("cleanup_finished", generation, result="permit_expired")
                    draining = None
                    observer.candidate = None
                    continue
                stage = draining["stage"]
                if stage == 0 and (not root_alive or now >= draining["until"]):
                    for pid in root_alive:
                        signal_owned(alive[pid], signal.SIGKILL, generation)
                    draining.update(stage=1, until=now + 1.0)
                elif stage == 1 and not root_alive:
                    # Stop parents before leaves: killing only cat can allow
                    # its waiting shell to run the rest of the tool script.
                    for record in alive.values():
                        signal_owned(record, signal.SIGTERM, generation)
                    draining.update(stage=2, until=now + 1.0)
                elif stage == 1 and now >= draining["until"]:
                    emit("cleanup_finished", generation, result="root_did_not_exit")
                    draining = None
                    observer.candidate = None
                elif stage == 2 and (not alive or now >= draining["until"]):
                    for record in alive.values():
                        signal_owned(record, signal.SIGKILL, generation)
                    draining.update(stage=3, until=now + 1.0)
                elif stage == 3 and (not alive or now >= draining["until"]):
                    cleanup_count += 1
                    emit(
                        "cleanup_finished",
                        generation,
                        result="drained" if not alive else "remaining_tool",
                        count=cleanup_count,
                    )
                    draining = None
                    observer.candidate = None
        # Do not wait for pipe EOF: an orphan tool can retain a write fd after
        # the CLI/wrapper has already exited. The controller finally reaps it.
        try:
            code = child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            child.kill()
            if host_stop_signal is not None:
                host_stop_signal = int(signal.SIGKILL)
            code = child.wait(timeout=2)
        drain_deadline = time.monotonic() + 0.2
        for key in list(selector.get_map().values()):
            try:
                while time.monotonic() < drain_deadline:
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        break
                    output.write(data)
                    if key.data == "stdout":
                        observer.feed(data, time.monotonic())
            except BlockingIOError:
                pass
        output.flush()
        emit(
            "cli_exited",
            None,
            returncode=code,
            cleanup_attempts=cleanup_attempts,
            cleanup_count=cleanup_count,
            cli_result=observer.result,
            first_limit_result=observer.first_limit_result,
            host_stop_signal=host_stop_signal,
        )
        return code, {
            "tool_cleanup_count": cleanup_count,
            "tool_cleanup_attempts": cleanup_attempts,
            "returncode": code,
            "cli_result": observer.result,
            "first_limit_result": observer.first_limit_result,
            "host_stop_signal": host_stop_signal,
        }
    finally:
        selector.close()
        child.stdout.close()
        child.stderr.close()
        for record in [*registered.values(), *([cli] if cli is not None else [])]:
            os.close(record.fd)
