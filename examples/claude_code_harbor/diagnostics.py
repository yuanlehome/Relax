# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bounded, best-effort diagnostics for Harbor agents and their local model
proxy.

This module records lifecycle metadata and archives completed trial evidence;
it never changes rewards or exports. It uses only the standard library and does
not initialize Ray or a model runtime.
"""

from __future__ import annotations

import asyncio
import errno
import fcntl
import json
import math
import os
import re
import socket
import stat
import traceback
import uuid
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator


MAX_FILE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024
SUMMARY_RESERVE_BYTES = 64 * 1024
MAX_SESSION_JSONL_FILES = 4
MAX_PROXY_FILE_BYTES = 64 * 1024
MAX_PROXY_EVENT_BYTES = 4 * 1024
MAX_PROXY_REQUEST_FILE_BYTES = 512 * 1024
_PROXY_REQUEST_PHASES = {
    "request_received",
    "response_headers_sent",
    "upstream_send_start",
    "upstream_response_complete",
    "capture_committed",
    "terminal_write_start",
    "terminal_flushed",
    "client_disconnected",
    "request_failed",
    "handler_transport_error",
    "handler_error",
    "handler_end",
    "proxy_closed",
}
_PROXY_REQUEST_OPERATIONS = {
    "prepare",
    "upstream_connect",
    "upstream_send",
    "upstream_read",
    "capture_record",
    "response_headers",
    "heartbeat_write",
    "await_upstream",
    "cancelled",
    "client_probe",
    "terminal_write",
}
_TRIAL = "harbor-trials/trial"
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_SECRET_KEY = re.compile(
    r"(?i)^(?:authorization|auth|token|credentials?|password|passwd|secret|api[_-]?key|auth[_-]?token|"
    r"access[_-]?token|refresh[_-]?token|.*[_-](?:password|secret|api[_-]?key|auth[_-]?token))$"
)
_SECRET_TEXT = re.compile(
    r"(?im)(\b(?:authorization|[\w-]*(?:api[_-]?key|auth[_-]?token|access[_-]?token|"
    r"password|passwd|secret|credentials?))\b[\"']?\s*[:=]\s*)[^\r\n]+"
)
_BEARER = re.compile(r"(?i)\bBearer\s+[^\s\"',;]+")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact_text(value: str) -> str:
    return _SECRET_TEXT.sub(r"\1[REDACTED]", _BEARER.sub("Bearer [REDACTED]", value))


def _append_proxy_data(filename: str, data: bytes) -> None:
    """Best-effort append to an allowlisted, regular file under the Session
    root."""
    session_dir = os.environ.get("RELAX_SESSION_IO_DIR")
    if not session_dir:
        return
    try:
        limit = {
            "proxy-errors.jsonl": MAX_PROXY_FILE_BYTES,
            "proxy-requests.jsonl": MAX_PROXY_REQUEST_FILE_BYTES,
        }[filename]
        root_fd = os.open(session_dir, _DIR_FLAGS)
        try:
            flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK
            fd = os.open(filename, flags, 0o600, dir_fd=root_fd)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_size + len(data) > limit:
                    return
                if os.write(fd, data) != len(data):
                    os.ftruncate(fd, info.st_size)
            finally:
                os.close(fd)
        finally:
            os.close(root_fd)
    except BaseException:
        return


def record_proxy_request(
    request_seq: int,
    fingerprint: str | None,
    phase: str,
    elapsed_ms: float,
    *,
    capture_committed: bool = False,
    status: int | None = None,
    error_type: str | None = None,
    operation: str | None = None,
) -> None:
    """Record only lifecycle metadata, never request bodies, headers, URLs or
    errors' text.

    Events may be dropped on lock contention, I/O failure or the fixed file
    cap. A successful flush means a local write completed, not that Claude
    consumed it.
    """
    try:
        if (
            type(request_seq) is not int
            or request_seq < 1
            or phase not in _PROXY_REQUEST_PHASES
            or not math.isfinite(elapsed_ms)
            or elapsed_ms < 0
        ):
            return
        event = {
            "version": 1,
            "timestamp": _now(),
            "request_seq": request_seq,
            "fingerprint": fingerprint if fingerprint and re.fullmatch(r"[0-9a-f]{32}", fingerprint) else None,
            "phase": phase,
            "elapsed_ms": round(elapsed_ms, 3),
            "capture_committed": capture_committed is True,
        }
        if type(status) is int and 0 <= status <= 999:
            event["status"] = status
        if error_type and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", error_type):
            event["error_type"] = error_type
        if operation in _PROXY_REQUEST_OPERATIONS:
            event["operation"] = operation
        data = (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
        if len(data) <= 1024:
            _append_proxy_data("proxy-requests.jsonl", data)
    except BaseException:
        return


def record_proxy_error(error_type: str, message: str, status: int | None = None) -> None:
    """Append one bounded error event; diagnostics must never change proxy
    behavior."""
    session_dir = os.environ.get("RELAX_SESSION_IO_DIR")
    if not session_dir:
        return
    try:
        redacted = _redact_text(str(message))
        event = {
            "timestamp": _now(),
            "session_id": os.environ.get("RELAX_SESSION_ID", "")[:200],
            "group_id": os.environ.get("RELAX_GROUP_ID", "")[:200],
            "status": (
                status if isinstance(status, int) and not isinstance(status, bool) and 0 <= status <= 999 else None
            ),
            "error_type": str(error_type)[:128],
            "message": redacted[:2048],
            "message_truncated": len(redacted) > 2048,
        }
        while True:
            data = (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
            if len(data) <= MAX_PROXY_EVENT_BYTES:
                break
            if not event["message"]:
                return
            event["message"] = event["message"][: len(event["message"]) // 2]
            event["message_truncated"] = True
        _append_proxy_data("proxy-errors.jsonl", data)
    except BaseException:
        return


def _redact_json(value: Any, depth: int = 0) -> Any:
    if depth > 24:
        return "[depth limit]"
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if _SECRET_KEY.fullmatch(str(key)) else _redact_json(item, depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_json(item, depth + 1) for item in value]
    return _redact_text(value) if isinstance(value, str) else value


def _safe_bytes(data: bytes, *, json_file: bool) -> bytes:
    text = data.decode("utf-8", errors="replace")
    if json_file:
        try:
            return (json.dumps(_redact_json(json.loads(text)), ensure_ascii=False) + "\n").encode("utf-8")
        except (ValueError, RecursionError):
            pass
    return _redact_text(text).encode("utf-8")


def _parts(relative: str) -> tuple[str, ...]:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("diagnostic path must stay inside its root")
    return path.parts


@contextmanager
def _directory(root_fd: int, parts: tuple[str, ...], *, create: bool = False) -> Iterator[int]:
    current = os.dup(root_fd)
    try:
        for part in parts:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=current)
                except FileExistsError:
                    pass
            child = os.open(part, _DIR_FLAGS, dir_fd=current)
            os.close(current)
            current = child
        yield current
    finally:
        os.close(current)


def _read(root_fd: int, relative: str, limit: int, *, tail: bool = False) -> tuple[bytes, dict[str, Any]]:
    entry: dict[str, Any] = {"path": relative, "selection": "tail" if tail else "head"}
    parts = _parts(relative)
    try:
        with _directory(root_fd, parts[:-1]) as parent:
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode):
                    entry.update(status="blocked", reason="not a regular file")
                    return b"", entry
                entry["source_bytes"] = info.st_size
                if tail and info.st_size > limit:
                    os.lseek(fd, info.st_size - limit, os.SEEK_SET)
                chunks = []
                remaining = limit
                while remaining:
                    chunk = os.read(fd, min(remaining, 65536))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                data = b"".join(chunks)
                entry.update(status="read", truncated=info.st_size > len(data))
                return data, entry
            finally:
                os.close(fd)
    except FileNotFoundError:
        entry["status"] = "missing"
    except OSError as error:
        entry.update(
            status="blocked" if error.errno in (errno.ELOOP, errno.ENOTDIR) else "unreadable",
            reason=type(error).__name__,
        )
    return b"", entry


def _json(root_fd: int, relative: str, notes: list[dict[str, Any]]) -> dict[str, Any]:
    data, entry = _read(root_fd, relative, MAX_FILE_BYTES)
    if entry["status"] == "read" and not entry["truncated"]:
        try:
            value = json.loads(data)
            if isinstance(value, dict):
                return value
            entry.update(status="invalid", reason="expected JSON object")
        except (ValueError, RecursionError) as error:
            entry.update(status="invalid", reason=type(error).__name__)
    notes.append(entry)
    return {}


def _destination() -> Path | None:
    explicit = os.environ.get("HARBOR_FAILURE_ROOT")
    if explicit:
        return Path(explicit).absolute()
    job_id = os.environ.get("RAY_JOB_ID")
    if not job_id:
        return None
    for variable in ("HARBOR_SIF_CACHE_ROOT", "HARBOR_TASK_ROOT"):
        directory = os.environ.get(variable)
        if not directory:
            continue
        try:
            fd = os.open(directory, _DIR_FLAGS)
        except FileNotFoundError:
            continue
        try:
            data, info = _read(fd, "diagnostics.json", 16384)
        finally:
            os.close(fd)
        if info["status"] == "missing":
            continue
        if info["status"] != "read" or info["truncated"]:
            raise ValueError("diagnostics.json is unreadable or exceeds 16 KiB")
        config = json.loads(data)
        if not isinstance(config, dict) or set(config) != {"root", "job_id", "enabled"}:
            raise ValueError("invalid diagnostics.json schema")
        # An existing local policy is authoritative, including disabled or different-job policies.
        if config.get("enabled") is not True or config.get("job_id") != job_id:
            return None
        root = config.get("root")
        if not isinstance(root, str) or not root.strip():
            raise ValueError("diagnostics.json root must be a nonempty path")
        return Path(root).absolute()
    return None


def _classification(error: BaseException | None, trial: dict[str, Any], cli: int | None, reward: float | None) -> str:
    cancelled = isinstance(error, (asyncio.CancelledError, KeyboardInterrupt)) or (
        isinstance(error, SystemExit) and error.code in (130, 143)
    )
    if cancelled:
        return "expected_cancellation"
    if error is not None or trial.get("exception_info"):
        return "runtime_exception"
    if cli is not None and cli != 0:
        return "cli_nonzero"
    if reward is not None:
        return "task_failure" if reward <= 0 else "success"
    return "unknown"


def _extra_files(source_fd: int) -> tuple[list[str], list[dict[str, Any]]]:
    selected: list[str] = []
    notes: list[dict[str, Any]] = []
    try:
        with _directory(source_fd, _parts(_TRIAL)) as fd:
            servers = sorted(name for name in os.listdir(fd) if re.fullmatch(r"apptainer-server-\d+\.log", name))
        selected.extend(f"{_TRIAL}/{name}" for name in servers[:4])
        if len(servers) > 4:
            notes.append({"path": f"{_TRIAL}/apptainer-server-*.log", "status": "selection_limit"})
    except OSError:
        notes.append({"path": f"{_TRIAL}/apptainer-server-*.log", "status": "missing_or_blocked"})

    visited = 0
    jsonl_count = 0

    def visit(relative: str, depth: int = 0) -> None:
        nonlocal visited, jsonl_count
        if visited >= 128 or depth > 8 or jsonl_count >= MAX_SESSION_JSONL_FILES:
            notes.append({"path": relative, "status": "selection_limit"})
            return
        try:
            with _directory(source_fd, _parts(relative)) as fd:
                for name in sorted(os.listdir(fd)):
                    visited += 1
                    if visited > 128 or jsonl_count >= MAX_SESSION_JSONL_FILES:
                        notes.append({"path": relative, "status": "selection_limit"})
                        break
                    info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                    child = f"{relative}/{name}"
                    if stat.S_ISLNK(info.st_mode):
                        notes.append({"path": child, "status": "blocked", "reason": "symlink"})
                    elif stat.S_ISDIR(info.st_mode):
                        visit(child, depth + 1)
                    elif stat.S_ISREG(info.st_mode) and name.endswith(".jsonl"):
                        selected.append(child)
                        jsonl_count += 1
        except OSError as error:
            notes.append({"path": relative, "status": "missing_or_blocked", "reason": type(error).__name__})

    for directory in (f"{_TRIAL}/agent/sessions", f"{_TRIAL}/agent/projects"):
        visit(directory)
    return selected, notes


def _write(root_fd: int, relative: str, data: bytes) -> None:
    parts = _parts(relative)
    with _directory(root_fd, parts[:-1], create=True) as parent:
        temporary = f".{parts[-1]}.{uuid.uuid4().hex}.tmp"
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(temporary, parts[-1], src_dir_fd=parent, dst_dir_fd=parent)
        finally:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass


def _archive(error: BaseException | None) -> Path | None:
    root = _destination()
    if root is None:
        return None
    session_id = os.environ.get("RELAX_SESSION_ID", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,200}", session_id) or session_id in (".", ".."):
        raise ValueError("invalid or missing RELAX_SESSION_ID")
    started_at = _now()
    source = Path(os.environ["RELAX_SESSION_IO_DIR"])
    source_fd = os.open(source, _DIR_FLAGS)
    try:
        return _archive_open(root, source_fd, session_id, started_at, error)
    finally:
        os.close(source_fd)


def _archive_open(root: Path, source_fd: int, session_id: str, started_at: str, error: BaseException | None) -> Path:
    notes: list[dict[str, Any]] = []
    payload = _json(source_fd, "session_input.json", notes)
    trial = _json(source_fd, f"{_TRIAL}/result.json", notes)
    cli_data, cli_info = _read(source_fd, f"{_TRIAL}/agent/claude-exit-code.txt", 128)
    try:
        cli = int(cli_data.strip()) if cli_info["status"] == "read" and not cli_info["truncated"] else None
    except ValueError:
        cli = None
        cli_info.update(status="invalid", reason="expected integer CLI exit code")
    if cli is None:
        notes.append(cli_info)
    verifier = trial.get("verifier_result") or {}
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    raw_reward = rewards.get("reward") if isinstance(rewards, dict) else None
    reward_source = "result.json" if raw_reward is not None else None
    if raw_reward is None:
        reward_data, reward_info = _read(source_fd, f"{_TRIAL}/verifier/reward.txt", 128)
        try:
            raw_reward = (
                float(reward_data) if reward_info["status"] == "read" and not reward_info.get("truncated") else None
            )
            reward_source = "verifier/reward.txt" if raw_reward is not None else None
        except ValueError:
            reward_info.update(status="invalid", reason="expected numeric reward")
        if raw_reward is None:
            notes.append(reward_info)
    reward = float(raw_reward) if isinstance(raw_reward, (int, float)) and not isinstance(raw_reward, bool) else None
    if reward is not None and not math.isfinite(reward):
        notes.append({"path": reward_source, "status": "invalid", "reason": "non-finite reward"})
        reward = None
    category = _classification(error, trial, cli, reward)
    copy_artifacts = (
        category in {"runtime_exception", "cli_nonzero", "task_failure"}
        or (cli is not None and cli != 0)
        or (reward is not None and reward <= 0)
    )
    metadata = payload.get("metadata") or {}
    harbor = (metadata.get("harbor") or {}) if isinstance(metadata, dict) else {}
    task = harbor.get("task") if isinstance(harbor, dict) else None
    exception_info = trial.get("exception_info") or {}
    summary: dict[str, Any] = {
        "schema_version": 1,
        "session_id": session_id,
        "group_id": os.environ.get("RELAX_GROUP_ID"),
        "mode": os.environ.get("RELAX_ROLLOUT_MODE"),
        "task": str(task)[:512] if task is not None else None,
        "node": socket.gethostname(),
        "node_ip": os.environ.get("RAY_NODE_IP_ADDRESS") or os.environ.get("NODE_IP"),
        "ray_job_id": os.environ.get("RAY_JOB_ID"),
        "classification": category,
        "reward": reward,
        "reward_source": reward_source,
        "cli_exit_code": cli,
        "trial_exception_type": exception_info.get("exception_type") if isinstance(exception_info, dict) else None,
        "exception": (
            {"type": type(error).__name__, "message": _redact_text(str(error))[:2048]} if error is not None else None
        ),
        "archive_method": "agent_main_after_asyncio_run",
        "archive_started_at": started_at,
        "metadata_notes": notes,
        "artifacts": [],
    }
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root_fd = os.open(root, _DIR_FLAGS)
    try:
        with _directory(root_fd, ("sessions", session_id), create=True) as destination_fd:
            existing, info = _read(destination_fd, "summary.json", SUMMARY_RESERVE_BYTES)
            if info["status"] == "read":
                # A completed archive is immutable; repeated calls are idempotent.
                if json.loads(existing).get("session_id") != session_id:
                    raise ValueError("existing diagnostic summary has a different session identity")
                return root / "sessions" / session_id / "summary.json"
            if os.listdir(destination_fd):
                raise ValueError("incomplete existing archive; refusing to append beyond its original budget")
            artifacts_bytes = 0
            if copy_artifacts:
                fixed = [
                    "session_input.json",
                    "proxy-errors.jsonl",
                    "proxy-requests.jsonl",
                    f"{_TRIAL}/result.json",
                    f"{_TRIAL}/exception.txt",
                    f"{_TRIAL}/trial.log",
                    f"{_TRIAL}/agent/claude-exit-code.txt",
                    f"{_TRIAL}/agent/claude-code.txt",
                    f"{_TRIAL}/verifier/reward.txt",
                    f"{_TRIAL}/verifier/reward.json",
                    f"{_TRIAL}/verifier/test-stdout.txt",
                    f"{_TRIAL}/verifier/test-stderr.txt",
                ]
                extras, selection_notes = _extra_files(source_fd)
                summary["artifacts"].extend(selection_notes)
                candidates = [(name, None) for name in fixed + extras]
                if error is not None:
                    trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
                    candidates.insert(0, ("agent-main-exception.txt", trace.encode("utf-8")))
                for relative, generated in candidates:
                    remaining = MAX_TOTAL_BYTES - SUMMARY_RESERVE_BYTES - artifacts_bytes
                    if remaining <= 0:
                        summary["artifacts"].append({"path": relative, "status": "total_budget_exhausted"})
                        continue
                    limit = min(MAX_FILE_BYTES, remaining)
                    if generated is None:
                        data, entry = _read(source_fd, relative, limit, tail=not relative.endswith(".json"))
                    else:
                        data = generated[:limit]
                        entry = {
                            "path": relative,
                            "status": "read",
                            "source_bytes": len(generated),
                            "selection": "head",
                            "truncated": len(generated) > limit,
                        }
                    if entry["status"] == "read":
                        if relative in {"proxy-errors.jsonl", "proxy-requests.jsonl"}:
                            safe = b"".join(_safe_bytes(line, json_file=True) for line in data.splitlines())
                        else:
                            safe = _safe_bytes(data, json_file=relative.endswith(".json"))
                        entry["redacted"] = safe != data
                        entry["truncated"] = entry["truncated"] or len(safe) > limit
                        safe = safe[:limit]
                        _write(destination_fd, f"artifacts/{relative}", safe)
                        entry.update(status="copied", archived_bytes=len(safe))
                        artifacts_bytes += len(safe)
                    summary["artifacts"].append(entry)
            incomplete = any(item["status"] != "copied" or item.get("truncated") for item in summary["artifacts"])
            summary["coverage"] = {
                "status": "metadata_only" if not copy_artifacts else "partial" if incomplete else "allowlist_complete",
                "complete": copy_artifacts and not incomplete,
                "scope": "Allowlisted artifacts only; absent, blocked, truncated and unselected files are recorded.",
                "max_file_bytes": MAX_FILE_BYTES,
                "max_total_bytes": MAX_TOTAL_BYTES,
                "artifact_bytes": artifacts_bytes,
            }
            summary["archived_at"] = _now()
            for _ in range(4):
                encoded = (json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
                total = artifacts_bytes + len(encoded)
                if summary["coverage"].get("total_bytes") == total:
                    break
                summary["coverage"]["total_bytes"] = total
            if len(encoded) > SUMMARY_RESERVE_BYTES or total > MAX_TOTAL_BYTES:
                raise ValueError("diagnostic summary exceeds its reserved budget")
            _write(destination_fd, "summary.json", encoded)
    finally:
        os.close(root_fd)
    return root / "sessions" / session_id / "summary.json"


def archive(error: BaseException | None = None) -> Path | None:
    """Archive final session evidence without ever replacing the caller's
    error.

    Call in the synchronous main entrypoint after asyncio.run returns or
    raises. Disabled or job-mismatched fallback configuration is a silent no-
    op.
    """
    try:
        return _archive(error)
    except BaseException as archive_error:
        try:
            warnings.warn(
                f"Harbor diagnostics archive failed: {type(archive_error).__name__}: "
                f"{_redact_text(str(archive_error))[:512]}",
                RuntimeWarning,
                stacklevel=2,
            )
        except BaseException:
            # Warning filters may promote warnings to exceptions. Preserve the caller's error even then.
            pass
        return None
