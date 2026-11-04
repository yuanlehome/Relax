# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Optional, short-lived admission policy for new Harbor training agents."""

from __future__ import annotations

import json
import math
import os
import re
import socket
import stat
import time
import warnings
from pathlib import Path
from typing import Any


_MAX_POLICY_BYTES = 16 * 1024
_MAX_POLICY_SECONDS = 30 * 60
_POLICY_KEYS = {"version", "policy_id", "enabled", "job_id", "failure_root", "issued_at", "expires_at", "nodes"}


class NodeAdmissionRejectedError(RuntimeError):
    """The current run explicitly declined a new task on this node."""


def _read_policy(root: Path) -> dict[str, Any]:
    fd = os.open(root / "node-admission.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_POLICY_BYTES:
            raise ValueError("admission policy must be a bounded regular file")
        data = source.read(_MAX_POLICY_BYTES + 1)
    if len(data) > _MAX_POLICY_BYTES:
        raise ValueError("admission policy exceeds size limit")
    policy = json.loads(data)
    if not isinstance(policy, dict) or set(policy) != _POLICY_KEYS:
        raise ValueError("invalid admission policy schema")
    if type(policy["version"]) is not int or policy["version"] != 1 or type(policy["enabled"]) is not bool:
        raise ValueError("invalid admission policy version or enabled flag")
    if not isinstance(policy["policy_id"], str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", policy["policy_id"]):
        raise ValueError("invalid admission policy identifier")
    for key in ("issued_at", "expires_at"):
        value = policy[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("invalid admission policy time")
    if not 0 < policy["expires_at"] - policy["issued_at"] <= _MAX_POLICY_SECONDS:
        raise ValueError("admission policy lifetime must not exceed 30 minutes")
    if not isinstance(policy["nodes"], list) or not policy["nodes"]:
        raise ValueError("admission policy requires explicit node/shard pairs")
    for node in policy["nodes"]:
        if not isinstance(node, dict) or set(node) != {"hostname", "shard"}:
            raise ValueError("invalid admission node schema")
        if not isinstance(node["hostname"], str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,253}", node["hostname"]):
            raise ValueError("invalid admission hostname")
        if not isinstance(node["shard"], str) or not re.fullmatch(r"agentic_session_shard_[0-9]+", node["shard"]):
            raise ValueError("invalid admission shard")
    return policy


def check_node_admission(*, task: str, now: float | None = None) -> None:
    """Fail before container creation only for an exact, unexpired training
    policy.

    Missing, invalid, expired or differently scoped policies leave normal
    execution unchanged. The exception uses the existing whole-group failure
    path; it must never be converted into a reward or a successful empty
    export.
    """
    root_value = os.environ.get("HARBOR_FAILURE_ROOT")
    job_id = os.environ.get("RAY_JOB_ID")
    session_id = os.environ.get("RELAX_SESSION_ID", "")
    if os.environ.get("RELAX_ROLLOUT_MODE") != "train" or not root_value or not job_id:
        return
    root = Path(root_value)
    if not root.is_absolute():
        return
    try:
        policy = _read_policy(root)
    except FileNotFoundError:
        return
    except (OSError, ValueError, TypeError, OverflowError) as error:
        try:
            warnings.warn(
                f"Ignoring invalid Harbor admission policy: {type(error).__name__}", RuntimeWarning, stacklevel=2
            )
        except Exception:
            # A warnings-as-errors policy must not turn a diagnostic into a dropped group.
            pass
        return
    at = time.time() if now is None else now
    if (
        not policy["enabled"]
        or policy["job_id"] != job_id
        or policy["failure_root"] != str(root)
        or not policy["issued_at"] <= at < policy["expires_at"]
    ):
        return
    hostname = socket.gethostname()
    for node in policy["nodes"]:
        if node["hostname"] == hostname and session_id.startswith(node["shard"] + ".session-"):
            raise NodeAdmissionRejectedError(
                f"node admission rejected; policy={policy['policy_id']}; job={job_id}; run={root.parent.name}; "
                f"node={hostname}; shard={node['shard']}; session={session_id}; "
                f"group={os.environ.get('RELAX_GROUP_ID', '')}; task={task}; expires_at={policy['expires_at']}"
            )
