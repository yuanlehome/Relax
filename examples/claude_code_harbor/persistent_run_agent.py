# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run the normal Harbor adapter and retain its exact successful output
bytes."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import socket
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def _write_private_bytes(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        os.fchmod(output.fileno(), 0o600)
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _check_existing(
    destination: Path,
    payload: bytes,
    session_id: str,
    session_sha256: str,
    payload_sha256: str,
) -> Path:
    if destination.is_symlink() or not destination.is_dir():
        raise FileExistsError("Canonical output destination is not an owned directory")
    payload_path = destination / "contexts.jsonl"
    identity_path = destination / "identity.json"
    for path in (payload_path, identity_path):
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            raise PermissionError("Existing canonical audit files must be regular files with mode 0600")
    if payload_path.read_bytes() != payload:
        raise ValueError("Existing canonical output bytes conflict with their content-addressed path")
    identity = json.loads(identity_path.read_bytes())
    expected = {
        "schema_version": 1,
        "session_id": session_id,
        "session_id_sha256": session_sha256,
        "payload_sha256": payload_sha256,
        "payload_bytes": len(payload),
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("Existing canonical output identity does not match this Session and payload")
    if not all(key in identity for key in ("created_at_utc", "hostname", "pid")):
        raise ValueError("Existing canonical output identity is incomplete")
    return payload_path


def persist_output(output_path: Path, audit_dir: Path, session_id: str) -> Path:
    """Publish a content-addressed payload and identity together, without
    decoding.

    The directory rename commits both files atomically. Repeated identical
    Session/payload pairs retain the first identity sidecar; different bytes
    use a different directory and never overwrite a previous payload.
    """
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("RELAX_SESSION_ID must be a nonempty string")
    payload = output_path.read_bytes()
    if not payload:
        raise ValueError("Successful canonical output must not be empty")
    session_sha256 = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    session_dir = audit_dir / session_sha256
    session_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    if session_dir.is_symlink() or not session_dir.is_dir():
        raise FileExistsError("Canonical Session destination is not an owned directory")
    destination = session_dir / payload_sha256
    if destination.exists() or destination.is_symlink():
        return _check_existing(destination, payload, session_id, session_sha256, payload_sha256)

    identity = {
        "schema_version": 1,
        "session_id": session_id,
        "session_id_sha256": session_sha256,
        "payload_sha256": payload_sha256,
        "payload_bytes": len(payload),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=session_dir))
    try:
        staging.chmod(0o700)
        _write_private_bytes(staging / "contexts.jsonl", payload)
        _write_private_bytes(
            staging / "identity.json",
            (json.dumps(identity, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        )
        try:
            os.rename(staging, destination)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            return _check_existing(destination, payload, session_id, session_sha256, payload_sha256)
        return destination / "contexts.jsonl"
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    # Keep the existing model/Gym loop, output validation and reward anchors.
    from examples.claude_code_harbor.run_agent import main as run_agent_main

    run_agent_main()
    audit_value = os.environ["HARBOR_AUDIT_CONTEXT_DIR"]
    if not audit_value.strip():
        raise ValueError("HARBOR_AUDIT_CONTEXT_DIR must be nonempty")
    payload_path = persist_output(
        Path(os.environ["RELAX_OUTPUT_JSON"]),
        Path(audit_value),
        os.environ["RELAX_SESSION_ID"],
    )
    logger.info(
        "Persisted canonical output session_sha256=%s payload_sha256=%s",
        payload_path.parent.parent.name,
        payload_path.parent.name,
    )


if __name__ == "__main__":
    main()
