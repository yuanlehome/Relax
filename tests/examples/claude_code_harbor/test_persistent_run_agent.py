# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only synthetic checks; no model, Gym, Ray or training process is
called."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import socket
import stat
import tempfile
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from examples.claude_code_harbor import persistent_run_agent as wrapper


class PersistentRunAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "session_output.json"
        self.audit = self.root / "audit"
        self.session_id = "synthetic-session/with-path-characters"
        self.payload = '{"name":"context_0","metadata":{"outcome_reward":0},"text":"中文"}\r\n \n'.encode()
        self.source.write_bytes(self.payload)

    def persist(self) -> Path:
        return wrapper.persist_output(self.source, self.audit, self.session_id)

    def test_persist_preserves_exact_bytes_identity_and_private_modes(self) -> None:
        previous_umask = os.umask(0)
        try:
            output = self.persist()
        finally:
            os.umask(previous_umask)
        self.assertEqual(output.read_bytes(), self.payload)
        self.assertEqual(self.source.read_bytes(), self.payload)
        session_hash = hashlib.sha256(self.session_id.encode()).hexdigest()
        payload_hash = hashlib.sha256(self.payload).hexdigest()
        self.assertEqual(output, self.audit / session_hash / payload_hash / "contexts.jsonl")
        identity_path = output.with_name("identity.json")
        identity = json.loads(identity_path.read_bytes())
        self.assertEqual(
            set(identity),
            {
                "schema_version",
                "session_id",
                "session_id_sha256",
                "payload_sha256",
                "payload_bytes",
                "created_at_utc",
                "hostname",
                "pid",
            },
        )
        self.assertEqual(identity["session_id"], self.session_id)
        self.assertEqual(identity["session_id_sha256"], session_hash)
        self.assertEqual(identity["payload_sha256"], payload_hash)
        self.assertEqual(identity["payload_bytes"], len(self.payload))
        self.assertEqual(identity["hostname"], socket.gethostname())
        self.assertEqual(identity["pid"], os.getpid())
        for path in (output, identity_path):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_identical_payload_is_idempotent_and_preserves_first_identity(self) -> None:
        output = self.persist()
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.parent.iterdir()}
        with mock.patch.object(wrapper.os, "getpid", return_value=999999):
            self.assertEqual(self.persist(), output)
        after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.parent.iterdir()}
        self.assertEqual(before, after)

    def test_different_payload_and_session_have_distinct_paths(self) -> None:
        first = self.persist()
        self.source.write_bytes(b'{"name":"context_1"}\n')
        second = self.persist()
        third = wrapper.persist_output(self.source, self.audit, "another-session")
        self.assertEqual(len({first, second, third}), 3)
        self.assertEqual(first.read_bytes(), self.payload)
        self.assertEqual(second.read_bytes(), self.source.read_bytes())
        self.assertEqual(third.read_bytes(), second.read_bytes())

    def test_concurrent_identical_payload_commits_one_complete_bundle(self) -> None:
        with ThreadPoolExecutor(max_workers=8) as executor:
            paths = list(executor.map(lambda _: self.persist(), range(16)))
        self.assertEqual(len(set(paths)), 1)
        output = paths[0]
        self.assertEqual(output.read_bytes(), self.payload)
        self.assertEqual(len(list(output.parent.parent.iterdir())), 1)
        self.assertEqual({p.name for p in output.parent.iterdir()}, {"contexts.jsonl", "identity.json"})

    def test_invalid_session_empty_payload_and_missing_output_fail(self) -> None:
        with self.assertRaises(ValueError):
            wrapper.persist_output(self.source, self.audit, " ")
        self.source.write_bytes(b"")
        with self.assertRaises(ValueError):
            self.persist()
        self.source.unlink()
        with self.assertRaises(FileNotFoundError):
            self.persist()

    def test_corrupt_existing_payload_is_not_overwritten(self) -> None:
        output = self.persist()
        output.write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            self.persist()
        self.assertEqual(output.read_bytes(), b"corrupt")

    def test_mismatched_sidecar_and_insecure_mode_fail(self) -> None:
        output = self.persist()
        identity = output.with_name("identity.json")
        initial = identity.read_bytes()
        metadata = json.loads(initial)
        metadata["session_id"] = "wrong-session"
        identity.write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
            self.persist()
        identity.write_bytes(initial)
        identity.chmod(0o644)
        with self.assertRaises(PermissionError):
            self.persist()

    def test_publish_failure_leaves_no_success_bundle(self) -> None:
        with mock.patch.object(wrapper.os, "rename", side_effect=OSError(errno.EIO, "synthetic I/O failure")):
            with self.assertRaises(OSError):
                self.persist()
        self.assertFalse(list(self.audit.glob("*/*")))

    def test_main_preserves_original_success_and_never_dumps_input_or_environment(self) -> None:
        original = types.ModuleType("examples.claude_code_harbor.run_agent")
        original.main = mock.Mock()
        environment = {
            "RELAX_OUTPUT_JSON": str(self.source),
            "RELAX_SESSION_ID": self.session_id,
            "HARBOR_AUDIT_CONTEXT_DIR": str(self.audit),
            "SYNTHETIC_SECRET": "must-never-be-persisted",
        }
        with (
            mock.patch.dict("sys.modules", {original.__name__: original}),
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch.object(wrapper, "logger") as logger,
        ):
            wrapper.main()
        original.main.assert_called_once_with()
        logger.info.assert_called_once()
        for path in self.audit.rglob("*"):
            if path.is_file():
                self.assertNotIn(b"must-never-be-persisted", path.read_bytes())

    def test_original_or_persistence_failure_never_reports_success(self) -> None:
        original = types.ModuleType("examples.claude_code_harbor.run_agent")
        original.main = mock.Mock(side_effect=RuntimeError("synthetic original failure"))
        with (
            mock.patch.dict("sys.modules", {original.__name__: original}),
            mock.patch.object(wrapper, "persist_output") as persist,
            mock.patch.object(wrapper, "logger") as logger,
        ):
            with self.assertRaises(RuntimeError):
                wrapper.main()
        persist.assert_not_called()
        logger.info.assert_not_called()

        original.main = mock.Mock()
        environment = {
            "RELAX_OUTPUT_JSON": str(self.source),
            "RELAX_SESSION_ID": self.session_id,
            "HARBOR_AUDIT_CONTEXT_DIR": str(self.audit),
        }
        with (
            mock.patch.dict("sys.modules", {original.__name__: original}),
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch.object(wrapper, "persist_output", side_effect=OSError("synthetic save failure")),
            mock.patch.object(wrapper, "logger") as logger,
        ):
            with self.assertRaises(OSError):
                wrapper.main()
        logger.info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
