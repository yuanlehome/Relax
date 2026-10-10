# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from relax_nemo_gym_example.app.protocol import PROTOCOL_VERSION, ProtocolValidationError, TrialResult, TrialStatus
from relax_nemo_gym_example.app.result import to_relax_output, write_relax_output
from relax_nemo_gym_example.service.registry import GatewayRegistry
from relax_nemo_gym_example.service.run_adapter import (
    AdapterResult,
    HttpNemoGymRunAdapter,
    TrialRunContext,
    _normalize_run_result,
)

from .test_gateway_service import ControlledAdapter, _request, _settings, _wait_for_status


def _context(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "messages": [{"role": "user", "content": "task"}, {"role": "assistant", "content": "answer"}],
        "tools": [],
        "chat_template_kwargs": {},
    }


def _terminal(metadata: Any) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": "request-one",
        "status": "completed",
        "reward": 0.0,
        "metadata": metadata,
    }


def test_result_metadata_is_optional_validated_and_copied() -> None:
    payload = _terminal({"contexts": [_context("before"), _context("after")]})
    result = TrialResult.from_payload(payload, expected_request_id="request-one")
    payload["metadata"]["contexts"].clear()
    assert len(result.metadata["contexts"]) == 2
    assert "contexts" not in repr(result)
    payload.pop("metadata")
    assert TrialResult.from_payload(payload, expected_request_id="request-one").metadata == {}


@pytest.mark.parametrize("metadata", [None, [], {"bad": float("nan")}])
def test_invalid_result_metadata_is_rejected_at_both_boundaries(metadata: Any) -> None:
    with pytest.raises((ValueError, ProtocolValidationError)):
        TrialResult.from_payload(_terminal(metadata), expected_request_id="request-one")
    with pytest.raises(ValueError):
        _normalize_run_result({"reward": 0.0, "metadata": metadata})


@pytest.mark.parametrize("capture", ["disabled", "success", "failure"])
async def test_http_adapter_preserves_contexts_across_artifact_paths(tmp_path: Path, capture: str) -> None:
    metadata = {"contexts": [_context("before"), _context("after")]}
    request = _request("request-one")
    request.metadata["capture_artifacts"] = capture != "disabled"
    spec = next(iter(_settings().environments.values()))
    adapter = HttpNemoGymRunAdapter(
        [spec],
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"reward": 0.0, "metadata": metadata})),
        artifact_root=tmp_path if capture == "success" else None,
    )
    try:
        handle = await adapter.start(TrialRunContext(request=request, spec=spec, rollout_id="capability-0"))
        result = await handle.wait()
        assert result.metadata == metadata
        if capture == "success":
            assert result.metrics["artifact_captured"] is True
        elif capture == "failure":
            assert result.metrics["artifact_captured"] is False
    finally:
        await adapter.close()


async def test_terminal_contexts_round_trip_through_registry_and_writer(tmp_path: Path) -> None:
    adapter = ControlledAdapter()
    registry = GatewayRegistry(settings=_settings(), adapter=adapter)
    metadata = {"contexts": [_context("before"), _context("after")], "task_id": "task-one"}
    await registry.start()
    try:
        snapshot = await registry.create(_request("request-one").to_payload())
        assert snapshot["metadata"] == {}
        await adapter.started.wait()
        assert (await registry.get("request-one"))["metadata"] == {}
        adapter.handles[0].result.set_result(
            AdapterResult(status=TrialStatus.COMPLETED, reward=0.0, metadata=metadata)
        )
        snapshot = await _wait_for_status(registry, "request-one", "completed")
        metadata["contexts"].clear()
        assert len(snapshot["metadata"]["contexts"]) == 2
        result = TrialResult.from_payload(snapshot, expected_request_id="request-one")
        path = tmp_path / "output.json"
        write_relax_output(path, to_relax_output(result))
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert [record["name"] for record in records] == ["before", "after"]
        assert all(record["metadata"]["outcome_reward"] == 0.0 for record in records)
        snapshot["metadata"]["contexts"].clear()
        assert len((await registry.get("request-one"))["metadata"]["contexts"]) == 2
    finally:
        await registry.close()


def test_context_writer_preserves_metadata_and_authoritative_provenance(tmp_path: Path) -> None:
    first = _context("before")
    first.update(reward=None, metadata={"nemo_gym": {"request_id": "spoof"}, "local": 1})
    metadata = {"contexts": [first, _context("after")], "task_id": "task-one", "nemo_gym": {"bad": True}}
    result = TrialResult(request_id="request-one", status=TrialStatus.COMPLETED, reward=0.5, metadata=metadata)
    original = copy.deepcopy(metadata)
    path = tmp_path / "output.json"
    write_relax_output(path, to_relax_output(result))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["reward"] for record in records] == [None, 0.5]
    assert records[0]["metadata"]["local"] == 1
    for record in records:
        assert record["metadata"]["task_id"] == "task-one"
        assert record["metadata"]["outcome_reward"] == 0.5
        assert record["metadata"]["nemo_gym"]["request_id"] == "request-one"
        assert "contexts" not in record["metadata"]
    assert result.metadata == original


@pytest.mark.parametrize(
    "contexts",
    [
        None,
        [],
        {},
        [_context("same"), _context("same")],
        [{**_context("a"), "messages": []}],
        [{**_context("a"), "tools": None}],
        [{**_context("a"), "metadata": {"outcome_reward": 0.0}}],
        [_context("a"), {**_context("b"), "metadata": {"bad": object()}}],
    ],
)
def test_invalid_contexts_never_fall_back_or_replace_previous_output(tmp_path: Path, contexts: Any) -> None:
    path = tmp_path / "output.json"
    path.write_text('{"old":true}', encoding="utf-8")
    with pytest.raises((ValueError, TypeError)):
        write_relax_output(path, {"reward": 0.5, "metadata": {"contexts": contexts}})
    assert json.loads(path.read_text()) == {"old": True}
    assert not path.with_name(".output.json.tmp").exists()


def test_context_absence_preserves_legacy_output(tmp_path: Path) -> None:
    path = tmp_path / "output.json"
    result = TrialResult(request_id="request-one", status=TrialStatus.COMPLETED, reward=1.0)
    payload = to_relax_output(result)
    write_relax_output(path, payload)
    assert json.loads(path.read_text()) == payload
