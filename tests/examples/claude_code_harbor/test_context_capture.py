# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise capture against the real Session API normalization
implementation."""

from __future__ import annotations

import copy

import pytest

from examples.claude_code_harbor.contexts import ContextCapture, canonical_context
from examples.claude_code_harbor.training_data import RESPONSE_MASK_KEY


pytest.importorskip("torch")
pytest.importorskip("ray")


def request(messages: list[dict], **kwargs: object) -> dict:
    return {"model": "proxy-model", "max_tokens": 8192, "messages": messages, **kwargs}


def response(text: str) -> dict:
    return {"type": "message", "role": "assistant", "content": [{"type": "text", "text": text}]}


def test_context_capture_keeps_one_maximal_linear_leaf() -> None:
    capture = ContextCapture()
    first = request([{"role": "user", "content": "solve"}])
    capture.record(first, response("start"))
    second = request(
        [*first["messages"], {"role": "assistant", "content": "start"}, {"role": "user", "content": "next"}]
    )
    capture.record(second, response("done"))
    # Reordered duplicate delivery cannot resurrect the shorter leaf.
    capture.record(first, response("start"))
    records = capture.export(1)
    assert len(records) == 1
    assert records[0]["messages"][-1] == {"role": "assistant", "content": "done"}
    assert len(records[0]["messages"]) == 4


def test_context_capture_preserves_pre_compaction_and_summary_contexts() -> None:
    capture = ContextCapture()
    capture.record(request([{"role": "user", "content": "old task"}]), response("old result"))
    capture.record(request([{"role": "user", "content": "Summarize this history"}]), response("summary"))
    capture.record(request([{"role": "user", "content": "Continue from summary"}]), response("new result"))
    records = capture.export(0.5, {"instance_id": "task"})
    assert [row["messages"][-1]["content"] for row in records] == ["old result", "summary", "new result"]
    assert all(
        row["metadata"]["instance_id"] == "task" and row["metadata"]["outcome_reward"] == 0.5 for row in records
    )


def test_context_capture_assigns_shared_generated_prefixes_one_training_owner() -> None:
    capture = ContextCapture()
    initial = [{"role": "user", "content": "task"}]
    capture.record(request(initial), response("shared"))
    shared = [*initial, {"role": "assistant", "content": "shared"}]
    capture.record(request([*shared, {"role": "user", "content": "branch one"}]), response("one"))
    capture.record(request([*shared, {"role": "user", "content": "branch two"}]), response("two"))
    records = capture.export(1)
    assert len(records) == 2
    plans = [record["metadata"][RESPONSE_MASK_KEY] for record in records]
    assert len(plans[0]["response_state_hashes"]) == 2
    assert len(plans[1]["response_state_hashes"]) == 2
    assert plans[0]["response_state_hashes"][0] == plans[1]["response_state_hashes"][0]
    assert len(plans[0]["train_response_hashes"]) == 2
    assert plans[1]["train_response_hashes"] == plans[1]["response_state_hashes"][1:]


def test_context_capture_uses_actual_anthropic_tool_and_reasoning_projection() -> None:
    raw_request = request(
        [{"role": "user", "content": "inspect repo"}],
        system=[{"type": "text", "text": "first"}, {"type": "text", "text": "second"}],
        tools=[{"name": "Bash", "description": "shell", "input_schema": {"type": "object"}}],
        thinking={"type": "enabled", "budget_tokens": 1024},
    )
    raw_response = {
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "inspect first", "signature": "signature"},
            {"type": "tool_use", "id": "call_1", "name": "Bash", "input": {"count": 1.0, "command": "pwd"}},
        ],
    }
    original = copy.deepcopy((raw_request, raw_response))
    context = canonical_context(raw_request, raw_response)
    assistant = context["messages"][-1]
    assert context["messages"][0] == {"role": "system", "content": "first\n\nsecond"}
    assert assistant["reasoning_content"] == "inspect first"
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"command":"pwd","count":1}'
    assert context["chat_template_kwargs"] == {"enable_thinking": True}
    assert context["tools"][0]["function"]["name"] == "Bash"
    assert (raw_request, raw_response) == original


@pytest.mark.parametrize("outcome", [True, float("nan"), float("inf")])
def test_context_capture_rejects_invalid_outcomes(outcome: float) -> None:
    capture = ContextCapture()
    capture.record(request([{"role": "user", "content": "task"}]), response("answer"))
    with pytest.raises(ValueError, match="finite"):
        capture.export(outcome)


def test_context_capture_requires_committed_response() -> None:
    with pytest.raises(ValueError, match="without a captured"):
        ContextCapture().export(1)
