# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Source-compatible normalization keeps transport noise out of lineage."""

from __future__ import annotations

import copy
from uuid import NAMESPACE_URL, uuid5

import pytest

from examples.claude_code_harbor.request_normalization import normalize_request


@pytest.mark.parametrize("as_blocks", [False, True])
def test_request_normalization_matches_source_stable_fields(as_blocks: bool) -> None:
    first = "x-anthropic-billing-header: cc_version=2.1.150.660; cc_entrypoint=cli; cch=abcde;"
    second = first.replace(".660;", ".FFF;").replace("cch=abcde", "cch=01234")
    content = [{"type": "text", "text": first, "cache_control": {"type": "ephemeral"}}] if as_blocks else first
    payload = {"system": content, "messages": [{"role": "system", "content": second}]}
    original = copy.deepcopy(payload)
    actual = normalize_request(payload, "rollout-1")
    cch = uuid5(NAMESPACE_URL, "rollout-1").hex[:5]
    suffix = uuid5(NAMESPACE_URL, "cc-version:rollout-1").hex[:3]
    expected = f"x-anthropic-billing-header: cc_version=2.1.150.{suffix}; cc_entrypoint=cli; cch={cch};"
    assert actual["messages"][0]["content"] == expected
    if as_blocks:
        assert actual["system"] == [{**content[0], "text": expected}]
    else:
        assert actual["system"] == expected
    assert payload == original
    assert normalize_request(actual, "rollout-1") == actual
    assert normalize_request(payload, "rollout-2") != actual


def test_request_normalization_preserves_semantic_versions_and_real_system_changes() -> None:
    first = {"system": "cc_version=2.1.150.abc; cch=01234; instructions one"}
    normalized = normalize_request(first, "rollout")
    assert normalize_request({"system": first["system"].replace("2.1.150", "2.1.151")}, "rollout") != normalized
    assert normalize_request({"system": first["system"].replace("one", "two")}, "rollout") != normalized
    assert normalize_request({"system": "cc_version=2.1.150; cc_entrypoint=cli;"}, "rollout")["system"] == (
        "cc_version=2.1.150; cc_entrypoint=cli;"
    )


def test_request_normalization_collapses_source_context_window_reminder_runs() -> None:
    reminder = "<total_tokens>200000</total_tokens>"
    payload = {"system": f"instructions\n{reminder}\n\n{reminder}\n{reminder}\nend"}
    assert normalize_request(payload, "rollout")["system"] == f"instructions\n{reminder}\nend"
    separate = {"system": f"{reminder}\nother instructions\n{reminder}"}
    assert normalize_request(separate, "rollout") == separate
    # Reproduce the source regex, including its first-value-wins behavior.
    differing = {"system": "<total_tokens>200000</total_tokens>\n<total_tokens>180000</total_tokens>"}
    assert normalize_request(differing, "rollout")["system"] == reminder


def test_request_normalization_collapses_reminders_across_system_text_blocks() -> None:
    reminder = "<total_tokens>200000</total_tokens>"
    payload = {
        "system": [
            {"type": "text", "text": f"instructions\n{reminder}", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": f"{reminder}\nend"},
        ]
    }
    assert normalize_request(payload, "rollout")["system"] == f"instructions\n{reminder}\nend"


def test_request_normalization_leaves_tools_observations_and_assistant_history_unchanged() -> None:
    value = "cc_version=2.1.150.abc; cch=abc; <total_tokens>1</total_tokens><total_tokens>1</total_tokens>"
    payload = {
        "messages": [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": value}]},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": value}]},
        ],
        "tools": [{"name": "Bash", "description": value, "input_schema": {"type": "object"}}],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    }
    assert normalize_request(payload, "rollout") == payload
