# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from examples.claude_code_harbor.advantage import advantage_func
from examples.claude_code_harbor.run_agent import read_explicit_records


def _metadata(outcome: Any, response_hash: str = "response-1") -> dict[str, Any]:
    return {
        "outcome_reward": outcome,
        "agentic_trace": {"turns": [{"resp_state_hash": response_hash}]},
    }


def _record(name: str, outcome: float) -> dict[str, Any]:
    return {
        "name": name,
        "messages": [{"role": "user", "content": "task"}, {"role": "assistant", "content": "done"}],
        "tools": [],
        "chat_template_kwargs": {},
        "metadata": {"outcome_reward": outcome},
    }


def test_grpo_uses_source_population_std_over_logical_sessions() -> None:
    sessions = [{"before": _metadata(1.0, "before"), "after": _metadata(1.0, "after")}]
    sessions += [{"only": _metadata(0.0)} for _ in range(7)]
    values = advantage_func(sessions)
    assert values[0] == pytest.approx({"before": math.sqrt(7), "after": math.sqrt(7)})
    for value in values[1:]:
        assert value == pytest.approx({"only": -1 / math.sqrt(7)})


def test_grpo_drops_zero_variance_and_keeps_tiny_nonzero_variance() -> None:
    assert advantage_func([{"a": _metadata(0.5)}, {"b": _metadata(0.5)}]) is None
    assert advantage_func([{"a": _metadata(0.0)}, {"b": _metadata(1e-8)}]) == [
        {"a": -5e-9},
        {"b": 5e-9},
    ]


@pytest.mark.parametrize("outcome", [None, True, float("nan"), float("inf"), "1"])
def test_grpo_rejects_invalid_outcomes(outcome: Any) -> None:
    with pytest.raises(ValueError, match="finite numeric"):
        advantage_func([{"a": _metadata(outcome)}, {"b": _metadata(0.0)}])


def test_grpo_rejects_inconsistent_outcomes_and_shared_response_prefix() -> None:
    with pytest.raises(ValueError, match="same outcome"):
        advantage_func([{"a": _metadata(0.0, "a"), "b": _metadata(1.0, "b")}, {"c": _metadata(0.0)}])
    with pytest.raises(ValueError, match="share generated responses"):
        advantage_func([{"a": _metadata(1.0), "b": _metadata(1.0)}, {"c": _metadata(0.0)}])


def test_explicit_output_accepts_context_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "output.json"
    records = [_record("before", 0.5), _record("after", 0.5)]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    assert read_explicit_records(path) == records


@pytest.mark.parametrize(
    "payload",
    [
        {"reward": 1.0, "metadata": {}},
        [_record("a", 1.0)],
        {**_record("a", 1.0), "tools": None},
        {**_record("a", 1.0), "metadata": {}},
    ],
)
def test_explicit_output_rejects_scalar_only_and_incomplete_contexts(tmp_path: Path, payload: Any) -> None:
    path = tmp_path / "output.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        read_explicit_records(path)


def test_wrapper_removes_disable_flag_and_passes_source_window(tmp_path: Path) -> None:
    output = tmp_path / "output.json"
    adapter = tmp_path / "fake_adapter.py"
    adapter.write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "assert 'DISABLE_COMPACT' not in os.environ\n"
        "assert os.environ['CLAUDE_CODE_AUTO_COMPACT_WINDOW'] == '200000'\n"
        "settings = json.loads(Path(os.environ['CLAUDE_CODE_SETTINGS_PATH']).read_text())\n"
        "assert 'DISABLE_COMPACT' not in settings['env']\n"
        f"Path(os.environ['RELAX_OUTPUT_JSON']).write_text(json.dumps({repr(_record('main', 1.0))}))\n",
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[3]
    subprocess.run(
        [sys.executable, "-m", "examples.claude_code_harbor.run_agent"],
        cwd=root,
        env={
            **os.environ,
            "DISABLE_COMPACT": "1",
            "CLAUDE_CODE_AGENT_COMMAND": shlex.join([sys.executable, str(adapter)]),
            "RELAX_OUTPUT_JSON": str(output),
        },
        check=True,
    )
    assert read_explicit_records(output) == [{**_record("main", 1.0), "reward": 1.0}]


def test_wrapper_materializes_metadata_contexts_with_one_reward_anchor(tmp_path: Path) -> None:
    output = tmp_path / "output.json"
    payload = {"reward": 0.5, "metadata": {"contexts": [_record("before", 0.5), _record("after", 0.5)]}}
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "import os\nfrom pathlib import Path\n"
        f"Path(os.environ['RELAX_OUTPUT_JSON']).write_text({json.dumps(payload)!r})\n",
        encoding="utf-8",
    )
    subprocess.run(
        [sys.executable, "-m", "examples.claude_code_harbor.run_agent"],
        cwd=Path(__file__).resolve().parents[3],
        env={
            **os.environ,
            "CLAUDE_CODE_AGENT_COMMAND": shlex.join([sys.executable, str(adapter)]),
            "RELAX_OUTPUT_JSON": str(output),
        },
        check=True,
    )
    records = read_explicit_records(output)
    assert [record["reward"] for record in records] == [None, 0.5]
    assert all(record["metadata"]["outcome_reward"] == 0.5 for record in records)
    assert len(output.read_text().splitlines()) == 2
