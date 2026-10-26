# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Boundary regressions for native CLI limits; no Docker or model service."""

from __future__ import annotations

import asyncio
import io
import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from examples.claude_code_harbor import sandbox_worker
from examples.claude_code_harbor.cli_lifecycle import TerminalObserver, monitor_cli


def events(*items: dict[str, Any]) -> bytes:
    return b"".join((json.dumps({"session_id": "main", **item}) + "\n").encode() for item in items)


INIT = {"type": "system", "subtype": "init"}
LIMIT = {"type": "result", "subtype": "error_max_turns", "is_error": True, "num_turns": 2}


@pytest.mark.parametrize(
    "later",
    [
        {"type": "assistant"},
        {"type": "user"},
        INIT,
        {"type": "result", "subtype": "error_during_execution", "is_error": True},
    ],
)
def test_expected_limit_cannot_survive_later_activity(later: dict[str, Any]) -> None:
    observer = TerminalObserver()
    observer.feed(events(INIT, LIMIT), 1)
    assert observer.result["subtype"] == "error_max_turns"
    assert observer.candidate is not None
    observer.feed(events(later), 2)
    assert observer.result is None or observer.result["subtype"] != "error_max_turns"


def test_limit_requires_matching_session_and_complete_valid_stdout() -> None:
    observer = TerminalObserver(max_turns=1)
    observer.feed(events(INIT, {**LIMIT, "session_id": "subagent"}), 1)
    assert observer.result is None
    assert observer.first_limit_result is None
    raw = events(LIMIT)
    observer.feed(raw[:10], 2)
    assert observer.result is None and observer.first_limit_result is None
    observer.feed(raw[10:], 3)
    assert observer.result["num_turns"] == 2
    assert observer.first_limit_result["num_turns"] == 2
    observer.feed(b"x" * (1024 * 1024 + 1) + b"\n", 4)
    assert observer.result is None
    assert observer.first_limit_result["num_turns"] == 2


def test_first_configured_limit_survives_activity_in_the_same_read() -> None:
    observer = TerminalObserver(max_turns=1)
    observer.feed(events(INIT, LIMIT, {"type": "assistant"}), 1)
    assert observer.result is None
    assert observer.candidate is None
    assert observer.first_limit_result["num_turns"] == 2
    observer.feed(events(INIT, {**LIMIT, "num_turns": 1}), 2)
    assert observer.first_limit_result["num_turns"] == 2


@pytest.mark.parametrize("turns", [None, True, 0, 3, "2", 2.0])
def test_invalid_limit_does_not_latch(turns: Any) -> None:
    observer = TerminalObserver(max_turns=1)
    observer.feed(events(INIT, {**LIMIT, "num_turns": turns}), 1)
    assert observer.first_limit_result is None


def test_private_stdout_limit_requires_host_stop_and_preserves_actual_exit(tmp_path: Path) -> None:
    payload = events(INIT, LIMIT, {"type": "user"})
    child = subprocess.Popen(
        [sys.executable, "-c", f"import os,time; os.write(1, {payload!r}); time.sleep(10)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    seen = []

    def progress(event: dict[str, Any]) -> None:
        seen.append(event)
        if event["phase"] == "max_turns_reached":
            assert child.poll() is None
            # This stands in for the host after it fences relay admissions.
            (tmp_path / "stop-agent").touch()

    try:
        code, summary = monitor_cli(
            child,
            io.BytesIO(),
            cli_executable=Path(sys.executable),
            control=tmp_path,
            progress=progress,
            max_turns=1,
        )
        assert code == -signal.SIGTERM
        assert summary["host_stop_signal"] == signal.SIGTERM
        assert summary["first_limit_result"]["num_turns"] == 2
        assert summary["cli_result"] is None
        assert sum(e["phase"] == "max_turns_reached" for e in seen) == 1
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


@pytest.mark.parametrize("stderr", [False, True])
def test_fast_exit_drain_observes_only_stdout(tmp_path: Path, stderr: bool) -> None:
    payload = events(INIT, LIMIT)
    child = subprocess.Popen(
        [sys.executable, "-c", f"import os; os.write({2 if stderr else 1}, {payload!r}); raise SystemExit(1)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    child.wait(timeout=10)  # Force the post-exit drain path, not the poll loop.
    code, summary = monitor_cli(
        child,
        io.BytesIO(),
        cli_executable=Path(sys.executable),
        control=tmp_path,
        progress=lambda event: None,
        max_turns=1,
    )
    assert code == 1
    assert (summary["cli_result"] is None) is stderr
    assert (summary["first_limit_result"] is None) is stderr
    assert summary["host_stop_signal"] is None
    if not stderr:
        assert summary["cli_result"]["subtype"] == "error_max_turns"


@pytest.mark.parametrize("instruction", ["solve this task", "/escalate literal task", " \n/escalate literal task"])
def test_worker_passes_real_limit_and_literal_task_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, instruction: str
) -> None:
    captured = {}
    real_path = Path
    monkeypatch.setattr(
        sandbox_worker,
        "Path",
        lambda p: real_path(tmp_path / "cli.log") if p == "/logs/agent/claude-code.txt" else real_path(p),
    )

    def popen(argv, **kwargs):
        captured.update(argv=argv, stdin=kwargs["stdin"].read())
        return object()

    monkeypatch.setattr(sandbox_worker.subprocess, "Popen", popen)
    monkeypatch.setattr(sandbox_worker, "monitor_cli", lambda *args, **kwargs: (1, {"cli_result": LIMIT}))
    monkeypatch.setattr(sandbox_worker, "cleanup_descendants", lambda: None)
    result = sandbox_worker.run_command(
        {
            "operation": "agent",
            "max_turns": 17,
            "instruction": instruction,
            "disallowed_tools": "",
            "env": {},
            "workdir": str(tmp_path),
        }
    )
    assert captured["argv"][captured["argv"].index("--max-turns") + 1] == "17"
    expected = "Task instruction:\n\n" + instruction if instruction.lstrip().startswith("/") else instruction
    assert captured["stdin"] == expected.encode()
    assert result["returncode"] == 1 and result["cli_result"] == LIMIT


@pytest.mark.parametrize("cap", [None, True, 0, -1, 1.5, "10"])
def test_worker_rejects_invalid_cap_before_starting_process(cap: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        sandbox_worker.run_command({"operation": "agent", "env": {}, "max_turns": cap})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code,result,accepted",
    [
        (1, {**LIMIT, "num_turns": 17}, True),
        (1, {**LIMIT, "num_turns": 18}, True),
        (1, {**LIMIT, "num_turns": 19}, False),
        (1, {**LIMIT, "num_turns": True}, False),
        (1, {**LIMIT, "num_turns": 16}, False),
        (137, {**LIMIT, "num_turns": 18}, False),
        (1, None, False),
    ],
)
async def test_adapter_accepts_only_expected_typed_limit(code: int, result: Any, accepted: bool) -> None:
    pytest.importorskip("responses_api_agents.claude_code_agent.app")
    from examples.claude_code_harbor.nemo_agent import _TASK, HarborClaudeCodeAgent, TaskContext

    class Sandbox:
        metadata = {
            "harbor_max_turns": 17,
            "backend_options": {
                "model_name": "test",
                "workdir": "/tmp",
                "agent_kwargs": {"disallowed_tools": "", "max_turns": 10},
            },
            "sandbox_extra_params": {},
        }

        async def request(self, *args, **kwargs):
            assert kwargs["max_turns"] == 17
            return {"returncode": code, "cli_result": result}

        def read_log(self, path):
            return "untrusted merged log"

    context = TaskContext(Sandbox(), SimpleNamespace(base_url="http://127.0.0.1", terminal_reason=None), "test")
    token = _TASK.set(context)
    try:
        call = HarborClaudeCodeAgent._run_claude_code(SimpleNamespace(config=SimpleNamespace(max_turns=200)), "task")
        if accepted:
            assert await call == ("untrusted merged log", "test")
            assert context.max_turns_reached
        else:
            with pytest.raises(RuntimeError, match="no training reward"):
                await call
    finally:
        _TASK.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fast_exit,code,stop_signal,harvest_error,accepted",
    [
        (False, -signal.SIGTERM, int(signal.SIGTERM), False, True),
        (False, -signal.SIGKILL, int(signal.SIGKILL), False, True),
        (False, 137, int(signal.SIGTERM), False, False),
        (False, -signal.SIGKILL, int(signal.SIGTERM), False, False),
        (False, -signal.SIGTERM, int(signal.SIGTERM), True, False),
        (True, 1, None, False, True),
    ],
)
async def test_latched_limit_fences_before_stop_and_harvests_accepted_calls(
    fast_exit: bool, code: int, stop_signal: int | None, harvest_error: bool, accepted: bool
) -> None:
    pytest.importorskip("responses_api_agents.claude_code_agent.app")
    from examples.claude_code_harbor.nemo_agent import _TASK, HarborClaudeCodeAgent, TaskContext

    actions: list[str] = []
    stopped = asyncio.Event()
    limit = {**LIMIT, "num_turns": 18}

    class Relay:
        base_url = "http://127.0.0.1"
        terminal_reason = None

        def begin_harvest(self) -> None:
            assert not stopped.is_set()
            actions.append("fence")

        async def finish_harvest(self) -> None:
            actions.append("harvest")
            await asyncio.sleep(0)
            if harvest_error:
                raise RuntimeError("accepted call failed")
            actions.append("canonical_capture")

    class Sandbox:
        metadata = {
            "harbor_max_turns": 17,
            "backend_options": {
                "model_name": "test",
                "workdir": "/tmp",
                "agent_kwargs": {"disallowed_tools": "", "max_turns": 10},
            },
            "sandbox_extra_params": {},
        }

        def __init__(self) -> None:
            self.agent_progress = []

        async def request(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            if not fast_exit:
                self.agent_progress.append({"phase": "max_turns_reached", "cli_result": limit})
                await asyncio.wait_for(stopped.wait(), timeout=2)
            return {
                "returncode": code,
                "first_limit_result": limit,
                "host_stop_signal": stop_signal,
                "cli_result": None,
            }

        def stop_agent(self) -> None:
            assert actions == ["fence"]
            actions.append("stop")
            stopped.set()

        def read_log(self, path: str) -> str:
            assert actions[-1] == "canonical_capture"
            return "untrusted merged log"

    context = TaskContext(Sandbox(), Relay(), "test")
    token = _TASK.set(context)
    try:
        call = HarborClaudeCodeAgent._run_claude_code(SimpleNamespace(config=SimpleNamespace(max_turns=200)), "task")
        if accepted:
            assert await call == ("untrusted merged log", "test")
            assert context.max_turns_reached
            assert context.first_limit_result == limit
            assert context.num_turns is None  # Do not invent a final CLI result.
            assert context.cli_returncode == code
        else:
            with pytest.raises(RuntimeError, match="accepted call failed|no training reward"):
                await call
            assert not context.max_turns_reached
        assert actions[:1] == ["fence"]
        assert actions.count("fence") == 1
        assert ("stop" in actions) is not fast_exit
    finally:
        _TASK.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("reward", [0.0, 1.0])
async def test_turn_limit_runs_real_verifier_and_exports_actual_reward(
    monkeypatch: pytest.MonkeyPatch, reward: float
) -> None:
    pytest.importorskip("responses_api_agents.claude_code_agent.app")
    from examples.claude_code_harbor import nemo_agent

    events_seen = []

    class Sandbox:
        def __init__(self, metadata):
            self.metadata = metadata

        async def start(self):
            pass

        async def verify(self):
            events_seen.append("verify")
            return {"reward": reward}

    class Capture:
        def export(self, actual, metadata):
            assert actual == reward
            assert metadata["termination_reason"] == "max_turns"
            return [{"metadata": {**metadata, "outcome_reward": actual}}]

    class Relay:
        terminal_reason = None
        capture = Capture()

        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        async def aclose(self):
            pass

    class Agent:
        _sandboxes = {}

        def _resolve_call_base_url(self, rollout_id):
            return "http://127.0.0.1"

        async def _create_response(self, *args, **kwargs):
            context = nemo_agent._TASK.get()
            context.max_turns_reached, context.max_turns, context.num_turns = True, 1, 2
            return SimpleNamespace(model_dump=lambda **kwargs: {})

        async def _ensure_cleanup(self, *args):
            events_seen.append("cleanup")

    monkeypatch.setattr(nemo_agent, "DockerSandbox", Sandbox)
    monkeypatch.setattr(nemo_agent, "LocalModelRelay", Relay)
    monkeypatch.setattr(
        nemo_agent, "ClaudeCodeAgentVerifyResponse", SimpleNamespace(model_validate=lambda value: value)
    )
    body = nemo_agent.ClaudeCodeAgentRunRequest.model_validate(
        {
            "responses_create_params": {"input": "task"},
            "reward_fn": "harbor_verifier",
            "blackbox_type": "harbor_agent",
            "instance_id": "task",
            "dataset": "test",
        }
    )
    result = await nemo_agent.HarborClaudeCodeAgent._run_task(Agent(), body, "test")
    assert result["reward"] == reward and result["truncated"] is True
    assert events_seen == ["verify", "cleanup"]
