# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run in the pinned NeMo environment; these tests do not start Docker."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest


pytest.importorskip("responses_api_agents.claude_code_agent.app")

from examples.claude_code_harbor.gym_service import create_service
from examples.claude_code_harbor.nemo_agent import _TASK, ClaudeCodeAgentRunRequest, HarborClaudeCodeAgent, TaskContext


@pytest.fixture(autouse=True)
def _management_token_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    token_file = tmp_path / "management-token"
    token_file.write_text("management-test-token\n")
    token_file.chmod(0o600)
    monkeypatch.setenv("NEMO_GYM_TOKEN_FILE", str(token_file))
    return token_file


def _agent(monkeypatch: pytest.MonkeyPatch) -> Any:
    for name, value in {
        "GYM_HOST": "127.0.0.1",
        "GYM_PORT": "9999",
        "GYM_CONCURRENCY": "1",
        "CLAUDE_CODE_DOCKER_BINARY": "/unused/docker",
        "CLAUDE_CODE_DOCKER_HOST": "unix:///unused/docker.sock",
        "NEMO_GYM_CALLBACK_ALLOWED_HOSTS": "127.0.0.1",
        "NEMO_GYM_COMMIT": "a85670eb167ba9b48cc53a36a070eed815e6c40d",
    }.items():
        monkeypatch.setenv(name, value)
    return create_service().state.cc_agent


@pytest.mark.asyncio
async def test_management_routes_require_token_but_readiness_and_callback_keep_their_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _agent(monkeypatch)
    app = create_service()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gym") as client:
        for method, path in (
            ("POST", "/v1/trials"),
            ("GET", "/v1/trials/unknown"),
            ("POST", "/v1/trials/unknown/abort"),
            ("POST", "/v1/trials/unknown/renew"),
            ("POST", "/cc/run"),
            ("GET", "/cc/cleanup/unknown"),
            ("POST", "/cc/cleanup/unknown"),
            ("POST", "/cc/v1/responses"),
            ("GET", "/cc"),
        ):
            response = await client.request(method, path)
            assert response.status_code == 401
            assert response.headers["WWW-Authenticate"] == "Bearer"
        assert (await client.get("/cc/readyz")).status_code == 200
        response = await client.post("/ng-rollout/missing/v1/messages", json={"messages": []})
        assert response.status_code == 410
        assert (
            await client.get("/cc/cleanup/unknown", headers={"Authorization": "Bearer wrong-token"})
        ).status_code == 401
        response = await client.get("/cc/cleanup/unknown", headers={"Authorization": "bearer management-test-token"})
        assert response.status_code == 200 and response.json()["clean"] is True
    assert app.state.registry._adapter._client.headers["Authorization"] == "Bearer management-test-token"


def test_oci_service_requires_private_management_token_file(monkeypatch: pytest.MonkeyPatch) -> None:
    _agent(monkeypatch)
    monkeypatch.delenv("NEMO_GYM_TOKEN_FILE")
    with pytest.raises(KeyError, match="NEMO_GYM_TOKEN_FILE"):
        create_service()


def test_gym_turn_limit_default_override_and_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HARBOR_MAX_TURNS", raising=False)
    assert _agent(monkeypatch).config.max_turns == 200
    monkeypatch.setenv("HARBOR_MAX_TURNS", "17")
    assert _agent(monkeypatch).config.max_turns == 17
    for invalid in ("0", "-1", "none", "1.5"):
        monkeypatch.setenv("HARBOR_MAX_TURNS", invalid)
        with pytest.raises(ValueError):
            create_service()


@pytest.mark.asyncio
async def test_abort_before_slot_prevents_late_container_start(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent(monkeypatch)
    agent.sem = asyncio.Semaphore(0)
    body = ClaudeCodeAgentRunRequest.model_validate(
        {"responses_create_params": {"input": "task"}, "_ng_task_index": "cancelled", "_ng_rollout_index": "0"}
    )
    task = asyncio.create_task(agent.run(None, body))
    await asyncio.sleep(0)
    assert (await agent.probe_cleanup("cancelled-0"))["clean"] is False
    assert (await agent.cleanup_rollout("cancelled-0"))["clean"] is True
    agent.sem.release()
    await asyncio.gather(task, return_exceptions=True)
    assert not agent._sandboxes
    with pytest.raises(ValueError, match="live NeMo"):
        await agent.run(None, body)


@pytest.mark.asyncio
async def test_cleanup_survives_repeated_abort_and_artifact_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = _agent(monkeypatch)
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("file")
    monkeypatch.setenv("CLAUDE_CODE_ARTIFACT_ROOT", str(blocker))
    entered, release = asyncio.Event(), asyncio.Event()

    class Sandbox:
        closed = False

        async def stop(self) -> None:
            entered.set()
            await release.wait()

        async def close(self) -> None:
            self.closed = True

    sandbox = Sandbox()
    agent._sandboxes["owned-0"] = sandbox
    cleanup = agent._ensure_cleanup("owned-0")
    await entered.wait()
    # Endpoint retries must share and shield this cleanup operation.
    aborts = [asyncio.create_task(agent.cleanup_rollout("owned-0")) for _ in range(2)]
    await asyncio.sleep(0)
    assert agent._ensure_cleanup("owned-0") is cleanup
    release.set()
    assert all(result["clean"] for result in await asyncio.gather(*aborts))
    assert sandbox.closed and not cleanup.cancelled()


@pytest.mark.asyncio
async def test_failed_owned_container_removal_can_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent(monkeypatch)
    monkeypatch.delenv("CLAUDE_CODE_ARTIFACT_ROOT", raising=False)

    class Sandbox:
        attempts = 0

        async def stop(self) -> None:
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("temporary daemon failure")

        async def close(self) -> None:
            pass

    sandbox = Sandbox()
    agent._sandboxes["retry-0"] = sandbox
    with pytest.raises(RuntimeError, match="temporary daemon"):
        await agent._ensure_cleanup("retry-0")
    assert (await agent.probe_cleanup("retry-0"))["clean"] is False
    assert (await agent.cleanup_rollout("retry-0"))["clean"] is True
    assert sandbox.attempts == 2


@pytest.mark.asyncio
async def test_repeated_cancellation_during_relay_close_preserves_owned_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent(monkeypatch)
    monkeypatch.delenv("CLAUDE_CODE_ARTIFACT_ROOT", raising=False)
    entered, release = asyncio.Event(), asyncio.Event()

    class Relay:
        async def aclose(self) -> None:
            entered.set()
            await release.wait()

    class Sandbox:
        stopped = False
        closed = False

        async def stop(self) -> None:
            self.stopped = True

        async def close(self) -> None:
            self.closed = True

    sandbox = Sandbox()
    agent._sandboxes["close-0"] = sandbox
    cleanup = agent._ensure_cleanup("close-0", Relay())
    await entered.wait()
    waiter = asyncio.create_task(agent.cleanup_rollout("close-0"))
    await asyncio.sleep(0)
    waiter.cancel()
    await asyncio.gather(waiter, return_exceptions=True)
    assert not cleanup.cancelled()
    assert not (await agent.probe_cleanup("close-0"))["clean"]
    release.set()
    await cleanup
    assert sandbox.stopped and sandbox.closed
    assert (await agent.probe_cleanup("close-0"))["clean"]


@pytest.mark.asyncio
async def test_image_eviction_timeout_preserves_cache_and_allows_staging_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from examples.claude_code_harbor import oci

    monkeypatch.setenv("CLAUDE_CODE_DOCKER_BINARY", "/synthetic/docker")
    monkeypatch.setenv("CLAUDE_CODE_DOCKER_HOST", "unix:///synthetic")
    monkeypatch.setattr(oci.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))
    sandbox = oci.DockerSandbox({})
    sandbox.cache = oci._ImageCache()
    sandbox.cache.limit = 0
    sandbox.image = "synthetic-image"
    sandbox.cache.references[sandbox.image] = 1
    calls = []

    async def command(*args: str, **kwargs: Any) -> str:
        calls.append(args)
        if args[0] == "image":
            raise asyncio.TimeoutError
        return ""

    monkeypatch.setattr(sandbox, "command", command)
    await sandbox.stop()
    assert sandbox.stopped and "synthetic-image" in sandbox.cache.unused
    previous = len(calls)
    await sandbox.close()
    assert len(calls) == previous and sandbox.closed and not tmp_path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("budget,returncode", [(0.01, -15), (None, 0), (None, 1)])
async def test_backend_budget_harvests_zero_but_does_not_mask_agent_failure(
    monkeypatch: pytest.MonkeyPatch, budget: float | None, returncode: int
) -> None:
    events: list[str] = []
    stopped = asyncio.Event()

    class Sandbox:
        metadata = {
            "backend_options": {"model_name": "test", "workdir": "/tmp", "agent_kwargs": {"disallowed_tools": []}},
            "sandbox_extra_params": {},
            "harbor_backend_timeout_s": budget,
        }

        async def request(self, *args: Any, **kwargs: Any) -> dict[str, int]:
            events.append("agent_started")
            if budget is not None:
                await stopped.wait()
            return {"returncode": returncode}

        def stop_agent(self) -> None:
            events.append("stop_agent")
            stopped.set()

        def read_log(self, path: str) -> str:
            return "synthetic output"

    class Relay:
        base_url = "http://127.0.0.1"
        terminal_reason = None

        def begin_harvest(self) -> None:
            events.append("fence")

        async def finish_harvest(self) -> None:
            events.append("drained")

    context = TaskContext(Sandbox(), Relay(), "synthetic")
    token = _TASK.set(context)
    try:
        call = HarborClaudeCodeAgent._run_claude_code(
            SimpleNamespace(config=SimpleNamespace(max_turns=200)), "synthetic task"
        )
        if budget is None and returncode:
            with pytest.raises(RuntimeError, match="no training reward"):
                await call
        else:
            output, model = await asyncio.wait_for(call, 3)
            assert model == "test"
            assert output == ("" if budget else "synthetic output")
        assert context.backend_timed_out is (budget is not None)
        assert events == (["agent_started", "fence", "stop_agent", "drained"] if budget else ["agent_started"])
    finally:
        _TASK.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("late_admission", [False, True])
async def test_cleanup_permission_retries_expired_lease_but_rejects_late_model_activity(
    monkeypatch: pytest.MonkeyPatch, late_admission: bool
) -> None:
    from examples.claude_code_harbor import nemo_agent

    clock = [100.0]
    monkeypatch.setattr(nemo_agent, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    finished = asyncio.Event()
    events = []

    class Sandbox:
        metadata = {
            "backend_options": {"model_name": "test", "workdir": "/tmp", "agent_kwargs": {"disallowed_tools": []}},
            "sandbox_extra_params": {},
        }
        agent_progress = [
            {"phase": "terminal_candidate", "generation": "g", "monotonic": 99.0},
            {"phase": "cleanup_ready", "generation": "g", "monotonic": 100.0},
        ]

        async def request(self, *args, **kwargs):
            if late_admission:
                await asyncio.sleep(0.3)
            else:
                await finished.wait()
            return {"returncode": 0}

        def permit_tool_cleanup(self, generation, deadline):
            events.append(("permit", generation, deadline))
            if sum(event[0] == "permit" for event in events) == 1:
                clock[0] = deadline + 1.0
            else:
                finished.set()

        def revoke_tool_cleanup(self):
            events.append(("revoke",))

        def read_log(self, path):
            return "natural exit"

    class Relay:
        base_url = "http://127.0.0.1"
        terminal_reason = None

        def model_activity(self, *, quiet_since):
            assert quiet_since == 99.0
            return None if late_admission else 7

        def begin_tool_cleanup(self, activity):
            assert activity == 7
            events.append(("begin", activity))
            return clock[0] + 10

        def end_tool_cleanup(self):
            events.append(("end",))

    context = TaskContext(Sandbox(), Relay(), "synthetic")
    token = _TASK.set(context)
    try:
        assert await asyncio.wait_for(
            HarborClaudeCodeAgent._run_claude_code(SimpleNamespace(config=SimpleNamespace(max_turns=200)), "task"), 3
        ) == (
            "natural exit",
            "test",
        )
        assert not context.backend_timed_out
        if late_admission:
            assert events == []
        else:
            assert [event[0] for event in events] == [
                "begin",
                "permit",
                "revoke",
                "end",
                "begin",
                "permit",
                "revoke",
                "end",
            ]
    finally:
        _TASK.reset(token)


@pytest.mark.asyncio
async def test_backend_timeout_skips_verifier_and_exports_canonical_contexts(monkeypatch: pytest.MonkeyPatch) -> None:
    from examples.claude_code_harbor import nemo_agent

    events: list[str] = []

    class Sandbox:
        def __init__(self, metadata: dict[str, Any]) -> None:
            self.metadata = metadata

        async def start(self) -> None:
            events.append("start")

        async def verify(self) -> dict[str, float]:
            raise AssertionError("Budget timeout must skip verifier")

    class Capture:
        def export(self, reward: float, metadata: dict[str, Any]) -> list[dict[str, Any]]:
            assert reward == 0 and metadata["termination_reason"] == "backend_timeout"
            return [{"metadata": {**metadata, "outcome_reward": reward}}]

    class Relay:
        terminal_reason = None
        capture = Capture()

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def start(self) -> None:
            pass

        async def aclose(self) -> None:
            events.append("relay_closed")

    class Agent:
        _sandboxes: dict[str, Any] = {}

        def _resolve_call_base_url(self, rollout_id: str) -> str:
            return "http://127.0.0.1"

        async def _create_response(self, *args: Any, **kwargs: Any) -> Any:
            _TASK.get().backend_timed_out = True
            return SimpleNamespace(model_dump=lambda **kwargs: {})

        async def _ensure_cleanup(self, rollout_id: str, relay: Any = None) -> None:
            events.append("cleanup")

    monkeypatch.setattr(nemo_agent, "DockerSandbox", Sandbox)
    monkeypatch.setattr(nemo_agent, "LocalModelRelay", Relay)
    # Inspect the exact final payload independently of OpenAI response schema.
    monkeypatch.setattr(
        nemo_agent, "ClaudeCodeAgentVerifyResponse", SimpleNamespace(model_validate=lambda value: value)
    )
    body = ClaudeCodeAgentRunRequest.model_validate(
        {
            "responses_create_params": {"input": "synthetic task"},
            "reward_fn": "harbor_verifier",
            "blackbox_type": "harbor_agent",
            "instance_id": "synthetic",
            "dataset": "synthetic",
        }
    )
    result = await HarborClaudeCodeAgent._run_task(Agent(), body, "synthetic")
    assert result["reward"] == 0 and result["truncated"] is True
    assert result["metadata"]["contexts"][0]["metadata"]["termination_reason"] == "backend_timeout"
    assert events[-1] == "cleanup"


@pytest.mark.asyncio
async def test_pinned_nemo_accepts_empty_timeout_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    async def empty_output(self: Any, *args: Any, **kwargs: Any) -> tuple[str, str]:
        return "", "synthetic"

    monkeypatch.setattr(HarborClaudeCodeAgent, "_run_claude_code", empty_output)
    agent = _agent(monkeypatch)
    body = ClaudeCodeAgentRunRequest.model_validate({"responses_create_params": {"input": "synthetic task"}})
    response = await agent._create_response(body.responses_create_params, rollout_id="synthetic")
    # This empty control-plane envelope is never used as a training context.
    assert response.output[0].content[0].text == ""
    assert response.usage.input_tokens == 0 and response.usage.output_tokens == 0
