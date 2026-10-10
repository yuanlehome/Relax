# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run in the pinned NeMo environment; these tests do not start Docker."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest


pytest.importorskip("responses_api_agents.claude_code_agent.app")

from examples.claude_code_harbor.gym_service import create_service
from examples.claude_code_harbor.nemo_agent import ClaudeCodeAgentRunRequest


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
