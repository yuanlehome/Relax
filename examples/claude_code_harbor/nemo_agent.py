# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Extend pinned NeMo Gym's CC harness with per-task OCI and Harbor scoring."""

from __future__ import annotations

import asyncio
import hashlib
import os
import secrets
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from nemo_gym.base_responses_api_model import maybe_rollout_id_from_run_body
from pydantic import PrivateAttr
from responses_api_agents.claude_code_agent.app import (
    ClaudeCodeAgent,
    ClaudeCodeAgentConfig,
    ClaudeCodeAgentRunRequest,
    ClaudeCodeAgentVerifyResponse,
)

from examples.claude_code_harbor.oci import DockerSandbox
from examples.claude_code_harbor.relay import LocalModelRelay
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


@dataclass
class TaskContext:
    sandbox: DockerSandbox
    relay: LocalModelRelay
    api_key: str


_TASK: ContextVar[TaskContext] = ContextVar("harbor_cc_task")


class HarborClaudeCodeAgent(ClaudeCodeAgent):
    """Keep NeMo CC response parsing; put its CLI and verifier in the task
    OCI."""

    config: ClaudeCodeAgentConfig
    _trials: dict[str, asyncio.Task[Any]] = PrivateAttr(default_factory=dict)
    _cancelled: set[str] = PrivateAttr(default_factory=set)
    _sandboxes: dict[str, DockerSandbox] = PrivateAttr(default_factory=dict)
    _cleanups: dict[str, asyncio.Task[None]] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        # Source CC is installed from the task's pinned assets inside the OCI;
        # upstream's host installation would run in the wrong environment.
        self.sem = asyncio.Semaphore(self.config.concurrency)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/cleanup/{rollout_id}")(self.cleanup_rollout)
        app.get("/cleanup/{rollout_id}")(self.probe_cleanup)
        return app

    async def cleanup_rollout(self, rollout_id: str) -> dict[str, Any]:
        first_abort = rollout_id not in self._cancelled
        self._cancelled.add(rollout_id)
        task = self._trials.get(rollout_id)
        if task is not None:
            if first_abort and rollout_id not in self._cleanups:
                task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), 30)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception:
                logger.warning("Aborted CC run failed during cleanup: %s", rollout_id)
        if rollout_id not in self._trials and rollout_id in self._sandboxes:
            try:
                await asyncio.wait_for(asyncio.shield(self._ensure_cleanup(rollout_id)), 30)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                logger.warning("OCI cleanup is not yet confirmed: %s", rollout_id)
        return await self.probe_cleanup(rollout_id)

    async def probe_cleanup(self, rollout_id: str) -> dict[str, Any]:
        return {
            "rollout_id": rollout_id,
            "clean": rollout_id not in self._trials and rollout_id not in self._sandboxes,
        }

    def _ensure_cleanup(self, rollout_id: str) -> asyncio.Task[None]:
        cleanup = self._cleanups.get(rollout_id)
        if cleanup is None or cleanup.done():
            if cleanup is not None and not cleanup.cancelled():
                cleanup.exception()
            cleanup = asyncio.create_task(self._clean_sandbox(rollout_id))
            self._cleanups[rollout_id] = cleanup
        return cleanup

    async def _clean_sandbox(self, rollout_id: str) -> None:
        sandbox = self._sandboxes[rollout_id]
        # Stop all container processes before reading agent-writable logs.
        await sandbox.stop()
        try:
            artifact_root = os.environ.get("CLAUDE_CODE_ARTIFACT_ROOT")
            if artifact_root:
                destination = Path(artifact_root) / hashlib.sha256(rollout_id.encode()).hexdigest()
                destination.mkdir(parents=True, exist_ok=True)
                for relative in (
                    "agent/setup.log",
                    "agent/claude-code.txt",
                    "verifier/test-output.txt",
                    "verifier/reward.txt",
                ):
                    try:
                        text = sandbox.read_log(relative)
                        (destination / Path(relative).name).write_text(text)
                    except (OSError, ValueError):
                        logger.warning("Cannot capture private OCI artifact %s", relative)
        except OSError:
            logger.warning("Cannot create private OCI artifact directory", exc_info=True)
        finally:
            await sandbox.close()
        self._sandboxes.pop(rollout_id, None)
        self._cleanups.pop(rollout_id, None)

    async def _run_claude_code(
        self,
        instruction: str,
        system_prompt: str | None = None,
        mcp_config: str | None = None,
        skills_path: str | None = None,
        rollout_id: str | None = None,
    ) -> tuple[str, str]:
        if system_prompt or mcp_config or skills_path:
            raise ValueError("Source Harbor CC uses its native tools and task instruction only")
        context = _TASK.get()
        metadata = context.sandbox.metadata
        model = metadata["backend_options"]["model_name"]
        env = {str(k): str(v) for k, v in metadata["sandbox_extra_params"].get("nex_envs", {}).items()}
        env.update(
            ANTHROPIC_BASE_URL=context.relay.base_url,
            ANTHROPIC_AUTH_TOKEN=context.api_key,
            ANTHROPIC_API_KEY=context.api_key,
            ANTHROPIC_MODEL=model,
            CLAUDE_CODE_SUBAGENT_MODEL=model,
            CLAUDE_CODE_AUTO_COMPACT_WINDOW="200000",
            CLAUDE_CODE_MAX_OUTPUT_TOKENS="8192",
            API_TIMEOUT_MS="2147483647",
            FORCE_AUTO_BACKGROUND_TASKS="1",
            ENABLE_BACKGROUND_TASKS="1",
            CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
            IS_SANDBOX="1",
            CLAUDE_CONFIG_DIR="/logs/agent/sessions",
        )
        for alias in ("SONNET", "OPUS", "HAIKU"):
            env[f"ANTHROPIC_DEFAULT_{alias}_MODEL"] = model
        env.pop("DISABLE_COMPACT", None)
        task = asyncio.create_task(
            context.sandbox.request(
                "agent",
                None,
                env=env,
                workdir=metadata["backend_options"]["workdir"],
                instruction=instruction,
                disallowed_tools=metadata["backend_options"]["agent_kwargs"]["disallowed_tools"],
            )
        )
        try:
            while not task.done():
                if context.relay.terminal_reason is not None:
                    context.sandbox.stop_agent()
                await asyncio.sleep(0.2)
            result = await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        if result["returncode"] != 0 and context.relay.terminal_reason not in {"max_steps", "context_overflow"}:
            raise RuntimeError("Claude Code failed; no training reward may be fabricated")
        stdout = context.sandbox.read_log("agent/claude-code.txt")
        return stdout, model

    async def run(self, request: Request, body: ClaudeCodeAgentRunRequest) -> ClaudeCodeAgentVerifyResponse:
        # Gateway callback identity is mandatory even when NeMo's optional
        # observability capture is disabled. Register before waiting for a slot.
        rollout_id = maybe_rollout_id_from_run_body(body)
        if not rollout_id or rollout_id in self._trials or rollout_id in self._cancelled:
            raise ValueError("A unique, live NeMo rollout identity is required")
        task = asyncio.current_task()
        assert task is not None
        self._trials[rollout_id] = task
        try:
            async with self.sem:
                if rollout_id in self._cancelled:
                    raise asyncio.CancelledError
                return await self._run_task(body, rollout_id)
        finally:
            self._trials.pop(rollout_id, None)

    async def _run_task(self, body: ClaudeCodeAgentRunRequest, rollout_id: str) -> ClaudeCodeAgentVerifyResponse:
        metadata = dict(body.model_extra or {})
        if metadata.get("reward_fn") != "harbor_verifier" or metadata.get("blackbox_type") != "harbor_agent":
            raise ValueError("This NeMo CC environment requires source Harbor routing")
        sandbox = DockerSandbox(metadata)
        self._sandboxes[rollout_id] = sandbox
        relay = None
        token = None
        try:
            api_key = secrets.token_urlsafe(32)
            relay = LocalModelRelay(self._resolve_call_base_url(rollout_id), api_key, normalization_id=rollout_id)
            token = _TASK.set(TaskContext(sandbox, relay, api_key))
            relay.start()
            await sandbox.start()
            response = await self._create_response(body.responses_create_params, rollout_id=rollout_id)
            relay.close()
            verification = await sandbox.verify()
            reward = verification["reward"]
            records = relay.capture.export(
                reward,
                {
                    "instance_id": metadata["instance_id"],
                    "dataset": metadata["dataset"],
                    "termination_reason": relay.terminal_reason or "completed",
                },
            )
            logger.info("NeMo Harbor CC task=%s contexts=%s reward=%s", metadata["instance_id"], len(records), reward)
            return ClaudeCodeAgentVerifyResponse.model_validate(
                {
                    **body.model_dump(),
                    "response": response.model_dump(mode="json"),
                    "reward": reward,
                    "metadata": {"contexts": records},
                }
            )
        finally:
            if relay is not None:
                relay.close()
            if token is not None:
                _TASK.reset(token)
            # This independently owned task survives repeated endpoint/client
            # cancellation. Failed removal retains the sandbox for retry.
            await asyncio.shield(self._ensure_cleanup(rollout_id))


if __name__ == "__main__":
    HarborClaudeCodeAgent.run_webserver()
