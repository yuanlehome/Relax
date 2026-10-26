# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Extend pinned NeMo Gym's CC harness with per-task OCI and Harbor scoring."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import secrets
import signal
import time
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

from examples.claude_code_harbor.cli_lifecycle import is_expected_turn_limit
from examples.claude_code_harbor.oci import DockerSandbox
from examples.claude_code_harbor.relay import LocalModelRelay
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


@dataclass
class TaskContext:
    sandbox: DockerSandbox
    relay: LocalModelRelay
    api_key: str
    backend_timed_out: bool = False
    max_turns: int | None = None
    num_turns: int | None = None
    max_turns_reached: bool = False
    first_limit_result: dict[str, Any] | None = None
    cli_returncode: int | None = None
    prompt_encoding: str = "plain"


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
        deadline = time.monotonic() + 30
        first_abort = rollout_id not in self._cancelled
        self._cancelled.add(rollout_id)
        task = self._trials.get(rollout_id)
        if task is not None:
            if first_abort and rollout_id not in self._cleanups:
                task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), max(0, deadline - time.monotonic()))
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception:
                logger.warning("Aborted CC run failed during cleanup: %s", rollout_id)
        if rollout_id not in self._trials and rollout_id in self._sandboxes:
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._ensure_cleanup(rollout_id)), max(0, deadline - time.monotonic())
                )
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                logger.warning("OCI cleanup is not yet confirmed: %s", rollout_id)
        return await self.probe_cleanup(rollout_id)

    async def probe_cleanup(self, rollout_id: str) -> dict[str, Any]:
        return {
            "rollout_id": rollout_id,
            "clean": rollout_id not in self._trials and rollout_id not in self._sandboxes,
        }

    def _ensure_cleanup(self, rollout_id: str, relay: LocalModelRelay | None = None) -> asyncio.Task[None]:
        cleanup = self._cleanups.get(rollout_id)
        if cleanup is None or cleanup.done():
            if cleanup is not None and not cleanup.cancelled():
                cleanup.exception()
            cleanup = asyncio.create_task(self._clean_sandbox(rollout_id, relay))
            self._cleanups[rollout_id] = cleanup
        return cleanup

    async def _clean_sandbox(self, rollout_id: str, relay: LocalModelRelay | None = None) -> None:
        sandbox = self._sandboxes[rollout_id]
        try:
            if relay is not None:
                await relay.aclose()
        finally:
            # The independent cleanup task owns both shutdowns even if the
            # request endpoint is cancelled repeatedly.
            await sandbox.stop()
            try:
                await asyncio.to_thread(self._capture_artifacts, sandbox, rollout_id)
            except OSError:
                logger.warning("Cannot create private OCI artifact directory", exc_info=True)
            finally:
                await sandbox.close()
        self._sandboxes.pop(rollout_id, None)
        self._cleanups.pop(rollout_id, None)

    @staticmethod
    def _capture_artifacts(sandbox: DockerSandbox, rollout_id: str) -> None:
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
            lifecycle = sandbox.root / "cli-lifecycle.jsonl"
            if lifecycle.is_file():
                (destination / lifecycle.name).write_text(lifecycle.read_text())
            # Host-side identity joins CLI / verifier evidence to Docker events.
            metadata = sandbox.metadata
            (destination / "trial.json").write_text(
                json.dumps(
                    {
                        "rollout_id": rollout_id,
                        "container_name": sandbox.name,
                        **{
                            key: metadata.get(key)
                            for key in (
                                "instance_id",
                                "dataset",
                                "sandbox_memory",
                                "sandbox_cpu",
                                "verifier_timeout",
                                "harbor_backend_timeout_s",
                                "harbor_max_turns",
                            )
                        },
                    },
                    indent=2,
                )
                + "\n"
            )

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
        max_turns = metadata.get(
            "harbor_max_turns", metadata["backend_options"]["agent_kwargs"].get("max_turns", self.config.max_turns)
        )
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns <= 0:
            raise ValueError("harbor_max_turns must be a positive integer")
        context.max_turns = max_turns
        context.prompt_encoding = "literal_prefix" if instruction.lstrip().startswith("/") else "plain"
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
        timeout = metadata.get("harbor_backend_timeout_s")
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("Harbor backend timeout must be a finite positive number")
        # Match the frozen business backend-call boundary: sandbox setup and
        # verifier time are outside this wall-clock budget.
        deadline = None if timeout is None else time.monotonic() + timeout
        task = asyncio.create_task(
            context.sandbox.request(
                "agent",
                None,
                env=env,
                workdir=metadata["backend_options"]["workdir"],
                instruction=instruction,
                disallowed_tools=metadata["backend_options"]["agent_kwargs"]["disallowed_tools"],
                max_turns=max_turns,
            )
        )
        candidate: tuple[str, int] | None = None
        cleanup_ready = False
        cleanup_permitted = False
        cleanup_lease = 0.0
        try:
            while not task.done():
                events = getattr(context.sandbox, "agent_progress", [])
                while events:
                    event = events.pop(0)
                    if event["phase"] == "max_turns_reached" and is_expected_turn_limit(
                        event.get("cli_result"), max_turns
                    ):
                        context.first_limit_result = dict(event["cli_result"])
                        # Seal admissions before asking the private controller
                        # to stop the CLI; accepted responses must be harvested.
                        context.relay.begin_harvest()
                        context.sandbox.stop_agent()
                        break
                    if event["phase"] in {"candidate_cancelled", "cleanup_finished"}:
                        context.sandbox.revoke_tool_cleanup()
                        context.relay.end_tool_cleanup()
                        candidate, cleanup_ready, cleanup_permitted = None, False, False
                    elif event["phase"] == "terminal_candidate":
                        if cleanup_permitted:
                            context.sandbox.revoke_tool_cleanup()
                            context.relay.end_tool_cleanup()
                        cleanup_permitted = False
                        activity = context.relay.model_activity(quiet_since=event["monotonic"])
                        candidate = (event["generation"], activity) if activity is not None else None
                        cleanup_ready = False
                    elif event["phase"] == "cleanup_ready" and candidate and candidate[0] == event["generation"]:
                        cleanup_ready = True
                if context.first_limit_result is not None:
                    break
                if cleanup_permitted and time.monotonic() >= cleanup_lease:
                    context.sandbox.revoke_tool_cleanup()
                    context.relay.end_tool_cleanup()
                    cleanup_permitted = False
                if candidate and cleanup_ready and not cleanup_permitted:
                    lease = context.relay.begin_tool_cleanup(candidate[1])
                    if lease is not None:
                        context.sandbox.permit_tool_cleanup(candidate[0], lease)
                        cleanup_lease = lease
                        cleanup_permitted = True
                if deadline is not None and time.monotonic() >= deadline:
                    # Fence before killing the CLI. An accepted model call must
                    # survive its client's disconnect and commit canonically.
                    context.relay.begin_harvest()
                    context.backend_timed_out = True
                    context.sandbox.stop_agent()
                    break
                if context.relay.terminal_reason is not None:
                    context.sandbox.stop_agent()
                await asyncio.sleep(0.2)
            result = await task
            if context.backend_timed_out:
                await context.relay.finish_harvest()
                return "", model
            receipt = result.get("first_limit_result")
            if context.first_limit_result is None and is_expected_turn_limit(receipt, max_turns):
                # A fast exit can complete before progress is polled. Preserve
                # the private stdout receipt and fence before exporting capture.
                context.first_limit_result = dict(receipt)
                context.relay.begin_harvest()
            if context.first_limit_result is not None:
                await context.relay.finish_harvest()
        finally:
            if cleanup_permitted:
                context.sandbox.revoke_tool_cleanup()
                context.relay.end_tool_cleanup()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        cli_result = result.get("cli_result") or {}
        context.cli_returncode = result["returncode"]
        num_turns = cli_result.get("num_turns")
        if isinstance(num_turns, int) and not isinstance(num_turns, bool) and num_turns >= 0:
            context.num_turns = num_turns
        stop_signal = result.get("host_stop_signal")
        controlled_stop = (
            type(stop_signal) is int
            and stop_signal in {int(signal.SIGTERM), int(signal.SIGKILL)}
            and result["returncode"] == -stop_signal
        )
        context.max_turns_reached = (result["returncode"] == 1 and is_expected_turn_limit(cli_result, max_turns)) or (
            context.first_limit_result is not None and (result["returncode"] in {0, 1} or controlled_stop)
        )
        if (
            result["returncode"] != 0
            and not context.max_turns_reached
            and context.relay.terminal_reason not in {"max_steps", "context_overflow"}
        ):
            raise RuntimeError(
                f"Claude Code failed (exit_code={result['returncode']}, turns={context.num_turns}); "
                "no training reward may be fabricated"
            )
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
            context = TaskContext(sandbox, relay, api_key)
            token = _TASK.set(context)
            relay.start()
            await sandbox.start()
            response = await self._create_response(body.responses_create_params, rollout_id=rollout_id)
            await relay.aclose()
            if context.backend_timed_out:
                reward = 0.0
            else:
                verification = await sandbox.verify()
                reward = verification["reward"]
            records = relay.capture.export(
                reward,
                {
                    "instance_id": metadata["instance_id"],
                    "dataset": metadata["dataset"],
                    "termination_reason": "backend_timeout"
                    if context.backend_timed_out
                    else "max_turns"
                    if context.max_turns_reached
                    else relay.terminal_reason or "completed",
                    "max_turns": context.max_turns,
                    "num_turns": context.num_turns,
                    "first_max_turns_result": context.first_limit_result,
                    "cli_returncode": context.cli_returncode,
                    "prompt_encoding": context.prompt_encoding,
                },
            )
            logger.info("NeMo Harbor CC task=%s contexts=%s reward=%s", metadata["instance_id"], len(records), reward)
            return ClaudeCodeAgentVerifyResponse.model_validate(
                {
                    **body.model_dump(),
                    "response": response.model_dump(mode="json"),
                    "reward": reward,
                    "truncated": context.backend_timed_out or context.max_turns_reached,
                    "metadata": {"contexts": records},
                }
            )
        finally:
            if token is not None:
                _TASK.reset(token)
            # This independently owned task survives repeated endpoint/client
            # cancellation. Failed removal retains the sandbox for retry.
            await asyncio.shield(self._ensure_cleanup(rollout_id, relay))


if __name__ == "__main__":
    HarborClaudeCodeAgent.run_webserver()
