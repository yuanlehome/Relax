# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run one Relax Session as one Harbor Claude Code trial."""

from __future__ import annotations

import asyncio
import json
import os
import signal
from pathlib import Path
from typing import Any

import yaml

from examples.claude_code_harbor.proxy import LocalModelProxy


os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


def _atomic_output(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8"
    )
    temporary.replace(path)


def _trial_config(task: Path, session_dir: Path, base_url: str) -> Any:
    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        ResourceMode,
        TaskConfig,
        TrialConfig,
        VerifierConfig,
    )

    config = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())
    session_id = os.environ["RELAX_SESSION_ID"]
    agent = AgentConfig(
        name="claude-code",
        model_name="anthropic/relax_model",
        model_base_url=base_url,
        model_api_key_name="ANTHROPIC_AUTH_TOKEN",
        override_timeout_sec=config["task"]["agent_timeout_sec"],
        env={
            "ANTHROPIC_AUTH_TOKEN": session_id,
            "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK": "1",
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": os.environ["MAX_OUTPUT_TOKENS"],
            "CLAUDE_CODE_MAX_RETRIES": "0",
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(config["claude_code"]["max_context_tokens"]),
            "CLAUDE_CODE_MODEL_CAPABILITIES": "-mid_conv_system",
            "CLAUDE_CODE_SHELL": "/bin/bash",
            "CLAUDE_CODE_SIMPLE_SYSTEM_PROMPT": "1",
            "CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS": "1",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
            "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
            "CLAUDE_CODE_DISABLE_CRON": "1",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
            "CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS": "1",
            "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1",
            "CLAUDE_CODE_DISABLE_WORKFLOWS": "1",
            "DISABLE_UPDATES": "1",
            "API_TIMEOUT_MS": "2147483647",
            "CLAUDE_STREAM_IDLE_TIMEOUT_MS": "2147483647",
        },
        kwargs={
            "max_turns": config["claude_code"]["max_turns"],
            "permission_mode": "dontAsk",
            "allowed_tools": config["claude_code"]["allowed_tools"],
            "disallowed_tools": config["claude_code"]["disallowed_tools"],
            "config": {
                "autoCompactEnabled": True,
                "skillOverrides": config["claude_code"]["skill_overrides"],
            },
        },
    )
    environment = EnvironmentConfig(
        import_path="examples.claude_code_harbor.harbor_environment:BoundedApptainerEnvironment",
        cpu_enforcement_policy=ResourceMode.LIMIT,
        memory_enforcement_policy=ResourceMode.LIMIT,
        override_cpus=config["task"]["cpus"],
        override_memory_mb=config["task"]["memory_mb"],
        kwargs={"pids_limit": config["apptainer"]["pids_limit"]},
    )
    return TrialConfig(
        task=TaskConfig(path=task),
        trial_name="trial",
        trials_dir=session_dir / "harbor-trials",
        agent=agent,
        environment=environment,
        verifier=VerifierConfig(override_timeout_sec=config["task"]["verifier_timeout_sec"]),
    )


async def _run_trial(config: Any, stop: asyncio.Event) -> Any:
    from harbor.trial.hooks import TrialEvent
    from harbor.trial.trial import Trial

    trial = await Trial.create(config)

    async def before_verification(_event):
        await trial.agent_environment.prepare_verification()

    trial.add_hook(TrialEvent.VERIFICATION_START, before_verification)
    task = asyncio.create_task(trial.run(), name=f"harbor-trial:{config.trial_name}")
    waiter = asyncio.create_task(stop.wait(), name=f"harbor-stop:{config.trial_name}")
    try:
        done, _ = await asyncio.wait((task, waiter), return_when=asyncio.FIRST_COMPLETED)
        if waiter in done and not task.done():
            task.cancel()
        result = await task
        if stop.is_set():
            raise asyncio.CancelledError
        return result
    finally:
        waiter.cancel()
        if not task.done():
            task.cancel()
        await asyncio.gather(waiter, task, return_exceptions=True)


async def main_async() -> None:
    input_path = Path(os.environ["RELAX_INPUT_JSON"])
    output_path = Path(os.environ["RELAX_OUTPUT_JSON"])
    session_dir = Path(os.environ["RELAX_SESSION_IO_DIR"]).resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    metadata = payload["metadata"]["harbor"]
    task_root = Path(os.environ["HARBOR_TASK_ROOT"]).resolve()
    task = task_root / metadata["task"]
    mode = os.environ["RELAX_ROLLOUT_MODE"]

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for value in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(value, stop.set)

    parent_pid = os.getppid()

    async def watch_parent() -> None:
        while os.getppid() == parent_pid:
            await asyncio.sleep(0.2)
        stop.set()

    parent_watch = asyncio.create_task(watch_parent(), name=f"relax-parent:{parent_pid}")

    proxy = LocalModelProxy(os.environ["RELAX_BASE_URL"], os.environ["RELAX_SESSION_ID"])
    proxy.start()

    try:
        result = await _run_trial(_trial_config(task, session_dir, proxy.base_url), stop)
        info = result.exception_info
        if info is not None and info.exception_type == "VerifierTimeoutError":
            reward = 0.0
        elif result.verifier_result is None:
            raise RuntimeError(
                f"Harbor Trial failed: {info.exception_type}: {info.exception_message}\n{info.exception_traceback}"
            )
        else:
            reward = result.verifier_result.rewards["reward"]
    finally:
        parent_watch.cancel()
        await asyncio.gather(parent_watch, return_exceptions=True)
        proxy.close()
    records = proxy.capture.export(reward)
    if mode == "eval":
        records = records[-1:]
    records[-1]["reward"] = reward
    _atomic_output(output_path, records)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
