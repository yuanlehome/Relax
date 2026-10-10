# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real NeMo CC + source OCI/SRT/Harbor plumbing check with a stub model.

Run with the prepared NeMo Python environment. This does not validate Qwen's
reasoning/tool parser, reward variance, GPU training, or optimizer updates.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import socket
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request

from examples.claude_code_harbor.gym_service import create_service
from examples.nemo_gym_agentic.app.client import ClientConfig, GatewayClient, build_trial_request, run_trial
from examples.nemo_gym_agentic.app.result import to_relax_output, write_relax_output
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


async def verify(dataset: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    token_path = output / "gateway.token"
    token_path.touch(mode=0o600, exist_ok=False)
    token_path.write_text(secrets.token_urlsafe(32))
    gym_port, model_port = free_port(), free_port()
    os.environ.update(
        GYM_HOST="127.0.0.1",
        GYM_PORT=str(gym_port),
        GYM_CONCURRENCY="1",
        NEMO_GYM_CALLBACK_ALLOWED_HOSTS="127.0.0.1",
        NEMO_GYM_COMMIT="a85670eb167ba9b48cc53a36a070eed815e6c40d",
        CLAUDE_CODE_ARTIFACT_ROOT=str(output / "artifacts"),
        CLAUDE_CODE_RESULT_ROOT=str(output / "results"),
        NEMO_GYM_TOKEN_FILE=str(token_path),
    )
    gym_app = create_service()
    model_app = FastAPI()
    calls: list[dict[str, Any]] = []
    credential = secrets.token_urlsafe(32)

    @model_app.post("/agentic_api/v1/messages")
    async def messages(request: Request) -> dict[str, Any]:
        assert request.headers.get("Authorization") == f"Bearer {credential}"
        payload = await request.json()
        assert payload["stream"] is False
        calls.append(payload)
        (output / "model-requests.json").write_text(json.dumps(calls, ensure_ascii=False))
        if len(calls) == 1:
            content = [
                {
                    "type": "tool_use",
                    "id": "toolu_relax_oci_probe",
                    "name": "Bash",
                    "input": {
                        "command": "pwd; printf 'RELAX_OCI_TOOL_OK\\n'; test ! -e /run/secrets/kubernetes.io/serviceaccount/token; cat /tmp/mimo_git_targets.json",
                        "description": "Check task environment and protected verifier assets",
                    },
                }
            ]
            stop = "tool_use"
            # Exercise the source CLI's auto-compaction path, without sending
            # hundreds of thousands of fixture tokens through the test model.
            input_tokens = 200000
        else:
            content = [
                {
                    "type": "text",
                    "text": "Integration smoke completed. The tool ran in /testbed; verifier assets remain protected. Continue from this summary.",
                }
            ]
            stop = "end_turn"
            input_tokens = 1000
        return {
            "id": f"msg_smoke_{len(calls)}",
            "type": "message",
            "role": "assistant",
            "model": "proxy-model",
            "content": content,
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 100},
        }

    servers = [
        uvicorn.Server(uvicorn.Config(gym_app, host="127.0.0.1", port=gym_port, log_level="warning")),
        uvicorn.Server(uvicorn.Config(model_app, host="127.0.0.1", port=model_port, log_level="warning")),
    ]
    tasks = [asyncio.create_task(server.serve()) for server in servers]
    try:
        for _ in range(100):
            if all(server.started for server in servers):
                break
            if any(task.done() for task in tasks):
                raise RuntimeError("An owned verification server exited during startup")
            await asyncio.sleep(0.1)
        else:
            raise TimeoutError("Owned verification servers did not become ready")
        with dataset.open() as source:
            row = json.loads(next(line for line in source if line.strip()))
        metadata = dict(row["metadata"])
        for key in ("reward_fn", "blackbox_type", "agent_mode"):
            if key in row:
                metadata.setdefault(key, row[key])
        session_input = {"messages": row["prompt"], "metadata": metadata}
        environment = {
            "RELAX_SESSION_ID": credential,
            "RELAX_SESSION_IO_DIR": str(output),
            "RELAX_BASE_URL": f"http://127.0.0.1:{model_port}/agentic_api",
            "NEMO_GYM_ENVIRONMENT": "harbor_cc",
            "NEMO_GYM_CONFIG": "claude-code-harbor-oci-v1",
            "NEMO_GYM_MODEL": "Qwen3.6-35B-A3B",
            "NEMO_GYM_DEADLINE_S": "3600",
        }
        config = ClientConfig(gateway_url=f"http://127.0.0.1:{gym_port}", token_file=token_path)
        async with GatewayClient(config) as client:
            result = await run_trial(
                client=client,
                request=build_trial_request(session_input, environment),
                stop_event=asyncio.Event(),
                poll_interval_s=1,
            )
        write_relax_output(output / "contexts.jsonl", to_relax_output(result))
        exported = [json.loads(line) for line in (output / "contexts.jsonl").read_text().splitlines()]
        assert len(calls) >= 2, "CC did not complete a model/tool loop"
        assert any("RELAX_OCI_TOOL_OK" in json.dumps(call["messages"]) for call in calls[1:]), (
            "Bash result was not returned to the model"
        )
        assert exported and all(record["metadata"]["outcome_reward"] == result.reward for record in exported)
        artifacts = list((output / "artifacts").glob("*/claude-code.txt"))
        assert len(artifacts) == 1
        events = [json.loads(line) for line in artifacts[0].read_text().splitlines() if line.startswith("{")]
        boundaries = [event for event in events if event.get("subtype") == "compact_boundary"]
        assert boundaries and boundaries[0]["compact_metadata"]["trigger"] == "auto"
        assert len(exported) >= 2, "Compaction must preserve multiple committed training contexts"
        assert any("Permission denied" in json.dumps(call["messages"]) for call in calls[1:])
        assert gym_app.state.registry.stats()["active_trials"] == 0
        assert all(not record.metadata for record in gym_app.state.registry._records.values())
        assert not gym_app.state.cc_agent._trials
        evidence = {
            "status": "PASS",
            "instance_id": metadata["instance_id"],
            "model_calls": len(calls),
            "contexts": len(exported),
            "auto_compaction": boundaries[0]["compact_metadata"],
            "protected_verifier_read_denied": True,
            "reward": result.reward,
            "active_trials": 0,
            "gpu_model_validated": False,
        }
        (output / "status.json").write_text(json.dumps(evidence, indent=2))
        logger.info("NeMo CC / OCI / Harbor verification: %s", evidence)
    finally:
        for server in servers:
            server.should_exit = True
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(verify(args.dataset, args.output))
