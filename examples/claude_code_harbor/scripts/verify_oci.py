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
import stat
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, Request

from examples.nemo_gym_agentic.app.client import ClientConfig, GatewayClient, build_trial_request, run_trial
from examples.nemo_gym_agentic.app.result import explicit_context_records, to_relax_output, write_relax_output
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
MAX_LOG_BYTES = 16 * 1024 * 1024


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket() as listener:
        listener.bind((host, 0))
        return listener.getsockname()[1]


def artifact_tail(path: Path) -> str:
    """Read only the bounded tail of a regular, private service artifact."""
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Verification artifact must be a regular file")
        offset = max(0, info.st_size - MAX_LOG_BYTES)
        source.seek(offset)
        data = source.read(MAX_LOG_BYTES)
    if offset:
        # The bounded tail can begin in the middle of a JSON event.
        data = data.partition(b"\n")[2]
    return data.decode("utf-8", errors="replace")


async def external_active_trials(gateway_url: str) -> int | None:
    """Observe public health counts without claiming private registry
    cleanup."""
    import httpx

    try:
        async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
            response = await client.get(gateway_url.rstrip("/") + "/healthz")
            response.raise_for_status()
            active = response.json().get("active_trials")
        return active if type(active) is int and active >= 0 else None
    except (httpx.HTTPError, ValueError, AttributeError):
        return None


async def verify(
    dataset: Path,
    output: Path,
    *,
    gateway_url: str | None = None,
    token_file: Path | None = None,
    callback_host: str | None = None,
    artifact_root: Path | None = None,
) -> None:
    import uvicorn

    external = gateway_url is not None
    if external:
        endpoint = urlsplit(gateway_url)
        if endpoint.scheme not in {"http", "https"} or not endpoint.hostname or endpoint.username or endpoint.password:
            raise ValueError("External Gateway URL must be an absolute HTTP(S) URL without credentials")
        if token_file is None or not callback_host or artifact_root is None:
            raise ValueError("External verification requires token-file, callback-host and artifact-root")
        if artifact_root.is_symlink() or (
            not artifact_root.is_dir() and (artifact_root.exists() or not artifact_root.parent.is_dir())
        ):
            raise ValueError("External artifact-root must identify the persistent Gym artifact directory")
    elif token_file is not None or callback_host is not None or artifact_root is not None:
        raise ValueError("External verification options require gateway-url")
    os.umask(0o077)
    if output.is_symlink():
        raise ValueError("Verification output must be a private directory")
    if output.exists():
        for existing in output.iterdir():
            if (
                existing.name != "verify.log"
                or existing.is_symlink()
                or not existing.is_file()
                or existing.stat().st_mode & 0o077
            ):
                raise ValueError("Verification output already contains evidence; choose a fresh directory")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.chmod(0o700)
    gym_app = None
    servers = []
    if not external:
        from examples.claude_code_harbor.gym_service import create_service

        token_file = output / "gateway.token"
        token_file.touch(mode=0o600, exist_ok=False)
        token_file.write_text(secrets.token_urlsafe(32))
        gym_port = free_port()
        callback_host = "127.0.0.1"
        artifact_root = output / "artifacts"
        gateway_url = f"http://127.0.0.1:{gym_port}"
        os.environ.update(
            GYM_HOST="127.0.0.1",
            GYM_PORT=str(gym_port),
            GYM_CONCURRENCY="1",
            NEMO_GYM_CALLBACK_ALLOWED_HOSTS="127.0.0.1",
            NEMO_GYM_COMMIT="a85670eb167ba9b48cc53a36a070eed815e6c40d",
            CLAUDE_CODE_ARTIFACT_ROOT=str(artifact_root),
            CLAUDE_CODE_RESULT_ROOT=str(output / "results"),
            NEMO_GYM_TOKEN_FILE=str(token_file),
        )
        gym_app = create_service()
        servers.append(uvicorn.Server(uvicorn.Config(gym_app, host="127.0.0.1", port=gym_port, log_level="warning")))
    assert callback_host is not None and artifact_root is not None and gateway_url is not None
    previous_artifacts = set(artifact_root.glob("*/claude-code.txt"))
    model_port = free_port(callback_host)
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

    servers.append(uvicorn.Server(uvicorn.Config(model_app, host=callback_host, port=model_port, log_level="warning")))
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
            "RELAX_BASE_URL": f"http://{callback_host}:{model_port}/agentic_api",
            "NEMO_GYM_ENVIRONMENT": "harbor_cc",
            "NEMO_GYM_CONFIG": "claude-code-harbor-oci-v1",
            "NEMO_GYM_MODEL": "Qwen3.6-35B-A3B",
            "NEMO_GYM_DEADLINE_S": "3600",
        }
        config = ClientConfig(gateway_url=gateway_url, token_file=token_file)
        async with GatewayClient(config) as client:
            result = await run_trial(
                client=client,
                request=build_trial_request(session_input, environment),
                stop_event=asyncio.Event(),
                poll_interval_s=1,
            )
        payload = to_relax_output(result)
        write_relax_output(output / "contexts.jsonl", payload)
        exported = explicit_context_records(payload)
        assert len(calls) >= 2, "CC did not complete a model/tool loop"
        assert any("RELAX_OCI_TOOL_OK" in json.dumps(call["messages"]) for call in calls[1:]), (
            "Bash result was not returned to the model"
        )
        assert exported and all(record["metadata"]["outcome_reward"] == result.reward for record in exported)
        artifacts = set(artifact_root.glob("*/claude-code.txt")) - previous_artifacts
        assert len(artifacts) == 1, "Expected exactly one new verification artifact"
        artifact = artifacts.pop()
        artifact.resolve(strict=True).relative_to(artifact_root.resolve(strict=True))
        events = [json.loads(line) for line in artifact_tail(artifact).splitlines() if line.startswith("{")]
        boundaries = [event for event in events if event.get("subtype") == "compact_boundary"]
        assert boundaries and boundaries[0]["compact_metadata"]["trigger"] == "auto"
        assert len(exported) >= 2, "Compaction must preserve multiple committed training contexts"
        assert any("Permission denied" in json.dumps(call["messages"]) for call in calls[1:])
        if gym_app is not None:
            assert gym_app.state.registry.stats()["active_trials"] == 0
            assert all(not record.metadata for record in gym_app.state.registry._records.values())
            assert not gym_app.state.cc_agent._trials
            active_trials = 0
            cleanup = "in_process_verified"
        else:
            active_trials = await external_active_trials(gateway_url)
            assert active_trials in (None, 0), "Persistent Gym still reports active trials"
            cleanup = "unverifiable_externally"
        evidence = {
            "status": "PASS",
            "model_calls": len(calls),
            "contexts": len(exported),
            "auto_compaction_verified": True,
            "protected_verifier_read_denied": True,
            "callback_calls": len(calls),
            "active_trials": active_trials,
            "cleanup_verification": cleanup,
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
    parser.add_argument("--gateway-url", help="Use an existing persistent Gym; no Gym lifecycle operations")
    parser.add_argument("--token-file", type=Path, help="Existing private Gateway token, consumed only by the client")
    parser.add_argument("--callback-host", help="Actual caller IP for the owned callback stub")
    parser.add_argument("--artifact-root", type=Path, help="Shared CLAUDE_CODE_ARTIFACT_ROOT of the selected Gym")
    args = parser.parse_args()
    asyncio.run(
        verify(
            args.dataset,
            args.output,
            gateway_url=args.gateway_url,
            token_file=args.token_file,
            callback_host=args.callback_host,
            artifact_root=args.artifact_root,
        )
    )
