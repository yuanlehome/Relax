# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Deploy the real NeMo CC harness and its Relax Gateway in one process.

This stateless HTTP deployment does not join the Relax Ray cluster. Each /run
owns a separate OCI environment; no training code executes in that container.
"""

from __future__ import annotations

import hmac
import json
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.server_utils import ServerClient
from omegaconf import OmegaConf
from responses_api_agents.claude_code_agent.app import ClaudeCodeAgentConfig

from examples.claude_code_harbor.nemo_agent import HarborClaudeCodeAgent
from examples.nemo_gym_agentic.app.client import gateway_auth_headers
from examples.nemo_gym_agentic.service.app import create_app
from examples.nemo_gym_agentic.service.config import GatewaySettings
from examples.nemo_gym_agentic.service.run_adapter import HttpNemoGymRunAdapter


def create_service() -> FastAPI:
    host = os.environ["GYM_HOST"]
    port = int(os.environ["GYM_PORT"])
    concurrency = int(os.environ.get("GYM_CONCURRENCY", "512"))
    max_turns = int(os.environ.get("HARBOR_MAX_TURNS", "200"))
    if max_turns <= 0:
        raise ValueError("HARBOR_MAX_TURNS must be a positive integer")
    auth_headers = gateway_auth_headers(os.environ["NEMO_GYM_TOKEN_FILE"])
    management_token = auth_headers["Authorization"][7:].encode("ascii")
    for name in ("CLAUDE_CODE_DOCKER_BINARY", "CLAUDE_CODE_DOCKER_HOST"):
        if not os.environ.get(name):
            raise ValueError(f"{name} must identify the deployment's OCI runtime")
    global_config = OmegaConf.create(
        {
            "policy_model": {"responses_api_models": {"relax_gateway_model": {"host": host, "port": port}}},
        }
    )
    config = ClaudeCodeAgentConfig(
        name="claude_code_harbor",
        entrypoint="nemo_agent.py",
        host=host,
        port=port,
        resources_server={"type": "resources_servers", "name": "harbor_verifier"},
        model_server={"type": "responses_api_models", "name": "policy_model"},
        model="anthropic/proxy-model",
        concurrency=concurrency,
        max_turns=max_turns,
        bare=False,
    )
    agent = HarborClaudeCodeAgent(
        config=config,
        server_client=ServerClient(
            head_server_config=BaseServerConfig(host=host, port=port),
            global_config_dict=global_config,
        ),
    )
    cc_app = agent.setup_webserver()
    agent.setup_liveness(cc_app)
    agent.setup_exception_middleware(cc_app)

    @cc_app.get("/readyz")
    async def cc_ready() -> dict[str, bool]:
        return {"ready": True}

    values = dict(os.environ)
    values["NEMO_GYM_GATEWAY_ENVIRONMENTS_JSON"] = json.dumps(
        {
            "claude-code-harbor-oci-v1": {
                "environment": "harbor_cc",
                "agent_name": config.name,
                "agent_url": f"http://{host}:{port}/cc",
                "readiness_urls": [f"http://{host}:{port}/cc/readyz"],
                "abort_url": f"http://{host}:{port}/cc/cleanup/{{rollout_id}}",
                "force_cleanup_url": f"http://{host}:{port}/cc/cleanup/{{rollout_id}}",
                "cleanup_probe_url": f"http://{host}:{port}/cc/cleanup/{{rollout_id}}",
                "interrupt_policy": "protected",
                "max_concurrency": concurrency,
                "queue_capacity": concurrency * 4,
                "max_deadline_s": 2147483,
            },
        }
    )
    values.setdefault("NEMO_GYM_CALLBACK_TIMEOUT_S", "2147483")
    result_root = os.environ.get("CLAUDE_CODE_RESULT_ROOT")
    settings = GatewaySettings.from_env(values)
    app = create_app(
        settings=settings,
        adapter=HttpNemoGymRunAdapter(
            settings.environments.values(), artifact_root=settings.artifact_root, headers=auth_headers
        ),
        terminal_metadata_dir=Path(result_root) if result_root else None,
        terminal_retention_s=3600,
    )

    @app.middleware("http")
    async def require_management_auth(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        path = request.url.path
        protected = path.startswith("/v1/trials") or (
            (path == "/cc" or path.startswith("/cc/")) and path.rstrip("/") != "/cc/readyz"
        )
        if protected:
            scheme, _, supplied = request.headers.get("Authorization", "").partition(" ")
            if scheme.lower() != "bearer" or not hmac.compare_digest(supplied.encode("utf-8"), management_token):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Management authentication required"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
        return await call_next(request)

    app.mount("/cc", cc_app)
    app.state.cc_agent = agent
    return app
