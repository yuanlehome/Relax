# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Check Harbor service construction without starting Ray, OCI or Claude
Code."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import secrets
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import httpx
import nemo_gym
import responses_api_agents.claude_code_agent.app as cc_module

from examples.claude_code_harbor.gym_service import create_service


async def check(prefix: Path) -> None:
    for module in (nemo_gym, cc_module):
        if not Path(module.__file__).resolve().is_relative_to((prefix / "nemo-gym").resolve()):
            raise RuntimeError("A Gym import resolved outside the newly built source")
    for name in ("torch", "ray"):
        spec = importlib.util.find_spec(name)
        if spec is None or Path(spec.origin).resolve().is_relative_to((prefix / "nemo-venv").resolve()):
            raise RuntimeError("Training packages must be inherited from the base image")
    with tempfile.TemporaryDirectory(prefix="harbor-build-check-") as directory:
        token = Path(directory) / "token"
        token.touch(mode=0o600)
        token.write_text(secrets.token_urlsafe(32))
        env = {
            "GYM_HOST": "127.0.0.1",
            "GYM_PORT": "27980",
            "GYM_CONCURRENCY": "1",
            "NEMO_GYM_TOKEN_FILE": str(token),
            "NEMO_GYM_COMMIT": "a85670eb167ba9b48cc53a36a070eed815e6c40d",
            "NEMO_GYM_CALLBACK_ALLOWED_HOSTS": "127.0.0.1",
            "CLAUDE_CODE_DOCKER_BINARY": "/unused/docker",
            "CLAUDE_CODE_DOCKER_HOST": "unix:///unused/docker.sock",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch(
                "responses_api_agents.claude_code_agent.app.ensure_claude_code",
                side_effect=AssertionError("Harbor must not install Claude Code on the Gym host"),
            ),
        ):
            app = create_service()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gym") as client:
                if (await client.get("/cc/readyz")).status_code != 200:
                    raise RuntimeError("Harbor readiness check failed")
                if (await client.get("/cc/cleanup/check")).status_code != 401:
                    raise RuntimeError("Harbor must reject unauthenticated management calls")
                response = await client.get(
                    "/cc/cleanup/check", headers={"Authorization": "Bearer " + token.read_text()}
                )
                if response.status_code != 200 or response.json()["clean"] is not True:
                    raise RuntimeError("Harbor authenticated cleanup probe failed")
            await app.state.registry._adapter.close()


if __name__ == "__main__":
    asyncio.run(check(Path(sys.argv[1])))
