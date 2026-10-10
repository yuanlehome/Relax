#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
: "${NEMO_GYM_ROOT:?Set the pinned NeMo Gym checkout}"
: "${NEMO_GYM_PYTHON:?Set the isolated NeMo Python executable}"
: "${GYM_HOST:?Set the address used by the node-local Gateway}"
: "${GYM_PORT:?Set an available Gateway port}"
: "${CLAUDE_CODE_DOCKER_BINARY:?Set the deployment's Docker CLI}"
: "${CLAUDE_CODE_DOCKER_HOST:?Set the deployment's private Docker endpoint}"
: "${CLAUDE_CODE_RESULT_ROOT:?Set a private directory for terminal contexts}"
: "${NEMO_GYM_COMMIT:?Set the verified NeMo source commit}"
: "${NEMO_GYM_TOKEN_FILE:?Set a private host-only Gateway token file}"
export PYTHONPATH="${RELAX_ROOT}:${NEMO_GYM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export GYM_CONCURRENCY="${GYM_CONCURRENCY:-512}"
cd "${RELAX_ROOT}"
exec "${NEMO_GYM_PYTHON}" -m uvicorn examples.claude_code_harbor.gym_service:create_service \
  --factory --host "${GYM_HOST}" --port "${GYM_PORT}" --workers 1
