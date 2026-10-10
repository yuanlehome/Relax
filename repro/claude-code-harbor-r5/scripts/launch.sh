#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Submit the r5 recipe only on a dedicated reproduction cluster.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="${BUNDLE_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
REPO="${REPO:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
: "${MODEL_DIR:?Set the cloud root containing Qwen3.6-35B-A3B/}"
: "${DATA_ROOT:?Set the prepared cloud task and asset root}"
: "${RAY_ADDRESS:?Set the reproduction Ray GCS address, head-IP:6379}"
: "${RAY_DASHBOARD_ADDRESS:?Set the reproduction Ray Jobs HTTP address}"
: "${EXP_NAME:?Set a stable NEW experiment name, or deliberately reuse it to resume}"
if [[ ! "${EXP_NAME}" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
  echo 'EXP_NAME must be a plain stable directory name' >&2; exit 2
fi
export RELAX="${REPO}"
export MEGATRON="${MEGATRON:-/root/Megatron-LM}"
test -d "${MEGATRON}/megatron" || { echo "Image Megatron checkout is missing" >&2; exit 2; }
export MODEL_DIR
export DATA_DIR="${DATA_ROOT}/aligned-150037-oci"
export SAVE_DIR="${SAVE_DIR:-${BUNDLE_DIR}/checkpoints}"
export PROJECT_NAME="${PROJECT_NAME:-Relax/agentic/claude-code-harbor}"
export NEMO_GYM_URL="${NEMO_GYM_URL:-http://127.0.0.1:${GYM_PORT:-27980}}"
export NEMO_GYM_ENVIRONMENT=harbor_cc
export NEMO_GYM_CONFIG=claude-code-harbor-oci-v1
export NEMO_GYM_TOKEN_FILE="${BUNDLE_DIR}/private/gateway.token"
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
export PYTHONPATH="${RELAX}:${MEGATRON}${PYTHONPATH:+:${PYTHONPATH}}"
unset RELAX_ENTRYPOINT_MODE
mkdir -p "${SAVE_DIR}"
# Historical archive checks remain available through verify_bundle.py. A fresh
# source build has no archived environment/code manifest to compare against.
python3 "${SCRIPT_DIR}/verify_inputs.py" --data-root "${DATA_ROOT}" \
  --model-dir "${MODEL_DIR}/Qwen3.6-35B-A3B"
test -s "${NEMO_GYM_TOKEN_FILE}"
curl --noproxy '*' -fsS "${NEMO_GYM_URL}/readyz" >/dev/null
cd "${RELAX}"
exec bash "${RELAX}/scripts/entrypoint/ray-job.sh" \
  "${RELAX}/examples/claude_code_harbor/scripts/run-qwen36-35B-A3B-128xgpu.sh" "$@"
