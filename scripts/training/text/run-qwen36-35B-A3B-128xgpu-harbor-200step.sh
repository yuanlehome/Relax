#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# 64 training GPUs + 32 rollout GPUs (4 prefill + 12 decode TP2 engines), with 128 GB/rank HiCache on prefill.
# Each rollout node runs 1P + 3D; 32 resident groups, with 16 * 16 engines = 256 shared model-call permits.
# Inspect without cluster cleanup: bash <this-script> --dry-run
# Launch through scripts/entrypoint/ray-job.sh after reviewing the dry-run output.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
DRY_RUN=0
EXTRA_ARGS=()
for arg in "$@"; do
   if [[ "$arg" == --dry-run ]]; then DRY_RUN=1; else EXTRA_ARGS+=("$arg"); fi
done

: "${MODEL_DIR:?Set MODEL_DIR to the parent of Qwen3.6-35B-A3B}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent shared artifact directory}"
: "${HARBOR_TASK_ROOT:?Set HARBOR_TASK_ROOT to the materialized Harbor tasks}"
: "${HARBOR_PYTHON:?Set HARBOR_PYTHON to the worker-local Harbor venv Python}"
: "${HARBOR_SIF_CACHE_ROOT:?Set HARBOR_SIF_CACHE_ROOT to the worker-local runtime/cache root}"
: "${RAY_ADDRESS:?Set RAY_ADDRESS to the existing cluster GCS address}"
: "${RAY_DASHBOARD_ADDRESS:?Set RAY_DASHBOARD_ADDRESS to the head HTTP Jobs address}"

NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
PROJECT_NAME="${PROJECT_NAME:-Relax/agentic/claude-code-harbor}"
EXP_NAME="${EXP_NAME:-qwen36-harbor-pevekcc-gbs512-n16-ctx262k-${NUM_ROLLOUT}step}"
RUN_ID="${RELAX_SUBMISSION_ID:-${EXP_NAME}-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${SAVE_DIR%/}/runs}"
RUN_DIR="${RUN_ROOT%/}/${RUN_ID}"
HF_CHECKPOINT="${MODEL_DIR%/}/Qwen3.6-35B-A3B"
PROMPT_DATA="${HARBOR_TASK_ROOT%/}/prompts.jsonl"
MAX_OUTPUT_TOKENS=8192
MAX_CONTEXT_TOKENS=262144
printf -v AGENT_COMMAND '%q -m examples.claude_code_harbor.agent' "${HARBOR_PYTHON}"
source "${RELAX_ROOT}/scripts/models/qwen36-35B-A3B.sh"

# Full token-level dumps are optional; scalar metrics, rollout audit, and checkpoints stay enabled.
DEBUG_ARGS=()
if [[ "${SAVE_FULL_DEBUG_DUMPS:-0}" == 1 ]]; then
   DEBUG_ARGS=(--dump-details "${RUN_DIR}/dump")
fi

TRAIN_ARGS=(
   --resource '{"actor":[1,64],"rollout":[1,32]}'
   --hybrid --max-staleness 4
   --num-data-storage-units 1 --num-iters-per-train-update 1 --update-weights-interval 1
   "${MODEL_ARGS[@]}"
   --hf-checkpoint "${HF_CHECKPOINT}"
   --load "${HARBOR_LOAD_DIR:-${HF_CHECKPOINT}}"
   --megatron-to-hf-mode bridge
   --save "${RUN_DIR}/checkpoints" --save-interval "${SAVE_INTERVAL}"
   --prompt-data "${PROMPT_DATA}" --input-key input --metadata-key metadata
   --rollout-shuffle --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size 32 --n-samples-per-prompt 16 --global-batch-size 512
   --over-sampling-batch-size 64 --num-steps-per-rollout 1
   --rollout-max-response-len "${MAX_OUTPUT_TOKENS}"
   --rollout-max-context-len "${MAX_CONTEXT_TOKENS}" --rollout-temperature 1.0 --balance-data
   --seq-length "${MAX_CONTEXT_TOKENS}"
   --use-agentic-rollout --agentic-prelaunch --agentic-session-lifecycle --agentic-program-admission
   --agentic-native-replay-dir "${RUN_DIR}/native-replay"
   --agent-command "${AGENT_COMMAND}" --agent-cwd "${RELAX_ROOT}"
   --agentic-concurrency "${HARBOR_AGENTIC_CONCURRENCY:-32}" --agent-timeout 10800
   --agentic-reasoning-parser qwen3-thinking --agentic-tool-call-parser qwen3_coder
   --agentic-custom-advantage-path examples.claude_code_harbor.advantage.advantage_func
   --agent-env "MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS}"
   "HARBOR_TASK_ROOT=${HARBOR_TASK_ROOT}" "HARBOR_SIF_CACHE_ROOT=${HARBOR_SIF_CACHE_ROOT}"
   --advantage-estimator grpo --kl-coef 0.0 --kl-loss-coef 0.0 --entropy-coef 0.0
   --eps-clip 0.2 --eps-clip-high 0.28 --use-tis
   --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1
   --adam-beta1 0.9 --adam-beta2 0.98
   --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
   --tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1
   --context-parallel-size 32 --expert-model-parallel-size 16 --expert-tensor-parallel-size 1
   --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
   --empty-unused-memory-level 1 --micro-batch-size 1 --use-dynamic-batch-size
   --max-tokens-per-gpu 8192 --log-probs-max-tokens-per-gpu 10240 --log-probs-chunk-size 32
   --qkv-format thd --moe-flex-dispatcher-backend deepep --moe-token-dispatcher-type flex
   --moe-router-load-balancing-type none --moe-aux-loss-coeff 0.0 --no-rope-fusion --recompute-loss-function
   --rollout-num-gpus-per-engine 2 --sglang-pp-size 1 --sglang-dp-size 1 --sglang-ep-size 1
   --sglang-config "${RELAX_ROOT}/examples/claude_code_harbor/pd-hicache-cpu-4p12d-spread128-tp2.yaml"
   --sglang-disaggregation-transfer-backend mooncake
   --sglang-mem-fraction-static 0.88 --sglang-context-length "${MAX_CONTEXT_TOKENS}"
   --sglang-max-running-requests 32 --sglang-server-concurrency "${HARBOR_SGLANG_SERVER_CONCURRENCY:-16}" --sglang-max-mamba-cache-size 384
   --sglang-enable-session-radix-cache --sglang-radix-eviction-policy priority
   --sglang-router-policy consistent_hashing --sglang-log-level warning
   --sglang-router-prefill-policy consistent_hashing --sglang-router-decode-policy power_of_two
   --use-clearml --use-metrics-service --no-use-tensorboard
   --tb-project-name "${PROJECT_NAME}" --tb-experiment-name "${RUN_ID}"
   --distributed-timeout-minutes 180 "${DEBUG_ARGS[@]}"
   --rollout-result-dir "${RUN_DIR}/audit/rollout_result" --skip-eval-before-train
   --attention-dropout 0.0 --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash
   "${EXTRA_ARGS[@]}"
)

# This branch runs before ray-job.sh, directory creation, or any Ray call.
if [[ "$DRY_RUN" == 1 ]]; then
   python3 - "${RELAX_ROOT}" "${RUN_ID}" "${RUN_DIR}" "${RAY_DASHBOARD_ADDRESS}" "${TRAIN_ARGS[@]}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

import yaml

root, run_id, run_dir, dashboard, *argv = sys.argv[1:]
config = yaml.safe_load((Path(root) / "examples/claude_code_harbor/config.yaml").read_text())
print(json.dumps({
    "mode": "dry-run-no-cluster-actions",
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    "tracked_diff": subprocess.check_output(["git", "diff", "--name-only"], cwd=root, text=True).splitlines(),
    "submission_id": run_id,
    "run_dir": run_dir,
    "dashboard": dashboard,
    "agent_config": config,
    "command": ["python3", "-m", "relax.entrypoints.train", *argv],
    "note": "CLI and agent configuration; framework defaults still require parser validation.",
}, ensure_ascii=False, indent=2))
PY
   exit 0
fi

if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
   export RELAX_SUBMISSION_ID="${RUN_ID}"
   exec bash "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh" "${BASH_SOURCE[0]}" "${EXTRA_ARGS[@]}"
fi
: "${RUNTIME_ENV_JSON:?The Relax entrypoint must supply RUNTIME_ENV_JSON}"
[[ -s "${HF_CHECKPOINT}/config.json" && -s "${PROMPT_DATA}" ]]
[[ ! -e "${RUN_DIR}" ]] || { echo "Run directory already exists: ${RUN_DIR}" >&2; exit 2; }
mkdir -p "${RUN_DIR}"
RAY_NO_WAIT="${RAY_NO_WAIT-1}"
ray job submit ${RAY_NO_WAIT:+--no-wait} \
   --submission-id="${RUN_ID}" --address="${RAY_DASHBOARD_ADDRESS}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" 2>&1 | tee "${RUN_DIR}/submission.log"
