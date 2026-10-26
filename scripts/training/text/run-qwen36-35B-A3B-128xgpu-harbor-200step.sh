#!/usr/bin/env bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Two hundred real optimizer updates: Qwen3.6-35B-A3B + Claude Code / Harbor.
# 128 H800 GPUs: 64 actor (EP16) + 64 rollout; 32 regular engines, each TP2 for attention and MoE (DP1 EP1).
# See examples/claude_code_harbor/README.md for source evidence, adapter requirements and semantic gaps.
#
# Usage (an existing 16-node, 8-GPU-per-node Ray cluster):
#   RAY_ADDRESS=<head-ip>:6379 MODEL_DIR=<model-root> DATA_DIR=<dataset-root> \
#   SAVE_DIR=<persistent-run-artifact-root> \
#   NEMO_GYM_URL=<gateway-url> NEMO_GYM_ENVIRONMENT=<registered-environment> \
#   NEMO_GYM_CONFIG=claude-code-harbor-oci-v1 \
#     bash scripts/entrypoint/ray-job.sh \
#       scripts/training/text/run-qwen36-35B-A3B-128xgpu-harbor-200step.sh

set -e
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
PROJECT_NAME="${PROJECT_NAME:-Relax/agentic/claude-code-harbor}"
EXP_NAME="${EXP_NAME:-qwen36-35b-a3b-claude-code-harbor-128xh800-200step}"
now=$(date "+%Y%m%d-%H%M%S")

: "${MODEL_DIR:?Set MODEL_DIR to the root containing Qwen3.6-35B-A3B/}"
: "${DATA_DIR:?Set DATA_DIR to the source Harbor dataset directory containing data.jsonl}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent shared run-artifact root}"
: "${RAY_ADDRESS:?Set RAY_ADDRESS to the existing Ray cluster GCS address}"
source "${RELAX_ROOT}/examples/claude_code_harbor/scripts/cloud_paths.sh"
harbor_resolve_paths MODEL_DIR DATA_DIR SAVE_DIR NEMO_GYM_TOKEN_FILE
# The node-local Gym service must already run the Harbor/CC OCI harness and
# return metadata.contexts. Its environment/config names are deployment-owned.
if [ -z "${CLAUDE_CODE_AGENT_COMMAND:-}" ]; then
   : "${NEMO_GYM_URL:?Set NEMO_GYM_URL to the existing NeMo Gym gateway}"
   : "${NEMO_GYM_ENVIRONMENT:?Set the registered Harbor environment name}"
   : "${NEMO_GYM_CONFIG:?Set the registered Harbor/CC/OCI config name}"
   : "${NEMO_GYM_TOKEN_FILE:?Set the private Gateway token file shared with the host clients}"
   printf -v CLAUDE_CODE_AGENT_COMMAND 'bash %q' "${RELAX_ROOT}/examples/nemo_gym_agentic/scripts/run_agent_app.sh"
fi
AGENT_COMMAND='python3 -m examples.claude_code_harbor.persistent_run_agent'

if [ ! -s "${MODEL_DIR}/Qwen3.6-35B-A3B/config.json" ] || [ ! -s "${DATA_DIR}/data.jsonl" ]; then
   echo "ERROR: model config or Harbor data.jsonl is missing or empty" >&2
   exit 2
fi

# Direct invocation follows the same existing-cluster entrypoint as the usage above.
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
   exec bash "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh" "${BASH_SOURCE[0]}" "$@"
fi
source "${RELAX_ROOT}/scripts/models/qwen36-35B-A3B.sh"
: "${RUNTIME_ENV_JSON:?RUNTIME_ENV_JSON must be supplied by the Relax entrypoint}"
# This run uses the default SGLang numerical/scheduling behavior.
RUNTIME_ENV_JSON="$(python3 - "${RUNTIME_ENV_JSON}" <<'PYENV'
import json
import sys
runtime = json.loads(sys.argv[1])
env = runtime.setdefault("env_vars", {})
env["HARBOR_LOG_GROUP_OUTCOMES"] = "1"
env["RAY_DEDUP_LOGS"] = "0"
for key in ("SGLANG_GDN_PREFILL_FINAL_STATE_ONLY", "SGLANG_FORCE_COARSE_WAR_BARRIER"):
    env.pop(key, None)
assert not any(key.startswith(("HARBOR_NUMERICAL_", "HARBOR_PD_TRACE_", "HARBOR_SYNC_DIAGNOSTIC")) for key in env)
print(json.dumps(runtime))
PYENV
)"
export RAY_NO_WAIT="${RAY_NO_WAIT-1}"

# Dressage promotes top-level task/reward routing fields into Sample.metadata.
# Preserve that contract before Relax's standard JSONL loader selects metadata.
PROMPT_SET="${SAVE_DIR}/${EXP_NAME}/data/harbor.jsonl"
python3 - "${DATA_DIR}/data.jsonl" "${PROMPT_SET}" "${HARBOR_MAX_TURNS:-150}" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path

source, destination = map(Path, sys.argv[1:3])
max_turns = int(sys.argv[3])
if max_turns <= 0:
    raise ValueError("HARBOR_MAX_TURNS must be a positive integer")
destination.parent.mkdir(parents=True, exist_ok=True)
temporary = None
prompt_indices = {}
try:
    with source.open(encoding="utf-8") as inputs, tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, delete=False
    ) as output:
        temporary = Path(output.name)
        for line in inputs:
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = dict(row.get("metadata") or {})
            instance_id = metadata.get("instance_id")
            if not isinstance(instance_id, str) or not instance_id:
                raise ValueError("Prompt-equal loss requires the source instance_id")
            metadata["_loss_prompt_index"] = prompt_indices.setdefault(instance_id, len(prompt_indices))
            metadata["harbor_backend_timeout_s"] = 7200
            # This travels through Gateway metadata into the remote OCI CLI;
            # an agent-process environment variable alone cannot configure it.
            metadata["harbor_max_turns"] = max_turns
            for key in (
                "agent_mode", "env_type", "env_args", "tool_set", "agent_id", "reward_fn", "blackbox_type", "task_type"
            ):
                if key in row:
                    metadata.setdefault(key, row[key])
            row["metadata"] = metadata
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, destination)
finally:
    if temporary is not None:
        temporary.unlink(missing_ok=True)
PY

if [ -n "${RAY_DASHBOARD_ADDRESS:-}" ]; then
   DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS}"
else
   : "${MASTER_ADDR:?Set MASTER_ADDR or RAY_DASHBOARD_ADDRESS to the Ray head}"
   DASHBOARD_ADDRESS="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT:-8265}"
fi
case "${DASHBOARD_ADDRESS}" in
   http://* | https://*) ;;
   *) echo "ERROR: RAY_DASHBOARD_ADDRESS must be an HTTP(S) Jobs URL" >&2; exit 2 ;;
esac

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/Qwen3.6-35B-A3B"
   --ref-load "${MODEL_DIR}/Qwen3.6-35B-A3B"
   --megatron-to-hf-mode bridge
   --load "${HARBOR_LOAD_DIR:-${MODEL_DIR}/Qwen3.6-35B-A3B}"
   --save "${SAVE_DIR}/${EXP_NAME}/checkpoints"
   --save-interval 50
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_SET}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout 200
   --rollout-batch-size 64
   --over-sampling-batch-size 96
   --n-samples-per-prompt 8
   --global-batch-size 512
   --num-steps-per-rollout 1
   --rollout-max-response-len 8192
   --rollout-max-context-len 262144
   --rollout-temperature 1.0
   --balance-data
)

AGENTIC_ARGS=(
   --use-agentic-rollout
   --agent-command "${AGENT_COMMAND}"
   --agent-cwd "${RELAX_ROOT}"
   --agentic-concurrency 96
   --agent-timeout 10800
   --agentic-reasoning-parser qwen3
   --agentic-tool-call-parser qwen3_coder
   --agentic-custom-advantage-path examples.claude_code_harbor.advantage.advantage_func
   --custom-convert-samples-to-train-data-path examples.claude_code_harbor.training_data.convert_samples_to_train_data
   --agent-env
   "CLAUDE_CODE_AGENT_COMMAND=${CLAUDE_CODE_AGENT_COMMAND}"
   "CLAUDE_CODE_SETTINGS_PATH=${SCRIPT_DIR}/../claude_settings.json"
   "CLAUDE_CODE_AUTO_COMPACT_WINDOW=200000"
   "DRESSAGE_BLACKBOX_MAX_STEPS=200"
   "HARBOR_AUDIT_CONTEXT_DIR=${SAVE_DIR}/${EXP_NAME}/audit/contexts"
   "NEMO_GYM_URL=${NEMO_GYM_URL:-}"
   "NEMO_GYM_ENVIRONMENT=${NEMO_GYM_ENVIRONMENT:-}"
   "NEMO_GYM_CONFIG=${NEMO_GYM_CONFIG:-}"
   "NEMO_GYM_TOKEN_FILE=${NEMO_GYM_TOKEN_FILE:-}"
   "NEMO_GYM_MODEL=Qwen3.6-35B-A3B"
   # Transport lease spans weight-sync pauses; Runtime owns active-time expiry.
   "NEMO_GYM_DEADLINE_S=2147483"
   "NEMO_GYM_INTERRUPT_POLICY=protected"
)
# Compaction requires explicit contexts. The custom function normalizes outcomes
# over logical Sessions and drops zero-variance Groups before replenishment.
# Do not apply the built-in filter over physical export rows.

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-coef 0.0
   --kl-loss-coef 0.0
   --entropy-coef 0.0
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
   --prompt-equal-loss
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 32
   # MCore's CP contract stays enabled; prompt-equal loss cancels its token denominator.
   --calculate-per-token-loss
   --expert-model-parallel-size 16
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   # Release cached allocations before first gradient-sync NCCL allocation.
   --empty-unused-memory-level 1
   --micro-batch-size 1
   --use-dynamic-batch-size
   # Source uses 10000; Relax requires multiples of TP * data-pad-size-multiplier.
   --max-tokens-per-gpu 8192
   --log-probs-max-tokens-per-gpu 10240
   --log-probs-chunk-size 32
   --qkv-format thd
   --moe-flex-dispatcher-backend deepep
   --moe-token-dispatcher-type flex
   --moe-router-load-balancing-type none
   --moe-aux-loss-coeff 0.0
   --no-rope-fusion
   --recompute-loss-function
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-config "${RELAX_ROOT}/examples/claude_code_harbor/unified32-tp2-no-hicache.yaml"
   --sglang-dp-size 1
   --sglang-ep-size 1
   --sglang-mem-fraction-static 0.75
   --sglang-context-length 262144
   --sglang-max-running-requests 64
   --sglang-server-concurrency 64
   --sglang-router-policy consistent_hashing
   --sglang-log-level warning
)

TRACKING_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${EXP_NAME}-${now}"
)

MISC_ARGS=(
   --distributed-timeout-minutes 180
   --rollout-result-dir "${SAVE_DIR}/${EXP_NAME}/audit/rollout_result"
   # Save model/optimizer/scheduler/RNG every 50 updates; keep all four checkpoints.
   # Canonical session evidence remains under the run-artifact root.
   --skip-eval-before-train
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

mkdir -p "${SCRIPT_DIR}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} \
   --submission-id="${RELAX_SUBMISSION_ID:-claude-code-harbor-${now}-${BASHPID}-${RANDOM}}" \
   --address="${DASHBOARD_ADDRESS}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor":[1,64],"rollout":[1,64]}' \
   --hybrid \
   --max-staleness 4 \
   --num-data-storage-units 1 \
   --num-iters-per-train-update 1 \
   --update-weights-interval 1 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${AGENTIC_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${TRACKING_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "$@" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
