#!/usr/bin/env bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# 16 H800 GPUs on two nodes: 8 actor GPUs + 8 rollout GPUs.
# Every Relax Session runs a Harbor Trial with shared-container verification.
#
# Usage (an existing 2-node, 8-GPU-per-node Ray cluster):
#   RAY_ADDRESS=<head-ip>:6379 MODEL_DIR=<model-root> HARBOR_TASK_ROOT=<task-root> \
#   HARBOR_PYTHON=<python-with-harbor> HARBOR_SIF_CACHE_ROOT=<worker-local-dir> \
#   SAVE_DIR=<checkpoint-root> \
#     bash scripts/entrypoint/ray-job.sh \
#       examples/claude_code_harbor/run_qwen36_35b_a3b_16xh800.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
PROJECT_NAME="${PROJECT_NAME:-Relax/agentic/claude-code-harbor}"
EXP_NAME="${EXP_NAME:-qwen36-35b-a3b-claude-code-harbor-16xh800-hybrid}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-32768}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-61440}"
now=$(date "+%Y%m%d-%H%M%S")
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/log}"
RUN_DIR="${RUN_ROOT%/}/${EXP_NAME}-${now}"

PROMPT_DATA="${HARBOR_TASK_ROOT%/}/prompts.jsonl"
printf -v AGENT_COMMAND '%q -m examples.claude_code_harbor.agent' "${HARBOR_PYTHON}"

source "${RELAX_ROOT}/scripts/models/qwen36-35B-A3B.sh"
DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:-http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT:-8265}}"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/Qwen3.6-35B-A3B"
   # --ref-load "${MODEL_DIR}/Qwen3.6-35B-A3B"
   --megatron-to-hf-mode bridge
   --load "${SAVE_DIR}/${EXP_NAME}"
   # --save "${SAVE_DIR}/${EXP_NAME}"
   # --save-interval 16
   # --max-actor-ckpt-to-keep 2
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --rollout-shuffle
   --num-rollout 128
   --rollout-batch-size 4
   --n-samples-per-prompt 16
   --global-batch-size 64
   --rollout-max-response-len "${MAX_OUTPUT_TOKENS}"
   --rollout-max-context-len "${MAX_CONTEXT_TOKENS}"
)

AGENTIC_ARGS=(
   --use-agentic-rollout
   --agentic-prelaunch
   --agentic-session-lifecycle
   --agent-command "${AGENT_COMMAND}"
   --agent-cwd "${RELAX_ROOT}"
   --agent-timeout 3600
   --agentic-reasoning-parser qwen3-thinking
   --agentic-tool-call-parser qwen3_coder
   --agentic-custom-advantage-path examples.claude_code_harbor.advantage.advantage_func
   --agent-env
   "MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS}"
   "HARBOR_TASK_ROOT=${HARBOR_TASK_ROOT}"
   "HARBOR_SIF_CACHE_ROOT=${HARBOR_SIF_CACHE_ROOT}"
)
# Compaction exports explicit contexts; advantage.py owns Session normalization
# and zero-variance filtering.

GRPO_ARGS=(
   --advantage-estimator grpo
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
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
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 4
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   # Release cached allocations before first gradient-sync NCCL allocation.
   --empty-unused-memory-level 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 15360
   --log-probs-max-tokens-per-gpu 8192
   --log-probs-chunk-size 32
   --qkv-format thd
   --moe-flex-dispatcher-backend deepep
   --moe-token-dispatcher-type flex
   --moe-router-load-balancing-type none
   --no-rope-fusion
   --recompute-loss-function
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.85
   --sglang-context-length "${MAX_CONTEXT_TOKENS}"
   --sglang-enable-session-radix-cache
   --sglang-radix-eviction-policy priority
   --sglang-router-policy consistent_hashing
   --sglang-log-level warning
)

TRACKING_ARGS=(
   --use-clearml
   --use-metrics-service
   --no-use-tensorboard
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${EXP_NAME}-${now}"
)

MISC_ARGS=(
   --skip-eval-before-train
   --dump-details "${RUN_DIR}/dump"
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

mkdir -p "${RUN_DIR}"
ray job submit ${RAY_NO_WAIT:+--no-wait} \
   --submission-id="${RELAX_SUBMISSION_ID:-claude-code-harbor-${now}-${BASHPID}-${RANDOM}}" \
   --address="${DASHBOARD_ADDRESS}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor":[1,8],"rollout":[1,8]}' \
   --hybrid \
   --max-staleness 4 \
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
   "$@" 2>&1 | tee "${RUN_DIR}/run.log"
