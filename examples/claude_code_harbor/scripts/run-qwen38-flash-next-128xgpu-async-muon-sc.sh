#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.8-Flash-Next Harbor RL: 64 actor + 64 rollout GPUs, fully async, SC + Muon.
# From the repository root, after preparing Harbor runtimes on all workers:
#   MODEL_DIR=<model-root> DATA_DIR=<harbor-materialized> SAVE_DIR=<checkpoint-root> \
#   HARBOR_PYTHON=<worker-python> HARBOR_SIF_CACHE_ROOT=<worker-local-cache> \
#   RAY_DASHBOARD_ADDRESS=http://<head>:8265 RELAX_SKIP_CLUSTER_CLEANUP=1 \
#     bash scripts/entrypoint/ray-job.sh \
#       examples/claude_code_harbor/scripts/run-qwen38-flash-next-128xgpu-async-muon-sc.sh
#
# Defaults: 64 prompt groups x 8 Sessions = 512 logical samples per update.
# Context length comes from the existing Harbor config.yaml (131072 tokens).
# Set NUM_ROLLOUT=2 for initial end-to-end validation. Checkpoints use a stable EXP_NAME.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
export RELAX="${RELAX_ROOT}"
export RELAX_SKIP_CLUSTER_CLEANUP=1
export RAY_NO_WAIT="${RAY_NO_WAIT-}"

: "${MODEL_DIR:?Set MODEL_DIR to the directory containing Qwen3.8-Flash-Next}"
: "${DATA_DIR:?Set DATA_DIR to the materialized Harbor dataset directory}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
: "${HARBOR_PYTHON:?Set HARBOR_PYTHON to the prepared Python path on every worker}"
: "${HARBOR_SIF_CACHE_ROOT:?Set HARBOR_SIF_CACHE_ROOT to the prepared worker-local cache}"
: "${RAY_DASHBOARD_ADDRESS:?Set RAY_DASHBOARD_ADDRESS to the existing Ray Jobs HTTP address}"

# Prefer the prepared worker-local Muon source; otherwise use the installed package.
if [[ -d "${HARBOR_SIF_CACHE_ROOT}/muon" ]]; then
    export PYTHONPATH="${HARBOR_SIF_CACHE_ROOT}/muon${PYTHONPATH:+:${PYTHONPATH}}"
fi

if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    source "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh"
fi
set +x

PROJECT_NAME="${PROJECT_NAME:-Relax/agentic/claude-code-harbor}"
EXP_NAME="${EXP_NAME:-qwen38-flash-next-cc-harness-harbor-128xgpu-async-muon-sc}"
NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
N_SAMPLES="${N_SAMPLES:-8}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE + (ROLLOUT_BATCH_SIZE + 3) / 4))}"
AGENTIC_RUNTIME_GROUPS="${AGENTIC_RUNTIME_GROUPS:-80}"
AGENTIC_PREPARE_GROUPS="${AGENTIC_PREPARE_GROUPS:-16}"
AGENTIC_CONCURRENCY="${AGENTIC_CONCURRENCY:-$((AGENTIC_RUNTIME_GROUPS + AGENTIC_PREPARE_GROUPS))}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-32768}"
MAX_CONTEXT_TOKENS="$(python3 - "${EXAMPLE_DIR}/config.yaml" <<'PY'
import sys
from pathlib import Path

import yaml

print(int(yaml.safe_load(Path(sys.argv[1]).read_text())["claude_code"]["max_context_tokens"]))
PY
)"
(( NUM_ROLLOUT > 0 && N_SAMPLES >= 2 && ROLLOUT_BATCH_SIZE > 0 ))
(( OVER_SAMPLING_BATCH_SIZE >= ROLLOUT_BATCH_SIZE && AGENTIC_CONCURRENCY > 0 ))
(( AGENTIC_RUNTIME_GROUPS > 0 && AGENTIC_PREPARE_GROUPS > 0 ))
(( AGENTIC_CONCURRENCY >= AGENTIC_RUNTIME_GROUPS + AGENTIC_PREPARE_GROUPS ))
(( MAX_CONTEXT_TOKENS > MAX_OUTPUT_TOKENS && MAX_OUTPUT_TOKENS > 0 ))
now="$(date +%Y%m%d-%H%M%S)"
SUBMISSION_ID="${RELAX_SUBMISSION_ID:-${EXP_NAME}-${now}-${BASHPID}}"
printf -v AGENT_COMMAND '%q -m examples.claude_code_harbor.agent' "${HARBOR_PYTHON}"

# Checkpoint PLE by HC stream while keeping its projection, norm, and convolution
# weights trainable. The ngram table remains frozen for CPU offload.
export RELAX_QWEN38_PLE_STREAMED=1
# Project only response rows and checkpoint bounded SC vocabulary blocks.
export RELAX_SCORE_CENTERING_RESPONSE_ONLY_LM_HEAD=1
export QWEN38_PLE_CPU_OFFLOAD=1
export MCORE_QSA_SPARSE_BACKEND=flex
export RELAX_QWEN38_QSA_FUSED_ATTENTION=triton_v2
source "${RELAX_ROOT}/scripts/models/qwen38-flash-next.sh"

export NVTE_FLASH_ATTN_V2=1
export NVTE_FLASH_ATTN_V3=0
export NVTE_FLASH_ATTN_V4=0
export EP_DISABLE_GIN=1
export NVSHMEM_DISABLE_NCCL=1
export NCCL_P2P_PXN_LEVEL=0
export NCCL_NVLS_ENABLE=0
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0
export SGLANG_RETURN_ORIGINAL_LOGPROB=0
unset MC_FORCE_TCP
RUNTIME_ENV_JSON="$(python3 - <<'PY'
import json
import os

runtime = json.loads(os.environ["RUNTIME_ENV_JSON"])
env = runtime.setdefault("env_vars", {})
env.pop("MC_FORCE_TCP", None)
env.update(
    # Disable Ray task-state telemetry only for this experiment's new processes.
    RAY_task_events_report_interval_ms="0",
    MOONCAKE_PROTOCOL="rdma",
    PYTORCH_ALLOC_CONF="expandable_segments:False",
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False",
)
for name in (
    "RELAX", "PYTHONPATH", "PATH", "MCORE_QSA_SPARSE_BACKEND",
    "RELAX_SCORE_CENTERING_RESPONSE_ONLY_LM_HEAD",
    "QWEN38_PLE_CPU_OFFLOAD", "NVTE_FLASH_ATTN_V2", "NVTE_FLASH_ATTN_V3",
    "NVTE_FLASH_ATTN_V4", "EP_DISABLE_GIN", "NVSHMEM_DISABLE_NCCL",
    "NCCL_P2P_PXN_LEVEL", "NCCL_NVLS_ENABLE",
    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM", "SGLANG_RETURN_ORIGINAL_LOGPROB",
):
    env[name] = os.environ[name]
env.update((name, value) for name, value in os.environ.items() if name.startswith("RELAX_QWEN38_"))
forwarded = [name for name in env.get("RELAX_PROPAGATE_ENV_VARS", "").split(",") if name and name != "MC_FORCE_TCP"]
env["RELAX_PROPAGATE_ENV_VARS"] = ",".join(dict.fromkeys([*forwarded, *env]))
print(json.dumps(runtime))
PY
)"

RESOURCE_ARGS=(
    --resource '{"actor":[1,64],"rollout":[1,64],"advantages":[1,0]}'
    --actor-num-nodes 8
    --actor-num-gpus-per-node 8
    --fully-async
    --max-staleness 2
    --num-data-storage-units 1
    --num-iters-per-train-update 1
    --update-weights-interval 1
    --update-weight-buffer-size 268435456
    --distributed-timeout-minutes 90
)

CKPT_ARGS=(
    --disable-s3-model-download
    --hf-checkpoint "${MODEL_DIR}/Qwen3.8-Flash-Next"
    --megatron-to-hf-mode bridge
    --load "${SAVE_DIR}/${EXP_NAME}"
    --save "${SAVE_DIR}/${EXP_NAME}"
    --save-interval 20
    --max-actor-ckpt-to-keep 2
)

ROLLOUT_ARGS=(
    --prompt-data "${DATA_DIR}/prompts.jsonl"
    --input-key input
    --metadata-key metadata
    --rollout-shuffle
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES}"
    --global-batch-size "$((ROLLOUT_BATCH_SIZE * N_SAMPLES))"
    --num-steps-per-rollout 1
    --rollout-temperature 1
    --rollout-max-response-len "${MAX_OUTPUT_TOKENS}"
    --rollout-max-context-len "${MAX_CONTEXT_TOKENS}"
    --balance-data
    --skip-eval-before-train
)

AGENTIC_ARGS=(
    --use-agentic-rollout
    --agentic-prelaunch
    --agentic-session-lifecycle
    --agentic-program-admission
    --agent-command "${AGENT_COMMAND}"
    --agent-cwd "${RELAX_ROOT}"
    --agent-timeout 7200
    --agentic-concurrency "${AGENTIC_CONCURRENCY}"
    --agentic-runtime-groups "${AGENTIC_RUNTIME_GROUPS}"
    --agentic-prepare-groups "${AGENTIC_PREPARE_GROUPS}"
    --agentic-admission-expected-decode-cap 4096
    --agentic-admission-max-wait-s 5
    --agentic-admission-pressure-threshold 0.92
    --agentic-reasoning-parser qwen3-thinking
    --agentic-tool-call-parser qwen3_coder
    --agentic-custom-advantage-path examples.claude_code_harbor.advantage.score_centering_advantage
    --agent-env
    "MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS}"
    "HARBOR_TASK_ROOT=${DATA_DIR}"
    "HARBOR_SIF_CACHE_ROOT=${HARBOR_SIF_CACHE_ROOT}"
)

SCORE_CENTERING_ARGS=(
    --advantage-estimator score_centering
    --rollout-top-logprobs-num 128
    --score-centering-is tis
    --score-centering-tis-clip 2
    --use-rollout-logprobs
    --get-mismatch-metrics
    --calculate-per-token-loss
    --entropy-coef 0
    --kl-coef 0
    --kl-loss-coef 0
)

OPTIMIZER_ARGS=(
    --optimizer muon
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --clip-grad 1
    --muon-momentum 0.95
    --muon-nesterov
    --muon-coefficient-type polar_express
    --muon-num-ns-steps 8
    --muon-scale-mode spectral
    --muon-extra-scale-factor 0.2
    --muon-tp-mode blockwise
    --adam-beta1 0.9
    --adam-beta2 0.98
    --use-distributed-optimizer
    --no-use-precision-aware-optimizer
    --no-store-param-remainders
    --chunked-optimizer-state-offload
    --optimizer-state-offload-fraction 1
    --optimizer-state-offload-chunk-size-mb 128
)

FREEZE_ARGS=(
    --freeze-vision-model
    --freeze-vision-projection
    --freeze-params-name-list
    'self_attention\.indexer\.'
    '(^|\.)per_layer_embedding\.ple_embedding\.ngram_embedding\.weight$'
)

PERF_ARGS=(
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 8
    --context-parallel-size 2
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --sequence-parallel
    --qkv-format thd
    --moe-token-dispatcher-type flex
    --moe-flex-dispatcher-backend deepep
    --moe-flex-dispatcher-num-sms 24
    --moe-shared-expert-overlap
    --moe-grouped-gemm
    --moe-router-load-balancing-type none
    --moe-aux-loss-coeff 0
    --seq-length "${MAX_CONTEXT_TOKENS}"
    --micro-batch-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "$((MAX_CONTEXT_TOKENS / 2))"
    --log-probs-max-tokens-per-gpu "$((MAX_CONTEXT_TOKENS / 2))"
    --log-probs-chunk-size 512
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --recompute-loss-function
    --empty-unused-memory-level 1
    --train-env-vars '{"PYTORCH_ALLOC_CONF":"expandable_segments:True","PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
    --bf16
    --accumulate-allreduce-grads-in-fp32
    --attention-backend flash
    --attention-softmax-in-fp32
    --attention-dropout 0
    --hidden-dropout 0
    --no-rope-fusion
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 8
    --sglang-config "${SCRIPT_DIR}/pd-hicache-cpu-2p6d-tp8.yaml"
    --sglang-served-model-name relax_model
    --sglang-load-format safetensors
    --sglang-pp-size 1
    --sglang-enable-dp-attention
    --sglang-dp-size 2
    --sglang-ep-size 8
    --sglang-kv-cache-dtype fp8_e4m3
    --sglang-mem-fraction-static 0.78
    --sglang-context-length "${MAX_CONTEXT_TOKENS}"
    --sglang-max-running-requests 128
    --sglang-max-mamba-cache-size 640
    --sglang-server-concurrency 128
    --sglang-disable-prefill-cuda-graph
    --sglang-cuda-graph-max-bs-decode 64
    --sglang-disaggregation-transfer-backend mooncake
    --sglang-enable-session-radix-cache
    --sglang-radix-eviction-policy priority
    --sglang-router-policy consistent_hashing
    --sglang-router-prefill-policy consistent_hashing
    --sglang-router-decode-policy session_least_load
    --sglang-router-health-check-timeout-secs 360
)

TRACKING_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)

[[ -s "${MODEL_DIR}/Qwen3.8-Flash-Next/model.safetensors.index.json" ]]
[[ -s "${DATA_DIR}/prompts.jsonl" && -x "${HARBOR_PYTHON}" ]]
mkdir -p "${SCRIPT_DIR}/log"
cd "${RELAX_ROOT}"
ray job submit ${RAY_NO_WAIT:+--no-wait} \
    --submission-id="${SUBMISSION_ID}" \
    --address="${RAY_DASHBOARD_ADDRESS}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    "${RESOURCE_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${AGENTIC_ARGS[@]}" \
    "${SCORE_CENTERING_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${FREEZE_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${TRACKING_ARGS[@]}" \
    "$@" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
