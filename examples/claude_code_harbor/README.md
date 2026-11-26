# Claude Code + Harbor 训练示例

## 环境配置

### 共享目录

```text
<shared-root>/
├── claude-code-harbor-r5/   # MiMo task 与 runtime 源资产
├── harbor-materialized/     # 生成的 Harbor task root
└── claude-code-harbor/
    ├── harbor-wheelhouse-0.24.0/
    └── claude-code-runtime-2.1.308-srt-0.0.77.tar.gz
```

模型和checkpoint目录必须在所有Ray节点使用相同绝对路径：

```text
${MODEL_DIR}/Qwen3.8-Flash-Next
${SAVE_DIR}/
```

### 生成 Harbor task root

```bash
cd /path/to/Relax

SOURCE_ROOT=/path/to/shared-root/claude-code-harbor-r5
HARBOR_TASK_ROOT=/path/to/shared-root/harbor-materialized

python3 -m examples.claude_code_harbor.prepare_data \
  --source "${SOURCE_ROOT}/aligned-150037-oci/data.jsonl" \
  --asset-root "${SOURCE_ROOT}" \
  --image-map "${SOURCE_ROOT}/ordinary-image-map.json" \
  --output-root "${HARBOR_TASK_ROOT}"
```

`HARBOR_TASK_ROOT`必须尚未存在。生成结果为：

```text
harbor-materialized/
├── prompts.jsonl
└── tasks/
```

### 每台 worker 安装 Harbor

Harbor venv安装在worker本地盘：

```bash
LOCAL_ROOT=/data/temp/claude-code-harbor
HARBOR_WHEELHOUSE=/path/to/shared-root/claude-code-harbor/harbor-wheelhouse-0.24.0

mkdir -p "${LOCAL_ROOT}"
python3 -m venv "${LOCAL_ROOT}/harbor-venv"

"${LOCAL_ROOT}/harbor-venv/bin/pip" install \
  --no-index \
  --find-links "${HARBOR_WHEELHOUSE}" \
  "harbor==0.24.0"
```

### 每台 worker 解压 Claude Code runtime

```bash
export HARBOR_SIF_CACHE_ROOT=/data/temp/claude-code-harbor
CLAUDE_RUNTIME=/path/to/shared-root/claude-code-harbor/claude-code-runtime-2.1.308-srt-0.0.77.tar.gz

mkdir -p "${HARBOR_SIF_CACHE_ROOT}"
tar -xzf "${CLAUDE_RUNTIME}" -C "${HARBOR_SIF_CACHE_ROOT}"
patch -d "${HARBOR_SIF_CACHE_ROOT}/runtime/usr/local/lib/node_modules/@anthropic-ai/sandbox-runtime" \
  -p1 < examples/claude_code_harbor/sandbox-runtime-0.0.77-apptainer.patch
```

Harbor venv和Claude Code runtime都安装在`/data/temp/claude-code-harbor/`。

无法访问共享目录时，可在每台worker自行安装，或在构建训练镜像时安装Harbor 0.24.0
（commit `77d2f153`）、Python 3.12.13、Node.js 22.20.0、Claude Code 2.1.308、Sandbox Runtime
0.0.77、bubblewrap 0.10.0、ripgrep 14.1.1和socat 1.8.0.0，并保持相同的本地目录布局。

### Apptainer

```bash
apptainer config global --set "max loop devices" 8192
apptainer config global --set "systemd cgroups" no
apptainer config global --set "sessiondir max size" 8192
apptainer config global --set "mount slave" no
sysctl -w user.max_user_namespaces=65536
```

`max loop devices`是Apptainer可使用的loop device上限；设备仍由内核按需创建，设置为8192不会在启动时预创建8192个设备。

外层Pod需要可写或已委派的`cpu`、`memory`和`pids` cgroup subtree。

## Qwen3.8-Flash-Next：128 卡全异步 SC + Muon

独立脚本 `scripts/run-qwen38-flash-next-128xgpu-async-muon-sc.sh` 固定使用 64 卡 actor 和 64 卡 rollout；优势计算使用 CPU。rollout 为 TP8 的 2 个 Prefill、6 个 Decode engine，Prefill HiCache 为每 rank 96 GB，为模型加载和主机侧开销保留余量。

`config.yaml` 的 `claude_code.max_context_tokens` 设为 131072，脚本读取该值并统一设置训练 sequence length、rollout 和 SGLang context length。默认每轮 64 个 prompt group、每组 8 个 session，共 512 个逻辑训练样本；SC 保留零方差组。

在已准备 SGLang、Megatron 和 Muon 的训练 Python 环境中，从仓库根目录运行。模型、数据和 checkpoint 在所有 Ray 节点使用相同的绝对路径；Harbor runtime、venv 和可选 Muon 源码在每台 worker 使用相同的本地路径：

```text
${MODEL_DIR}/Qwen3.8-Flash-Next/
${DATA_DIR}/prompts.jsonl
${DATA_DIR}/tasks/
${SAVE_DIR}/${EXP_NAME}/
${HARBOR_SIF_CACHE_ROOT}/harbor-venv/bin/python
${HARBOR_SIF_CACHE_ROOT}/runtime/
${HARBOR_SIF_CACHE_ROOT}/muon/   # 可选：存在时自动加入 PYTHONPATH
```

如果未部署上述 Muon 源码目录，则使用训练环境中已安装的 Muon。Harbor runtime 和系统依赖按本文前面的环境配置准备。

先用完整 128 卡拓扑运行两轮小批量验收，每轮 2 个 prompt group × 8 个 session = 16 个逻辑训练样本：

```bash
export MODEL_DIR=/path/to/models
export DATA_DIR=/path/to/harbor-materialized
export SAVE_DIR=/path/to/checkpoints
export HARBOR_SIF_CACHE_ROOT=/path/to/worker-local/claude-code-harbor
export HARBOR_PYTHON="${HARBOR_SIF_CACHE_ROOT}/harbor-venv/bin/python"
export RAY_DASHBOARD_ADDRESS=http://ray-head:8265
export RELAX_SKIP_CLUSTER_CLEANUP=1
export EXP_NAME=qwen38-flash-next-cc-harness-harbor-128xgpu-async-muon-sc

NUM_ROLLOUT=2 ROLLOUT_BATCH_SIZE=2 N_SAMPLES=8 \
OVER_SAMPLING_BATCH_SIZE=3 AGENTIC_CONCURRENCY=4 \
bash scripts/entrypoint/ray-job.sh \
  examples/claude_code_harbor/scripts/run-qwen38-flash-next-128xgpu-async-muon-sc.sh \
  --save-interval 1
```

验收应确认至少完成 2 次 optimizer update、真实 Harbor 任务评分、训练权重同步到 rollout，以及 `${SAVE_DIR}/${EXP_NAME}` 下成功写入 checkpoint。通过后去掉小批量环境覆盖即可使用默认 512 样本批量；默认运行 200 轮，每 20 轮保存，最多保留 2 个 actor checkpoint。

`--load` 和 `--save` 都使用稳定的 `${SAVE_DIR}/${EXP_NAME}`，重启时保持这两个变量即可从已有 checkpoint 恢复；实验日志名带时间戳，但 checkpoint 路径不变。默认阻塞提交时，提交与 driver 输出位于 `examples/claude_code_harbor/scripts/log/${EXP_NAME}-<timestamp>.log`；设置 `RAY_NO_WAIT=1` 时该文件只保留提交输出，driver 日志用 `ray job logs --address="$RAY_DASHBOARD_ADDRESS" <job-id>` 查看。训练使用 ClearML 和 metrics service。ClearML 配置和凭证需要对 Ray worker 可见。

脚本不管理 GPU 占卡进程。本次验收由 `tmp` 内的外部监督器负责运行前释放占卡，以及结束或失败后恢复占卡。
