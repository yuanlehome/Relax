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
${MODEL_DIR}/Qwen3.6-35B-A3B
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
LOCAL_ROOT=/data/temp/wulumeng/claude-code-harbor
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
export HARBOR_SIF_CACHE_ROOT=/data/temp/wulumeng/claude-code-harbor
CLAUDE_RUNTIME=/path/to/shared-root/claude-code-harbor/claude-code-runtime-2.1.308-srt-0.0.77.tar.gz

mkdir -p "${HARBOR_SIF_CACHE_ROOT}"
tar -xzf "${CLAUDE_RUNTIME}" -C "${HARBOR_SIF_CACHE_ROOT}"
patch -d "${HARBOR_SIF_CACHE_ROOT}/runtime/usr/local/lib/node_modules/@anthropic-ai/sandbox-runtime" \
  -p1 < examples/claude_code_harbor/sandbox-runtime-0.0.77-apptainer.patch
```

Harbor venv和Claude Code runtime都安装在`/data/temp/wulumeng/claude-code-harbor/`。

无法访问共享目录时，可在每台worker自行安装，或在构建训练镜像时安装Harbor 0.24.0
（commit `77d2f153`）、Python 3.12.13、Node.js 22.20.0、Claude Code 2.1.308、Sandbox Runtime
0.0.77、bubblewrap 0.10.0、ripgrep 14.1.1和socat 1.8.0.0，并保持相同的本地目录布局。

### Apptainer

```bash
apptainer config global --set "systemd cgroups" no
apptainer config global --set "sessiondir max size" 8192
apptainer config global --set "mount slave" no
sysctl -w user.max_user_namespaces=65536
```

外层Pod需要可写或已委派的`cpu`、`memory`和`pids` cgroup subtree。

## 启动任务

当前脚本使用两个8-GPU Ray节点：actor使用8张GPU，rollout使用8张GPU。

```bash
export MODEL_DIR=/path/to/models
export SAVE_DIR=/path/to/checkpoints
export RUN_ROOT=/path/to/shared-root/claude-code-harbor/runs

export HARBOR_PYTHON=/data/temp/wulumeng/claude-code-harbor/harbor-venv/bin/python
export HARBOR_TASK_ROOT=/path/to/shared-root/harbor-materialized
export HARBOR_SIF_CACHE_ROOT=/data/temp/wulumeng/claude-code-harbor

export RAY_ADDRESS=<ray-head>:6379
export RAY_DASHBOARD_ADDRESS=http://<ray-head>:8265

bash scripts/entrypoint/ray-job.sh \
  examples/claude_code_harbor/run_qwen36_35b_a3b_16xh800.sh
```

ClearML配置或credential需要对Ray worker可见。

每次启动在`${RUN_ROOT}/${EXP_NAME}-<timestamp>/`下写入`run.log`和`dump/`。

默认脚本读取`${SAVE_DIR}/${EXP_NAME}`作为恢复路径。需要写入checkpoint时，启用启动脚本中的
`--save`、`--save-interval`和`--max-actor-ckpt-to-keep`。
