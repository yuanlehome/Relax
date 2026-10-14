# Claude Code / Harbor：代码、配置与一键运行

本示例把 Claude Code 的真实 OCI 工具执行接入 Relax Agentic rollout，用 Qwen3.6-35B-A3B 做 GRPO 训练。Claude Code 负责工具循环，NeMo Gym 管理任务环境与 verifier；模型请求回到 Relax 的 SGLang，训练使用 Megatron。

本文对应 **2026-10-14 的 loss/timeout 修复版本**。这次文档提交记录了工作区和独立部署 release 中的实现；其中 `run-harbor.sh`、`prompt_loss.py` 等实现尚未随本文提交。运行前必须使用包含这些文件及配套改动的完整代码目录，不能仅凭本文所在 Git commit 认定实现齐备。当前代码位置、文件 SHA-256 和运行交接记录分别见仓库中的 `reports/harbor-code-inventory-20261014.json`、`reports/harbor-handoff-20261014.md`。

## 从哪里开始

正常使用顺序是：准备一套独占的 Ray 集群与共享输入，填写路径，运行配置检查，然后在 **Ray head 上执行一次** 一键入口。入口使用现有集群，不负责新建 QS Job/Trial、下载模型或安装基础训练镜像。

需要的环境：

- 16 个 GPU worker，每个 8 张 H800，共 128 GPU；通常另有一个 CPU head。
- 已包含 PyTorch、CUDA、Ray、SGLang 和 Megatron 的 Relax 训练镜像。Gym builder 使用 Linux CPython 3.12 系统 Python；Megatron 默认位于 `/root/Megatron-LM`。
- 每个节点以相同绝对路径访问完整代码、原始数据及其引用上传文件、模型、prepared data 和 bundle。
- 原始数据版本为 `20261009-150037-mimo-v26-code-claude_code-srt-gitclean-w200k`。准备逻辑会校验其摘要、2683 条记录、verifier 目录和上传资产；不能随意换一个同名 JSONL。
- 模型目录内有配置、tokenizer 和全部权重分片；本模型有 26 个分片。
- 至少一个 GPU worker 支持 user namespace 和可用 FUSE。其余节点可通过认证 forwarder 访问 native Gym，数量由探测决定。
- Gym 端口和跨节点 callback 可达，任务镜像仓库可访问。离线 OCI 包只免去工具下载，不包含任务镜像或模型。

::: warning 专用集群
后续 `scripts/entrypoint/ray-job.sh` 包含 Relax 的旧训练进程、Serve 和 placement group 清理。入口会检查其他活动 Job、Serve、PG 和 Ray 可用 GPU，但不会检查所有物理 GPU 进程。提交前另外核对物理显存及 compute 进程，并在整个准入与提交期间保持集群独占。
:::

## 一键运行

### 两个输入的简便方式

从共享盘上的可写目录执行，路径替换为实际挂载位置：

```bash
export CLOUD_ROOT=/shared/cloud
export SAVE_DIR=/shared/cloud/harbor/checkpoints
export EXP_NAME=harbor-qwen36-loss-timeout-run001

bash /shared/cloud/Relax/examples/claude_code_harbor/scripts/run-harbor.sh \
  --source-root /shared/cloud/inputs/20261009-150037-mimo-v26-code-claude_code-srt-gitclean-w200k \
  --model-dir /shared/cloud/models/Qwen3.6-35B-A3B \
  --check-config

# 配置检查通过后，只执行一次正常启动。
bash /shared/cloud/Relax/examples/claude_code_harbor/scripts/run-harbor.sh \
  --source-root /shared/cloud/inputs/20261009-150037-mimo-v26-code-claude_code-srt-gitclean-w200k \
  --model-dir /shared/cloud/models/Qwen3.6-35B-A3B
```

`--model-dir` 接受模型目录或其父目录。显式设置 `CLOUD_ROOT` 可以避免依赖原始数据目录布局的自动推导。若移动了源项目布局，还需配置下节中的 wheel 路径及必要的 `ORIGINAL_CLOUD_ROOT`；只挂载数据目录可能缺少它引用的共享上传文件。

默认在 `SAVE_DIR` 的父目录下生成 `harbor-runs/<时间>-<随机后缀>/prepared` 和 `bundle`。若未设置 `SAVE_DIR`，默认是当前工作目录下的 `harbor-checkpoints`，且该默认位置必须位于共享 `CLOUD_ROOT` 内。检查点位于 `SAVE_DIR/EXP_NAME`，不会放在随机运行目录中。已有同名检查点目录会被拒绝，不会静默恢复。

### 使用显式配置文件

为了保留明确的输出位置，推荐复制 `examples/claude_code_harbor/scripts/harbor.env.example` 到 Git 外的私有文件。示例：

```bash
# /private/config/harbor.env
CLOUD_ROOT=/shared/cloud
SOURCE_ROOT=inputs/20261009-150037-mimo-v26-code-claude_code-srt-gitclean-w200k
MODEL_DIR=models
DATA_ROOT=harbor/runs/run001/prepared
BUNDLE_DIR=harbor/runs/run001/bundle
SAVE_DIR=harbor/checkpoints
EXP_NAME=harbor-qwen36-loss-timeout-run001
PROJECT_NAME=Relax/agentic/claude-code-harbor
BBS_WHEEL_PATH=inputs/wheels/dressage_blackbox_server-1.0.0-py3-none-any.whl
DRESSAGE_WHEEL_PATH=inputs/wheels/dressage-0.1.0-py3-none-any.whl
```

```bash
bash /shared/cloud/Relax/examples/claude_code_harbor/scripts/run-harbor.sh \
  --env-file /private/config/harbor.env --check-config

bash /shared/cloud/Relax/examples/claude_code_harbor/scripts/run-harbor.sh \
  --env-file /private/config/harbor.env
```

`--env-file` 通过 shell `source` 执行并自动 export，必须是可信文件；有空格的值需要引号。凭证通过现有安全环境提供，不写入文档、配置模板或 Git。入口强制使用自身 checkout 作为 `REPO`，避免混用两个版本的 helpers。

`--check-config` 检查设置和目录冲突，不安装、不连接 Ray，也不验证完整输入、GPU 空闲或服务健康；配置文件自身仍会执行。`--prepare-only` 会准备数据、Gym、OCI 和私有认证文件，但不启动服务或训练。它创建的 bundle 不能再用于正常一键启动；之后应使用新 `BUNDLE_DIR` 并显式复用 prepared data/Gym。

长任务应由 QS RayJob 入口或持久 Ray outer driver 承载。临时 `qs exec` 会话不能替代 owner 的生命周期。不要在每个 worker 启动一份，也不要在网络返回不明时再次点击提交。

## 入口到底执行了什么

```text
run-harbor.sh
  -> bootstrap_qs.py
     -> prepare_data.py -> verify_inputs.py
     -> build/check Gym -> install OCI -> create private token
     -> identify head / workers -> check exclusive cluster
     -> probe native OCI capability
     -> first native Gym + cross-node callback verification
     -> remaining native Gyms + forwarders -> verify every node
     -> write child submission ID to bootstrap-status.json
     -> repro/claude-code-harbor-r5/scripts/launch.sh
        -> scripts/entrypoint/ray-job.sh
        -> run-qwen36-35B-A3B-128xgpu.sh
        -> relax.entrypoints.train
     -> observe child via Jobs API -> retain services
```

每个存活节点各有一个不占训练 GPU 的 `HarborNode` actor，负责持有节点服务。native 节点运行私有 OCI 和 Gym；forwarder 节点提供本地 Gym 入口并转发到 native 节点。先验证首个 native 节点的真实跨节点模型 callback，再扩展并逐一验证所有节点。受控验证使用 stub 模型，它的 model calls、contexts、callback 次数都不是训练进度。

Gym 从固定 NeMo commit 和精确版本依赖构建，基础 Torch/Ray/CUDA 继承训练镜像。Claude Code、Node、SRT 及原始两个 wheel 作为任务上传资产在 OCI 内安装，不安装到 Relax 主机环境。原始数据保持不变，prepared data 中的资产路径改为相对路径；普通 OCI 镜像映射通过已校验的命名规则产生，不转换镜像内容。

## 路径与服务配置

以下为一键入口配置，不是所有 Relax 参数的全局默认值。

| 设置 | 默认或要求 | 用途与注意点 |
| --- | --- | --- |
| `CLOUD_ROOT` | env 模式必填；两输入模式可按原始布局推导 | 相对路径的共享盘根 |
| `SOURCE_ROOT` | 默认必填 | 原始数据及引用资产；不能用 prepared data 冒充 |
| `MODEL_DIR` | env 模式必须是模型父目录 | 其下应有 `Qwen3.6-35B-A3B/`；CLI 参数才同时接受模型目录 |
| `DATA_ROOT` | 新目录 | 存放 prepared data；训练实际使用其下 `aligned-150037-oci/data.jsonl` |
| `BUNDLE_DIR` | 每次新目录 | 环境、服务状态、验证结果、私有日志 |
| `SAVE_DIR` / `EXP_NAME` | 稳定共享根 / 唯一实验名 | 检查点根为二者拼接；输出不能与输入或其他输出重叠 |
| `BBS_WHEEL_PATH` / `DRESSAGE_WHEEL_PATH` | 可自动发现或显式设置 | 原始两个 wheel；自动发现时按 `source-assets.json` 校验；显式路径需事先自行核对尺寸与 SHA-256 |
| `ORIGINAL_CLOUD_ROOT` | 需要时显式指定 | 源项目目录重排时，帮助将历史绝对路径映射到新挂载 |
| `HARBOR_REUSE_DATA` | `0` | `1` 表示显式复用 prepared data；仍做输入校验。本次从原始数据重跑应保持 `0` |
| `GYM_RUNTIME_DIR` | 新 bundle 的 `runtime/gym` | 设置已有目录即原位复用；只检查，日志写到新 bundle；不能移动 venv |
| `OCI_PACKAGES_DIR` | 未设置则下载固定包 | 离线目录须有锁定的四个包；新 bundle 仍重新安装 OCI |
| `MEGATRON` | `/root/Megatron-LM` | 镜像中已有 checkout；入口不安装 Megatron |
| `RAY_ADDRESS` | `auto` | 已有集群 GCS；连接后取实际地址 |
| `RAY_DASHBOARD_ADDRESS` | head 地址，端口默认 `8265` | Jobs API 的 HTTP(S) 地址；可用 `RAY_DASHBOARD_PORT` 调整推导端口 |
| `GYM_PORT` / `GYM_UPLINK_PORT` | `27980` / `27981` | 节点本地入口 / native uplink；必须不同且空闲 |
| `OCI_ROOT` | 自动短随机本地目录 | 私有 Docker 状态及 Unix socket；不要使用过长共享盘路径 |
| `PROJECT_NAME` | `Relax/agentic/claude-code-harbor` | ClearML 项目；任务名是 `EXP_NAME` 加时间戳 |
| `PYTHON` | `python3` | wrapper 使用的 Python，可指向训练镜像系统 Python |

显式复用 Gym 与离线包时仍使用新 bundle 和新的 OCI 安装。不要 symlink 整个旧 bundle。完整的 `prepare-only` 后复用方式是设置旧 `DATA_ROOT`、`HARBOR_REUSE_DATA=1`、旧原位 `GYM_RUNTIME_DIR`，另选全新 `BUNDLE_DIR` 和未占用的 `SAVE_DIR/EXP_NAME`。

## 训练参数及含义

参数来自 `examples/claude_code_harbor/scripts/run-qwen36-35B-A3B-128xgpu.sh`。`run-harbor.sh` 只接收前述入口参数，不接受 `--lr`、`--num-rollout` 等任意训练参数。要制作变体，应修改并审查 recipe，再走正常入口；低层 `launch.sh` 的参数透传不承担完整环境准备和服务验证。

| 类别 | 本 recipe 设置 | 含义 |
| --- | --- | --- |
| GPU / 模式 | actor 64 + rollout 64，`--hybrid` | 共 128 H800；无独立 eval GPU 池；ref/actor forward 等复用训练侧 |
| 更新计划 | 128 rollout；每 rollout 1 step；每 update 1 iter | 完整训练目标为 128 个实际 optimizer 更新；1 不是只跑一次训练 |
| 逻辑 batch | 32 prompts × 8 Sessions；GBS 256 | context fanout 不扩大逻辑 batch；筛选和跨轮结转可能改变实际尝试 Session 数 |
| 并发 | `--agentic-concurrency 64` | 以 prompt group 为单位，配置可对应 512 个逻辑 Sessions 的并发预算 |
| GRPO | 8 samples；clip `0.2 / 0.28`；TIS 开启 | 每组逻辑 Session 归一化；同 reward 整组过滤补采 |
| 正则 | KL、KL loss、entropy 系数均 `0` | 本 recipe 不添加这些正则项 |
| Loss | `--prompt-equal-loss` 和 `--calculate-per-token-loss` 同时启用 | 目标按 prompt 归一化；后者保留 MCore CP 接口，详见下节 |
| 优化器 | Adam；LR `1e-6` 恒定；WD `0.1`；betas `0.9 / 0.98` | CPU optimizer offload、异步搬运、precision-aware optimizer |
| 并行 | TP 2 / PP 1 / CP 32 / EP 8 / ETP 1 | 静态 CP；sequence parallel 开启 |
| 激活 / packing | full uniform 重算 1 层；micro batch 1；动态 batch；THD | `max-tokens-per-gpu=8192` 是训练 packing 预算 |
| Log-prob | token 预算 `10240`；chunk `32` | 控制 log-prob forward 峰值，不是生成长度 |
| 生成 | temperature `1.0`；每次请求 response 最长 `8192`；context `262144` | 多轮 Session 总输出可能包含多个请求 |
| 工具循环 | compaction window `200000`；最多 `200` 步 | context 压缩与 agent 步数限制，不等于 optimizer step |
| SGLang | 8 GPU/engine；静态显存比例 `0.85`；最大运行请求 `256`；server concurrency `512` | 路由策略 `consistent_hashing` |
| 权重更新 / staleness | 每 1 次同步；`max-staleness=4` | 当前是队列领先限制，并非业务 token 版本检查 |
| MoE | flex + DeepEP；load balancing `none`；aux coeff `0.0` | prompt-equal 模式拒绝非零 auxiliary loss |
| 保存 | 每 `16` 次保存；保留 `2` 个 actor checkpoint | load/save 使用同一稳定目录；入口不自动恢复 |
| 跟踪 | ClearML + metrics service | 跳过训练前 eval；当前 recipe 没有独立评测配置 |

### Loss 如何对齐

对完整 optimizer batch，令 `B` 为配置的 GBS，`P` 为不同源 prompt 的数量，`M_p` 为 prompt `p` 下全部逻辑 Session 的原始有效 response token 总数。每个 prompt 的分母为：

```text
D_p = M_p * P / B
L = (1 / B_eff) * sum_context [sum_token(m' * token_loss) / max(D_p, 1)]
```

`B_eff` 是本次实际逻辑 batch 大小。通常 `B_eff=B` 且不触发分母下限时，等价于先对每个 prompt 的 token loss 求均值，再对 prompt 等权平均。TIS 可以改变分子中的有效 mask `m'`，但不改变原始 `D_p`。

元数据先按 Session 去重，再跨不含 CP 的 DP group 聚合；compaction 产生多个 context 和 CP 切片都不会重复增加 Session 权重。后端仍使用 token-normalized MCore 接口，reducer 的额外缩放抵消其 token 分母，所以不能直接去掉 `--calculate-per-token-loss`。

这条路径限制为 Megatron + Agentic + hybrid + GRPO + `policy_loss` + 静态 CP，且每 rollout 一步、每 update 一 iter；不支持 OPD、MTP、非零 MoE auxiliary loss 或自定义 PG reducer。参数校验兼容 Megatron 返回的标量、list、tuple 零系数，拒绝非法值和非零项。

### 7200 秒与 10800 秒分别约束什么

| 设置 | 计时范围 | 到期行为 |
| --- | --- | --- |
| `harbor_backend_timeout_s=7200` | sandbox agent request 发起后的墙钟；暂停也计时；不含此前 OCI 准备和之后 verifier | 关闭新 model 请求、停止 CLI、收完已接受 canonical 请求；保留真实 tokens/logprobs，导出 `TRUNCATED`、reward 0，跳过 verifier |
| `--agent-timeout 10800` | 外层 ManagedCommand active-time budget；generation gate 暂停时暂停计时 | 外层安全截止并报执行错误；不能当作业务 reward 0 |
| `NEMO_GYM_DEADLINE_S=2147483` | Gym 客户端 transport lease | 允许跨训练暂停保持请求，不替代上面两种截止机制 |

7200 秒之后的已受理模型请求仍需收取结果，因此整个 Session 不保证恰在第 7200 秒退出。没有已捕获的真实 response 时不能伪造可训练轨迹；HTTP 非 200、transport 错误和普通 CLI 故障仍报错。`max_steps`、`context_overflow` 的受控终止仍可走 verifier，不自动等同于 backend timeout。

### 与业务任务的对齐边界

参考为 Job 317717 / Trial 2156263 的冻结配置。此次选择对齐 prompt loss 和 **7200 秒墙钟** 超时行为；保留 128 H800，以及现有 staleness。

业务 GPU 分配为 actor 64 / rollout 128 / eval 32，共 224；本 recipe 为 64 / 64，无独立 eval 池。业务按轨迹最旧生成 token 的模型版本计算年龄，超过 4 或版本缺失时丢整组并补采。当前 `--max-staleness 4` 检查 `current_rollout_id + 1 - oldest_partition_id <= 4`，超过后等待队列消费。两者数字相同但语义不同。

训练 packing 预算也存在 `8192` 与业务 `10000` 的差别。因此本示例不代表与业务完全同配置，也不保证相同 reward 曲线。

## 如何看训练与 reward

先读 `BUNDLE_DIR/bootstrap-status.json` 的最新时间、stage 和 `training_submission_id`，再通过 Jobs API 查询这个 child。状态文件采用增量更新，历史字段可能仍存在；不能把旧的健康字段当作当前状态。

| 位置 | 内容 |
| --- | --- |
| `BUNDLE_DIR/bootstrap-status.json` | 阶段、唯一 child ID、Jobs 状态、服务健康 |
| `BUNDLE_DIR/bootstrap_logs/` | prepare/install/verify 的私有日志，及 `training.log`、`failure.log` |
| `BUNDLE_DIR/service_state/<node-ip>/` | 节点服务、PID、受控验证状态及私有 artifacts |
| `SAVE_DIR/EXP_NAME/data/harbor.jsonl` | recipe 增补 routing、prompt ID、timeout 元数据后的训练输入 |
| `SAVE_DIR/EXP_NAME/` | 实际 checkpoint 及 tracker；有普通文件不等于有 checkpoint |
| `examples/claude_code_harbor/scripts/log/` | recipe 的额外 tee 日志，因此该目录需可写 |

ClearML 项目由 `PROJECT_NAME` 指定，实验名由 `EXP_NAME` 加启动时间组成。按该名称和启动时间查找 task，并核对日志中的 ClearML task 链接；不要把同一个 QS Trial 的多个 attempt 当成一个 ClearML run。当前 recipe 使用 ClearML，不代表业务任务也一定上报 ClearML。

实际 optimizer 更新要核验 step 与 `train/step` 一致、有限且正的 `grad_norm`、`global_batch_size=256`、无 skip，并按索引去重。`Finish rollout`、Session 数、context row 数或 optimizer cache release 次数都不能代替更新次数。完整成功需同时满足 128 个实际更新、唯一 child `SUCCEEDED`、最终 checkpoint 可读。

reward 的统计单位同样需要区分：

- 一个 prompt 生成 8 个逻辑 Session；一个 Session 可以导出多个 context，所有 context 共享该 Session 的 `outcome_reward` 和 advantage。
- 仅末尾 context 保留一个 raw reward anchor，其余 `reward=None`；按 anchor 汇总可得到逻辑轨迹 reward。直接平均全部 context 的 `outcome_reward` 会按 context 数加权。
- GRPO 使用逻辑 Session 的总体标准差，无附加 denominator epsilon；全组 reward 相同则过滤。标准差大于 `1e-6` 才除，否则保留 centered reward。
- 对照业务应优先使用实际入训 batch 的 `raw_reward_trajectory_mean`。生成轮与入训 batch 边界不同；动态筛选后的 reward 也不是固定评测成功率。
- reward 下降能描述采样结果变差，但仅凭一条曲线不能唯一归因于 loss、timeout 或 staleness。应对照同一 batch 边界下的零奖励比例、超时与其他失败、筛选/补采比例及任务构成。

## 停止、失败与恢复

child 结束后，outer 会进入 `services_retained` 并继续持有普通 Ray actors。outer `RUNNING`、服务健康或阶段到达 `services_retained` 都不表示训练成功。若启动失败发生在服务启动之后，owner 也可能保留服务供检查。

通用入口拒绝已存在的输出目录，独占创建 bundle 和私有阶段日志，并在调用 launcher 前原子记录 child ID。它没有自动恢复、自动重试或完整进程清理功能。此次运维中的 `outer-submit-intent.json`、runroot `launch-record.json` 是外部提交控制器的额外防重记录，不是通用 `run-harbor.sh` 自带的锁。

恢复前依次确认：当前 QS/Pod 是否可用、准确 outer/child 身份和终态、已完成更新、实际 checkpoint/tracker、旧服务和私有 OCI 是否仍占用资源。接口超时不等于提交失败，应先查原 ID。若有 checkpoint，需要专门核对恢复配置；对旧目录直接再次运行一键入口不会续训。若 QS Pod 已消失，最后一次快照只能标为历史观测，当前 child/服务状态应记为未知，不能沿用 `RUNNING` 或推断最终进度。

| 现象 | 优先检查 |
| --- | --- |
| 输出目录已存在 | 是否重复启动；先核对原 child，不覆盖目录 |
| Gym 构建下载失败 | 检查私有构建日志、NeMo Git/pip 网络；已有完成 runtime 可显式原位复用 |
| OCI 包下载失败 | 按 `oci-packages.json` 准备锁定的离线 OCI 包并校验 |
| OCI socket 路径过长 | 使用默认短本地 `OCI_ROOT`，不要把 socket 放到深层共享盘路径 |
| 无 native worker / callback 不通 | user namespace、FUSE、服务端口和真实跨节点路径；readyz 通过不足以证明 callback 可用 |
| 一直停在节点验证 | 查看最新 stage/时间和验证状态；stub 计数尚不代表训练提交 |
| child 参数校验失败 | recipe 与 Megatron parser 契约，尤其 aux coefficient 的 list 形式；不要直接重复同一 launch |
| 训练等待队列 | 联合查看最近日志时间、Session/accounting 与真实 step；累计旧 warning 不等于当前阻塞 |

## 代码索引

除另有前缀外，下表路径位于 `examples/claude_code_harbor/`。它列出运行相关实现；工作区快照的文件版本见代码 inventory。

| 代码 | 职责 |
| --- | --- |
| `scripts/run-harbor.sh`、`scripts/harbor.env.example` | 公开一键入口、可信环境文件模板 |
| `scripts/bootstrap_qs.py` | 准备环境、集群准入、HarborNode、节点验证、提交 child、保留服务 |
| `scripts/prepare_data.py`、`scripts/source-assets.json` | 固定数据和原始 wheel 的解析、校验与 prepared data 构建 |
| `asset_paths.py`、`scripts/cloud_paths.sh` | 云盘路径解析与搬迁 |
| `scripts/prepare_data_via_qs.py` | 两个 QS Pod 间的准备与传输辅助；不是正常单集群入口 |
| `scripts/build_gym_environment.py`、`scripts/check_gym_environment.py`、`scripts/install_gym_runtime.sh` | 固定 NeMo 源码与 Gym 环境构建/检查 |
| `scripts/gym-requirements.txt`、`scripts/gym-base-requirements.txt` | overlay 与继承基础依赖的精确版本 |
| `scripts/fetch_oci_packages.py`、`scripts/oci-packages.json`、`scripts/install_oci_runtime.sh`、`scripts/oci_paths.py` | OCI 包校验、私有安装和短 socket 路径 |
| `scripts/start_gym.sh`、`scripts/forward_gym.py`、`gym_service.py` | 常驻 Gym 和认证转发服务 |
| `nemo_agent.py` | Harbor任务生命周期、7200秒截止、verifier及结果导出 |
| `oci.py`、`sandbox_worker.py` | 私有 OCI 任务执行与 sandbox 命令协议 |
| `scripts/install_task_runtime.sh`、`mimo_git_reflog.patch` | 在任务容器内安装原始运行资产和 Git 兼容修复 |
| `relay.py`、`request_normalization.py` | CLI请求规范化、调用 Relax模型、截止后收取已受理结果 |
| `contexts.py` | 捕获 canonical contexts、context边界和唯一 response 训练归属 |
| `run_agent.py`、`claude_settings.json` | Agentic进程桥、JSONL解析、context输出与CLI设置 |
| `training_data.py`、`advantage.py` | context转训练row、重复response mask、逻辑Session GRPO与动态筛选 |
| `scripts/verify_oci.py` | 受控真实OCI、工具循环、compaction、verifier保护和callback验证 |
| `scripts/run-qwen36-35B-A3B-128xgpu.sh` | 固定训练recipe、训练输入元数据、ClearML、child提交 |

其他目录中的接入点：

| 路径 | 职责 |
| --- | --- |
| `repro/claude-code-harbor-r5/scripts/launch.sh`、`start_node.sh` | 一键仍调用的训练提交和节点服务脚本；目录名r5不表示当前实验名也是r5 |
| 同目录 `verify_inputs.py`、`create_gateway_token.py` | 输入完整性与主机端私有认证文件 |
| 同目录 `prepare_environment.py`、`verify_bundle.py` | 历史bundle准备/核验辅助；新源码构建不依赖旧环境归档 |
| `examples/nemo_gym_agentic/` | Gym thin client、受控agent进程和通用协议适配 |
| `relax/agentic/` | Session、generation gate、runtime、组生命周期与导出 |
| `relax/backends/megatron/prompt_loss.py` | prompt分母聚合、契约校验和后端归一化补偿 |
| `relax/backends/megatron/actor.py`、`data.py`、`model.py`、`loss.py`、`cp_utils.py` | 完整batch元数据、packing、CP与loss连接 |
| `relax/utils/arguments.py`、`utils.py`、`data/stream_dataloader.py` | opt-in参数校验及元数据传递 |
| `relax/components/rollout.py` | 现有队列staleness控制，本次未改其语义 |
| `scripts/models/qwen36-35B-A3B.sh`、`scripts/entrypoint/ray-job.sh` | 模型结构与集群训练入口 |
| `tests/examples/claude_code_harbor/`、`tests/backends/megatron/test_prompt_equal_loss.py` | 对应模块回归与prompt loss数值检查 |

理解训练侧资源和权重同步可继续阅读 [Hybrid训练](../guide/hybrid-training.md)。运行环境的独立构建命令位于 `examples/claude_code_harbor/ENVIRONMENT.md`；其中历史说明应结合本文的版本边界使用。
