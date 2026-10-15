# Harbor / Claude Code 128 H800：验证与调效果交接

记录时点：2026-10-15T18:43:26+08:00。运行状态会变化，下面的 v3 状态是此时快照。

这是运行证据的交接快照。文件位置直接使用本次运行的云盘绝对路径；代码和文档随 Git 发布，tensor、日志、原生 verifier 结果及验证记录保存在云盘。平台地址仍用 `${PLATFORM_WEB}`、`${CLEARML_WEB}`、`${RAY_ADDRESS}`、`${RAY_DASHBOARD_ADDRESS}` 和 `${IMAGE_REPOSITORY}` 表示，完整连接配置见 `/mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax/repro/runs/harbor-dump2-2220455-v3/handoff/recorded-launch.sh`。文中的运行状态对应所列快照时点，不能用来判断实时状态。

## 发布说明

本次将实际 v3 冻结运行副本中尚未提交的 Harbor、prompt 等权/CP 对齐及两步 raw dump 实现发布到 `codex/claude-code-agentic-128xh800`。保留 Git HEAD `1ec9ba96d5ffce14c98daa82f51df6be6e39ba48` 已有较新指南和导航，不回退到运行副本的旧文档。逐文件 deployed / published SHA-256 见 [代码清单](harbor-code-inventory-20261015.json)。大型 tensor、日志、原生 verifier 结果和部署环境记录继续保存在共享目录。

发布版修复了一处兼容问题：仅当显式选择 `relax.utils.training.train_dump_utils.enable_raw_policy_capture` hook 且处于前两次更新时，训练 dump 才采用 raw 审计格式；普通 `--save-debug-train-data` 及后续更新保留原有解码格式。rollout raw 格式同样由该 hook 显式启用。新增 persistent-agent 和通用/raw dump 分派回归测试。其余差异包括仓库 hooks 的必要格式调整，完整差异以代码清单和 Git diff 为准。

这次发布没有修改或重启 v3 冻结运行副本，也未恢复原长跑；不能把发布 commit 视为 v3 的逐字节部署身份。v3 自然结束后的只读审计仍按下文收尾器运行。发布检查结果另见代码清单的 `validation` 字段及 `/mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax/repro/runs/harbor-dump2-2220455-v3/handoff/publication-validation/`；CPU 检查不替代多节点 GPU 效果验收。

## 交接结论

**首步实际数值链路验证通过；两步诊断尚未完成；没有满足“超过20步并持续reward上涨”的效果验收。**

- 原长跑完成真实 optimizer step0–7，共8次更新，已停止并保存完整 `iter_0000007`。没有自动恢复。
- dump2-v2 从初始HF模型运行，完成1次真实更新。首步token/logprob/mask完整落盘并经过CPU逐token审计；第二次更新之前因Gloo健康检查超时失败。v2没有iter0/iter1训练checkpoint，不能把数值dump当成可恢复模型。
- dump2-v3 已提交，当前 `RUNNING`，Ray job `35000000`。目前已确认 optimizer steps 为 `[]`；当前没有v3 reward可供判断。目标仅为2次更新，并按每步保存配置运行。
- 未证明梯度/optimizer完整回放正确；没有独立重跑真实请求tokenizer；没有固定评测集上的学习改善结论。

## 当前跑的是什么

| 项目                    | 当前值                                                                                                              |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------- |
| QS                      | [Job330975 / Trial2220455](${PLATFORM_WEB}/qs/model/production/job/trial/330975/2220455)                            |
| 资源                    | 16×8 H800，actor64 + rollout64；pevek队列，Stable独占                                                               |
| 实际优先级              | 最近QS回读10；原提交请求20，不能把最高优视为已验证                                                                  |
| 镜像                    | `${IMAGE_REPOSITORY}:qs-20260922-42511f6a`                                                                          |
| QS容器启动命令          | `sleep infinity`；实际训练由下面Ray child提交                                                                       |
| Ray head / Jobs API     | `${RAY_ADDRESS}` / `${RAY_DASHBOARD_ADDRESS}`                                                                       |
| 最新submission          | `harbor-dump2-2220455-20261015-v3`                                                                                  |
| 最新Ray job             | `35000000`                                                                                                          |
| 模型                    | `/mnt/tidalfs-bdsz01/dataset/pevek/checkpoints/opensource/Qwen3.6-35B-A3B`                                          |
| 当前代码副本            | `/mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax/repro/releases/b582cfc811ed-alignfix-2220455-dump2-v3-20261015` |
| 完整代码manifest SHA256 | `8162be534c2c886d70206359329b371bc9115fd558ce0a1e4352cd396922ba5e`                                                  |
| 训练脚本                | `scripts/training/text/run-qwen36-35B-A3B-128xgpu-harbor-dump2.sh`                                                  |
| 当前产物根目录          | `/mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax/repro/runs/harbor-dump2-2220455-v3`                             |

用户云盘仓库：`/mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax`，当前分支 `codex/claude-code-agentic-128xh800`，工作树HEAD `1ec9ba96d5ffce14c98daa82f51df6be6e39ba48`。实际运行代码使用上面的冻结副本；manifest记录的base commit为 `b582cfc811ed6dc9b7bb085cc3e07488993eb800`，继承已有alignfix和dump overlay。精确文件内容见 [code manifest](/mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax/repro/runs/harbor-dump2-2220455-v3/provenance/code-manifest.json)。该运行快照生成时，尚未提交或推送用户工作树；本次发布内容见上方“发布说明”。

当前与v2相比，新增训练脚本至 `scripts/training/text/`，配置改为 `--distributed-timeout-minutes 180`、`--save-interval 1`；模型、数据、并行拓扑、batch及训练数学沿用v2冻结代码。脚本经过 `bash -n` 和实际参数检查，当前新实验从初始HF加载。

关键配置：hybrid fully_async、TP2 / CP32 / EP8 / PP1 / DP(exclCP)1；GRPO；32 Groups×8 Sessions=GBS256；每次rollout mini仅1次optimizer update；prompt等权；TIS `[0,2]`；LR=1e-6；entropy coefficient=0；KL coefficient=0；dropout=0；staleness=4；recompute-num-layers=1；num-rollout=2。不要把40层网络误记为recompute-num-layers40。

## 启动命令

完整实际环境与提交命令已记录在 [recorded-launch.sh](/mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax/repro/runs/harbor-dump2-2220455-v3/handoff/recorded-launch.sh)。它包含已有Gym的URL、环境/config名称及token文件路径，不包含token内容；当前job已在运行，文件用于交接和复现记录。

入口命令为：

```bash
cd /mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax/repro/releases/b582cfc811ed-alignfix-2220455-dump2-v3-20261015
RAY_NO_WAIT=1 RAY_ADDRESS=${RAY_ADDRESS} \
RAY_DASHBOARD_ADDRESS=${RAY_DASHBOARD_ADDRESS} \
bash scripts/entrypoint/ray-job.sh scripts/training/text/run-qwen36-35B-A3B-128xgpu-harbor-dump2.sh
```

上面短命令需配合记录文件中的MODEL_DIR、DATA_DIR、SAVE_DIR、EXP_NAME、Gym及代码环境变量。完整展开的 `python3 -m relax.entrypoints.train ...` 见 [effective-training-command.txt](/mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax/repro/runs/harbor-dump2-2220455-v3/handoff/effective-training-command.txt)。

用户原始数据集：`/mnt/tidalfs-bdsz01/dataset/pevek01/youhj/codes/swe/data/rl_dataset/20261009-150037-mimo-v26-code-claude_code-srt-gitclean-w200k`。

实际DATA_DIR：`/mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax/repro/runs/harbor-validation-2220455/prepared/aligned-150037-oci`。准备数据用于OCI执行和字段路由；原始data.jsonl快照与用户原始文件字节SHA一致：`6285be128a1d0a1d77a203b1a262a5a12ca5e6bc3f7ff62b8e910b36ba12cfda`。证据见 [original-source-data-equality.json](/mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax/repro/runs/harbor-dump2-2220455-v2/provenance/original-source-data-equality.json)。

## ClearML、日志与checkpoint

| 运行         | 状态/更新次数                      | ClearML                                                                                                                           | 日志                                                                                                                                       |
| ------------ | ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| 最新dump2-v3 | `RUNNING`；此时未完成首步          | [最新ClearML](${CLEARML_WEB}/projects/5c0d02bc627a45f0b07524fff5697359/experiments/1d34851c104e4f11b61a4e1969591a64/output/log)   | [driver.log](/mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax/repro/runs/harbor-dump2-2220455-v3/control/driver.log)                     |
| dump2-v2     | FAILED；真实step0完成，step1未完成 | [v2 ClearML](${CLEARML_WEB}/projects/5c0d02bc627a45f0b07524fff5697359/experiments/ec88d196f0be4e789c225a89e554ee1a/output/log)    | [v2 driver.log](/mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax/repro/runs/harbor-dump2-2220455-v2/control/driver.log)                  |
| 原长跑       | STOPPED；真实step0–7完成           | [原长跑ClearML](${CLEARML_WEB}/projects/5c0d02bc627a45f0b07524fff5697359/experiments/a60a201ab5ee46b9985caca37a16f1d7/output/log) | [training.log](/mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax/repro/runs/harbor-validation-2220455/bundle/bootstrap_logs/training.log) |

最新ClearML实验名：`harbor-2220455-qwen36-35b-a3b-128h800-dump2-v3-20261015-183806`。

原长跑可恢复checkpoint：

`/mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax/repro/runs/harbor-validation-2220455/checkpoints/harbor-2220455-qwen36-35b-a3b-128h800/iter_0000007`

64次save RPC、64个shard、120232个引用extent、native common optimizer/scheduler/RNG元数据均已核验。回执：[validated-receipt.json](/mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax/repro/runs/harbor-validation-2220455/bundle/manual-checkpoint-step7-v2/validated-receipt.json)。v3是新初始模型实验，未加载这个step7 checkpoint。

v3预期保存到 `checkpoints/harbor-2220455-qwen36-35b-a3b-128h800-dump2-v3/iter_0000000`、`iter_0000001`；是否实际完成必须看checkpoint及最终job状态。目标完成后停止本次两步诊断；QS/Gym仍保留，原20+步目标处于暂停状态。

## 已经验证的内容

1. v2 step0：725 physical contexts / 256 Sessions / 32 Groups。每行tokens/hash/长度/ownership mask精确关联canonical及raw；raw0提供659行，raw1提供66行。每个response只有一个训练owner，重复历史响应被mask排除，reward只有一个Session锚点。
2. 32CP完整重组：全部response坐标恰好覆盖一次，无重叠、无遗漏；重组rollout LP与raw逐位一致。有效mask1为4,981,276 tokens。
3. old actor LP与fresh policy-forward LP在全部26,911,779 response坐标逐token完全相等，mask1位置也完全相等。这符合单次更新前同权重forward；`ppo_kl=0`、`pg_clipfrac=0`本身不能作为没训练的证据。grad_norm=0.23970008，有限且正。
4. 实际32prompt ID及有效mask独立重算prompt等权分母一致；32×8 Session advantage均值0、总体方差1，广播关系通过。
5. PG/entropy/TIS等13项CPU重算与实际训练日志通过，最大绝对误差约6.28e-8；160个逐CP captured loss标量比对通过。
6. v2返回的424/424 Sessions，canonical→已归档native结果→唯一CLI见证→实际verifier reward一致。148成功、276失败，全部为真实binary verifier reward；53个staged任务的verifier文件与sealed manifest一致。精确关联到step0的256条reward为89/256；另外168条只是已返回，未据此声称进入optimizer。

native verifier returncode被运行adapter丢弃，无法补证；53个原始tests绝对路径在本机不可达，因此这里只证明staged verifier与sealed manifest一致。data.jsonl字节一致不能替代所有原始tests文件字节核验。CLI日志并非完整模型转录；canonical/native上下文精确关联已通过，CLI全assistant body覆盖不足的两例详见证据。

审计工具曾把CPU标量Tensor `group_index` 当字典key，导致0/725匹配；已仅在离线工具规范成整数，725/725唯一匹配，10项CPU回归检查通过。它不是训练tokens错位的证据。

## Reward与其他指标

原长跑按**实际入训batch、去重Session**重建；每步256条：

| optimizer step | 成功数  | reward均值 | contexts/trajectory |
| -------------- | ------- | ---------- | ------------------- |
| 0              | 97/256  | 0.37890625 | 2.76172             |
| 1              | 105/256 | 0.41015625 | 2.87500             |
| 2              | 105/256 | 0.41015625 | 2.65625             |
| 3              | 126/256 | 0.49218750 | 2.99609             |
| 4              | 103/256 | 0.40234375 | 2.77734             |
| 5              | 91/256  | 0.35546875 | 2.96875             |
| 6              | 107/256 | 0.41796875 | 2.80859             |
| 7              | 95/256  | 0.37109375 | 2.96094             |

8步合计829/2048=0.40478516；存在波动，没有足够证据证明持续上涨。未达到超过20步验收。这里step是optimizer编号，不是generation window。

v2唯一入训step0：89/256=**0.34765625**，contexts/trajectory=**2.83203125**，比原先3–4的目标偏低。raw window0的0.33482143、window1的0.365属于不同返回窗口，不能拿这两个数当作两次optimizer更新后的reward上涨。

| v2 step0指标                     | 实测            |
| -------------------------------- | --------------- |
| `pg_loss`                        | -0.005498058257 |
| `entropy_loss`                   | 0.334672554     |
| `train_rollout_logprob_abs_diff` | 0.03748605899   |
| `train_rollout_prob_abs_diff`    | 0.01153321902   |
| `tis`                            | 1.000046243     |
| `tis_clipfrac`                   | 0.001688670909  |
| `ppo_kl`                         | 0               |
| `pg_clipfrac`                    | 0               |
| `grad_norm`                      | 0.239700079     |

原长跑8个train step：logprob abs diff约0.0379–0.0407，entropy约0.2817–0.3570，grad_norm约0.1993–0.2950，TIS clip fraction约0.00175–0.00206。原长跑返回窗口0–8的response_len均值约7.3k–11.0k、repetition_frac约0.9%–3.6%；这些是窗口/context口径。v2 window0/1对应response_len均值10303/9106、repetition_frac约2.43%/3.99%，不替代实际optimizer batch口径。

`train_data.common.rewards`是custom advantage，不是binary outcome。业务reward取去重Session的outcome锚点；mask0和重复上下文不应计入有效loss token数。训练loss及entropy使用prompt等权reducer，与普通token均值不同。

## 已证实的问题、仍怀疑的点与接手优先级

### P0：执行完成与清理、采样分母

v2有8个自然执行失败Group：4个有实际OCI OOM事件证据，2个历史底层事件已被挤出，2个后续未深入底层原因。8GiB/4CPU限制在实际容器核实；不可将8个都叫OOM。普通agent_error走整Group丢弃补采，没有冒充verifier reward0。完整尝试/全对/全错/执行失败/timeout分母仍未重建，效果比较可能受过滤和任务构成影响。

同一Group中的两条CLI在17:23/17:26已写success终态，18:21仍活着；SRT/claude.real进程未退出，worker在child.poll循环，尚未进入verifier。Docker socket与容器运行状态正常。没有发手工信号，容器后来由训练失败后的正常取消路径回收。这是实际生命周期长尾证据；CLI为何不退出尚未唯一定位。建议先在recipe层收敛terminal-exit grace/非零错误回收方案，并保留原Group错误语义。

### P0：post-step同步门控超时

首步于17:56:39完成；rank0继续轮询rollout readiness，最终producer debt尚待window2的88条补采、当时仅完成72条；其余rank提前进入健康flags的Gloo all_reduce。18:26:56等待1800000ms超时，触发shutdown；18:27:00的 `KeyError: train:prepare_group_70`为清理级联。不是第二步actor backward失败，也不是已证明的NCCL权重同步错误。

v3现有参数180分钟覆盖该Gloo组，并在gate之前按每步保存。它解决超时预算不匹配，不能证明上游CLI生命周期问题已经修好。后续应观察实际producer门控和权重发布，不应只看optimizer日志出现。

### P1：train/rollout LP尾部

mask1普通token LP均差0.03550994，中位数3.06e-5，p99约0.5067，最大18.92531；>1有10395tokens（0.2087%），>5有180，>10有31。>5均为actor LP更负。top20 token、±2邻域、mask、raw LP、turn绑定均匹配，没有统一±1/±2移位，也未集中在CP或packing边界。

TIS `[0,2]`中这些尾部并未触发clamp，而由ratio自身大幅降权；>5部分signed PG贡献约1.324e-8。贡献小不能证明尾部正常。候选包括源头LP/token绑定、真实请求tokenization、Qwen hybrid/DeltaNet的prefill/decode或内核数值差；当前无唯一归因。建议用记录的top token/context做小规模定点对照。

### P1：效果判断方法

优先完成v3两步和最终checkpoint/权重发布对账，再做20+步与固定题集评测。重建每Group尝试、过滤、执行失败和timeout的reward sum/count；按实际入训batch统计，补任务类别、轨迹长度、context数和生成版本年龄。原8步或本次1步的波动不能用来归因LR、TIS或prompt权重的效果差异。

## 主要证据和接手入口

- [首步整合token/logprob/mask，约1.09GB](/mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax/repro/runs/harbor-dump2-2220455-v2/analysis/partial-step0/token-logprob-mask-step-0.pt)；SHA256 `4b196c9dda62ef4370d186bb5a5f2f0d9713d705a2cd55937749db0f5b12bfef`。
- [tensor字段与CPU读取示例](/mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax/repro/runs/harbor-dump2-2220455-v2/analysis/TENSOR_FORMAT_ZH.md)。
- [首步总审计](/mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax/repro/runs/harbor-dump2-2220455-v2/analysis/partial-step0/forensic-audit.json)、[32CP数值报告](/mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax/repro/runs/harbor-dump2-2220455-v2/analysis/partial-step0/numeric/cp32-numeric-audit.json)。
- [424条原生reward报告](/mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax/repro/runs/harbor-dump2-2220455-v2/analysis/native-verifier-all-returned-v3/native-reward-report.json)、[step0精确native关联](/mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax/repro/runs/harbor-dump2-2220455-v2/analysis/native-verifier-all-returned-v3/optimizer0-native-join.json)。
- [LP尾部详细报告](/mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax/repro/runs/harbor-dump2-2220455-v2/analysis/independent-step0-numeric/lp-tail/README.zh.md)。
- [Gloo首因报告](/mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax/repro/runs/harbor-dump2-2220455-v2/analysis/v2-gloo-timeout-primary-cause.json)。
- [CLI终态长尾](/mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax/repro/runs/harbor-dump2-2220455-v2/analysis/agentic-audit-evidence-v2/CLI_TERMINAL_HANG_ZH.md)、[8个失败Group清单](/mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax/repro/runs/harbor-dump2-2220455-v2/analysis/agentic-audit-evidence-v2/driver-failure-inventory-final.json)。
- [原始handoff参考](harbor-handoff-20261014.md)、[Claude Code Harbor指南](../docs/zh/examples/claude-code-harbor.md)。
- [本交接时点的机器可读快照](/mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax/repro/runs/harbor-dump2-2220455-v3/handoff/status-and-validation-snapshot.json)。

v3后续dump：`checkpoints/harbor-2220455-qwen36-35b-a3b-128h800-dump2-v3/audit/`，包括canonical contexts、raw rollout windows、32CP train shards和policy capture。只读监控已在运行；工具快照在 `/mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax/repro/runs/harbor-dump2-2220455-v3/handoff/observer-tools/`，请勿重复启动。共享 driver 日志由 head 已脱离终端的进程持续镜像；原生结果存在 1h TTL，现有收尾器每 5 分钟增量保全到云盘。发布检查的日志、CPU harness 及方法说明见 `/mnt/tidalfs-bdsz01/dataset/yuzhe1/xiaoliang/Relax/repro/runs/harbor-dump2-2220455-v3/handoff/publication-validation/`。

保护服务：原Gym保活submission `harbor-validation-2220455-20261014`（job08）、17个实际creator为job04的HarborNode、QS keepalive与私有Docker/forwarders。v2退出后已确认263个诊断Actor全DEAD、3PG removed、17Pod managed进程/OCI清零、128GPU物理compute process为0；随后v3开始使用这些训练资源。

后台只读收尾器已配置：v3自然成功退出后，将生成 `analysis/final/forensic-audit.json`、两个整合tensor文件及 `analysis/native-verifier-final/native-reward-report.json`。它每5分钟增量保全原生归档；状态见 `control/automatic-audit-status.json`。若job或审计失败，会明确留下失败状态。最终checkpoint完整性和GPU/外部资源收敛仍需单独核验。
