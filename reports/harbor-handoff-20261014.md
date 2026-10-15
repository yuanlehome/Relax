# Harbor 代码与运行交接：2026-10-14

后续运行及实现发布见 [2026-10-15 交接](harbor-handoff-20261015.md)；本文保留原时点事实。

这是一次运行交接快照，不是训练成功报告。为符合仓库敏感信息检查，下文部署路径使用配置占位符，内部平台以 Job/Trial/task ID 定位；完整挂载位置保存在本机私有交接副本。`${LOCAL_REPO}` 为本机 checkout，`${SHARED_REPO}` 为共享盘 checkout，`${SOURCE_PROJECT}` 为原始数据项目，`${MODEL_PARENT}` 为模型父目录，`${CONTROL_ROOT}` 为本地监控证据根。操作与配置说明见 [中文指南](../docs/zh/examples/claude-code-harbor.md) / [English guide](../docs/en/examples/claude-code-harbor.md)。文件版本见 [代码清单](harbor-code-inventory-20261014.json)。后续新 attempt 应追加记录，不能覆盖历史身份。

## 代码在哪里

- 工作区：`${LOCAL_REPO}`。
- 分支：`codex/claude-code-agentic-128xh800`，远端为内部 `origin`。
- 部署基线：`b582cfc811ed6dc9b7bb085cc3e07488993eb800`，在此基线上应用 29 个明确 overlay，形成当前独立 release。
- release 共有 1608 个代码文件、12 个 symlink。manifest SHA-256：`e88446c396651480a86ef8b882193b017b1714593dbd8e42ec99b0fcd1c3a863`。
- 本次发布的是文档、导航和代码校验清单。工作区的实现改动及 untracked 实现文件没有一并提交；不能把 Git 基线本身当作完整最新实现。inventory 的 `release_overlay`、`working_tree_status`、`matches_release` 用于逐文件辨认；清单不包含文件内容、密钥或任务数据。

当前独立部署目录：

```text
${SHARED_REPO}/repro/releases/b582cfc811ed-alignfix-2213885-20261014
```

部署是 base + overlay，不是就地改原 release。原始、loss/timeout 对齐、validator 修复三个目录均须保留，不能互相覆盖。运行目录对原 release 的代码哈希依赖应一起核验。

## 当前选择

- 使用 Relax + Qwen3.6-35B-A3B + Harbor/Claude Code。
- 保留 128 H800：actor 64、rollout 64，无独立 eval 池。
- loss 按业务 Job317717/Trial2156263 的 prompt 分母和 clamp 对齐；TIS 只影响分子。
- timeout 采用业务实际 7200 秒墙钟，暂停期间也继续计时；保留已有真实轨迹，超时 reward 0。
- staleness 保持现有 TransferQueue 领先限制，没有实现业务最旧 token 版本年龄规则。
- 完整目标仍为 128 个实际 optimizer 更新、唯一 child SUCCEEDED、最终 checkpoint 可读。
- 不新建 Job/Trial、不修改 priority/ACL，不重复已有提交。当前 Trial 已终止，需要可用资源后才能继续；这份文档不构成再次启动命令。

## 运行身份与当前状态

| Attempt                    | 结果与边界                                                                                                                                                         |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `oneclick-20261014-113919` | 按用户要求于 17:11 停止 child；已确认 7 次真实更新（step 0–6），无 iter checkpoint/tracker；原 outer 已停止清理。不能称完整训练成功                                |
| `oneclick-20261014-184504` | child 于 19:27 提交、19:28 在 parse_args 前置 loss 校验失败，未进入训练 main；实际 0 更新、0 rollout、无 iter checkpoint。错误为 `[0.0]` 被当作启用 auxiliary loss |
| `oneclick-20261014-195454` | 19:54:54 正常一键 outer 提交。最后可达 runtime 观测 20:10:22：节点验证 5/17、child NOT_SUBMITTED、0 个已观测更新；此后 QS 终止，最终 child/训练进度未知            |

QS Job330975 / Trial2213885 于北京时间 20:14:58 进入 Deleting，20:16:28 Terminated，20:18 回读确认 Pod NOT_FOUND。助手未执行这次终止，原因未核验。不能用最后的 0 更新推断最终 0，也不能沿用旧服务健康或 outer RUNNING 作为当前状态。

在内部 QS 平台按 Job330975 / Trial2213885 查证。

当前修复 attempt 的身份仅用于查证，禁止重复提交：

```text
outer: harbor-alignfix-2213885-loss-timeout-20261014
sleep: qs-330975-2213885-ai-1829764-xsmk2
attempt: oneclick-20261014-195454
experiment: harbor-2213885-qwen36-35b-a3b-128h800-loss-timeout-fix
runroot:
${SHARED_REPO}/repro/runs/harbor-alignfix-2213885-loss-timeout
```

`prepared` 和 `bundle` 在该 runroot 的 attempt 下。唯一 child ID 应从该 bundle 的 `bootstrap-status.json` / 原子提交记录读取，不能从历史 child ID 推导。外部提交器通过本地 `outer-submit-intent.json` 和远端 `launch-record.json` 的 O_EXCL 创建防止重复 dispatch；通用入口本身没有这两把锁。

## 本次原始输入

```text
source-root:
${SOURCE_PROJECT}/data/rl_dataset/20261009-150037-mimo-v26-code-claude_code-srt-gitclean-w200k
model:
${MODEL_PARENT}/Qwen3.6-35B-A3B
```

原始输入重新 prepare，不设 `HARBOR_REUSE_DATA=1`。模型参数 CLI 可以传上述模型目录，env 的 `MODEL_DIR` 则传其父目录。

允许复用的旧 Gym 与四个离线 OCI 包分别位于：

```text
${SHARED_REPO}/repro/runs/harbor-repro-20261010-203604/bundle/runtime/gym
${SHARED_REPO}/repro/runs/harbor-repro-20261010-203604/bundle/runtime/oci-runtime.packages
```

Gym 原位只检查，使用新的私有 check-log，不移动 venv、不改旧 build.log。离线包重验尺寸和 SHA 后安装新 OCI；不能 symlink 整个旧 bundle。这些路径需要原共享盘挂载可达；h800-dev 不保证挂载源数据，不借用其 GPU。

## 修复和验证证据

7200 秒 timeout、context 导出与 prompt-equal loss 已在工作区实现并打包。最后一次恢复相对前一 release 只修 `prompt_loss.py` 的 validator 和相应测试：支持 scalar/list/tuple 零系数，拒绝非零、NaN、Inf、bool、空或非法值，不改 loss 公式、Megatron parser、GPU 或 staleness。

保存的 CPU 验证共 45 个主测试：loss 4（subtests 共 40 条报告）、relay 14、gym 10、data 17，无 skip；loss 修复在实际 head CPU 环境重测，其他未改套件沿用此前证据。实际 recipe 完整参数的 parse-only preflight 通过，CUDA 未初始化。针对性 pre-commit、bash -n、diff-check 已通过。

这些不等于多节点 GPU 数值验收。最终 recovery 在服务验证期间失去 QS Pod，未获得当前实现的完整训练结果；该项仍未完成。

### 已修复、保留与待验收

此状态针对最新工作区和 `alignfix` release，不适用于旧 r5；本次文档提交仍不包含实现文件。

| 项目                | 状态                                                                                                          |
| ------------------- | ------------------------------------------------------------------------------------------------------------- |
| Loss 权重           | 已实现并启用业务 prompt 等权及完整 batch 的 CP/DP 归一化；CPU 数值通过，多卡效果待验收                        |
| Timeout             | 已实现 7200 秒墙钟、暂停继续计时、保留真实轨迹并记 reward 0；没有真实 response 或普通 HTTP/CLI 故障仍报错     |
| JSONL Unicode 分隔  | 已用 `split("\n")` 修复 `splitlines()` 的错误分隔；旧运行影响数量未知                                         |
| BBS 初始化          | 新 prepare 保留原始 `150037` 方式，消除已记录的历史替换差异；执行环境整体等价未验收                           |
| Drop 统计           | 未补齐：全同 reward 的 custom advantage 分支只写日志，不计入通用 dynamic-filter drop 指标；失败组去重分母缺失 |
| 失败丢组            | 仍存在：单个受控 Session 失败可能丢掉整组 siblings，从后续题目补采；不是同题自动重试                          |
| Reward 统计口径     | 仍需区分逻辑 Session 的 raw anchor 与按 context 加权的 metadata outcome；两者不能混作一条学习曲线             |
| OOM/API400/输出超限 | 未逐项修复验证，不能视作随 loss/timeout 改动一起解决                                                          |
| Staleness / GPU     | 按用户选择保留现状，不实现业务 token 版本门控或 224 GPU 分配                                                  |
| 训练效果            | 当前实现尚无完整多卡训练和 reward 改善验收，不能称上述疑点全部解决                                            |

源码依据包括 `relax/agentic/pipeline/reward.py` 的 custom advantage 与 final filter 分支、`relax/agentic/pipeline/runtime.py` 的整组收集/丢弃，以及 `relax/agentic/rollout.py` 的 metadata 聚合。完整解释和补证步骤见中英文指南的“修复状态与未完成项”及 reward 归因部分。

本地证据根：

```text
R=${CONTROL_ROOT}
C=R/recovery-run-2213885-20261014-1937
```

| 文件                                                              | 用途                                                                     |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `C/code-manifest.json`、`validation.json`、`parse-preflight.json` | 部署内容、CPU测试与真实参数契约证据                                      |
| `C/monitor-state.json`                                            | 当前终止状态及历史 runtime 观察边界                                      |
| `C/last-progress-before-qs-termination-20261014.json`             | 最后成功 runtime 快照                                                    |
| `C/old-original-stopped-monitor-state.json`                       | 原始 7 更新的独立历史                                                    |
| `C/old-alignment-failed-monitor-state.json`                       | 对齐 attempt 的 pre-main 失败历史                                        |
| `R/alignment-run-2213885-20261014/terminal-failure-summary.json`  | list 零系数误判的根因                                                    |
| `R/reward-baseline-2156263.json`                                  | 业务前 19 个实际入训 batch 的 reward 基线                                |
| `R/r5-reward-diagnostic-20261014.json`                            | 历史 r5 窗口统计、业务筛选前后分母及归因限制；本地私有证据，不随文档上传 |

Pod 缺失时不要执行会 assert 的 `C/read-progress.py`，也不要让 updater 把旧快照写成实时 RUNNING。运行恢复前先核对准确 child、checkpoint 和已完成进度，再确定新身份；不得直接重跑任何历史 submit/deploy/launch/stop/cleanup 脚本。

## 历史 ClearML 与 reward 对比

已确认的较长 Relax r5 run：

- QS Job329061 / Trial2183823，Ray submission `claude-code-harbor-128h800-20261009-oci-r5-t8192`。
- ClearML project ID `5c0d02bc627a45f0b07524fff5697359`，task ID `3552454ddf064c4785c6763050586777`。在内部 ClearML 中定位该 task；它不是当前 loss/timeout 修复 run。
- 截至 2026-10-10 11:27 的历史记录核验了 step 0–18，共 19 个实际更新；不能据此称已跑完 128。
- 前 19 个入训 batch：r5 reward 为 `1811/4864 = 0.3723273`，业务为 `1917/4864 = 0.3941201`，相差约 2.18 个百分点。两者不是固定评测集成功率，不能据此唯一归因于某个配置差异。
- 业务前 5 batch 均值 `0.39765625`、末 5 `0.40546875`，这段没有明确持续上涨；完整业务曲线未核验。
- 业务 Job317717 / Trial2156263 在 10 月 9 日附近那次长 run 的 ClearML 链接仍未确认。QS 效果面板未接入、ClearML 搜索未匹配，不应编造链接或把 r5 链接充作业务链接。

旧 `build-reward-comparison.py` 是原来 6 更新/8 生成轮的快照脚本，不能直接当作当前 run 的通用统计程序。后续分析仍应按同一实际入训 batch 的逻辑 trajectory reward 比较，并保留过滤、超时、任务构成与 context 加权的口径差异。

### Reward 复核与剩余排查

以下均为历史 r5 和业务运行各自实际入训 batch 的逻辑轨迹 reward，所有 batch 各含 256 个 Session；不是当前修复 attempt，也不是同题、同生成版本配对评测。

| 窗口（step）       | r5         | 业务       |
| ------------------ | ---------- | ---------- |
| 前 5 批（0–4）     | 38.28125%  | 39.765625% |
| 第 6–10 批（5–9）  | 33.90625%  | 39.21875%  |
| 末 5 批（14–18）   | 39.140625% | 40.546875% |
| 全部 19 批（0–18） | 37.232730% | 39.412007% |

r5 中段下探、后段恢复；前五到末五批增加 0.859375 个百分点，业务增加 0.78125 个百分点。这个窗口不足以证明哪一方持续改善或退化。旧 r5 部署 manifest 对应 `ab845150d` 的 loss 路径按 token 加权，与业务 prompt 等权的差异已确认；其对 reward 差距的贡献尚未量化。最新 loss 修复不能回溯解释旧曲线为已对齐结果。

业务前五批共检查 286 组/2288 条轨迹：reward 总和 621，筛选前均值 27.141608%；丢弃 14 个全对组、112 个全错组后，保留 160 组/1280 条轨迹，reward 总和 509，均值 39.765625%。组数与 reward 总和守恒已核对。这说明筛选会显著改变观察到的均值，不说明它已解释 r5 与业务的差距；r5 缺少对应的完整尝试分母。

现有证据降低了“完全没更新”“advantage 符号反了”“大量 TIS token 被屏蔽”的优先级：旧 r5 前 19 次 `grad_norm` 均有限且正（约 0.137–0.200），TIS clip fraction 平均约 0.1806%；旧代码 CPU 检查中成功/失败优势符号正确，增加 context fanout 不改变 Session advantage。这些检查尚未证明后续所有 rollout 都使用了已发布的新权重。

下一步应按 group 去重重建尝试、全对/全错/其他恒定 reward 过滤、执行失败、待完成与入训去向；联合统计筛选前后 reward sum/count、未知 outcome、任务类别、context 数和时长。错误日志次数和 OCI OOM 事件数不等于失败 Session/group 数。再按实际入训 batch 核对每题有效 token 数/loss 权重、模型生成版本年龄和权重发布，最后在可用资源与有效授权下用初始模型及现存 checkpoint 做固定题集评测。以上是待完成的鉴别步骤，当前没有唯一根因结论，也没有新训练效果已改善的结论。
