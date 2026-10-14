# Claude Code / Harbor: code, configuration and one-command execution

This recipe connects Claude Code's real OCI tool execution to Relax Agentic rollout and trains Qwen3.6-35B-A3B with GRPO. Claude Code drives the tool loop, NeMo Gym manages the task environment and verifier, SGLang serves model requests through Relax, and Megatron trains the model.

This guide describes the **2026-10-14 loss/timeout repair snapshot** in the working tree and deployed release. Implementations including `run-harbor.sh`, `prompt_loss.py` and their supporting changes are not committed with this documentation update. Use a complete code directory containing that implementation; the Git commit containing this guide alone is not a runnable release of it. Consult `reports/harbor-code-inventory-20261014.json` for file SHA-256 values and `reports/harbor-handoff-20261014.md` for the operational handoff and release locations.

## Where to start

Provision an exclusive Ray cluster and shared inputs, configure paths, check the configuration, then run the entrypoint **once on the Ray head**. It uses an existing cluster; it does not create a QS Job/Trial, download model weights or install the base training image.

Requirements:

- Sixteen GPU workers, each with eight H800 GPUs, for 128 GPUs in total. A separate CPU head is typical.
- A Relax training image containing PyTorch, CUDA, Ray, SGLang and Megatron. The Gym builder uses Linux CPython 3.12 system Python; Megatron defaults to `/root/Megatron-LM`.
- Identical absolute mount paths on every node for the complete code, original data and referenced uploads, model, prepared data and bundle.
- The original `20261009-150037-mimo-v26-code-claude_code-srt-gitclean-w200k` dataset. Preparation validates its digest, 2683 records, verifier directories and uploads; another JSONL with the same name is not interchangeable.
- Model configuration, tokenizer and all weight shards; this model has 26 shards.
- At least one GPU worker with usable user namespaces and FUSE. Other nodes can reach a native Gym through authenticated forwarders; probing determines the allocation.
- Available service ports, reachable cross-node callbacks and access to task image registries. Offline OCI packages contain tools, not task images or model weights.

::: warning Exclusive cluster required
The downstream `scripts/entrypoint/ray-job.sh` cleans up old Relax training processes, Serve applications and placement groups. Admission checks other active Jobs, Serve, PGs and Ray GPU availability, but not every physical GPU process. Separately inspect physical memory and compute processes before submission, and keep the cluster exclusive throughout admission and submission.
:::

## One-command execution

### Two-input mode

Run from a writable shared directory, replacing these paths with your mounts:

```bash
export CLOUD_ROOT=/shared/cloud
export SAVE_DIR=/shared/cloud/harbor/checkpoints
export EXP_NAME=harbor-qwen36-loss-timeout-run001

bash /shared/cloud/Relax/examples/claude_code_harbor/scripts/run-harbor.sh \
  --source-root /shared/cloud/inputs/20261009-150037-mimo-v26-code-claude_code-srt-gitclean-w200k \
  --model-dir /shared/cloud/models/Qwen3.6-35B-A3B \
  --check-config

# After checking the configuration, launch exactly once.
bash /shared/cloud/Relax/examples/claude_code_harbor/scripts/run-harbor.sh \
  --source-root /shared/cloud/inputs/20261009-150037-mimo-v26-code-claude_code-srt-gitclean-w200k \
  --model-dir /shared/cloud/models/Qwen3.6-35B-A3B
```

`--model-dir` accepts the model directory or its parent. Setting `CLOUD_ROOT` explicitly avoids relying on inference from the original dataset layout. If that layout has moved, also configure the wheel paths below and, when needed, `ORIGINAL_CLOUD_ROOT`. Mounting only the dataset directory can omit referenced shared uploads.

By default, prepared data and the bundle go under `harbor-runs/<timestamp>-<random>/` beside `SAVE_DIR`. Without an explicit `SAVE_DIR`, it defaults to `harbor-checkpoints` under the current directory, which must be within shared `CLOUD_ROOT`. Checkpoints use the stable `SAVE_DIR/EXP_NAME`, not the random run directory. An existing experiment checkpoint directory is rejected rather than silently resumed.

### Explicit environment file

For predictable output paths, copy `examples/claude_code_harbor/scripts/harbor.env.example` to a private file outside Git:

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

The file is trusted shell code executed with `source` and automatic export. Quote values containing spaces. Supply credentials through the existing secure environment, never through documentation, templates or Git. The wrapper sets `REPO` to its own checkout so helpers cannot accidentally come from another version.

`--check-config` checks settings and output collisions without installing or connecting to Ray. It does not establish complete inputs, idle GPUs or healthy services; the environment file itself is still executed. `--prepare-only` prepares data, Gym, OCI and a private authentication file without services or training. A subsequent normal launch needs a new `BUNDLE_DIR` and explicit reuse of the prepared data/Gym.

Use a QS RayJob entrypoint or a persistent Ray outer driver for unattended execution. A transient `qs exec` session does not provide the owner's lifetime. Do not launch once per worker or repeat submission after an ambiguous network response.

## Execution flow

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

Every alive node gets a `HarborNode` actor that holds its services without reserving training GPUs. Native nodes host private OCI and Gym; forwarder nodes provide a local endpoint routed to a native node. The first native service must pass a real cross-node model callback before expansion and verification of every node. Controlled verification uses a stub model: its model calls, contexts and callbacks are not training progress.

Gym is built from a fixed NeMo commit and exact package versions, inheriting Torch/Ray/CUDA from the training image. Claude Code, Node, SRT and the original two wheels are task uploads installed inside OCI, not into the Relax host environment. Original data remains untouched; prepared asset paths become relative. The ordinary OCI image map follows validated naming rules without converting image contents.

## Paths and service settings

These are entrypoint settings, not global defaults for every Relax recipe.

| Setting | Default or requirement | Meaning |
| --- | --- | --- |
| `CLOUD_ROOT` | Required in env mode; inferable from original layout in two-input mode | Shared root for relative paths |
| `SOURCE_ROOT` | Required by default | Original data and referenced assets, not prepared data |
| `MODEL_DIR` | Model parent directory in env mode | Must contain `Qwen3.6-35B-A3B/`; only the CLI option also accepts the model directory |
| `DATA_ROOT` | New directory | Prepared inputs; training uses `aligned-150037-oci/data.jsonl` beneath it |
| `BUNDLE_DIR` | New for every invocation | Environment, service state, verification and private logs |
| `SAVE_DIR` / `EXP_NAME` | Stable shared root / unique experiment name | Their combination is the checkpoint root; outputs must not overlap inputs or each other |
| `BBS_WHEEL_PATH` / `DRESSAGE_WHEEL_PATH` | Discovered or supplied explicitly | Original wheels; discovery checks `source-assets.json`; verify sizes/SHA-256 yourself before supplying explicit paths |
| `ORIGINAL_CLOUD_ROOT` | Set when needed | Maps historical absolute paths after rearranging the source project layout |
| `HARBOR_REUSE_DATA` | `0` | `1` explicitly reuses prepared data, still validating inputs; keep `0` for this reproduction from original data |
| `GYM_RUNTIME_DIR` | New bundle's `runtime/gym` | An existing path enables in-place reuse/checks with new logs; do not move the venv |
| `OCI_PACKAGES_DIR` | Download pinned packages if unset | Optional directory containing the four locked packages; OCI is still installed into the new bundle |
| `MEGATRON` | `/root/Megatron-LM` | Existing image checkout, not installed by this entrypoint |
| `RAY_ADDRESS` | `auto` | Existing GCS address, replaced with the actual address after connecting |
| `RAY_DASHBOARD_ADDRESS` | Head address, default port `8265` | HTTP(S) Jobs API; `RAY_DASHBOARD_PORT` changes the inferred port |
| `GYM_PORT` / `GYM_UPLINK_PORT` | `27980` / `27981` | Local endpoint / native uplink; must be distinct and available |
| `OCI_ROOT` | Short random local path | Private Docker state and Unix sockets; avoid deep shared-storage paths |
| `PROJECT_NAME` | `Relax/agentic/claude-code-harbor` | ClearML project; task name is `EXP_NAME` plus a timestamp |
| `PYTHON` | `python3` | Python running the wrapper, normally the training image's system Python |

Reusing Gym or offline packages still uses a new bundle and OCI installation. Do not symlink an entire previous bundle. After `--prepare-only`, set the existing `DATA_ROOT`, `HARBOR_REUSE_DATA=1` and optionally the original `GYM_RUNTIME_DIR`, together with a new `BUNDLE_DIR` and unused `SAVE_DIR/EXP_NAME`.

## Training configuration

The source is `examples/claude_code_harbor/scripts/run-qwen36-35B-A3B-128xgpu.sh`. The public wrapper accepts only its documented entrypoint options, not arbitrary training flags such as `--lr` or `--num-rollout`. For a variant, edit and review the recipe, then use the normal entrypoint. Low-level `launch.sh` forwards arguments but does not perform the complete environment/service bootstrap.

| Category | Recipe setting | Meaning |
| --- | --- | --- |
| GPUs / mode | 64 actor + 64 rollout; `--hybrid` | 128 H800, no separate eval GPU pool; training GPUs also handle ref/actor forward work |
| Update schedule | 128 rollouts; 1 step/rollout; 1 iter/update | Target is 128 actual optimizer updates; the value 1 does not mean a one-update run |
| Logical batch | 32 prompts × 8 Sessions; GBS 256 | Context fanout does not enlarge the logical batch; filtering/carryover can change attempted Session counts |
| Concurrency | `--agentic-concurrency 64` | Prompt-group concurrency, corresponding to a configured budget of 512 logical Sessions |
| GRPO | 8 samples; clip `0.2 / 0.28`; TIS enabled | Normalize logical Sessions within each group; filter and replenish equal-reward groups |
| Regularization | KL, KL loss and entropy coefficients `0` | These terms are not added in this recipe |
| Loss | Both `--prompt-equal-loss` and `--calculate-per-token-loss` | Prompt-normalized objective with the MCore CP interface retained |
| Optimizer | Adam; constant LR `1e-6`; WD `0.1`; betas `0.9 / 0.98` | CPU optimizer offload, overlapped transfers and precision-aware optimizer |
| Parallelism | TP 2 / PP 1 / CP 32 / EP 8 / ETP 1 | Static CP and sequence parallelism |
| Activations / packing | Full uniform recompute, 1 layer; micro batch 1; dynamic batch; THD | `max-tokens-per-gpu=8192` is a training packing budget |
| Log-prob | Token budget `10240`; chunk `32` | Controls forward memory, not generation length |
| Generation | Temperature `1.0`; response `8192` per request; context `262144` | A multi-turn Session can contain multiple requests |
| Tool loop | Compaction window `200000`; at most `200` steps | Context and agent limits, not optimizer steps |
| SGLang | 8 GPUs/engine; static memory fraction `0.85`; running requests `256`; server concurrency `512` | `consistent_hashing` routing |
| Weight update / staleness | Sync every 1 update; `max-staleness=4` | Queue-ahead constraint, not business token-version validation |
| MoE | Flex + DeepEP; load balancing `none`; aux coefficient `0.0` | Prompt-equal mode rejects nonzero auxiliary loss |
| Saving | Every `16`; retain `2` actor checkpoints | Same stable load/save root; no automatic bootstrap resume |
| Tracking | ClearML + metrics service | Skip pre-training eval; no dedicated evaluation configuration in this recipe |

### Prompt-equal loss

For a closed optimizer batch, let `B` be configured GBS, `P` the number of distinct source prompts, and `M_p` the total original valid response tokens across logical Sessions for prompt `p`:

```text
D_p = M_p * P / B
L = (1 / B_eff) * sum_context [sum_token(m' * token_loss) / max(D_p, 1)]
```

`B_eff` is the actual logical batch size. With `B_eff=B` and no denominator clamp, this averages token loss within each prompt and then weights prompts equally. TIS may change the numerator mask `m'`, but never the original denominator `D_p`.

Metadata is deduplicated by Session and gathered over the DP group excluding CP. Neither context fanout nor CP slices duplicate Session weight. The backend retains MCore's token-normalized interface, while the reducer compensates for its token denominator. Do not simply remove `--calculate-per-token-loss`.

This path requires Megatron, Agentic, hybrid, GRPO, `policy_loss`, static CP, one step per rollout and one iter per update. It excludes OPD, MTP, nonzero MoE auxiliary loss and custom PG reducers. Validation accepts zero coefficients as a scalar, list or tuple, and rejects invalid values and nonzero entries.

### The 7200-second and 10800-second limits

| Setting | Clock boundary | Expiry behavior |
| --- | --- | --- |
| `harbor_backend_timeout_s=7200` | Wall clock from the sandbox agent request; includes pauses, excludes preceding OCI preparation and subsequent verifier | Reject new model requests, stop CLI, harvest accepted canonical requests; retain real tokens/logprobs, export `TRUNCATED`, reward 0, skip verifier |
| `--agent-timeout 10800` | Outer ManagedCommand active-time budget; generation-gate pauses stop this clock | Outer safety cutoff with an execution error; not a business reward-0 timeout |
| `NEMO_GYM_DEADLINE_S=2147483` | Gym client transport lease | Allows the request to span training pauses; does not replace either limit above |

Accepted model requests may still need to finish after the 7200-second limit, so it is not an exact wall-clock bound on the entire Session. Without a captured real response, no trainable trajectory is fabricated. Non-200 HTTP responses, transport errors and ordinary CLI failures remain errors. Controlled `max_steps` or `context_overflow` termination can still run the verifier and is not automatically a backend timeout.

### Alignment with the business run

The reference is the frozen configuration of Job 317717 / Trial 2156263. The selected scope aligns prompt loss and **7200-second wall-clock** timeout behavior, retaining 128 H800 and existing staleness.

The business allocation is 64 actor / 128 rollout / 32 eval GPUs, totaling 224; this recipe uses 64 / 64 with no independent eval pool. Business staleness uses the model version of the oldest generated token and drops/replenishes the whole group when its age exceeds 4 or a version is missing. This implementation checks `current_rollout_id + 1 - oldest_partition_id <= 4` and otherwise waits for queue consumption. Equal numeric settings do not make these semantics equivalent.

Training packing budgets also differ: `8192` here versus `10000` in the reference. This is not full configuration equivalence and does not promise matching reward curves.

## Reading training progress and reward

Read the latest timestamp, stage and `training_submission_id` in `BUNDLE_DIR/bootstrap-status.json`, then query that child through Jobs API. State fields are updated incrementally and historical fields can remain; an old health value is not a current observation.

| Location | Contents |
| --- | --- |
| `BUNDLE_DIR/bootstrap-status.json` | Stage, unique child ID, Jobs status and service health |
| `BUNDLE_DIR/bootstrap_logs/` | Private preparation/install/verification logs, `training.log` and `failure.log` |
| `BUNDLE_DIR/service_state/<node-ip>/` | Node services, PIDs, controlled verification and private artifacts |
| `SAVE_DIR/EXP_NAME/data/harbor.jsonl` | Training input with routing, prompt ID and timeout metadata |
| `SAVE_DIR/EXP_NAME/` | Actual checkpoints and tracker; ordinary readable files are not checkpoints |
| `examples/claude_code_harbor/scripts/log/` | Additional recipe tee logs; this directory must be writable |

ClearML uses `PROJECT_NAME` and the timestamped `EXP_NAME`. Search by both name and start time, then verify the ClearML task link in logs. Multiple attempts within one QS Trial are separate runs. ClearML tracking in this recipe does not imply that the business task also reports to ClearML.

Confirm optimizer updates from actual step records with matching `train/step`, finite positive `grad_norm`, `global_batch_size=256` and no skip, deduplicating by step index. `Finish rollout`, Sessions, physical context rows and optimizer cache releases do not count as updates. Full success requires all three: 128 actual updates, the unique child `SUCCEEDED`, and a readable final checkpoint.

Reward needs an equally precise unit:

- A prompt produces eight logical Sessions. Each Session may export multiple contexts sharing its `outcome_reward` and advantage.
- Only the final context retains a raw reward anchor; other contexts have `reward=None`. Anchor aggregation measures logical trajectory reward. Averaging every context's `outcome_reward` instead weights by context count.
- GRPO uses the population standard deviation of logical Session rewards without an additive denominator epsilon. Equal-reward groups are filtered; division occurs only when the standard deviation exceeds `1e-6`, otherwise centered rewards are retained.
- Prefer actual training-batch `raw_reward_trajectory_mean` when comparing with the business run. Generation rounds have different boundaries; dynamically filtered reward is not a fixed-evaluation success rate.
- Falling reward describes worse sampled outcomes, but the curve alone cannot uniquely attribute the cause to loss, timeout or staleness. Compare zero-reward proportions, timeout/other failures, filtering/replenishment and task composition using the same batch boundaries.

## Failure, stopping and recovery

After child termination, the outer driver enters `services_retained` and holds ordinary Ray actors. Outer `RUNNING`, healthy services and `services_retained` do not establish training success. Startup failure after services begin can also retain the owner for inspection.

The general entrypoint rejects existing outputs, creates the bundle/private stage logs exclusively and atomically records the child ID before calling the launcher. It has no automatic resume, retry or complete process cleanup. Operational `outer-submit-intent.json` and runroot `launch-record.json` files are additional safeguards of the external submission controller, not built-in locks in `run-harbor.sh`.

Before recovery, establish current QS/Pod availability, exact outer/child identity and terminal state, completed updates, actual checkpoints/tracker, and remaining services/private OCI resources. A network timeout does not prove submission failed; query the original ID first. Existing checkpoints need deliberate recovery configuration; rerunning this entrypoint against old directories is not resume. If QS Pods disappear, retain the last snapshot as historical evidence and mark current child/service status unknown rather than carrying forward `RUNNING` or inferring final progress.

| Symptom | First checks |
| --- | --- |
| Output already exists | Possible duplicate invocation; inspect the original child before touching directories |
| Gym build/download fails | Inspect private logs and NeMo Git/pip connectivity; explicitly reuse a completed runtime if available |
| OCI package download fails | Supply offline packages matching `oci-packages.json` and verify them |
| OCI socket path too long | Use the default short local `OCI_ROOT`, not a deep shared-storage path |
| No native worker / callback unreachable | User namespace, FUSE, ports and cross-node route; readyz alone does not test callbacks |
| Node verification continues | Latest stage/time and verification results; stub counts do not mean training was submitted |
| Child argument validation fails | Recipe/Megatron parser contract, including list-valued aux coefficients; do not repeat the same launch |
| Training waits for queue data | Recent log times, Session accounting and real steps together; accumulated old warnings are not current blocking evidence |

## Code map

Paths in the first table are relative to `examples/claude_code_harbor/`. The inventory identifies the exact working-tree snapshot.

| Code | Responsibility |
| --- | --- |
| `scripts/run-harbor.sh`, `scripts/harbor.env.example` | Public wrapper and trusted environment template |
| `scripts/bootstrap_qs.py` | Preparation, admission, HarborNode ownership, verification, child submission and retained services |
| `scripts/prepare_data.py`, `scripts/source-assets.json` | Original dataset/wheel discovery, validation and prepared data |
| `asset_paths.py`, `scripts/cloud_paths.sh` | Shared-storage path resolution and relocation |
| `scripts/prepare_data_via_qs.py` | Preparation/transfer helper across two QS Pods, not the normal single-cluster entrypoint |
| `scripts/build_gym_environment.py`, `scripts/check_gym_environment.py`, `scripts/install_gym_runtime.sh` | Fixed NeMo source and Gym environment build/check |
| `scripts/gym-requirements.txt`, `scripts/gym-base-requirements.txt` | Exact overlay and inherited base package versions |
| `scripts/fetch_oci_packages.py`, `scripts/oci-packages.json`, `scripts/install_oci_runtime.sh`, `scripts/oci_paths.py` | OCI package verification, private installation and short socket paths |
| `scripts/start_gym.sh`, `scripts/forward_gym.py`, `gym_service.py` | Resident Gym and authenticated forwarding |
| `nemo_agent.py` | Harbor task lifecycle, 7200-second deadline, verifier and exports |
| `oci.py`, `sandbox_worker.py` | Private OCI execution and sandbox command protocol |
| `scripts/install_task_runtime.sh`, `mimo_git_reflog.patch` | Original runtime assets and Git compatibility patch inside tasks |
| `relay.py`, `request_normalization.py` | CLI normalization, Relax model requests and post-deadline harvest |
| `contexts.py` | Canonical context capture, context boundaries and unique response training ownership |
| `run_agent.py`, `claude_settings.json` | Agentic process bridge, JSONL parsing, context output and CLI settings |
| `training_data.py`, `advantage.py` | Training rows, duplicate-response masks, logical-Session GRPO and filtering |
| `scripts/verify_oci.py` | Controlled OCI/tool loop/compaction/verifier protection/callback verification |
| `scripts/run-qwen36-35B-A3B-128xgpu.sh` | Training recipe, input metadata, ClearML and child submission |

Integration points elsewhere:

| Path | Responsibility |
| --- | --- |
| `repro/claude-code-harbor-r5/scripts/launch.sh`, `start_node.sh` | Submission/node services still used by bootstrap; the r5 directory name does not name the current experiment |
| `verify_inputs.py`, `create_gateway_token.py` in the same directory | Input completeness and private host authentication |
| `prepare_environment.py`, `verify_bundle.py` in the same directory | Historical bundle helpers; new source builds do not require old environment archives |
| `examples/nemo_gym_agentic/` | Gym thin client, managed agent process and shared protocol |
| `relax/agentic/` | Sessions, generation gates, runtime, group lifecycle and exports |
| `relax/backends/megatron/prompt_loss.py` | Prompt denominators, contract validation and backend normalization compensation |
| `relax/backends/megatron/actor.py`, `data.py`, `model.py`, `loss.py`, `cp_utils.py` | Closed-batch metadata, packing, CP and loss integration |
| `relax/utils/arguments.py`, `utils.py`, `data/stream_dataloader.py` | Opt-in validation and metadata propagation |
| `relax/components/rollout.py` | Existing queue staleness, whose semantics remain unchanged |
| `scripts/models/qwen36-35B-A3B.sh`, `scripts/entrypoint/ray-job.sh` | Model architecture and cluster training entrypoint |
| `tests/examples/claude_code_harbor/`, `tests/backends/megatron/test_prompt_equal_loss.py` | Module regressions and prompt-loss numerical checks |

See [Hybrid training](../guide/hybrid-training.md) for resource and weight synchronization behavior. Separate environment build commands live in `examples/claude_code_harbor/ENVIRONMENT.md`; interpret historical material there with this guide's version boundary.
