# Harbor Gym environment build

This recipe builds its Gym environment from code and package indexes. It does
not need the r5 `nemo-site-packages`, saved NeMo checkout, saved Megatron checkout,
or `observed-python-packages.json` archive.

## What belongs where

| Component                                   | Source / preparation                                                                                                                                                        |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Relax, PyTorch, CUDA, SGLang, Ray, Megatron | User-selected Relax training image. Megatron defaults to `/root/Megatron-LM`.                                                                                               |
| NeMo Gym                                    | Automatically fetch commit `a85670eb167ba9b48cc53a36a070eed815e6c40d`; sparse checkout includes NeMo core and the Claude Code agent.                                        |
| Gym Python packages                         | New `--system-site-packages` venv. Install the 27 exact versions in `scripts/gym-requirements.txt`; validate the inherited versions in `scripts/gym-base-requirements.txt`. |
| Claude Code / Node / SRT                    | Original dataset upload assets, installed by `install_task_runtime.sh` **inside each task OCI**. No host Claude Code installation.                                          |
| Task data, verifier files, model            | User-mounted cloud inputs. Prepare data separately with `scripts/prepare_data.py`.                                                                                          |
| Docker / fuse-overlayfs                     | `scripts/install_oci_runtime.sh` downloads pinned Docker/Ubuntu packages, checks SHA-256, and extracts into a private prefix. Offline packages are also supported.          |
| Task images                                 | Prepared `ordinary-image-map.json` resolves source references; `oci.py` pulls images on demand.                                                                             |

The dependency files record the effective r5 Harbor runtime. This is a **Harbor
service subset**, not a complete install of every upstream Gym recipe/CLI extra.
In particular, upstream's full dependency metadata has a different Anthropic
upper bound. We use a source `.pth` and explicit packages rather than resolve
all of `nemo-gym`'s extras. Ray and Torch are inherited, never reinstalled.

## Build in the training image

Use Linux x86_64, the same Relax image as r5, and its system CPython 3.12. A
different image must satisfy the checked-in base package contract; the installer
fails on mismatch instead of upgrading training dependencies. Git with partial
clone/sparse checkout support and Python `venv`/`pip` must be available.
The build needs access to GitHub and a package index containing the pinned
versions. Standard proxy variables and `PIP_INDEX_URL` can select your network
route/mirror. The builder ignores pip configuration files and installation
redirect variables, retaining only explicit network/index environment options.
If the training image already configures an internal mirror, select it before
building (the value is not printed):

```bash
export PIP_INDEX_URL="$(python3 -m pip config get global.index-url)"
# Only if the image already declares a trusted internal HTTP index:
export PIP_TRUSTED_HOST="$(python3 -m pip config get global.trusted-host)"
```

Do not put credentials into scripts or Git.

Set the paths for your own mount (the values below are placeholders):

```bash
export REPO=/your/cloud/Relax
export BUNDLE_DIR=/your/cloud/harbor-reproduction
mkdir -p "$BUNDLE_DIR"

cd "$REPO"
bash examples/claude_code_harbor/scripts/install_gym_runtime.sh \
  --repo "$REPO" --prefix "$BUNDLE_DIR/runtime/gym"
```

The compatibility entrypoint runs the same builder:

```bash
python3 repro/claude-code-harbor-r5/scripts/prepare_environment.py \
  --repo "$REPO" --bundle "$BUNDLE_DIR"
```

Use **one** of these commands. The prefix must not already exist. A failed
build leaves a private `build.log` for diagnosis; retry at a new prefix. Nothing
overwrites or stops a running Gym. Build once at its final shared path; venvs
contain absolute paths and must not be moved. Nodes reusing it need the same
image and mount path. Alternatively, build once per node at a node-local path
and set `GYM_RUNTIME_DIR` for `start_node.sh`.

Output:

```text
runtime/gym/
  nemo-gym/       fixed upstream source, including Git revision metadata
  nemo-venv/      independent Gym package overlay
  build.json     successful build evidence and effective package versions
  build.log      private installation diagnostics
```

None of these generated files belong in Git. The builder checks base Python
package versions before and after installation, imports the actual Harbor
service, constructs it with the host Claude installer disabled, and checks
readiness/authentication/cleanup routes in memory. It starts no Ray cluster,
Docker daemon, task container, model server or training job.

For an existing environment, check without installing:

```bash
bash examples/claude_code_harbor/scripts/install_gym_runtime.sh \
  --repo "$REPO" --prefix "$BUNDLE_DIR/runtime/gym" --check
```

A Git mirror can be supplied with `--source-url`; the commit remains fixed.
Build logs can include package-index diagnostics, so treat them as private.

## Connect the other prepared inputs

After separately preparing data/model:

```bash
export DATA_ROOT=/your/cloud/prepared-harbor-data
export MODEL_DIR=/your/cloud/models  # Contains Qwen3.6-35B-A3B/
bash examples/claude_code_harbor/scripts/install_oci_runtime.sh \
  --prefix "$BUNDLE_DIR/runtime/oci-runtime"
python3 repro/claude-code-harbor-r5/scripts/create_gateway_token.py \
  --bundle "$BUNDLE_DIR"
```

The OCI installer downloads Docker 28.5.2 from the [Docker static archive](https://download.docker.com/linux/static/stable/x86_64/),
plus fuse-overlayfs 1.13-1 and fuse3/libfuse3-3 3.14.0-5build1 from the
[Ubuntu archive](https://archive.ubuntu.com/ubuntu/). The checked-in
`scripts/oci-packages.json` locks filenames, sizes and SHA-256 values to r5's
packages. Both downloaded and offline packages must pass these checks.
The default cache is `PREFIX.packages`; use `--cache-dir DIR` to share a cache.
A corrupt existing cache file causes failure and is not silently replaced.

For a network mirror, pass `--docker-base-url` (static stable x86_64 directory)
and/or `--ubuntu-base-url` (Ubuntu root containing `pool/`); hashes remain fixed.
Standard HTTPS proxy settings are supported. For offline nodes, download on a
connected machine, then mount/copy the verified package directory:

```bash
python3 examples/claude_code_harbor/scripts/fetch_oci_packages.py \
  --assets-dir /your/cloud/oci-packages
bash examples/claude_code_harbor/scripts/install_oci_runtime.sh \
  --assets-dir /your/cloud/oci-packages --prefix "$BUNDLE_DIR/runtime/oci-runtime"
```

Installation needs Linux x86_64, Python 3.12 and `dpkg-deb` from the training
image. It does not run `apt`, modify system packages, start Docker, or grant
host privileges. The node/Pod must separately provide accessible `/dev/fuse`
and permitted user namespaces/mount operations for the FUSE storage driver.

On each **new, idle, OCI-capable** reproduction node, with the shared paths above:

```bash
export NODE_IP=<this-node-ip>
export NEMO_GYM_CALLBACK_ALLOWED_HOSTS=<new-serve-head-ip>
export OCI_ROOT=/tmp/harbor-new-run
bash "$REPO/repro/claude-code-harbor-r5/scripts/start_node.sh"
```

`start_node.sh` checks the built Gym before starting its private Docker daemon
and Gym process. The default service is node-local `127.0.0.1:27980`. Nodes
without user namespaces need a separately configured forwarder; this script
fails rather than guessing another node's address.

Once the reproduction Ray cluster and all required node-local Gym endpoints
are ready, run on its head:

```bash
export MEGATRON=/root/Megatron-LM
export RAY_ADDRESS=<new-ray-head-ip>:6379
export RAY_DASHBOARD_ADDRESS=http://<new-ray-head-ip>:8265
export EXP_NAME=harbor-new-reproduction
bash "$REPO/repro/claude-code-harbor-r5/scripts/launch.sh"
```

The launch helper validates all 2683 task rows, prepared asset sizes and references,
JSONL/runtime checksums, model shard presence, the private token and Gym readiness. It uses image Megatron and no longer requires the historical
`bundle-manifest.json`. `verify_bundle.py` remains a separate audit tool for
people who possess the original archive; it is not a fresh-build prerequisite.

## Validation boundary

The build check verifies package/source/service compatibility, not GPU training
or learning quality. For the real task-container/tool/verifier/auto-compaction
chain, run `scripts/verify_oci.py` with the built Gym Python, prepared data and an
owned OCI runtime. This uses a stub model; GPU/optimizer validation remains a
separate training run. Models, data and the ordinary image map remain inputs
to their existing preparation stages; OCI tool packages can be downloaded automatically.
