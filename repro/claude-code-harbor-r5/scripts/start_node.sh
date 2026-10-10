#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Run on each OCI-capable node of a NEW, idle reproduction cluster.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="${BUNDLE_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
RELAX_ROOT="${REPO:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
: "${DATA_ROOT:?Set the prepared cloud task and asset root}"
: "${NODE_IP:?Set the IP of this node}"
: "${NEMO_GYM_CALLBACK_ALLOWED_HOSTS:?Set the new Serve head IP or hostname}"
GYM_PORT="${GYM_PORT:-27980}"
OCI_ROOT="${OCI_ROOT:-/tmp/relax-harbor-r5-repro}"
NODE_STATE="${BUNDLE_DIR}/service_state/${NODE_IP}"
GYM_RUNTIME_DIR="${GYM_RUNTIME_DIR:-${BUNDLE_DIR}/runtime/gym}"
source "${SCRIPT_DIR}/../../../examples/claude_code_harbor/scripts/cloud_paths.sh"
harbor_resolve_paths RELAX_ROOT BUNDLE_DIR DATA_ROOT OCI_ROOT NODE_STATE GYM_RUNTIME_DIR
export NEMO_GYM_ROOT="${GYM_RUNTIME_DIR}/nemo-gym"
export NEMO_GYM_PYTHON="${GYM_RUNTIME_DIR}/nemo-venv/bin/python"
export NEMO_GYM_COMMIT="a85670eb167ba9b48cc53a36a070eed815e6c40d"
export NEMO_GYM_TOKEN_FILE="${BUNDLE_DIR}/private/gateway.token"
export NEMO_GYM_CALLBACK_ALLOWED_HOSTS
export GYM_HOST=127.0.0.1 GYM_PORT GYM_CONCURRENCY=512
export CLAUDE_CODE_DOCKER_BINARY="${BUNDLE_DIR}/runtime/oci-runtime/docker/docker"
export CLAUDE_CODE_DOCKER_HOST="unix://${OCI_ROOT}/docker.sock"
export CLAUDE_CODE_IMAGE_MAP="${DATA_ROOT}/ordinary-image-map.json"
export CLAUDE_CODE_DATA_ROOT="${DATA_ROOT}"
export CLAUDE_CODE_RESULT_ROOT="${NODE_STATE}/results"
export CLAUDE_CODE_ARTIFACT_ROOT="${NODE_STATE}/artifacts"
export NEMO_GYM_ARTIFACT_ROOT="${NODE_STATE}/gateway-artifacts"
export CLAUDE_CODE_MAX_CACHED_IMAGES=4
export PATH="${BUNDLE_DIR}/runtime/oci-runtime/fuse/usr/bin:${BUNDLE_DIR}/runtime/oci-runtime/docker:${PATH}"
export LD_LIBRARY_PATH="${BUNDLE_DIR}/runtime/oci-runtime/fuse/usr/lib/x86_64-linux-gnu:${BUNDLE_DIR}/runtime/oci-runtime/fuse/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
test -s "${NEMO_GYM_TOKEN_FILE}"
python3 "${RELAX_ROOT}/examples/claude_code_harbor/scripts/build_gym_environment.py" \
  --prefix "${GYM_RUNTIME_DIR}" --repo "${RELAX_ROOT}" --check
test -s "${CLAUDE_CODE_IMAGE_MAP}"
unshare -Ur true || { echo 'User namespace unavailable: use the fixed forwarder commands in README.' >&2; exit 2; }
# Refuse to take over live ports or an existing Docker root. No stop/kill actions.
python3 - "${GYM_PORT}" "${OCI_ROOT}" <<'PY'
import socket,sys
from pathlib import Path
with socket.socket() as probe:
    probe.bind(('127.0.0.1',int(sys.argv[1])))
if (Path(sys.argv[2])/'docker.sock').exists() or (Path(sys.argv[2])/'daemon.pid').exists():
    raise SystemExit('Existing private Docker state: choose a fresh OCI_ROOT; no processes stopped')
PY
mkdir -p "${NODE_STATE}" "${OCI_ROOT}"
chmod 700 "${NODE_STATE}" "${OCI_ROOT}"
nohup "${BUNDLE_DIR}/runtime/oci-runtime/docker/dockerd" \
  --host="${CLAUDE_CODE_DOCKER_HOST}" --data-root="${OCI_ROOT}/data" \
  --exec-root="${OCI_ROOT}/exec" --pidfile="${OCI_ROOT}/daemon.pid" \
  --storage-driver=fuse-overlayfs --iptables=false --ip6tables=false --bridge=none \
  --ip-masq=false --userland-proxy=false --exec-opt native.cgroupdriver=cgroupfs \
  >"${NODE_STATE}/dockerd.log" 2>&1 </dev/null &
ready=
for attempt in {1..60}; do
  if driver=$("${CLAUDE_CODE_DOCKER_BINARY}" --host="${CLAUDE_CODE_DOCKER_HOST}" info --format '{{.Driver}}' 2>/dev/null); then
    [ "${driver}" = fuse-overlayfs ] || exit 2
    ready=1; break
  fi
  sleep 1
done
[ -n "${ready}" ] || { echo 'Private Docker did not become ready; see cloud dockerd.log' >&2; exit 2; }
nohup bash "${RELAX_ROOT}/examples/claude_code_harbor/scripts/start_gym.sh" \
  >"${NODE_STATE}/gateway.log" 2>&1 </dev/null &
echo "$!" >"${NODE_STATE}/gateway.pid"
for attempt in {1..60}; do
  if curl --noproxy '*' -fsS "http://127.0.0.1:${GYM_PORT}/readyz" >/dev/null; then
    echo 'Node Gym ready; service logs and results are under the cloud bundle'; exit 0
  fi
  sleep 1
done
echo 'Gym did not become ready; see cloud gateway.log' >&2
exit 2
