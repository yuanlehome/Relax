#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Executes inside each task container with prepared, read-only uploads.
set -euo pipefail
printf '[harbor-bootstrap] stage=runtime-setup elapsed=%ss\n' "$SECONDS" >&2
mkdir -p /usr/local/bin /root/.local/bin
export PATH="/opt/srt-bins/bin:/run/agent-runtime/usr/local/bin:/usr/local/bin:$PATH"
# Older adapters do not stage/bind this optional helper. Keep their launch valid.
if [[ -f /staging/node_cpu_limit.cjs && -r /run/relax-node-cpus.cjs ]]; then
  export NODE_OPTIONS="--require=/run/relax-node-cpus.cjs${NODE_OPTIONS:+ $NODE_OPTIONS}"
  printf '[harbor-bootstrap] node_cpu_compat=enabled elapsed=%ss\n' "$SECONDS" >&2
fi
ln -sfn /run/agent-runtime/usr/local/bin/curl /usr/local/bin/curl
ln -sfn ../lib/node_modules/@anthropic-ai/sandbox-runtime/dist/cli.js /usr/local/bin/srt
ln -sfn /usr/local/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe /usr/local/bin/claude.real
cp /staging/claude_wrapper.sh /usr/local/bin/claude
chmod 755 /usr/local/bin/claude
ln -sfn /usr/local/bin/claude /root/.local/bin/claude

cp /run/agent-runtime/tmp/mimo_git_runtime.py /tmp/mimo_git_runtime.py
printf '[harbor-bootstrap] stage=git-prepare elapsed=%ss\n' "$SECONDS" >&2
# Already-imported adapters may not stage the optional diagnostic entrypoint.
if [[ -f /staging/git_prepare.py ]]; then
  git_prepare=(/opt/python/bin/python3.12 -I /staging/git_prepare.py /tmp/mimo_git_runtime.py)
else
  git_prepare=(/opt/python/bin/python3.12 -I /tmp/mimo_git_runtime.py)
fi
"${git_prepare[@]}" prepare \
  --config /staging/git_targets.json \
  --state /opt/mimo-verifier-state

grep -qw srtllm.internal /etc/hosts || printf '127.0.0.1 srtllm.internal\n' >>/etc/hosts

printf '[harbor-bootstrap] stage=execution-server elapsed=%ss\n' "$SECONDS" >&2
exec /opt/python/bin/python3.12 -I "$@"
