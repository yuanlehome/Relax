#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Executes inside each task container with prepared, read-only uploads.
set -euo pipefail
mkdir -p /usr/local/bin /root/.local/bin
export PATH="/opt/srt-bins/bin:/run/agent-runtime/usr/local/bin:/usr/local/bin:$PATH"
ln -sfn /run/agent-runtime/usr/local/bin/curl /usr/local/bin/curl
ln -sfn ../lib/node_modules/@anthropic-ai/sandbox-runtime/dist/cli.js /usr/local/bin/srt
ln -sfn /usr/local/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe /usr/local/bin/claude.real
cp /staging/claude_wrapper.sh /usr/local/bin/claude
chmod 755 /usr/local/bin/claude
ln -sfn /usr/local/bin/claude /root/.local/bin/claude

cp /run/agent-runtime/tmp/mimo_git_runtime.py /tmp/mimo_git_runtime.py
/opt/python/bin/python3.12 -I /tmp/mimo_git_runtime.py prepare \
  --config /staging/git_targets.json \
  --state /opt/mimo-verifier-state

grep -qw srtllm.internal /etc/hosts || printf '127.0.0.1 srtllm.internal\n' >>/etc/hosts

exec /opt/python/bin/python3.12 -I "$@"
