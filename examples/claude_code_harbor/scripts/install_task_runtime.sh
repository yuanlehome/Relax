#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Runs only after the task's original uploads have been copied into its OCI.
set -euo pipefail

if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then
    cat <<'EOF'
Usage: install_task_runtime.sh

Called automatically by the Harbor OCI worker, inside a disposable task
container. Installs the original Claude Code, Node, SRT, and wrapper from
the task uploads prepared by prepare_data.py. No network download is used.

Inputs: /tmp/{cc,node,srt-bins,srt-npm}-chunks and
        /tmp/claude_srt_wrapper.sh
Outputs: /usr/local/bin/{claude,claude.real,node,srt},
         /usr/local/lib/node_modules, /opt/srt-bins

The task image supplies its project dependencies. Git initialization and
verifier execution are separate OCI-worker steps. Dressage/BBS wheels are
preserved source assets but are not installed or imported by this adapter.
Do not run this installer on the training host.
EOF
    exit 0
fi
[ "$#" -eq 0 ] || { echo 'Unexpected installer argument.' >&2; exit 2; }
[ "${RELAX_HARBOR_PRIVATE_CONTAINER:-}" = 1 ] && [ -d /run/relax-control ] || {
    echo 'This installer must run inside a Harbor task container.' >&2; exit 2;
}

mkdir -p /opt/srt-bins /usr/local/lib/node_modules/@anthropic-ai
cat /tmp/srt-bins-chunks/srt-bins-chunk-* | tar xzf - -C /opt/srt-bins
chmod 755 /opt/srt-bins/bin/* /opt/srt-bins/real/*
export PATH="/opt/srt-bins/bin:/usr/local/bin:$PATH"
cat /tmp/cc-chunks/cc-chunk-* | tar xzf - -C /usr/local/lib/node_modules/@anthropic-ai
cat /tmp/node-chunks/node-chunk-* | xz -dc | tar xf - -C /usr/local --strip-components=1
cat /tmp/srt-npm-chunks/srt-npm-chunk-* | tar xzf - -C /usr/local/lib/node_modules
ln -sfn ../lib/node_modules/@anthropic-ai/sandbox-runtime/dist/cli.js /usr/local/bin/srt
REAL=/usr/local/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe
test -x "$REAL"
cp -a "$REAL" /usr/local/bin/claude.real
install -m 0755 /tmp/claude_srt_wrapper.sh /usr/local/bin/claude
node --version
/usr/local/bin/claude --version
test -x /opt/srt-bins/bin/bwrap
test -x /opt/srt-bins/bin/socat
test -x /opt/srt-bins/bin/rg
