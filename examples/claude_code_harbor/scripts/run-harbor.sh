#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Prepare Harbor and run the existing 128-H800 recipe on a dedicated Ray head.
# Usage: bash examples/claude_code_harbor/scripts/run-harbor.sh --env-file /path/to/harbor.env
set +x
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HARBOR_REPO="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
HARBOR_ARGS=()
while (($#)); do
    case "$1" in
        --env-file)
            if (($# < 2)) || [[ ! -f "$2" ]]; then
                echo "ERROR: --env-file requires an existing, trusted shell configuration file" >&2
                exit 2
            fi
            # This is a user-authored shell file, not a downloaded dotenv file.
            set -a
            source "$2"
            set +a
            shift 2
            ;;
        --check-config|--prepare-only|--help|-h)
            HARBOR_ARGS+=("$1")
            shift
            ;;
        --source-root|--model-dir)
            if (($# < 2)); then
                echo "ERROR: --source-root and --model-dir each require a path" >&2
                exit 2
            fi
            HARBOR_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            echo "ERROR: use --source-root DIR --model-dir DIR, --env-file FILE, --check-config or --prepare-only" >&2
            exit 2
            ;;
    esac
done
# The wrapper and all helpers must come from the same checkout.
export REPO="${HARBOR_REPO}"
export PYTHONPATH="${HARBOR_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON:-python3}" -u -m examples.claude_code_harbor.scripts.bootstrap_qs "${HARBOR_ARGS[@]}"
