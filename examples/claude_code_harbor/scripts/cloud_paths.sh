#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Resolve user-owned host roots once, before changing cwd or starting services.
harbor_resolve_paths() {
  local name value
  for name in "$@"; do
    value="${!name:-}"
    [ -n "${value}" ] || continue
    value=$(python3 - "${value}" "${CLOUD_ROOT:-}" <<'PY'
import sys
from pathlib import Path

value = Path(sys.argv[1]).expanduser()
if not value.is_absolute() and sys.argv[2]:
    value = Path(sys.argv[2]).expanduser() / value
print(value.resolve())
PY
    ) || return
    printf -v "${name}" '%s' "${value}"
    export "${name}"
  done
}
