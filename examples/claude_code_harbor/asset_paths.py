# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Resolve prepared host assets independently of the Gym process cwd."""

from __future__ import annotations

import os
from pathlib import Path


def asset_path(value: str, root: Path | None = None) -> Path:
    path = Path(value)
    if not value or ".." in path.parts:
        raise ValueError("Task asset path is empty or contains parent traversal")
    configured = os.environ.get("CLAUDE_CODE_DATA_ROOT")
    root = root if root is not None else Path(configured) if configured else None
    if root is None:
        if not path.is_absolute():
            raise ValueError("Relative task assets require CLAUDE_CODE_DATA_ROOT on the Gym node")
        # Preserve existing datasets used by running services without this env.
        return path
    if not root.is_absolute():
        raise ValueError("CLAUDE_CODE_DATA_ROOT must be an absolute host root")
    candidate = root / path
    if candidate.is_symlink() or not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError("Task asset is a symlink or outside CLAUDE_CODE_DATA_ROOT")
    return candidate
