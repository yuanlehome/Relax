# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build Session advantages from exported outcome rewards."""

from __future__ import annotations

import math
from typing import Any


def advantage_func(metadata_by_session: list[dict[str, dict[str, Any]]]) -> list[dict[str, float]] | None:
    """Normalize one outcome per logical Session and broadcast it to its
    contexts."""
    outcomes: list[float] = [next(iter(exports.values()))["outcome_reward"] for exports in metadata_by_session]
    if max(outcomes) == min(outcomes):
        return None
    mean = sum(outcomes) / len(outcomes)
    centered = [value - mean for value in outcomes]
    std = math.sqrt(sum(value * value for value in centered) / len(centered))
    advantages = [value / std for value in centered]
    return [{name: advantage for name in exports} for exports, advantage in zip(metadata_by_session, advantages)]
