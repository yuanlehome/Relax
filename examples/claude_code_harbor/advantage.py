# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build Session advantages from exported outcome rewards."""

from __future__ import annotations

import math
from typing import Any, Optional


def advantage_func(metadata_by_session: list[dict[str, dict[str, Any]]]) -> Optional[list[dict[str, float]]]:
    """Normalize one outcome per logical Session and broadcast it to its
    contexts."""
    outcomes: list[float] = [next(iter(exports.values()))["outcome_reward"] for exports in metadata_by_session]
    if max(outcomes) == min(outcomes):
        return None
    mean = sum(outcomes) / len(outcomes)
    centered = [value - mean for value in outcomes]
    std = math.sqrt(sum(value * value for value in centered) / len(centered))
    advantages = [value / (std + 1e-6) for value in centered]
    return [{name: advantage for name in exports} for exports, advantage in zip(metadata_by_session, advantages)]


def score_centering_advantage(
    metadata_by_session: list[dict[str, dict[str, Any]]],
) -> Optional[list[dict[str, float]]]:
    """Reject zero-variance groups; center each Session's outcome and broadcast
    it to that Session's exports."""
    outcomes: list[float] = [next(iter(exports.values()))["outcome_reward"] for exports in metadata_by_session]
    if max(outcomes) == min(outcomes):
        return None
    mean = sum(outcomes) / len(outcomes)
    return [
        {name: outcome - mean for name in exports}
        for exports, outcome in zip(metadata_by_session, outcomes, strict=True)
    ]
