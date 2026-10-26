# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Dressage-compatible GRPO normalization over logical Claude Code Sessions."""

from __future__ import annotations

import json
import math
import os
from typing import Any

from examples.claude_code_harbor.training_data import validate_response_masks
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def advantage_func(metadata_by_session: list[dict[str, dict[str, Any]]]) -> list[dict[str, float]] | None:
    """Broadcast one trajectory advantage to its explicitly exported contexts.

    Every export must carry the same finite ``outcome_reward`` for its Session.
    Context fanout never contributes additional GRPO siblings. The source uses
    population standard deviation, without an additive denominator epsilon.
    """
    if len(metadata_by_session) < 2:
        raise ValueError("Claude Code GRPO requires at least two logical Sessions")

    outcomes: list[float] = []
    for exports in metadata_by_session:
        if not exports:
            raise ValueError("Each Session must explicitly export at least one context")
        values: list[float] = []
        validate_response_masks(exports)
        for metadata in exports.values():
            value = metadata.get("outcome_reward")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Each export requires a finite numeric outcome_reward")
            values.append(float(value))
        if any(value != values[0] for value in values):
            raise ValueError("Contexts from one Session must carry the same outcome_reward")
        outcomes.append(values[0])

    # Observe logical outcomes before filtering; never count context fanout as siblings.
    if os.environ.get("HARBOR_LOG_GROUP_OUTCOMES") == "1":
        all_zero = all(value == 0 for value in outcomes)
        all_one = all(value == 1 for value in outcomes)
        keep = max(outcomes) != min(outcomes)
        logger.info(
            "HARBOR_GROUP_OUTCOMES %s",
            json.dumps(
                {
                    "sessions": len(outcomes),
                    "outcomes": outcomes,
                    "reward_mean": sum(outcomes) / len(outcomes),
                    "positive_count": sum(value == 1 for value in outcomes),
                    "classification": "all_zero"
                    if all_zero
                    else "all_one"
                    if all_one
                    else "mixed"
                    if keep
                    else "constant",
                    "keep": keep,
                },
                sort_keys=True,
            ),
        )
    # This is the source dynamic-sampling rule, evaluated before normalization.
    if max(outcomes) == min(outcomes):
        return None
    mean = sum(outcomes) / len(outcomes)
    centered = [value - mean for value in outcomes]
    std = math.sqrt(sum(value * value for value in centered) / len(centered))
    advantages = [value / std for value in centered] if std > 1e-6 else centered
    return [{name: advantage for name in exports} for exports, advantage in zip(metadata_by_session, advantages)]
