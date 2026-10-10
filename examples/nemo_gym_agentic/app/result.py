# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Map a terminal NeMo Gym result into Relax's managed-session output."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from .protocol import TrialResult, TrialStatus


def to_relax_output(result: TrialResult) -> dict[str, Any]:
    if result.status not in {TrialStatus.COMPLETED, TrialStatus.TRUNCATED}:
        raise ValueError(f"Cannot materialize non-success terminal status: {result.status.value}")
    if result.reward is None:
        # A Gateway deadline can end the trial before its verifier runs. Exit
        # as an agent failure so Relax drops the group instead of invoking RM.
        raise ValueError(
            f"NeMo Gym trial has no reward: request_id={result.request_id} "
            f"status={result.status.value} error_code={result.error_code or 'missing_reward'}"
        )

    reward = copy.deepcopy(result.reward)
    metadata = copy.deepcopy(result.metadata)
    metadata["nemo_gym"] = {
        "request_id": result.request_id,
        "status": result.status.value,
        "metrics": copy.deepcopy(result.metrics),
    }
    if result.artifact_ref is not None:
        metadata["nemo_gym"]["artifact_ref"] = result.artifact_ref
    if result.error is not None:
        safe_error = {
            key: value for key in ("code", "type") if isinstance((value := result.error.get(key)), str) and value
        }
        if safe_error:
            metadata["nemo_gym"]["error"] = safe_error

    if isinstance(reward, dict) and "scalar" in reward:
        scalar = reward.get("scalar")
        if scalar is not None and not isinstance(scalar, bool) and isinstance(scalar, (int, float)):
            metadata["nemo_gym"]["reward_components"] = copy.deepcopy(reward.get("components", {}))
            reward = float(scalar)

    return {
        "reward": reward,
        "metadata": metadata,
    }


def explicit_context_records(payload: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Expand harness-owned canonical contexts, without reconstructing history.

    Absence of the contexts key preserves the legacy implicit export. Once the
    key is present, malformed contexts must fail instead of falling back.
    Explicit context rewards are preserved; missing rewards inherit the trial
    reward. Recipes using logical-session advantages can choose one reward
    anchor before writing the records.
    """
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Result metadata must be a JSON object")
    if "contexts" not in metadata:
        return None
    contexts = metadata["contexts"]
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("metadata.contexts must be a nonempty array of canonical context objects")
    shared_metadata = {key: copy.deepcopy(value) for key, value in metadata.items() if key != "contexts"}
    reward = payload.get("reward")
    scalar = reward.get("scalar") if isinstance(reward, dict) else reward
    has_scalar = not isinstance(scalar, bool) and isinstance(scalar, (int, float)) and math.isfinite(scalar)
    records: list[dict[str, Any]] = []
    names: set[str] = set()
    for context in contexts:
        if not isinstance(context, dict):
            raise ValueError("Every context must be a JSON object")
        name = context.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Every context must have a unique nonempty name")
        names.add(name)
        if not isinstance(context.get("messages"), list) or not context["messages"]:
            raise ValueError("Every context requires its complete canonical messages")
        if not isinstance(context.get("tools"), list) or not isinstance(context.get("chat_template_kwargs"), dict):
            raise ValueError("Every context requires its exact tools and chat_template_kwargs")
        local_metadata = context.get("metadata", {})
        if not isinstance(local_metadata, dict) or "contexts" in local_metadata:
            raise ValueError("Context metadata must be an object without nested contexts")
        if has_scalar:
            for source in (shared_metadata, local_metadata):
                if "outcome_reward" in source and (
                    isinstance(source["outcome_reward"], bool) or source["outcome_reward"] != scalar
                ):
                    raise ValueError("Context outcome_reward conflicts with the trial reward")
        record = copy.deepcopy(context)
        record["metadata"] = {**copy.deepcopy(shared_metadata), **copy.deepcopy(local_metadata)}
        if has_scalar:
            record["metadata"]["outcome_reward"] = float(scalar)
        # Gateway provenance is authoritative even if the harness context
        # contains a business metadata field with the same name.
        if "nemo_gym" in shared_metadata:
            record["metadata"]["nemo_gym"] = copy.deepcopy(shared_metadata["nemo_gym"])
        record.setdefault("reward", copy.deepcopy(reward))
        records.append(record)
    return records


def write_relax_output(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically publish output, tolerating Relax removing a cancelled session
    dir."""

    output_path = Path(path)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        records = explicit_context_records(payload)
        serialized = "\n".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            for record in (records if records is not None else [payload])
        )
        temporary_path.write_text(
            serialized,
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
    except FileNotFoundError:
        temporary_path.unlink(missing_ok=True)
        return
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
