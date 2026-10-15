# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Apply harness-declared response ownership at the existing training
boundary."""

from __future__ import annotations

import copy
import math
from typing import Any


RESPONSE_MASK_KEY = "claude_code_response_mask"


def validate_response_masks(exports: dict[str, dict[str, Any]]) -> dict[str, list[bool]]:
    """Require every committed response to have exactly one training owner."""
    plans = [metadata.get(RESPONSE_MASK_KEY) for metadata in exports.values()]
    planned = any(plan is not None for plan in plans)
    if planned and any(not isinstance(plan, dict) for plan in plans):
        raise ValueError("Every context must declare the same response ownership contract")
    names = plans[0].get("context_names") if planned else None
    if planned and (
        not isinstance(names, list)
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
        or set(names) != set(exports)
    ):
        raise ValueError("Response masking requires every named context from the Session")
    owners: dict[str, int] = {}
    result: dict[str, list[bool]] = {}
    leaves: set[str] = set()
    for name, metadata in exports.items():
        trace = metadata.get("agentic_trace", {})
        turns = trace.get("turns") if isinstance(trace, dict) else None
        if not isinstance(turns, list) or not turns:
            raise ValueError("Export is missing the committed agentic_trace.turns supplied by Relax")
        hashes = [turn.get("resp_state_hash") if isinstance(turn, dict) else None for turn in turns]
        if any(not isinstance(value, str) or not value for value in hashes) or len(set(hashes)) != len(hashes):
            raise ValueError("Each committed turn requires a unique resp_state_hash")
        plan = metadata.get(RESPONSE_MASK_KEY)
        if planned:
            if (
                plan.get("version") != 1
                or plan.get("context_name") != name
                or plan.get("context_names") != names
                or plan.get("leaf_state_hash") != trace.get("leaf_state_hash")
                or plan.get("response_state_hashes") != hashes
                or trace.get("turn_count") != len(turns)
                or [turn.get("turn_idx") for turn in turns] != list(range(len(turns)))
            ):
                raise ValueError("Harness response ownership does not match the complete committed exports")
            leaf = plan.get("leaf_state_hash")
            if not isinstance(leaf, str) or not leaf or leaf in leaves:
                raise ValueError("Explicit context leaves must be unique committed states")
            leaves.add(leaf)
            selected = plan.get("train_response_hashes")
            if (
                not isinstance(selected, list)
                or not selected
                or any(not isinstance(value, str) for value in selected)
                or len(set(selected)) != len(selected)
                or selected != [value for value in hashes if value in selected]
            ):
                raise ValueError("Training responses must be a nonempty ordered subset of committed responses")
        else:
            selected = hashes
        result[name] = [value in selected for value in hashes]
        for value in hashes:
            owners[value] = owners.get(value, 0) + int(value in selected)
    if any(count != 1 for count in owners.values()):
        raise ValueError("Exports share generated responses without exactly one declared training owner")
    return result


def convert_samples_to_train_data(args: Any, samples: list[Any]) -> Any:
    """Mask duplicate response spans before the standard converter counts
    tokens.

    Work on shallow Sample copies with independent masks, so retries are
    idempotent and rollout dumps retain the original committed lineage.
    """
    from relax.utils.logging_utils import get_logger
    from relax.utils.utils import convert_samples_to_train_data as default_convert

    by_session: dict[str, dict[str, Any]] = {}
    logical_indices: dict[int, str] = {}
    for sample in samples:
        if sample.remove_sample:
            raise ValueError("Response ownership cannot include a context removed from training")
        trace = sample.metadata.get("agentic_trace", {})
        session_id = sample.session_id
        if not session_id or trace.get("session_id") != session_id or sample.index is None:
            raise ValueError("Response masking requires the committed Session and logical sample identity")
        if logical_indices.setdefault(sample.index, session_id) != session_id:
            raise ValueError("One logical sample index cannot identify different Sessions")
        plan = sample.metadata.get(RESPONSE_MASK_KEY)
        name = plan.get("context_name") if isinstance(plan, dict) else str(len(by_session.get(session_id, {})))
        exports = by_session.setdefault(session_id, {})
        if name in exports:
            raise ValueError("A Session cannot transfer the same named export twice")
        if exports:
            first = next(iter(exports.values()))
            if (first.index, first.group_index) != (sample.index, sample.group_index):
                raise ValueError("All contexts from a Session must share one logical sample and Group index")
        exports[name] = sample

    copies = {id(sample): copy.copy(sample) for sample in samples}
    masked_tokens = 0
    unique_tokens = 0
    for exports in by_session.values():
        decisions = validate_response_masks({name: sample.metadata for name, sample in exports.items()})
        committed: dict[str, tuple[list[int], list[float]]] = {}
        for name, sample in exports.items():
            mask = sample.loss_mask
            if (
                not isinstance(mask, list)
                or len(mask) != sample.response_length
                or any(value not in (0, 1) for value in mask)
                or sample.rollout_log_probs is None
                or len(sample.rollout_log_probs) != sample.response_length
                or any(not math.isfinite(value) for value in sample.rollout_log_probs)
                or len(sample.tokens) < sample.response_length
            ):
                raise ValueError("Committed response mask, tokens and rollout logprobs must align")
            copied_mask = list(mask)
            response_tokens = sample.tokens[len(sample.tokens) - sample.response_length :]
            cursor = 0
            turns = sample.metadata["agentic_trace"]["turns"]
            for turn, keep in zip(turns, decisions[name], strict=True):
                length = turn.get("response_length")
                if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
                    raise ValueError("Each committed response must have a positive token length")
                while cursor < len(mask) and mask[cursor] == 0:
                    cursor += 1
                end = cursor + length
                if end > len(mask) or any(value != 1 for value in mask[cursor:end]):
                    raise ValueError("Committed turn lengths do not align with response mask spans")
                value = (response_tokens[cursor:end], sample.rollout_log_probs[cursor:end])
                response_hash = turn["resp_state_hash"]
                if committed.setdefault(response_hash, value) != value:
                    raise ValueError("Shared committed responses have inconsistent tokens or rollout logprobs")
                if keep:
                    unique_tokens += length
                else:
                    copied_mask[cursor:end] = [0] * length
                    masked_tokens += length
                cursor = end
            if any(mask[cursor:]):
                raise ValueError("Response mask contains tokens outside the committed turn trace")
            copies[id(sample)].loss_mask = copied_mask
            if sample.metadata.get("termination_reason") == "backend_timeout":
                copies[id(sample)].status = type(sample).Status.TRUNCATED

    get_logger(__name__).info(
        "CC response masks: sessions=%s rows=%s unique_response_tokens=%s duplicated_response_tokens_masked=%s",
        len(by_session),
        len(samples),
        unique_tokens,
        masked_tokens,
    )
    converted_args = copy.copy(args)
    converted_args.custom_convert_samples_to_train_data_path = None
    return default_convert(converted_args, [copies[id(sample)] for sample in samples])
