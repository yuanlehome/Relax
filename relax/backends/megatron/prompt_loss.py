# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Prompt denominators for a closed Agentic optimizer batch."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def prompt_denominators(
    rank_rows: list[list[tuple[int, int, int]]], global_batch_size: int
) -> tuple[dict[int, float], int]:
    """Reduce CPU metadata once per logical Session, never per context or CP
    rank.

    Each tuple contains (logical sample index, stable prompt index, original
    Session mask sum). The frozen source denominator is M_prompt * N_prompt / GBS.
    TIS changes the numerator mask without changing this denominator.
    """
    if isinstance(global_batch_size, bool) or not isinstance(global_batch_size, int) or global_batch_size <= 0:
        raise ValueError("Prompt loss requires a positive configured global batch size")
    sessions: dict[int, tuple[int, int]] = {}
    prompt_tokens: dict[int, int] = defaultdict(int)
    for rows in rank_rows:
        local: dict[int, tuple[int, int]] = {}
        for sample_index, prompt_index, mask_sum in rows:
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (sample_index, prompt_index, mask_sum)
            ):
                raise ValueError("Prompt loss metadata must contain nonnegative CPU integers")
            value = (prompt_index, mask_sum)
            if local.setdefault(sample_index, value) != value:
                raise ValueError("Contexts from a Session disagree on prompt or original token count")
        if sessions.keys() & local.keys():
            raise ValueError("A logical Session must not be duplicated across data parallel ranks")
        sessions.update(local)
        for prompt_index, mask_sum in local.values():
            prompt_tokens[prompt_index] += mask_sum
    if not sessions:
        raise ValueError("Prompt loss requires a nonempty closed optimizer batch")
    scale = len(prompt_tokens) / global_batch_size
    return {prompt: count * scale for prompt, count in prompt_tokens.items()}, sum(prompt_tokens.values())


def apply_prompt_denominators(batch: dict[str, Any], global_batch_size: int, process_group: Any) -> None:
    """Gather over the existing DP-without-CP Gloo group on every member
    rank."""
    import torch.distributed as dist

    error = None
    try:
        rows = list(
            zip(batch["sample_indices"], batch["loss_prompt_ids"], batch["sample_index_mask_sums"], strict=True)
        )
        # Validate before communication, but publish errors collectively so peers
        # cannot hang waiting for a rank that failed local metadata validation.
        if rows:
            prompt_denominators([rows], global_batch_size)
    except (KeyError, TypeError, ValueError) as exc:
        rows = []
        error = type(exc).__name__
    payloads: list[Any] = [None] * dist.get_world_size(process_group)
    dist.all_gather_object(payloads, (error, rows), group=process_group)
    if any(item[0] is not None for item in payloads):
        raise ValueError("Invalid prompt loss metadata on a data parallel rank")
    denominators, total_tokens = prompt_denominators([item[1] for item in payloads], global_batch_size)
    batch["sample_index_mask_sums"] = [denominators[prompt] for prompt in batch.pop("loss_prompt_ids")]
    # A per-row CPU column survives packing and microbatch slicing. It cancels
    # MCore's required token normalization without altering model configuration.
    batch["prompt_loss_total_tokens"] = [total_tokens] * len(batch["sample_indices"])


def validate_prompt_equal_loss(args: Any) -> None:
    """Limit the opt-in mode to its validated closed-batch execution path."""
    if not getattr(args, "prompt_equal_loss", False):
        return
    # Megatron's parser returns a list for explicit --moe-aux-loss-coeff
    # values, but its default is a scalar. A nonempty zero list is still disabled.
    aux_coefficients = getattr(args, "moe_aux_loss_coeff", 0.0)
    if not isinstance(aux_coefficients, (list, tuple)):
        aux_coefficients = (aux_coefficients,)
    zero_aux_loss = bool(aux_coefficients) and all(
        not isinstance(value, bool) and isinstance(value, (int, float)) and value == 0 for value in aux_coefficients
    )
    if not (
        getattr(args, "train_backend", None) == "megatron"
        and getattr(args, "use_agentic_rollout", False)
        and getattr(args, "hybrid", False)
        and getattr(args, "advantage_estimator", None) == "grpo"
        and getattr(args, "loss_type", None) == "policy_loss"
        and getattr(args, "calculate_per_token_loss", False)
        and not getattr(args, "dynamic_context_parallel", False)
        and getattr(args, "num_steps_per_rollout", None) == 1
        and getattr(args, "num_iters_per_train_update", None) == 1
        and not getattr(args, "use_opd", False)
        and not getattr(args, "enable_mtp_training", False)
        and zero_aux_loss
        and not getattr(args, "custom_pg_loss_reducer_function_path", None)
    ):
        raise ValueError(
            "--prompt-equal-loss requires Agentic GRPO hybrid, static CP, one optimizer step and one rollout mini, "
            "the token-normalized MCore backend, and no OPD, MTP, MoE auxiliary loss or custom PG reducer"
        )


def scale_prompt_reducer(reducer: Any, batch: dict[str, Any]) -> Any:
    """Cancel the backend's token denominator while retaining its CP contract.

    MCore sums microbatch gradients and divides them by the original global
    token count S. Returning S/B_eff times the source reducer therefore gives
    exactly its non-token objective R/B_eff, including fractional D_P clamps.
    """
    totals = batch["prompt_loss_total_tokens"]
    effective_batch_size = batch["dynamic_global_batch_size"]
    if not totals or any(value != totals[0] for value in totals) or effective_batch_size <= 0:
        raise ValueError("Prompt loss requires consistent closed-batch normalization metadata")
    scale = totals[0] / effective_batch_size
    return lambda values: reducer(values) * scale
