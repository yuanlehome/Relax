# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import ast
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image

from relax.utils.logging_utils import get_logger
from relax.utils.multimodal.stats import get_sample_multimodal_stats


logger = get_logger(__name__)


_RAW_CAPTURE_STEPS = {(0, 0), (1, 0)}
_RAW_CAPTURE_MANAGER = None
_RAW_CAPTURE_DIRECTORY: Path | None = None
_LAST_BOUND_SCHEDULE_SEQUENCE = 0
_SCHEDULE_PREFIX = "After dynamic batching, num_microbatches: "
_SCHEDULE_PATTERN = re.compile(
    r"After dynamic batching, num_microbatches: (\[[^\n]*\]), "
    r"micro_batch_indices: (\[[^\n]*\]), dummy_microbatches=(\d+)"
)


class _RawScheduleObserver(logging.Handler):
    """Observe the existing CPU schedule message without changing logging."""

    relax_raw_schedule_observer_v1 = True

    def __init__(self) -> None:
        super().__init__()
        self._snapshot_lock = threading.Lock()
        self._sequence = 0
        self._latest: dict[str, Any] | None = None

    def emit(self, record: logging.LogRecord) -> None:
        if record.name != "relax.backends.megatron.data":
            return
        message = record.getMessage()
        if not message.startswith(_SCHEDULE_PREFIX):
            return
        # The message is already built by data.py. Do not inspect tensors or
        # parse large lists on the model's logging path.
        with self._snapshot_lock:
            self._sequence += 1
            self._latest = {
                "sequence": self._sequence,
                "created": record.created,
                "pid": record.process,
                "message": message,
            }

    def snapshot(self) -> dict[str, Any] | None:
        with self._snapshot_lock:
            return None if self._latest is None else dict(self._latest)


def _install_raw_schedule_observer() -> _RawScheduleObserver:
    data_logger = get_logger("relax.backends.megatron.data")
    for handler in data_logger.handlers:
        if getattr(handler, "relax_raw_schedule_observer_v1", False):
            return handler
    observer = _RawScheduleObserver()
    data_logger.addHandler(observer)
    return observer


# Must be installed at module import, before the first get_data_iterator.
# Never alter levels, filters, propagation or existing handlers.
_RAW_SCHEDULE_OBSERVER = _install_raw_schedule_observer()


def _raw_topology() -> dict[str, int]:
    from megatron.core import mpu

    if not torch.distributed.is_initialized():
        raise RuntimeError("Raw numerical dumps require an initialized training topology")
    return {
        "global_rank": torch.distributed.get_rank(),
        "world_size": torch.distributed.get_world_size(),
        "tp_rank": mpu.get_tensor_model_parallel_rank(),
        "tp_world_size": mpu.get_tensor_model_parallel_world_size(),
        "pp_rank": mpu.get_pipeline_model_parallel_rank(),
        "pp_world_size": mpu.get_pipeline_model_parallel_world_size(),
        "cp_rank": mpu.get_context_parallel_rank(),
        "cp_world_size": mpu.get_context_parallel_world_size(),
        "dp_rank_excluding_cp": mpu.get_data_parallel_rank(with_context_parallel=False),
        "dp_world_size_excluding_cp": mpu.get_data_parallel_world_size(with_context_parallel=False),
    }


def _raw_dump_producer() -> bool:
    from megatron.core import mpu

    return mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage(ignore_virtual=True)


def _raw_diagnostic_root(args: Any, rank: int) -> Path:
    template = args.save_debug_train_data
    if template is None:
        raise ValueError("The raw capture hook requires --save-debug-train-data")
    # A stable root even when the caller puts rollout_id in a directory name.
    return Path(template.format(rollout_id=0, rank=rank)).parent / "raw_policy_capture"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _raw_cpu(value: Any) -> Any:
    """Called only at a dump boundary, never from the before-step hook."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _raw_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_raw_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_raw_cpu(item) for item in value)
    return value


def _raw_algorithm_args(args: Any) -> dict[str, Any]:
    keys = (
        "train_backend",
        "loss_type",
        "advantage_estimator",
        "global_batch_size",
        "n_samples_per_prompt",
        "num_steps_per_rollout",
        "num_iters_per_train_update",
        "prompt_equal_loss",
        "calculate_per_token_loss",
        "normalize_advantages",
        "rewards_normalization",
        "use_rollout_logprobs",
        "keep_old_actor",
        "use_tis",
        "tis_clip_low",
        "tis_clip",
        "get_mismatch_metrics",
        "eps_clip",
        "eps_clip_high",
        "entropy_coef",
        "kl_coef",
        "kl_loss_coef",
        "use_kl_loss",
        "kl_loss_type",
        "rollout_temperature",
        "qkv_format",
        "data_pad_size_multiplier",
        "allgather_cp",
        "is_vl_model",
        "uses_unsplit_forward",
        "dynamic_context_parallel",
        "use_dynamic_batch_size",
        "use_dynamic_global_batch_size",
        "max_tokens_per_gpu",
        "log_probs_max_tokens_per_gpu",
        "per_rank_fetch",
        "hybrid",
        "fully_async",
    )
    return {key: getattr(args, key, None) for key in keys}


def _raw_plan_path(args: Any, topology: dict[str, int], rollout_id: int, step_id: int = 0) -> Path:
    root = _raw_diagnostic_root(args, topology["global_rank"])
    return root / f"rank-{topology['global_rank']}" / f"schedule-{rollout_id}-{step_id}.json"


def _raw_parse_schedule(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    if snapshot is None:
        raise RuntimeError("No actual dynamic microbatch schedule was observed before the train hook")
    match = _SCHEDULE_PATTERN.fullmatch(snapshot["message"])
    if match is None:
        raise RuntimeError("The observed microbatch schedule format does not match this diagnostic release")
    counts, indices = (ast.literal_eval(match.group(i)) for i in (1, 2))
    dummy_count = int(match.group(3))
    valid_integer = lambda value: isinstance(value, int) and not isinstance(value, bool) and value >= 0
    if (
        dummy_count != 0
        or not isinstance(counts, list)
        or len(counts) != 1
        or not valid_integer(counts[0])
        or counts[0] == 0
        or not isinstance(indices, list)
        or len(indices) != counts[0]
        or any(not isinstance(row, list) or not row or any(not valid_integer(i) for i in row) for row in indices)
    ):
        raise RuntimeError("Raw capture requires one rollout mini, a complete plan, and zero dummy microbatches")
    flattened = [index for group in indices for index in group]
    if sorted(flattened) != list(range(len(flattened))):
        raise RuntimeError("The actual microbatch plan does not cover consecutive local row ordinals exactly once")
    return {
        "observer_sequence": snapshot["sequence"],
        "record_created": snapshot["created"],
        "record_pid": snapshot["pid"],
        "raw_message_sha256": hashlib.sha256(snapshot["message"].encode()).hexdigest(),
        "num_microbatches": counts,
        "micro_batch_indices": indices,
        "dummy_microbatches": dummy_count,
        "local_row_count": len(flattened),
    }


def _raw_capture_is_nonempty(record: Any) -> bool:
    return bool(
        record.samples
        or record.tensors
        or record.expected
        or record.stages
        or record.captured_micro_batch_ids
        or record.response_lengths
        or record.total_lengths
        or record.sample_micro_batch_ids
        or record.loss_masks_tensor is not None
        or record.group_indices_tensor is not None
        or record.rewards is not None
    )


def enable_raw_policy_capture(
    args: Any, rollout_id: int, step_id: int, model: Any, optimizer: Any, scheduler: Any
) -> None:
    """Existing before-step hook: TP0/PP-last only, with no new collective."""
    del model, optimizer, scheduler
    global _RAW_CAPTURE_MANAGER, _RAW_CAPTURE_DIRECTORY, _LAST_BOUND_SCHEDULE_SEQUENCE

    if not _raw_dump_producer() or (int(rollout_id), int(step_id)) not in _RAW_CAPTURE_STEPS:
        return
    from relax.utils.replay import capture, capture_hooks
    from relax.utils.replay.schema import ActorStepId, Identity

    topology = _raw_topology()
    if (
        topology["dp_world_size_excluding_cp"] != 1
        or topology["pp_world_size"] != 1
        or topology["tp_world_size"] != 2
        or topology["cp_world_size"] != 32
        or getattr(args, "dynamic_context_parallel", False)
    ):
        raise ValueError("This two-step diagnostic is restricted to DP1 (excluding CP), PP1, TP2, static CP32")
    current = capture.current_step()
    if current is not None and _raw_capture_is_nonempty(current):
        raise RuntimeError("Refusing to replace a nonempty existing policy capture accumulator")
    if current is not None and current.actor_step_id != (int(rollout_id), int(step_id)):
        raise RuntimeError("The empty built-in capture accumulator belongs to a different actor step")
    plan = _raw_parse_schedule(_RAW_SCHEDULE_OBSERVER.snapshot())
    if plan["observer_sequence"] <= _LAST_BOUND_SCHEDULE_SEQUENCE:
        raise RuntimeError("A new optimizer step did not observe a new actual microbatch schedule")
    rank_directory = _raw_diagnostic_root(args, topology["global_rank"]) / f"rank-{topology['global_rank']}"
    active = capture.active()
    if active is not None and active is not _RAW_CAPTURE_MANAGER:
        raise RuntimeError("Another capture manager is active; do not enable global RELAX_REPLAY_CAPTURE for this run")
    if _RAW_CAPTURE_MANAGER is None:
        _RAW_CAPTURE_MANAGER = capture.enable(
            capture.CaptureConfig(
                enabled=True,
                output_dir=rank_directory,
                rank=topology["global_rank"],
                expected_ranks=[topology["global_rank"]],
                selected_steps=set(_RAW_CAPTURE_STEPS),
            )
        )
        _RAW_CAPTURE_DIRECTORY = rank_directory
    elif active is not _RAW_CAPTURE_MANAGER or _RAW_CAPTURE_DIRECTORY != rank_directory:
        raise RuntimeError("The raw policy capture manager changed identity or directory")
    bundle_id = f"raw-policy-{rollout_id}-{step_id}"
    plan.update(
        {
            "schema_version": 1,
            "kind": "raw_policy_schedule",
            "rollout_id": int(rollout_id),
            "step_id": int(step_id),
            "topology": topology,
            "algorithm_args": _raw_algorithm_args(args),
            "capture_bundle": str(rank_directory / bundle_id),
        }
    )
    _atomic_json(_raw_plan_path(args, topology, int(rollout_id), int(step_id)), plan)
    # model.py has already called begin_step_for. Its empty record used the
    # default DP(+CP) world size; replace it with the explicit DP-without-CP
    # identity. No tensor is copied or converted here.
    capture.begin_step(
        (int(rollout_id), int(step_id)),
        identity=Identity(
            actor_step_id=ActorStepId(int(rollout_id), int(step_id)),
            rank={
                "dp": topology["dp_world_size_excluding_cp"],
                "tp": topology["tp_world_size"],
                "pp": topology["pp_world_size"],
                "cp": topology["cp_world_size"],
            },
        ),
        config=capture_hooks.build_config_from_args(args),
        bundle_id=bundle_id,
    )
    _LAST_BOUND_SCHEDULE_SEQUENCE = plan["observer_sequence"]


def _raw_padding_lengths(args: Any, rollout_data: dict[str, Any]) -> list[int] | None:
    from relax.backends.megatron.cp_utils import maybe_padded_total_lengths

    return maybe_padded_total_lengths(
        rollout_data["total_lengths"],
        args.qkv_format,
        getattr(args, "is_vl_model", False)
        or "multimodal_train_inputs" in rollout_data
        or getattr(args, "uses_unsplit_forward", False),
    )


def _raw_local_response_lengths(
    args: Any, rollout_data: dict[str, Any], topology: dict[str, int], padded: list[int] | None
) -> list[int]:
    from relax.backends.megatron.cp_utils import get_logits_and_tokens_offset_with_cp

    result = []
    for index, (total, response) in enumerate(
        zip(rollout_data["total_lengths"], rollout_data["response_lengths"], strict=True)
    ):
        _, _, logit_ranges, _ = get_logits_and_tokens_offset_with_cp(
            total,
            response,
            args.qkv_format,
            rollout_data["max_seq_lens"][index] if args.qkv_format == "bshd" else None,
            padded_total_length=None if padded is None else padded[index],
            dynamic_cp_size=topology["cp_world_size"],
            dynamic_cp_rank=topology["cp_rank"],
        )
        result.append(sum(end - start for start, end in logit_ranges))
    return result


def _raw_validate_capture(
    plan: dict[str, Any],
    topology: dict[str, int],
    rollout_data: dict[str, Any],
    local: dict[str, Any],
    local_lengths: list[int],
    common: dict[str, Any] | None,
) -> dict[str, Any]:
    from relax.utils.replay import capture

    manager = _RAW_CAPTURE_MANAGER
    if manager is None or capture.active() is not manager:
        raise RuntimeError("The required raw policy capture manager was not enabled")
    if capture.current_step() is not None:
        raise RuntimeError("The policy capture accumulator is still open at the post-train dump boundary")
    manager.flush()
    if manager.error_count or manager.dropped_count:
        raise RuntimeError("Policy capture writer reported an error or dropped record")
    bundle = Path(plan["capture_bundle"])
    rank = topology["global_rank"]
    if not (bundle / "COMPLETE").is_file() or not (bundle / f"COMPLETE.{rank}").is_file():
        raise RuntimeError("The rank-local policy capture did not publish COMPLETE sentinels")
    index = json.loads((bundle / "index.json").read_text())
    manifest = json.loads((bundle / "manifest.json").read_text())
    if index["identity"]["actor_step_id"] != {"rollout_id": plan["rollout_id"], "step_id": plan["step_id"]}:
        raise RuntimeError("Captured actor-step identity differs from the persisted schedule")
    if index["identity"]["rank"] != {"dp": 1, "tp": 2, "pp": 1, "cp": 32}:
        raise RuntimeError("Captured parallel world sizes differ from the diagnostic topology")
    if manifest["stage_contracts"]["loss.policy"]["capability"] != "unsupported":
        raise RuntimeError("Unexpected generic replay capability; this diagnostic must not claim CP32 replay support")
    expected_ids = {capture.format_micro_batch_id(i) for i in range(len(plan["micro_batch_indices"]))}
    samples = index["samples"]
    if {sample["micro_batch_id"] for sample in samples} != expected_ids:
        raise RuntimeError("Captured microbatch IDs do not match the actual schedule")
    positions = {identifier: 0 for identifier in expected_ids}
    ordered_rows = []
    for sample in samples:
        identifier = sample["micro_batch_id"]
        mb = int(identifier.removeprefix("mb-"))
        position = positions[identifier]
        if position >= len(plan["micro_batch_indices"][mb]):
            raise RuntimeError("Capture contains excess rows for a microbatch")
        row = plan["micro_batch_indices"][mb][position]
        positions[identifier] += 1
        ordered_rows.append(row)
        if (sample["total_length"], sample["response_length"]) != (
            rollout_data["total_lengths"][row],
            rollout_data["response_lengths"][row],
        ):
            raise RuntimeError("Captured row lengths do not match the actual microbatch row binding")
        if len(sample["loss_mask"]) != rollout_data["response_lengths"][row]:
            raise RuntimeError("Capture did not preserve the full response mask")
        if common is not None and not torch.equal(
            torch.as_tensor(sample["loss_mask"]), torch.as_tensor(common["loss_masks"][row])
        ):
            raise RuntimeError("Captured full masks differ from post-ownership training masks")
    if sorted(ordered_rows) != list(range(plan["local_row_count"])):
        raise RuntimeError("Capture rows do not cover each actual training row exactly once")
    captured_length = sum(local_lengths[row] for row in ordered_rows)
    for name in ("old_log_probs", "log_probs", "entropy", "advantages"):
        if manifest["payloads"][name]["shape"] != [captured_length]:
            raise RuntimeError("Captured CP-local tensor shape differs from the explicit CP row slices")
    comparisons = {}
    for capture_name, local_name in (("old_log_probs", "log_probs"), ("advantages", "advantages")):
        captured = torch.load(bundle / "payloads" / f"{capture_name}.pt", map_location="cpu", weights_only=True)
        expected = torch.cat([torch.as_tensor(local[local_name][row]).reshape(-1) for row in ordered_rows])
        comparisons[capture_name] = torch.equal(captured, expected)
        if not comparisons[capture_name]:
            raise RuntimeError("Captured policy fields do not match the persisted actual row binding")
    return {
        "bundle": str(bundle),
        "complete": True,
        "writer_error_count": manager.error_count,
        "writer_dropped_count": manager.dropped_count,
        "capture_row_to_local_row": ordered_rows,
        "cp_local_tensor_length": captured_length,
        "exact_cpu_comparisons": comparisons,
        "capability": "raw tensors only; generic V2 CP32 replay unsupported",
    }


def _decode_tokens_to_chars(tokens: List[int], tokenizer) -> List[str]:
    """Decode each token to its corresponding string representation."""
    if tokenizer is None or not tokens:
        return []
    try:
        decoded_chars = []
        for token_id in tokens:
            try:
                char = tokenizer.decode([token_id], skip_special_tokens=False)
                decoded_chars.append(char)
            except Exception:
                decoded_chars.append(f"<{token_id}>")
        return decoded_chars
    except Exception as e:
        logger.warning(f"Failed to decode tokens: {e}")
        return []


def _tensor_to_list(tensor_or_list):
    """Convert tensor to list if needed."""
    if isinstance(tensor_or_list, torch.Tensor):
        return tensor_or_list.tolist()
    return tensor_or_list


def _reorganize_train_data_by_sample(rollout_data: Dict[str, Any], tokenizer=None) -> List[Dict[str, Any]]:
    """Reorganize train data from field-oriented to sample-oriented format.

    Args:
        rollout_data: Dict with keys like 'tokens', 'rewards', etc., each containing lists of values
        tokenizer: Optional tokenizer for decoding tokens

    Returns:
        List of sample dicts, each containing all fields for that sample
    """
    # Determine the number of samples from any available field
    num_samples = 0
    for key, value in rollout_data.items():
        if isinstance(value, (list, tuple)) and len(value) > 0:
            num_samples = len(value)
            break

    if num_samples == 0:
        return []

    # Use rollout_data keys for per-sample fields to keep extensibility
    per_sample_fields = list(rollout_data.keys())
    logger.info(
        "Reorganize train data with fields: %s",
        ", ".join(per_sample_fields) if per_sample_fields else "<empty>",
    )

    samples = []
    for i in range(num_samples):
        sample = {"sample_index": i}

        for field in per_sample_fields:
            if field in rollout_data:
                value = rollout_data[field]
                if isinstance(value, (list, tuple)) and i < len(value):
                    sample[field] = _tensor_to_list(value[i])
                elif isinstance(value, (list, tuple)):
                    sample[field] = None

        # Add decoded_tokens if tokens are available
        if "tokens" in sample and sample["tokens"] is not None:
            tokens = sample["tokens"]
            if tokenizer:
                sample["decoded_tokens"] = _decode_tokens_to_chars(tokens, tokenizer)
            else:
                sample["decoded_tokens"] = []

        samples.append(sample)

    return samples


def _save_raw_debug_train_data(args, *, rollout_id, rollout_data, tokenizer=None):
    """Raw, field-oriented two-step dump at the existing post-train
    boundary."""
    del tokenizer
    path_template = args.save_debug_train_data
    if path_template is None or not _raw_dump_producer():
        return
    from relax.utils.replay import capture

    topology = _raw_topology()
    manager = _RAW_CAPTURE_MANAGER
    if manager is None or capture.active() is not manager:
        raise RuntimeError("Raw training dump requires the configured before-train-step capture hook")
    manager.flush()
    if manager.error_count or manager.dropped_count:
        raise RuntimeError("Policy capture writer reported errors/drops before the post-train dump")
    path = Path(path_template.format(rollout_id=rollout_id, rank=topology["global_rank"]))
    plan_path = _raw_plan_path(args, topology, int(rollout_id))
    if not plan_path.is_file():
        raise RuntimeError("The actual train microbatch plan was not persisted by the before-step hook")
    plan = json.loads(plan_path.read_text())
    if plan["topology"] != topology or plan["rollout_id"] != int(rollout_id) or plan["step_id"] != 0:
        raise RuntimeError("Persisted train schedule identity/topology mismatch")
    rows = len(rollout_data["total_lengths"])
    required = (
        "tokens",
        "loss_masks",
        "response_lengths",
        "total_lengths",
        "sample_indices",
        "group_index",
        "sample_index_mask_sums",
        "prompt_loss_total_tokens",
        "rewards",
        "log_probs",
        "rollout_log_probs",
        "advantages",
        "returns",
    )
    if rows != plan["local_row_count"] or rows == 0:
        raise RuntimeError("Actual train row count does not match the persisted microbatch plan")
    for name in required:
        value = rollout_data.get(name)
        if not isinstance(value, (list, tuple)) or len(value) != rows:
            raise RuntimeError(f"Required training field {name!r} is missing or not row-aligned")
    for i in range(rows):
        if len(rollout_data["tokens"][i]) != rollout_data["total_lengths"][i]:
            raise RuntimeError("Actual training token sequence length mismatch")
        if len(rollout_data["loss_masks"][i]) != rollout_data["response_lengths"][i]:
            raise RuntimeError("Actual post-ownership response mask length mismatch")
    padded = _raw_padding_lengths(args, rollout_data)
    local_lengths = _raw_local_response_lengths(args, rollout_data, topology, padded)
    local_names = (
        "log_probs",
        "rollout_log_probs",
        "advantages",
        "returns",
        "ref_log_probs",
        "values",
        "teacher_log_probs",
    )
    local = {name: _raw_cpu(rollout_data[name]) for name in local_names if name in rollout_data}
    for name, values in local.items():
        if len(values) != rows or any(
            torch.as_tensor(value).numel() != local_lengths[i] for i, value in enumerate(values)
        ):
            raise RuntimeError(f"CP-local field {name!r} does not match explicit response shard lengths")
    binding_names = (
        "sample_indices",
        "group_index",
        "response_lengths",
        "total_lengths",
        "sample_index_mask_sums",
        "prompt_loss_total_tokens",
        "max_seq_lens",
    )
    binding = {name: _raw_cpu(rollout_data[name]) for name in binding_names if name in rollout_data}
    binding.update(
        {
            "local_row_ordinals": list(range(rows)),
            "cp_local_response_lengths": local_lengths,
            "padded_total_lengths": padded,
            "key_fields": ["rollout_id", "dp_rank_excluding_cp", "local_row_ordinal"],
        }
    )
    common = None
    if topology["cp_rank"] == 0:
        common_names = (
            "tokens",
            "loss_masks",
            "rewards",
            "sample_indices",
            "group_index",
            "response_lengths",
            "total_lengths",
            "sample_index_mask_sums",
            "prompt_loss_total_tokens",
            "max_seq_lens",
            "loss_prompt_ids",
            "dynamic_global_batch_size",
            "__rollout_mini_local_sample_counts__",
            "__rollout_mini_global_sample_counts__",
            "__rollout_mini_prompt_group_counts__",
        )
        common = {name: _raw_cpu(rollout_data[name]) for name in common_names if name in rollout_data}
        # Boundary metadata are not per-row columns. Preserve all declared
        # rollout-mini scalar/list metadata instead of silently dropping them.
        for name, value in rollout_data.items():
            if "rollout_mini" in name:
                common[name] = _raw_cpu(value)
        common["effective_global_batch_size"] = _raw_cpu(
            rollout_data.get("dynamic_global_batch_size", args.global_batch_size)
        )
        common["padded_total_lengths"] = padded
        common["token_sha256"] = [
            hashlib.sha256(torch.as_tensor(tokens).contiguous().numpy().tobytes()).hexdigest()
            for tokens in common["tokens"]
        ]
    evidence = _raw_validate_capture(plan, topology, rollout_data, local, local_lengths, common)
    payload = {
        "schema_version": 1,
        "kind": "raw_agentic_hybrid_train",
        "rollout_id": int(rollout_id),
        "rank": topology["global_rank"],
        "topology": topology,
        "binding": binding,
        "common": common,
        "local": local,
        "algorithm_args": _raw_algorithm_args(args),
        "actual_schedule": plan,
        "policy_capture": evidence,
        "common_lookup": {
            "rollout_id": int(rollout_id),
            "dp_rank_excluding_cp": topology["dp_rank_excluding_cp"],
            "cp_rank": 0,
            "tp_rank": 0,
            "pp_rank": topology["pp_rank"],
        },
        "semantics": {
            "rewards": "custom Session advantages, not raw outcome rewards",
            "sample_index_mask_sums": "post-apply_prompt_denominators D_p, not original Session mask sums",
            "loss_masks": "full response masks after harness ownership deduplication",
            "log_probs": "pre-update actor forward; fresh train forward is in policy_capture bundle",
        },
    }
    _atomic_torch_save(path, payload)
    _atomic_json(
        Path(str(path) + ".manifest.json"),
        {
            "schema_version": 1,
            "rollout_id": int(rollout_id),
            "topology": topology,
            "rows": rows,
            "full_common_written": common is not None,
            "local_fields": list(local),
            "cp_local_response_lengths": local_lengths,
            "actual_schedule": str(plan_path),
            "policy_capture": evidence,
            "torch_file": str(path),
        },
    )
    logger.info("Saved raw numerical training dump to %s (CP=%s, rows=%s)", path, topology["cp_rank"], rows)


def _save_raw_debug_rollout_data(args, data, rollout_id: int, evaluation: bool, tokenizer=None) -> None:
    """Save original Sample dictionaries without tokenizer work or decoding."""
    del tokenizer
    path_template = args.save_debug_rollout_data
    if path_template is None:
        return
    path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
    if evaluation:
        samples = [sample.to_dict() for info in data.values() for sample in info["samples"]]
    else:
        samples = [sample.to_dict() for sample in data]
    _atomic_torch_save(path, {"rollout_id": rollout_id, "samples": samples})
    logger.info("Saved raw rollout dictionaries to %s", path)


def _raw_dump_requested(args: Any) -> bool:
    return getattr(args, "custom_megatron_before_train_step_hook_path", None) == (
        "relax.utils.training.train_dump_utils.enable_raw_policy_capture"
    )


def save_debug_train_data(args, *, rollout_id, rollout_data, tokenizer=None):
    """Save debug train data reorganized by sample with tokenizer decoding.

    Args:
        args: Arguments containing save_debug_train_data path template
        rollout_id: The rollout ID
        rollout_data: Dict with training data organized by field
        tokenizer: Optional tokenizer for decoding tokens. If provided, will be used
                  to decode tokens to readable strings. Pass from caller context
                  (e.g., Actor.tokenizer) for efficiency.
    """
    if _raw_dump_requested(args) and (int(rollout_id), 0) in _RAW_CAPTURE_STEPS:
        _save_raw_debug_train_data(args, rollout_id=rollout_id, rollout_data=rollout_data, tokenizer=tokenizer)
        return
    if (path_template := args.save_debug_train_data) is not None:
        rank = torch.distributed.get_rank()
        path = Path(path_template.format(rollout_id=rollout_id, rank=rank))
        logger.info(f"Save debug train data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        # Reorganize data by sample
        samples = _reorganize_train_data_by_sample(rollout_data, tokenizer)

        torch.save(
            dict(
                rollout_id=rollout_id,
                rank=rank,
                samples=samples,
            ),
            path,
        )


def save_debug_rollout_data(args, data, rollout_id: int, evaluation: bool, tokenizer=None) -> None:
    """Save debug rollout data with tokenizer decoding support.

    Args:
        args: Arguments containing save_debug_rollout_data path template
        data: Either a list of Sample objects, or a dict (for evaluation data)
        rollout_id: The rollout ID
        evaluation: Whether this is evaluation data
        tokenizer: Optional tokenizer for decoding tokens. If provided, will be used
                  to decode tokens to readable strings. Pass from caller context
                  (e.g., GenerateState.tokenizer) for efficiency.
    """
    if _raw_dump_requested(args):
        _save_raw_debug_rollout_data(args, data, rollout_id, evaluation, tokenizer)
        return
    if (path_template := args.save_debug_rollout_data) is not None:
        path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
        logger.info(f"Save debug rollout data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        if evaluation:
            # Evaluation data is a dict with dataset_name -> info structure
            samples_list = [sample.to_dict() for dataset_name, info in data.items() for sample in info["samples"]]
        else:
            # Regular rollout data is a list of Sample objects
            samples_list = [sample.to_dict() for sample in data]

        # Add decoded_tokens field for each sample if tokenizer is provided
        for sample_dict in samples_list:
            tokens = sample_dict.get("tokens", [])
            if tokens and tokenizer:
                sample_dict["decoded_tokens"] = _decode_tokens_to_chars(tokens, tokenizer)
            else:
                sample_dict["decoded_tokens"] = []

        dump_data = dict(samples=samples_list)
        torch.save(dict(rollout_id=rollout_id, **dump_data), path)


def _summarize_multimodal_inputs(mm_inputs: dict) -> dict:
    """Summarize raw multimodal inputs (images, videos, audio) for logging."""
    summary = {}
    for key, value in mm_inputs.items():
        if isinstance(value, list):
            items = []
            for item in value:
                try:
                    if isinstance(item, Image.Image):
                        items.append(str(item))
                        continue
                except ImportError:
                    pass
                items.append(str(type(item).__name__))
            summary[key] = items if items else len(value)
        else:
            summary[key] = str(type(value).__name__)
    return summary


def _summarize_multimodal_train_inputs(mm_train_inputs: dict) -> dict:
    """Summarize processed multimodal train inputs (tensors) for logging."""
    import torch

    summary = {}
    for key, value in mm_train_inputs.items():
        if isinstance(value, torch.Tensor):
            summary[key] = f"Tensor(shape={list(value.shape)}, dtype={value.dtype})"
        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            summary[key] = [f"Tensor(shape={list(t.shape)}, dtype={t.dtype})" for t in value]
        else:
            summary[key] = str(type(value).__name__)
    return summary


def _sample_to_summary_record(sample, rollout_id: int, idx: int, dataset_name: str | None = None) -> dict:
    total_length = len(sample.tokens) if sample.tokens else 0
    response_length = sample.response_length
    prompt_length = max(total_length - response_length, 0)
    multimodal_stats = get_sample_multimodal_stats(sample)
    metadata = sample.metadata or {}
    record = {
        "rollout_id": rollout_id,
        "sample_index": idx,
        "prompt": sample.prompt,
        "response": sample.response,
        "reward": sample.reward,
        "prompt_length": prompt_length,
        "response_length": response_length,
        "total_length": total_length,
        "prompt_token_count": prompt_length,
        "response_token_count": response_length,
        "total_token_count": total_length,
        "image_count": multimodal_stats["image_count"],
        "image_token_count": multimodal_stats["image_token_count"],
        "multimodal_token_count": multimodal_stats["multimodal_token_count"],
        "agent_turns": metadata.get("rollout_turns", 1),
        "status": sample.status.value if hasattr(sample.status, "value") else str(sample.status),
        "group_index": sample.group_index,
    }
    if sample.label is not None:
        record["label"] = sample.label
    if sample.multimodal_inputs is not None:
        record["multimodal_inputs"] = _summarize_multimodal_inputs(sample.multimodal_inputs)
    if sample.multimodal_train_inputs is not None:
        record["multimodal_train_inputs"] = _summarize_multimodal_train_inputs(sample.multimodal_train_inputs)
    if dataset_name is not None:
        record["dataset"] = dataset_name
    return record


def _write_summary_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_rollout_result_jsonl(args, rollout_id: int, samples: list) -> None:
    """Save a lightweight per-step rollout result as a JSONL file.

    Always-on: writes one JSONL per rollout step with prompt, response,
    reward and sequence length for every sample.

    Args:
        args: Arguments containing ``rollout_result_dir``.
        rollout_id: The rollout ID (used as step number in the filename).
        samples: A list of :class:`Sample` objects from the rollout batch.
    """
    result_dir = getattr(args, "rollout_result_dir", None)
    if result_dir is None:
        return

    path = Path(result_dir) / "train" / f"{rollout_id}.jsonl"
    records = [_sample_to_summary_record(s, rollout_id, i) for i, s in enumerate(samples)]

    try:
        _write_summary_jsonl(path, records)
        logger.info(f"Saved rollout result ({len(records)} samples) to {path}")
    except Exception as e:
        logger.warning(f"Failed to save rollout result to {path}: {e}")


def save_eval_summary_jsonl(args, rollout_id: int, data: dict) -> None:
    """Save a lightweight per-step eval summary as a JSONL file.

    Similar to :func:`save_rollout_result_jsonl` but for evaluation rollouts.
    Each record includes an extra ``dataset`` field indicating which eval
    dataset the sample belongs to.

    Args:
        args: Arguments containing ``rollout_result_dir``.
        rollout_id: The rollout ID.
        data: Eval data dict — ``{dataset_name: {"samples": list[Sample], ...}}``.
    """
    result_dir = getattr(args, "rollout_result_dir", None)
    if result_dir is None:
        return

    eval_dir = Path(result_dir) / "eval"
    path = eval_dir / f"{rollout_id}.jsonl"

    records = []
    for dataset_name, info in data.items():
        samples = info.get("samples")
        if not samples:
            continue
        for i, s in enumerate(samples):
            records.append(_sample_to_summary_record(s, rollout_id, i, dataset_name=dataset_name))

    if not records:
        return

    try:
        _write_summary_jsonl(path, records)
        logger.info(f"Saved eval summary ({len(records)} samples) to {path}")
    except Exception as e:
        logger.warning(f"Failed to save eval summary to {path}: {e}")
