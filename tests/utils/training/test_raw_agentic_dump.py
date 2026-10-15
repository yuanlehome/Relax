# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Diagnostic raw dumps must not change the generic debug-dump contract."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from relax.utils.training import train_dump_utils as dumps


RAW_HOOK = "relax.utils.training.train_dump_utils.enable_raw_policy_capture"


@pytest.mark.parametrize("hook,step", [(None, 0), ("another.module.hook", 0), (RAW_HOOK, 2)])
def test_generic_train_dump_preserves_decoding_without_megatron(tmp_path: Path, hook: str | None, step: int) -> None:
    args = SimpleNamespace(
        save_debug_train_data=str(tmp_path / "train-{rollout_id}-{rank}.pt"),
        custom_megatron_before_train_step_hook_path=hook,
    )
    tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: f"token-{tokens[0]}")
    batch = {"tokens": [torch.tensor([4, 5])], "rewards": [1.0]}
    with (
        mock.patch.object(torch.distributed, "get_rank", return_value=7),
        mock.patch.object(
            dumps, "_raw_dump_producer", side_effect=AssertionError("Generic dump must not inspect Megatron topology")
        ),
    ):
        dumps.save_debug_train_data(args, rollout_id=step, rollout_data=batch, tokenizer=tokenizer)
    result = torch.load(tmp_path / f"train-{step}-7.pt", weights_only=True)
    assert result == {
        "rollout_id": step,
        "rank": 7,
        "samples": [{"sample_index": 0, "tokens": [4, 5], "rewards": 1.0, "decoded_tokens": ["token-4", "token-5"]}],
    }


@pytest.mark.parametrize("step", [0, 1])
def test_explicit_raw_hook_dispatches_the_first_two_train_steps(step: int) -> None:
    args = SimpleNamespace(custom_megatron_before_train_step_hook_path=RAW_HOOK)
    batch = {"sentinel": object()}
    with mock.patch.object(dumps, "_save_raw_debug_train_data") as raw:
        dumps.save_debug_train_data(args, rollout_id=step, rollout_data=batch)
    raw.assert_called_once_with(args, rollout_id=step, rollout_data=batch, tokenizer=None)


@pytest.mark.parametrize("raw", [False, True])
def test_rollout_dump_decoding_is_opt_in_to_raw_mode(tmp_path: Path, raw: bool) -> None:
    args = SimpleNamespace(
        save_debug_rollout_data=str(tmp_path / "rollout-{rollout_id}.pt"),
        custom_megatron_before_train_step_hook_path=RAW_HOOK if raw else None,
    )
    tokenizer = mock.Mock()
    tokenizer.decode.return_value = "decoded"
    sample = SimpleNamespace(to_dict=lambda: {"tokens": [4, 5], "reward": 1.0})
    # Backfill windows can exceed the two optimizer steps and still need raw data.
    dumps.save_debug_rollout_data(args, [sample], rollout_id=2, evaluation=False, tokenizer=tokenizer)
    result = torch.load(tmp_path / "rollout-2.pt", weights_only=True)
    assert result["samples"][0]["tokens"] == [4, 5]
    assert ("decoded_tokens" not in result["samples"][0]) == raw
    assert tokenizer.decode.call_count == (0 if raw else 2)


def test_raw_schedule_rejects_missing_duplicate_or_dummy_rows() -> None:
    prefix = "After dynamic batching, num_microbatches: "
    for message in [
        prefix + "[1], micro_batch_indices: [[0, 0]], dummy_microbatches=0",
        prefix + "[1], micro_batch_indices: [[0]], dummy_microbatches=1",
    ]:
        with pytest.raises(RuntimeError):
            dumps._raw_parse_schedule({"message": message})
    with pytest.raises(RuntimeError):
        dumps._raw_parse_schedule(None)
