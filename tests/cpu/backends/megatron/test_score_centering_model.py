# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SC output-head activation must reject incompatible Megatron schedules."""

from argparse import Namespace

import pytest

from relax.backends.megatron.model import _use_score_centering_response_head


def _args(**overrides):
    values = dict(
        advantage_estimator="score_centering", recompute_loss_function=True, recompute_loss_function_use_reentrant=True
    )
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize(
    "option",
    [
        "overlap_grad_reduce",
        "defer_embedding_wgrad_compute",
        "overlap_moe_expert_parallel_comm",
        "enable_mtp_training",
        "fp8",
        "fp4",
    ],
)
def test_sc_response_head_rejects_incompatible_schedule(monkeypatch, option):
    monkeypatch.setenv("RELAX_SCORE_CENTERING_RESPONSE_ONLY_LM_HEAD", "1")
    with pytest.raises(ValueError, match=option):
        _use_score_centering_response_head(_args(**{option: True}))


@pytest.mark.parametrize("option", ["recompute_loss_function", "recompute_loss_function_use_reentrant"])
def test_sc_response_head_requires_reentrant_checkpoint(monkeypatch, option):
    monkeypatch.setenv("RELAX_SCORE_CENTERING_RESPONSE_ONLY_LM_HEAD", "1")
    with pytest.raises(ValueError, match="reentrant loss recomputation"):
        _use_score_centering_response_head(_args(**{option: False}))


def test_sc_response_head_is_opt_in_and_leaves_other_algorithms_unchanged(monkeypatch):
    monkeypatch.setenv("RELAX_SCORE_CENTERING_RESPONSE_ONLY_LM_HEAD", "0")
    assert not _use_score_centering_response_head(_args(overlap_grad_reduce=True))
    monkeypatch.setenv("RELAX_SCORE_CENTERING_RESPONSE_ONLY_LM_HEAD", "1")
    assert _use_score_centering_response_head(_args())
    assert not _use_score_centering_response_head(_args(advantage_estimator="grpo", overlap_grad_reduce=True))
