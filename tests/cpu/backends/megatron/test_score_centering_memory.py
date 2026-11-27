# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Actual SC loss/projection paths; dense logits are an independent numerical
oracle."""

import pytest
import torch
import torch.nn.functional as F
from test_score_centering_loss import _args, _backend, _score_inputs


@pytest.mark.parametrize(
    "dtype,entropy_coef,frozen_trunk",
    [
        (torch.float32, 0.0, False),
        (torch.float32, 0.2, False),
        (torch.bfloat16, 0.2, False),
        (torch.float32, 0.0, True),
    ],
)
def test_response_head_matches_dense_values_and_all_gradients(monkeypatch, dtype, entropy_coef, frozen_trunk):
    loss, _ = _backend(monkeypatch)
    args = _args(log_probs_chunk_size=2, entropy_coef=entropy_coef)
    _, tokens, ids = _score_inputs()
    torch.manual_seed(817)
    hidden = torch.randn(1, 8, 4, dtype=dtype).requires_grad_(not frozen_trunk)
    weight = torch.randn(8, 4, dtype=dtype, requires_grad=True)
    expected_hidden = hidden.detach().clone().requires_grad_(not frozen_trunk)
    expected_weight = weight.detach().clone().requires_grad_()
    dense = F.linear(expected_hidden, expected_weight)
    logp = (dense[0, 1:7, :5].float() / args.rollout_temperature).log_softmax(-1)
    expected = (logp.gather(-1, tokens[2:, None]).squeeze(-1), logp.gather(-1, ids), -(logp.exp() * logp).sum(-1))
    projected_rows = []

    def head(block):
        projected_rows.append(block.shape[0])
        return F.linear(block, weight), None

    _, result = loss.get_log_probs_and_entropy(
        hidden,
        args=args,
        unconcat_tokens=[tokens],
        total_lengths=[8],
        response_lengths=[6],
        gather_topk_token_ids=[ids],
        with_entropy=True,
        lm_head_forward=head,
    )
    actual = tuple(result[k][0] for k in ("log_probs", "topk_log_probs", "entropy"))
    rtol, atol = (2e-2, 8e-3) if dtype == torch.bfloat16 else (2e-5, 2e-6)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=rtol, atol=atol)
    (actual[0].sum() + actual[1].sum() + entropy_coef * actual[2].sum()).backward()
    (expected[0].sum() + expected[1].sum() + entropy_coef * expected[2].sum()).backward()
    torch.testing.assert_close(weight.grad, expected_weight.grad, rtol=rtol, atol=atol)
    assert torch.count_nonzero(weight.grad) and torch.isfinite(weight.grad).all()
    if frozen_trunk:
        assert hidden.grad is None
    else:
        torch.testing.assert_close(hidden.grad, expected_hidden.grad, rtol=rtol, atol=atol)
        assert not torch.count_nonzero(hidden.grad[:, :1])
        assert not torch.count_nonzero(hidden.grad[:, 7:])
    assert projected_rows and max(projected_rows) <= 2
    assert sum(projected_rows) == 12  # six response rows, forward and checkpoint replay
    if entropy_coef == 0:
        assert not actual[2].requires_grad


@pytest.mark.parametrize("entropy_coef", [0.0, 0.1])
def test_response_head_policy_loss_keeps_host_batch_and_current_metrics(monkeypatch, entropy_coef):
    loss, _ = _backend(monkeypatch)
    monkeypatch.setattr(loss, "compute_policy_opd_loss", lambda **kwargs: (None, {}))
    _, tokens, ids = _score_inputs()
    args = _args(entropy_coef=entropy_coef, log_probs_chunk_size=2, recompute_loss_function=True)
    batch = {
        "advantages": [torch.tensor([1.0, -1.0, 2.0, 1.0, -0.5, 0.5])],
        "log_probs": [torch.full((6,), -2.0)],
        "rollout_log_probs": [torch.full((6,), -2.0)],
        "rollout_topk_token_ids": [ids],
        "rollout_topk_log_probs": [torch.full((6, 2), -2.5)],
        "rollout_topk_support_complete": [torch.zeros(6, dtype=torch.bool)],
        "unconcat_tokens": [tokens],
        "total_lengths": [8],
        "response_lengths": [6],
        "loss_masks": [torch.ones(6)],
    }
    originals = dict(batch)
    torch.manual_seed(91)
    values, grads, metrics = [], [], []
    for chunked in (False, True):
        hidden = torch.randn(1, 8, 4, generator=torch.Generator().manual_seed(17), requires_grad=True)
        weight = torch.randn(8, 4, generator=torch.Generator().manual_seed(18), requires_grad=True)

        def head(block):
            return F.linear(block, weight), None

        value, stats = loss._compute_loss_and_metrics(
            args,
            batch,
            hidden if chunked else F.linear(hidden, weight),
            lambda x: x.sum(),
            is_embedding=False,
            lm_head_forward=head if chunked else None,
        )
        value.backward()
        values.append(value.detach())
        metrics.append(stats)
        grads.append((hidden.grad, weight.grad))
        assert all(batch[key] is before for key, before in originals.items())
        assert batch["rollout_topk_token_ids"][0].device.type == "cpu"
    torch.testing.assert_close(values[0], values[1], rtol=2e-5, atol=2e-6)
    assert set(metrics[0]) == set(metrics[1])
    for key in metrics[0]:
        torch.testing.assert_close(metrics[0][key], metrics[1][key], rtol=2e-5, atol=2e-6)
    for a, b in zip(grads[0], grads[1], strict=True):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)


def test_empty_cp_response_preserves_decoder_graph_without_projection(monkeypatch):
    loss, cp = _backend(monkeypatch, rank=1, cp_size=2)
    args = _args(log_probs_chunk_size=2)
    hidden = torch.randn(1, 4, 4, requires_grad=True)
    tokens = torch.arange(8) % 5
    _, result = loss.get_log_probs_and_entropy(
        hidden,
        args=args,
        unconcat_tokens=[tokens],
        total_lengths=[8],
        response_lengths=[1],
        gather_topk_token_ids=[torch.empty((0, 2), dtype=torch.long)],
        with_entropy=True,
        cp_output_is_zigzag=True,
        lm_head_forward=lambda block: pytest.fail("empty response must not project"),
    )
    total = sum(value.sum() for values in result.values() for value in values)
    total.backward()
    assert hidden.grad is not None and not torch.count_nonzero(hidden.grad)
    assert all(value.numel() == 0 for values in result.values() for value in values)


def _response_head_distributed_worker(rank, rendezvous):
    from datetime import timedelta

    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=90)
    )
    try:
        for kind in ("tp", "cp", "cp_empty"):
            with pytest.MonkeyPatch.context() as patch:
                loss, _ = _backend(
                    patch, rank=rank, tp_size=2 if kind == "tp" else 1, cp_size=1 if kind == "tp" else 2
                )
                args = _args(log_probs_chunk_size=2, entropy_coef=0.1)
                _, tokens, ids = _score_inputs()
                total_response = 1 if kind == "cp_empty" else 6
                ids = ids[-total_response:]
                full_hidden = torch.randn(1, 8, 4, generator=torch.Generator().manual_seed(19))
                full_weight = torch.randn(8, 4, generator=torch.Generator().manual_seed(20))
                oracle_hidden = full_hidden.clone().requires_grad_()
                oracle_weight = full_weight.clone().requires_grad_()
                first = 8 - total_response - 1
                logp = (F.linear(oracle_hidden, oracle_weight)[0, first:7, :5] / args.rollout_temperature).log_softmax(
                    -1
                )
                expected = (
                    logp.gather(-1, tokens[-total_response:, None]).squeeze(-1),
                    logp.gather(-1, ids),
                    -(logp.exp() * logp).sum(-1),
                )
                if kind == "tp":
                    owned = list(range(8))
                    score_rows = list(range(total_response))
                    weight = full_weight[rank * 4 : (rank + 1) * 4].clone().requires_grad_()
                else:
                    owned = [rank * 2, rank * 2 + 1, 6 - rank * 2, 7 - rank * 2]
                    score_rows = [i for i in range(total_response) if first + i in owned]
                    weight = full_weight.clone().requires_grad_()
                hidden = full_hidden[:, owned].clone().requires_grad_()

                def head(block):
                    return F.linear(block, weight), None

                _, result = loss.get_log_probs_and_entropy(
                    hidden,
                    args=args,
                    unconcat_tokens=[tokens],
                    total_lengths=[8],
                    response_lengths=[total_response],
                    gather_topk_token_ids=[ids[score_rows]],
                    with_entropy=True,
                    cp_output_is_zigzag=kind != "tp",
                    lm_head_forward=head,
                )
                actual = tuple(result[k][0] for k in ("log_probs", "topk_log_probs", "entropy"))
                for value, reference in zip(actual, expected, strict=True):
                    torch.testing.assert_close(value, reference[score_rows], rtol=2e-5, atol=2e-6)
                sum(value.sum() for value in actual).backward()
                sum(value.sum() for value in expected).backward()
                # Megatron owns these cross-rank gradient reductions. Exercise the real
                # Gloo reductions here; this fixture does not claim native fused wgrad coverage.
                if kind == "tp":
                    dist.all_reduce(hidden.grad)
                    expected_w = oracle_weight.grad[rank * 4 : (rank + 1) * 4]
                else:
                    if weight.grad is None:
                        weight.grad = torch.zeros_like(weight)
                    dist.all_reduce(weight.grad)
                    expected_w = oracle_weight.grad
                torch.testing.assert_close(hidden.grad, oracle_hidden.grad[:, owned], rtol=3e-5, atol=3e-6)
                torch.testing.assert_close(weight.grad, expected_w, rtol=3e-5, atol=3e-6)
    finally:
        dist.destroy_process_group()


def test_response_head_real_tp_cp_collectives_and_empty_rank(tmp_path):
    torch.multiprocessing.spawn(
        _response_head_distributed_worker, args=(str(tmp_path / "response-head-gloo"),), nprocs=2, join=True
    )
