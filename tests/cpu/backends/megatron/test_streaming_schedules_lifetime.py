# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Run the complete streaming schedule with real autograd and isolated P2P.

The schedule's reference lifetime is the system under test. Transport/model
adapters are local CPU boundaries; this does not claim distributed correctness.
"""

from __future__ import annotations

import gc
import importlib.util
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "relax/backends/megatron/streaming_schedules.py"


def load_schedule(monkeypatch, device="cpu", source=SOURCE):
    spec = importlib.util.spec_from_file_location("warmup_lifetime_subject", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "torch",
        SimpleNamespace(int=torch.int, zeros=lambda *a, **kw: torch.zeros(*a, **{**kw, "device": device})),
    )
    monkeypatch.setattr(module, "get_model_config", lambda model: model.config)
    monkeypatch.setattr(module, "clear_embedding_activation_buffer", lambda *a: None)
    monkeypatch.setattr(module, "finish_embedding_wgrad_compute", lambda *a: None)
    monkeypatch.setattr(module, "get_tensor_shapes", lambda **kw: [])
    monkeypatch.setattr(module, "deallocate_output_tensor", lambda *a: None)
    monkeypatch.setattr(module, "check_first_val_step", lambda *a: False)
    return module


def exercise_schedule(monkeypatch, count, forward_only, stage, *, device="cpu", source=SOURCE, expect_retained=False):
    module = load_schedule(monkeypatch, device=device, source=source)
    payload_refs, input_refs, position_refs = [], [], []
    events, backward_ids, released_at_boundary = [], [], []
    memory_boundaries = []
    is_cuda = torch.device(device).type == "cuda"
    payload_shape = (8192, 1024) if is_cuda else (1024, 128)
    model = SimpleNamespace(weight=torch.nn.Parameter(torch.tensor(0.5, device=device)))
    max_warmup = 7 - stage
    warmup_count = min(count, max_warmup)

    def assert_consumed_payloads_released():
        gc.collect()
        alive = [index for index in range(warmup_count) if payload_refs[index]() is not None]
        released_at_boundary.append(alive)
        expected_alive = list(range(warmup_count)) if expect_retained else []
        if is_cuda:
            torch.cuda.synchronize()
        memory_boundaries.append(
            {
                "alive_payload_indices": alive,
                "allocated_bytes": torch.cuda.memory_allocated() if is_cuda else None,
                "alive_payload_bytes": sum(
                    payload_refs[index]().numel() * payload_refs[index]().element_size() for index in alive
                ),
            }
        )
        assert alive == expected_alive, f"consumed warmup payloads still held by schedule: {alive}"

    class Producer:
        index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.index == count:
                raise StopIteration
            index = self.index
            self.index += 1
            data = {
                "index": index,
                "unused_sc_payload": torch.ones(payload_shape, device=device),
                "x": torch.full((4,), float(index + 1), requires_grad=True, device=device),
                "packed": {"positions": torch.arange(4, dtype=torch.float32, device=device)},
            }
            payload_refs.append(weakref.ref(data["unused_sc_payload"]))
            input_refs.append(weakref.ref(data["x"]))
            position_refs.append(weakref.ref(data["packed"]["positions"]))
            return data, SimpleNamespace(extra_info={"index": index})

    def local_forward(forward_fn, iterator, model, num_microbatches, inputs, store, config, **kw):
        data, _ = next(iterator)
        index = data["index"]
        if index == max_warmup:
            # This is after all warmup forwards but before any backward.
            assert_consumed_payloads_released()
            if not forward_only:
                assert all(ref() is not None for ref in input_refs[:warmup_count])
                assert all(ref() is not None for ref in position_refs[:warmup_count])
        events.append(("forward", index))
        positions = data["packed"]["positions"]

        def recompute(x):
            # Actual checkpoint closure retains independent packed metadata.
            return (x * 2 + positions).square().sum() * model.weight

        if forward_only:
            with torch.no_grad():
                output = recompute(data["x"])
        else:
            output = checkpoint(recompute, data["x"], use_reentrant=False)
        store.append({"index": index, "value": output.detach()})
        return output, torch.tensor(4)

    def local_backward(inputs, output, grad, config):
        index = len(backward_ids)
        x = input_refs[index]()
        positions = position_refs[index]()
        assert x is not None and positions is not None, "required backward data released too early"
        output.backward()
        expected = 2 * (torch.full((4,), float(index + 1)) * 2 + torch.arange(4)) * 2 * 0.5
        torch.testing.assert_close(x.grad.cpu(), expected, rtol=0, atol=0)
        backward_ids.append(index)
        events.append(("backward", index))
        return None

    def finalize(*args, **kwargs):
        # Short streams never enter a steady forward; check after cooldown.
        if count <= max_warmup:
            assert_consumed_payloads_released()

    model.config = SimpleNamespace(
        no_sync_func=None,
        finalize_model_grads_func=finalize,
        calculate_per_token_loss=False,
        overlap_p2p_comm=False,
        grad_sync_func=None,
        deallocate_pipeline_outputs=False,
    )
    communicator = SimpleNamespace(
        total_stages=8,
        current_stage=stage,
        is_pp_first_stage=stage == 0,
        is_pp_last_stage=stage == 7,
        recv_forward=lambda *args: None,
        send_forward=lambda *args: None,
        send_forward_recv_backward=lambda *args: None,
        send_backward_recv_forward=lambda *args: None,
        send_backward=lambda *args: None,
        recv_backward=lambda *args: None,
    )
    groups = SimpleNamespace(cp=SimpleNamespace(size=lambda: 2), tp=object())
    monkeypatch.setattr(module, "forward_step", local_forward)
    monkeypatch.setattr(module, "backward_step", local_backward)
    result = module.streaming_forward_backward_pipelining_without_interleaving(
        forward_step_func=None,
        data_iterator=Producer(),
        model=model,
        num_microbatches=999,
        seq_length=128,
        micro_batch_size=1,
        forward_only=forward_only,
        p2p_communicator=communicator,
        pg_collection=groups,
    )
    assert [row["index"] for row in result] == list(range(count))
    assert [i for op, i in events if op == "forward"] == list(range(count))
    assert backward_ids == ([] if forward_only else list(range(count)))
    if not forward_only:
        if count:
            expected_weight_grad = sum(
                ((torch.full((4,), float(i + 1)) * 2 + torch.arange(4)).square().sum()).item() for i in range(count)
            )
            assert model.weight.grad.item() == expected_weight_grad
        else:
            assert model.weight.grad is None
    assert released_at_boundary
    gc.collect()
    assert all(ref() is None for ref in payload_refs)
    return {
        "count": count,
        "forward_only": forward_only,
        "stage": stage,
        "forward_ids": [i for op, i in events if op == "forward"],
        "backward_ids": backward_ids,
        "weight_grad": None if model.weight.grad is None else model.weight.grad.item(),
        "memory_boundaries": memory_boundaries,
        "all_payload_weakrefs_dead_after_schedule": True,
    }


@pytest.mark.parametrize(
    ("count", "forward_only", "stage"),
    [(10, False, 0), (3, False, 0), (10, True, 0), (0, False, 0), (3, False, 7)],
    ids=["steady-autograd", "short-cooldown", "forward-only", "empty", "zero-warmup"],
)
def test_consumed_warmup_payload_lifetime_and_gradients(monkeypatch, count, forward_only, stage):
    exercise_schedule(monkeypatch, count, forward_only, stage)
