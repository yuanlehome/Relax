# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Actual-module CPU tests; only TQ/process-group/metric sink boundaries are
isolated."""

import gc
import importlib.util
import weakref
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from torch.utils._pytree import tree_flatten


ROOT = Path(__file__).resolve().parents[4]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def stream(monkeypatch):
    original = load_file("existing_stream_tests", ROOT / "tests/cpu/utils/data/test_streaming_tq_iterator.py")
    return original._load_stream_module(monkeypatch)


def iterator(stream, *, train=True, window=0):
    return stream.StreamingTQIterator(
        args=Namespace(),
        tq_client=object(),
        data_fields=["tokens"],
        rollout_id=3,
        token_budget=128,
        loss_scale=0.25,
        all_consumed_fn=lambda: False,
        dp_rank=1,
        dp_size=2,
        window_quota=8,
        rollout_mini_index=window,
        start_batch_index=window * 100,
        per_round_dummy=train,
    )


def emit(it, data):
    return it._emit_data_batch(data, Namespace(extra_info={}), tq_wait=0.0)


def terminate(stream, monkeypatch, it):
    monkeypatch.setattr(
        stream, "get_data_from_transfer_queue", lambda **kw: (None, Namespace(extra_info={"stream_end": True}))
    )
    with pytest.raises(StopIteration):
        next(it)


def test_snapshot_detaches_all_tensor_fields_without_changing_training_rows(stream):
    it = iterator(stream)
    floating = torch.arange(6.0, requires_grad=True).reshape(2, 3)
    ids = torch.arange(6, dtype=torch.int64).reshape(2, 3)
    data = {
        "tokens": [torch.arange(5)],
        "total_lengths": [5],
        "response_lengths": [2],
        "loss_masks": [torch.tensor([1, 0], dtype=torch.int32)],
        "rollout_topk_token_ids": [ids],
        "rollout_topk_log_probs": [floating],
        "rollout_topk_support_complete": [torch.tensor([True, False])],
        "multimodal_train_inputs": [{"values": (floating, None), "grid": torch.tensor([1, 2])}],
        "rollout_replay_refs": [{"region": "synthetic", "sequence": 4}],
        "metadata": {"labels": ["safe", None], "scalar": 3.5},
    }
    real, meta = emit(it, data)
    archived, archived_meta = it.get_buffer()[0]
    assert meta is archived_meta
    assert real is not data and real["tokens"][0] is data["tokens"][0]
    assert real["rollout_topk_log_probs"][0] is floating
    assert set(archived) == set(data)
    assert archived["metadata"] == data["metadata"]
    original_leaves, original_spec = tree_flatten(data)
    archive_leaves, archive_spec = tree_flatten(archived)
    assert original_spec == archive_spec
    for original, saved in zip(original_leaves, archive_leaves, strict=True):
        if isinstance(original, torch.Tensor):
            assert saved.device.type == "cpu"
            assert saved.requires_grad is False and saved.grad_fn is None
            assert saved.dtype == original.dtype and saved.shape == original.shape
            assert saved.data_ptr() != original.data_ptr()
            torch.testing.assert_close(saved, original.detach(), rtol=0, atol=0)
    with torch.no_grad():
        floating.add_(100)
        ids.add_(20)
    assert archived["rollout_topk_log_probs"][0][0, 0].item() == 0
    assert archived["rollout_topk_token_ids"][0][0, 0].item() == 0


def test_history_drops_original_tensor_ownership_and_eos_drops_last_template(stream, monkeypatch):
    it = iterator(stream)
    refs = []
    for value in (1.0, 2.0, 3.0):
        row = torch.tensor([value])
        refs.append(weakref.ref(row))
        returned = emit(it, {"tokens": [row]})
        del row, returned
        gc.collect()
    assert [ref() is not None for ref in refs] == [False, False, True]
    assert len(it.get_buffer()) == 3
    terminate(stream, monkeypatch, it)
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert [data["tokens"][0].item() for data, _ in it.get_buffer()] == [1, 2, 3]


def test_dummy_uses_latest_training_template_and_preserves_window_accounting(stream, monkeypatch):
    first = {"tokens": [torch.tensor([1, 2])], "response_lengths": [1], "log_probs": [torch.tensor([-1.0])]}
    last = {"tokens": [torch.tensor([3, 4, 5])], "response_lengths": [2], "log_probs": [torch.tensor([-2.0, -3.0])]}
    events = iter(
        [
            (first, Namespace(extra_info={})),
            (None, Namespace(extra_info={})),
            (last, Namespace(extra_info={})),
            (None, Namespace(extra_info={"dummy_round": True})),
            (None, Namespace(extra_info={"dummy_round": True})),
            (None, Namespace(extra_info={"stream_end": True})),
        ]
    )
    monkeypatch.setattr(stream, "get_data_from_transfer_queue", lambda **kw: next(events))
    monkeypatch.setattr(stream.time, "sleep", lambda _: None)
    it = iterator(stream, window=2)
    real1, _ = next(it)
    real2, _ = next(it)
    real2["log_probs"] = [torch.tensor([100.0])]
    dummies = list(it)
    assert [batch["micro_batch_index"] for batch, _ in dummies] == [2, 3]
    assert all(batch["__is_dummy__"] and batch["__loss_scale__"] == 0.25 for batch, _ in dummies)
    assert all(batch["tokens"][0] is last["tokens"][0] for batch, _ in dummies)
    assert all(batch["log_probs"][0] is last["log_probs"][0] for batch, _ in dummies)
    assert it._sample_count == 2 and it._batch_index == 204 and it._mb_count == 4
    assert len(it.get_buffer()) == 2
    empty = iterator(stream, window=3)
    with pytest.raises(RuntimeError, match="zero real samples"):
        empty._emit_dummy_batch()
    assert empty.get_buffer() == []


def test_forward_only_buffer_alias_and_dummy_behavior_remain_unchanged(stream):
    it = iterator(stream, train=False)
    data = {"tokens": [torch.tensor([7, 8])], "log_probs": [torch.tensor([-1.0])]}
    returned, meta = emit(it, data)
    assert it.get_buffer()[0][0] is data
    returned["log_probs"][0].add_(1)
    assert it.get_buffer()[0][0]["log_probs"][0].item() == 0
    dummy, dummy_meta = it._make_dummy_batch()
    assert dummy_meta is meta and dummy["tokens"][0] is data["tokens"][0]
    assert dummy["__is_dummy__"]


@pytest.mark.parametrize("cp_size,local_values,expected", [(1, list(range(1, 9)), 3.6), (2, [7, 8], 3.0)])
def test_cpu_archive_merge_logger_cp_denominators_and_debug_roundtrip(
    stream, monkeypatch, tmp_path, cp_size, local_values, expected
):
    data_tests = load_file("existing_data_tests", ROOT / "tests/cpu/backends/megatron/test_data_vpp.py")
    data_mod = data_tests._load_data_module(monkeypatch)
    mpu = data_mod.mpu
    for name, fn in {
        "get_tensor_model_parallel_rank": lambda: 0,
        "is_pipeline_last_stage": lambda: True,
        "get_context_parallel_world_size": lambda: cp_size,
        "get_context_parallel_rank": lambda: 0,
        "get_data_parallel_world_size": lambda **kw: 1,
        "get_data_parallel_group": lambda **kw: object(),
    }.items():
        monkeypatch.setattr(mpu, name, fn, raising=False)
    monkeypatch.setattr(data_mod.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(data_mod.dist, "all_reduce", lambda *a, **kw: None)
    captured = {}

    def sink(kind, args, step, metrics, *rest):
        captured[kind] = metrics

    monkeypatch.setattr(data_mod, "gather_log_data", sink)

    it = iterator(stream)
    payload = {
        "tokens": [torch.arange(12)],
        "total_lengths": [12],
        "response_lengths": [8],
        "loss_masks": [torch.ones(8, dtype=torch.int32)],
        "sample_index_mask_sums": [10],
        "sample_indices": [0],
        "advantages": [torch.tensor(local_values, dtype=torch.float32)],
        "rollout_topk_token_ids": [torch.arange(len(local_values) * 2).reshape(-1, 2)],
        "rollout_topk_log_probs": [torch.zeros(len(local_values), 2)],
        "rollout_topk_support_complete": [torch.ones(len(local_values), dtype=torch.bool)],
        "rewards": [1.0],
    }
    emit(it, payload)
    from relax.utils.utils import merge_dict_list

    merged = merge_dict_list(it.get_buffer())
    args = Namespace(
        qkv_format="thd",
        dynamic_context_parallel=False,
        is_vl_model=False,
        uses_unsplit_forward=False,
        use_opd=False,
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        ci_test=False,
        log_multi_turn=True,
        rollout_max_response_len=128,
        save_debug_train_data=str(tmp_path / "{rollout_id}_{rank}.pt"),
    )
    data_mod.log_rollout_data(3, args, merged)
    assert captured["rollout"]["advantages"] == pytest.approx(expected)
    assert captured["multi_turn"]["wo_obs_response_length/response_length_mean"] == 8
    assert captured["rollout"]["total_lengths/max"] == 12
    from relax.utils.training import train_dump_utils

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    train_dump_utils.save_debug_train_data(args, rollout_id=3, rollout_data=merged)
    dump = torch.load(tmp_path / "3_0.pt", weights_only=False)
    assert len(dump["samples"]) == 1
    assert dump["samples"][0]["tokens"] == list(range(12))
    assert dump["samples"][0]["advantages"] == local_values
