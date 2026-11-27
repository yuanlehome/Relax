# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Trainable PLE numerical and packed-checkpoint ownership regressions.

Real shipped PLE and MCore checkpoint code run on CPU. Distributed transports
are supplied already-gathered tensors; these tests do not prove NCCL behavior.
The fixed CPU tolerances below cover each trainable parameter and input
gradient. GPU HC value-gradient sums may use a different FP32 addition order
from native broadcast reduction; these fixtures do not claim cross-backend
bitwise identity.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


TRAINABLE = {
    "key_proj.weight",
    "value_proj.weight",
    "norm_key.weight",
    "norm_query.weight",
    "norm_conv.weight",
    "conv1d.weight",
}


@pytest.fixture
def setup(monkeypatch):
    native = pytest.importorskip("megatron.core.transformer.per_layer_embedding")
    from megatron.core.ssm import mamba_context_parallel as cp
    from megatron.core.tensor_parallel import mappings
    from megatron.core.tensor_parallel import random as tp_random

    from relax.models.qwen3_8.ops import ple_stream as streamed

    state = SimpleNamespace(cp_size=1, cp_rank=0, tp_rank=0, tp_size=1)
    monkeypatch.setattr(native.parallel_state, "get_context_parallel_world_size", lambda: state.cp_size)
    monkeypatch.setattr(streamed.parallel_state, "get_context_parallel_world_size", lambda: state.cp_size)
    # CPU fixture supplies the already gathered tensors; this is not a transport test.
    monkeypatch.setattr(cp, "reconstruct_tensor_cp", lambda value, *_a, **_k: value)
    monkeypatch.setattr(cp, "split_tensor_cp", lambda value, *_a, **_k: value)
    monkeypatch.setattr(mappings, "gather_from_sequence_parallel_region", lambda value, **_k: value)
    monkeypatch.setenv("RELAX_QWEN38_PLE_STREAMED", "0")
    monkeypatch.setenv("RELAX_QWEN38_PLE_FUSED_CONV", "0")
    monkeypatch.setenv("RELAX_QWEN38_PLE_CP_FUSED_CONV", "0")
    monkeypatch.setenv("QWEN38_PLE_INPUT_HALO_CP", "0")
    monkeypatch.setenv("RELAX_QWEN38_QSA_RECOMPUTE_REPLAY", "1")
    # The actual MCore CheckpointFunction runs. Only unavailable CUDA RNG storage
    # is adapted to CPU, and the PLE path itself has no random operation.
    monkeypatch.setattr(tp_random, "_get_all_rng_states", lambda: (torch.get_rng_state(),))
    monkeypatch.setattr(tp_random, "_set_all_rng_states", lambda cpu: torch.set_rng_state(cpu))
    monkeypatch.setattr(tp_random, "_fork_rng", lambda: torch.random.fork_rng(devices=[]))
    return native, streamed, state, cp, tp_random


def build(native, state, dtype, *, sequence_length=32, channels=8, embedding_dim=12):
    # Instantiate the actual shipped class without its distributed GPU constructor.
    model = native.PerLayerEmbedding.__new__(native.PerLayerEmbedding)
    nn.Module.__init__(model)
    model.n = 4
    model.hidden_size = channels
    model.norm_eps = 1e-6
    model.config = SimpleNamespace(sequence_parallel=False)
    model.tp_group = SimpleNamespace(size=lambda: state.tp_size, rank=lambda: state.tp_rank)
    model.conv_kernel_size, model.conv_dilation, model.halo = 4, 3, 9
    model._ngram_ids = torch.arange(sequence_length)
    model._position_in_sequence = torch.cat((torch.arange(8), torch.arange(sequence_length - 8))).unsqueeze(0)
    model._cp_packed_seq_params = None
    model.ple_embedding = nn.Embedding(sequence_length, embedding_dim, dtype=dtype)
    model.ple_embedding.requires_grad_(False)
    model.ple_embedding.register_forward_hook(lambda _m, _i, output: output.unsqueeze(1))
    model.key_proj = nn.Linear(embedding_dim, 4 * channels, bias=False, dtype=dtype)
    model.value_proj = nn.Linear(embedding_dim, channels, bias=False, dtype=dtype)
    for name in ("norm_key", "norm_query", "norm_conv"):
        module = nn.Module()
        module.weight = nn.Parameter((torch.randn(4 * channels) * 0.15).to(dtype))
        setattr(model, name, module)
    model.conv1d = nn.Conv1d(4 * channels, 4 * channels, 4, groups=4 * channels, bias=False, dtype=dtype)
    return model


def grads(model):
    result = {}
    assert {name for name, p in model.named_parameters() if p.requires_grad} == TRAINABLE
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert torch.count_nonzero(parameter.grad), name
            result[name] = parameter.grad.detach().clone()
        else:
            assert parameter.grad is None, name
    return result


def compare(actual, expected, dtype, label):
    # Preserve the existing BF16 numerical budget. Input and all parameter
    # gradients are checked separately, including cancellation near zero.
    rtol, atol = (2e-5, 2e-6) if dtype == torch.float32 else (2e-2, 6e-3)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=label)


def outer_call(mode, function, hidden, tp_random):
    if mode == "direct":
        return function(hidden)
    if mode == "torch_reentrant":
        return checkpoint(function, hidden, use_reentrant=True, preserve_rng_state=False)
    if mode == "torch_nonreentrant":
        return checkpoint(function, hidden, use_reentrant=False, preserve_rng_state=False)
    assert mode == "mcore"
    return tp_random.checkpoint(function, False, hidden)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("outer", ["direct", "torch_reentrant", "torch_nonreentrant", "mcore"])
def test_trainable_stream_preserves_outputs_and_all_gradients(setup, monkeypatch, dtype, outer):
    native, streamed, state, _cp, tp_random = setup
    torch.manual_seed(314)
    model = build(native, state, dtype)
    hidden = torch.randn(32, 1, 32, dtype=dtype, requires_grad=True)
    upstream = torch.randn_like(hidden)
    expected = model(hidden)  # Actual full native reference, CP=1, unstreamed.
    expected.backward(upstream)
    expected_hidden_grad = hidden.grad.clone()
    expected_grads = grads(model)
    model.zero_grad(set_to_none=True)
    hidden.grad = None

    state.cp_size = 2
    model._cp_packed_seq_params = object()
    monkeypatch.setattr(streamed, "_STREAM_CHUNK_ROWS", 5)  # forces the old overlapping-halo failure mode
    monkeypatch.setenv("RELAX_QWEN38_PLE_STREAMED", "1")
    actual = outer_call(outer, model, hidden, tp_random)
    actual.backward(upstream)
    compare(actual, expected.detach(), dtype, "output")
    compare(hidden.grad, expected_hidden_grad, dtype, "hidden gradient")
    for name, actual_grad in grads(model).items():
        compare(actual_grad, expected_grads[name], dtype, name)


@pytest.mark.parametrize("tp_size", [1, 4])
def test_partial_frozen_cp_sp_gradient_ownership(setup, monkeypatch, tp_size):
    native, streamed, state, cp, _random = setup
    torch.manual_seed(90)
    model = build(native, state, torch.float32)
    hidden = torch.randn(32, 1, 32, requires_grad=True)
    upstream = torch.randn_like(hidden)  # nonuniform: detects duplicated reduction/scatter
    expected = model(hidden)
    expected.backward(upstream)
    expected_hidden = hidden.grad.clone()
    expected_grads = grads(model)
    hidden.grad = None
    model.zero_grad(set_to_none=True)

    cp_indices = []
    for rank in range(2):
        ids, start = [], 0
        for length in (8, 24):
            chunks = torch.arange(start, start + length).chunk(4)
            ids.extend((chunks[rank], chunks[3 - rank]))
            start += length
        cp_indices.append(torch.cat(ids))
    state.cp_size, state.tp_size = 2, tp_size
    model.config.sequence_parallel = True
    model._cp_packed_seq_params = object()
    monkeypatch.setattr(cp, "split_tensor_cp", lambda value, *_a, **_k: value[cp_indices[state.cp_rank]])
    monkeypatch.setattr(streamed, "_STREAM_CHUNK_ROWS", 5)
    monkeypatch.setenv("RELAX_QWEN38_PLE_STREAMED", "1")
    losses = []
    for cp_rank in range(2):
        state.cp_rank = cp_rank
        for tp_rank in range(tp_size):
            state.tp_rank = tp_rank
            local = model(hidden)
            indices = cp_indices[cp_rank].chunk(tp_size)[tp_rank]
            compare(local, expected.detach()[indices], torch.float32, "local output")
            losses.append((local * upstream[indices]).sum())
    sum(losses).backward()
    compare(hidden.grad, expected_hidden, torch.float32, "input ownership")
    for name, gradient in grads(model).items():
        compare(gradient, expected_grads[name], torch.float32, name)


def test_ngram_table_must_remain_frozen(setup, monkeypatch):
    native, _streamed, state, _cp, _random = setup
    model = build(native, state, torch.float32)
    model.ple_embedding.requires_grad_(True)
    state.cp_size = 2
    model._cp_packed_seq_params = object()
    monkeypatch.setenv("RELAX_QWEN38_PLE_STREAMED", "1")
    with pytest.raises(RuntimeError, match="frozen"):
        model(torch.randn(32, 1, 32, requires_grad=True))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_actual_bridge_checkpoint_restores_two_microbatch_states(setup, monkeypatch, dtype):
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_block import Qwen3VLTransformerBlock

    native, streamed, state, _cp, _random = setup
    torch.manual_seed(919)
    # Cross the convolution channel-block boundary while retaining two packed batches.
    model = build(native, state, dtype, channels=257)
    values = [torch.randn(32, 1, 4 * model.hidden_size, dtype=dtype) for _ in range(2)]
    upstream = [torch.randn_like(value) for value in values]
    ids = [torch.arange(32), torch.arange(31, -1, -1)]
    positions = [
        torch.cat((torch.arange(8), torch.arange(24))).unsqueeze(0),
        torch.cat((torch.arange(12), torch.arange(20))).unsqueeze(0),
    ]

    expected_inputs = [value.clone().requires_grad_() for value in values]
    expected_outputs = []
    for index in range(2):
        model.restore_checkpoint_state((ids[index], positions[index], None))
        expected_outputs.append(model(expected_inputs[index]))
    for index in (1, 0):
        expected_outputs[index].backward(upstream[index])
    expected_grads = grads(model)
    expected_input_grads = [value.grad.clone() for value in expected_inputs]
    model.zero_grad(set_to_none=True)

    class PLEOnlyLayer(nn.Module):
        # Expensive attention/MLP are excluded, but the production bridge's
        # state capture/restore and actual PLE forward/backward run unchanged.
        def __init__(self, ple):
            super().__init__()
            self.per_layer_embedding = ple
            self.layer_number = 1

        def forward(self, hidden_states, context=None, **kwargs):
            return self.per_layer_embedding(hidden_states), context

    block = Qwen3VLTransformerBlock.__new__(Qwen3VLTransformerBlock)
    nn.Module.__init__(block)
    block.config = SimpleNamespace(
        fp8=False, distribute_saved_activations=False, recompute_method="uniform", recompute_num_layers=1
    )
    block.num_layers_per_pipeline_rank = 1
    block.layers = nn.ModuleList([PLEOnlyLayer(model)])
    block.pre_process = False
    block.tp_group = model.tp_group
    state.cp_size = 2
    monkeypatch.setenv("RELAX_QWEN38_PLE_STREAMED", "1")
    monkeypatch.setattr(streamed, "_STREAM_CHUNK_ROWS", 5)
    actual_inputs = [value.clone().requires_grad_() for value in values]
    actual_outputs = []
    for index in range(2):
        packed = object()
        model.restore_checkpoint_state((ids[index], positions[index], packed))
        actual_outputs.append(
            block._checkpointed_forward(actual_inputs[index], None, None, None, None, None, packed, None, False)
        )
    model.restore_checkpoint_state((None, None, None))  # later batch/clear state
    for index in (1, 0):
        actual_outputs[index].backward(upstream[index])
        assert model.capture_checkpoint_state() == (None, None, None)
        compare(actual_outputs[index], expected_outputs[index].detach(), dtype, "checkpoint output")
        compare(actual_inputs[index].grad, expected_input_grads[index], dtype, "checkpoint input gradient")
    for name, gradient in grads(model).items():
        compare(gradient, expected_grads[name], dtype, name)
