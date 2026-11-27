# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in memory-reduced PLE forward for packed Qwen4Exp context parallelism.

Only the ngram embedding table must remain frozen. Trainable projection, norm,
or convolution weights use one full-sequence checkpoint per HC stream to avoid
rounding gradients separately across overlapping convolution halos. Fully
frozen PLE instead uses 8192-row sequence chunks with a left halo within each
stream. Trainable depthwise convolution additionally checkpoints independent
channel blocks. Both paths preserve packed-document boundaries. Trainable-
stream memory still scales with sequence length, and the original
implementation remains the default.
"""

import math

import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.transformer.hyper_connection import gated_residual_group_rmsnorm
from megatron.core.transformer.per_layer_embedding import PerLayerEmbedding
from torch.utils.checkpoint import checkpoint


_STREAM_CHUNK_ROWS = 8192
_CONV_CHUNK_CHANNELS = 256


def _stream_conv_channels(
    x: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    halo: int,
    dilation: int,
) -> torch.Tensor:
    """Depthwise convolution for independent channels, with the original tap
    order."""
    s_local = x.shape[0]
    x_ext = F.pad(x, (0, 0, 0, 0, halo, 0))
    result = torch.zeros_like(x, dtype=torch.float32)
    for tap in range(weight.shape[-1]):
        offset = (weight.shape[-1] - 1 - tap) * dilation
        source = x_ext[halo - offset : halo - offset + s_local]
        valid = (positions >= offset).unsqueeze(-1)
        result.add_(torch.where(valid, source.float(), 0.0) * weight[:, tap].float())
    return result.to(x.dtype)


def _stream_conv(ple: PerLayerEmbedding, x: torch.Tensor, positions: torch.Tensor, stream: int) -> torch.Tensor:
    """Checkpoint independent channel blocks without splitting the sequence."""
    channels = ple.hidden_size
    start = stream * channels
    weight = ple.conv1d.weight[start : start + channels].squeeze(1).to(x.dtype)
    if not any(parameter.requires_grad for parameter in ple.parameters()) or channels <= _CONV_CHUNK_CHANNELS:
        return _stream_conv_channels(x, weight, positions, ple.halo, ple.conv_dilation)

    # SplitBackward joins disjoint channel gradients without overlapping halos
    # or repeated full-width SliceBackward allocations. Norm/gate stay full-width.
    pieces = []
    for x_part, weight_part in zip(x.split(_CONV_CHUNK_CHANNELS, dim=-1), weight.split(_CONV_CHUNK_CHANNELS, dim=0)):
        pieces.append(
            checkpoint(
                _stream_conv_channels,
                x_part,
                weight_part,
                positions,
                ple.halo,
                ple.conv_dilation,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        )
    return torch.cat(pieces, dim=-1)


def _stream_forward(
    ple: PerLayerEmbedding,
    hidden: torch.Tensor,
    key_raw: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    stream: int,
) -> torch.Tensor:
    channels = ple.hidden_size
    start = stream * channels
    stop = start + channels
    key = gated_residual_group_rmsnorm(key_raw, ple.norm_key.weight[start:stop], 1, ple.norm_eps)
    query = gated_residual_group_rmsnorm(hidden, ple.norm_query.weight[start:stop], 1, ple.norm_eps)
    gate = (key.float() * query.float()).sum(dim=-1, keepdim=True) / math.sqrt(channels)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated_value = (torch.sigmoid(gate) * value.float()).to(hidden.dtype)
    normed = gated_residual_group_rmsnorm(gated_value, ple.norm_conv.weight[start:stop], 1, ple.norm_eps)
    conv_out = _stream_conv(ple, normed, positions, stream)
    return gated_value + F.silu(conv_out.float()).to(hidden.dtype)


def _streamed_forward(self: PerLayerEmbedding, hidden_states: torch.Tensor) -> torch.Tensor:
    if self._ngram_ids is None:
        raise RuntimeError("PerLayerEmbedding.prepare() must be called before forward().")
    if any(parameter.requires_grad for parameter in self.ple_embedding.parameters()):
        raise RuntimeError("RELAX_QWEN38_PLE_STREAMED requires the ngram embedding table to be frozen")

    from megatron.core.ssm.mamba_context_parallel import reconstruct_tensor_cp, split_tensor_cp
    from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

    cp_size = parallel_state.get_context_parallel_world_size()
    if cp_size <= 1 or self._cp_packed_seq_params is None:
        raise RuntimeError("RELAX_QWEN38_PLE_STREAMED currently requires packed THD with CP>1")

    sp_replicated = self.config.sequence_parallel and self.tp_group.size() > 1
    if sp_replicated:
        hidden_states = gather_from_sequence_parallel_region(hidden_states, group=self.tp_group)
    hidden_states = reconstruct_tensor_cp(hidden_states, self._cp_packed_seq_params, dim=0, differentiable=True)
    sequence_length, batch_size, hc_hidden_size = hidden_states.shape
    channels = self.hidden_size
    if hc_hidden_size != self.n * channels:
        raise RuntimeError("PLE hidden stream width mismatch")

    embeddings = self.ple_embedding(self._ngram_ids).to(hidden_states.dtype)
    if sp_replicated:
        embeddings = gather_from_sequence_parallel_region(embeddings, group=self.tp_group)
    if embeddings.shape[0] != sequence_length:
        raise RuntimeError("PLE embeddings and hidden sequence lengths differ")
    key_raw = self.key_proj(embeddings)
    value = self.value_proj(embeddings)
    trainable_ple = any(parameter.requires_grad for parameter in self.parameters())
    if trainable_ple:
        # Sum contributions from all HC streams in FP32 before one BF16 cast.
        value = value.float()
    # Recomputing overlapping halos rounds internal BF16 gradients separately.
    # Whole HC streams avoid repeated halo rounding. The shared value gradient
    # accumulates in FP32; its addition order can differ from the native HC sum.
    chunk_rows = max(sequence_length, 1) if trainable_ple else _STREAM_CHUNK_ROWS
    positions = self._local_positions(sequence_length)

    pieces = []
    for stream in range(self.n):
        start = stream * channels
        stop = start + channels
        hidden_stream = hidden_states[..., start:stop]
        key_stream = key_raw[..., start:stop]
        chunks = []
        for begin in range(0, max(sequence_length, 1), chunk_rows):
            end = min(begin + chunk_rows, sequence_length)
            left = max(0, begin - self.halo)
            trim = begin - left
            # Keep document-relative positions unchanged: the original mask
            # still prevents halo reads across packed document boundaries.
            # Bind trim/stream because checkpoint replay occurs after this loop.
            piece = checkpoint(
                lambda h, k, v, p, stream=stream, trim=trim: _stream_forward(self, h, k, v, p, stream)[trim:],
                hidden_stream[left:end],
                key_stream[left:end],
                value[left:end],
                positions[left:end],
                use_reentrant=False,
                preserve_rng_state=False,
            )
            chunks.append(piece)
        pieces.append(torch.cat(chunks, dim=0))
        del chunks, piece
    result = torch.cat(pieces, dim=-1)
    result = split_tensor_cp(result, self._cp_packed_seq_params, dim=0)
    if sp_replicated:
        # Keep each TP replica's loss local; the input gather already sums gradients.
        result = result.chunk(self.tp_group.size(), dim=0)[self.tp_group.rank()].contiguous()
    return result
