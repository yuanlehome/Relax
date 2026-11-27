# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""SC recipe filtering through the real resident pipeline, without Ray or
GPUs."""

from __future__ import annotations

import asyncio
import pickle
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from examples.claude_code_harbor.advantage import advantage_func, score_centering_advantage
from relax.agentic.pipeline import SampleExport, SessionExport
from relax.agentic.pipeline.prepare import PrepareDomain
from relax.agentic.pipeline.reward import RewardDomain
from relax.agentic.pipeline.runtime import RuntimeDomain, _SessionShardBinding
from relax.agentic.pipeline.transfer import TransferDomain
from relax.agentic.profile import TRACE_KEY
from relax.utils.types import Sample
from tests.cpu.components.test_agentic_streaming_handoff import _args as _conversion_args
from tests.cpu.components.test_agentic_streaming_handoff import _samples
from tests.cpu.test_agentic_rollout import AgenticResidentPipeline, _runtime_args


def _metadata(rewards):
    return [
        {f"context-{export}": {"outcome_reward": reward} for export in range(2 if member == 0 else 1)}
        for member, reward in enumerate(rewards)
    ]


@pytest.mark.parametrize("reward", [0.0, 1.0])
def test_sc_rejects_uniform_reward_groups(reward):
    metadata = _metadata([reward] * 8)
    assert score_centering_advantage(metadata) is None
    assert advantage_func(metadata) is None


def test_sc_centers_logical_sessions_without_std_normalization():
    result = score_centering_advantage(_metadata([1.0] + [0.0] * 7))
    assert result == [
        {"context-0": 0.875, "context-1": 0.875},
        *[{"context-0": -0.125} for _ in range(7)],
    ]
    # Retain the existing GRPO interface and its distinct normalization.
    grpo = advantage_func(_metadata([1.0] + [0.0] * 7))
    assert grpo[0]["context-0"] > 2.6


@pytest.mark.asyncio
async def test_sc_rejection_replenishes_512_logical_sessions(monkeypatch):
    values = vars(_runtime_args())
    values.update(vars(_conversion_args()))
    values.update(
        agentic_custom_advantage_path="examples.claude_code_harbor.advantage.score_centering_advantage",
        rollout_batch_size=64,
        over_sampling_batch_size=64,
        rollout_global_dataset=False,
        agentic_runtime_groups=80,
        agentic_prepare_groups=16,
        agentic_concurrency=96,
        agentic_prelaunch=False,
        use_dynamic_global_batch_size=False,
        num_rollout=1,
        max_staleness=2,
        sglang_profile=False,
        debug_rollout_only=False,
        save_debug_rollout_data=None,
        reward_key=None,
    )
    args = SimpleNamespace(**values)
    # The external data source contains the observed 6 all-one / 19 all-zero
    # groups, followed by 64 mixed groups. Only production Prepare pulls from it.
    pending = deque(
        [
            Sample(
                index=group * 8 + member,
                group_index=group,
                metadata={"outcome_reward": float(group < 6 or (group >= 25 and member == 0))},
            )
            for member in range(8)
        ]
        for group in range(89)
    )
    fetched = []
    fetch_sizes = []

    async def get_samples(count):
        assert pending, "pipeline requested more groups after filling the effective batch"
        result = [pending.popleft() for _ in range(min(count, len(pending)))]
        fetched.extend(group[0].group_index for group in result)
        fetch_sizes.append(len(result))
        return result

    data_source = SimpleNamespace(get_samples=SimpleNamespace(remote=get_samples))
    specs_by_group = {}
    released = []

    async def start_group(mode, group_id, specs):
        assert mode == "train"
        specs_by_group[group_id] = specs
        shard.group_streams[group_id].first_request_barrier.set_result(True)

    async def lease_group(group_id, rollout_id):
        assert rollout_id == 0
        specs = specs_by_group[group_id]
        rows = _samples(specs[0].group_index)
        for spec, result in zip(specs, shard.group_streams[group_id].session_results, strict=True):
            exports = []
            for row in rows:
                if row.index != spec.index:
                    continue
                row.session_id = spec.session_id
                row.metadata = {**spec.metadata, TRACE_KEY: {"turns": []}}
                row.reward = row.metadata["outcome_reward"]
                row.custom_advantage = None
                exports.append(SampleExport(name=f"context-{len(exports)}", sample=row))
            result.completion.set_result(SessionExport(exports=tuple(exports)))

    async def release_group(group_id):
        released.append(group_id)
        del specs_by_group[group_id]

    # Replace only the external Ray Shard boundary. Prepare/Runtime/Reward/
    # Transfer, rejection, refill scheduling, and tensor conversion stay real.
    handle = SimpleNamespace(
        start_group=SimpleNamespace(remote=start_group),
        lease_group=SimpleNamespace(remote=lease_group),
        release_group=SimpleNamespace(remote=release_group),
        drop_group=SimpleNamespace(remote=AsyncMock()),
        resume_generation=SimpleNamespace(remote=AsyncMock(return_value=0)),
        pause_generation=SimpleNamespace(remote=AsyncMock(return_value=0)),
        trim_memory=SimpleNamespace(remote=AsyncMock()),
    )
    shard = _SessionShardBinding(actor_name="cpu-test-shard", handle=handle)
    runtime = RuntimeDomain(args, "train", [shard], concurrency=96)
    monkeypatch.setattr(runtime, "_start_shard_observers", lambda: None)
    writes = []

    async def async_put(*, data, partition_id, custom_meta, is_last):
        assert partition_id == "train_0"
        # Exercise the serialized TensorDict payload, with only TQ network I/O
        # replaced. Duplicate rows remain one logical Session identity.
        payload = pickle.loads(pickle.dumps(data))
        writes.append((payload, is_last))

    transfer = TransferDomain(args=args)
    pipeline = AgenticResidentPipeline(
        args,
        PrepareDomain("train", data_source, runtime),
        runtime,
        RewardDomain(args, "train", group_filter=None),
        transfer,
        SimpleNamespace(async_put=async_put),
    )
    try:
        output = await asyncio.wait_for(pipeline.run_step(0), timeout=20)
        assert len(output.samples) == 64
        assert fetched == list(range(89))
        assert len(fetch_sizes) > 1
        assert not pending
        # Concurrent ready groups may be coalesced in one TQ write.
        assert 1 < len(writes) <= 64
        assert [is_last for _, is_last in writes] == [False] * (len(writes) - 1) + [True]
        assert len(set(writes[0][0]["sample_indices"].tolist())) < 512
        indices = [index for data, _ in writes for index in data["sample_indices"].tolist()]
        assert len(indices) == 576
        assert set(indices) == set(range(200, 712))
        for group in output.samples:
            assert len(group) == 9
            assert len({row.index for row in group}) == 8
            for row in group:
                assert row.custom_advantage == (0.875 if row.index % 8 == 0 else -0.125)
        assert transfer.total_debt == 0
    finally:
        await asyncio.wait_for(pipeline.shutdown(), timeout=20)
    assert runtime.resident_group_count == 0
    assert len(set(released)) == 89
    assert not specs_by_group
