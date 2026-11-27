# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real Pipeline/Prepare/Transfer scheduling with controlled external group
work."""

import asyncio
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from relax.agentic.pipeline import GroupExport, SampleExport, SessionExport
from relax.agentic.pipeline.prepare import PrepareDomain
from relax.agentic.pipeline.runtime import RuntimeDomain
from relax.agentic.pipeline.transfer import TransferDomain
from relax.utils.types import Sample
from tests.cpu import test_agentic_rollout as existing


pytestmark = pytest.mark.asyncio
pipeline_module = __import__("relax.agentic.rollout", fromlist=["AgenticResidentPipeline"])


async def until(predicate):
    async with asyncio.timeout(8):
        while not predicate():
            await asyncio.sleep(0.001)


class ControlledRuntime:
    teacher_routing_enabled = False
    rollout_mode = "train"

    def __init__(self, args):
        self.args = args
        self.concurrency = args.agentic_concurrency
        self.streams = {}
        self.leased = []
        self.pauses = 0
        self.resumes = []
        self.peak_runtime = 0
        self.peak_resident = 0
        self.permits = asyncio.Semaphore(self.concurrency)
        self.interrupted_group_ids = ()
        self.changed = lambda: None

    @property
    def resident_group_count(self):
        return len(self.streams)

    @property
    def runtime_group_count(self):
        return sum(stream.admitted and not stream.result.done() for stream in self.streams.values())

    def set_progress_callback(self, callback):
        self.changed = callback

    def observe_group_input(self, _group):
        pass

    async def prepare_group(self, group, decision):
        await self.permits.acquire()
        if decision.done() and not decision.result():
            self.permits.release()
            return None
        decision.set_result(True)
        stream = SimpleNamespace(
            group_id=group.group_id,
            samples=group.samples,
            result=asyncio.get_running_loop().create_future(),
            admitted=False,
        )
        self.streams[stream.group_id] = stream
        self.peak_resident = max(self.peak_resident, self.resident_group_count)
        assert self.resident_group_count <= 96
        return stream

    async def lease_group(self, stream, rollout_id):
        assert not stream.admitted
        stream.admitted = True
        self.leased.append((stream.group_id, rollout_id))
        self.peak_runtime = max(self.peak_runtime, self.runtime_group_count)
        assert self.runtime_group_count <= 80
        self.changed()

    async def collect_group(self, stream):
        try:
            return await stream.result
        finally:
            self.retire(stream)

    def retire(self, stream):
        if self.streams.pop(stream.group_id, None) is not None:
            self.permits.release()
        self.changed()

    async def drop_group(self, stream):
        if not stream.result.done():
            stream.result.set_result(None)
        self.retire(stream)

    def running(self):
        return [stream for stream in self.streams.values() if stream.admitted and not stream.result.done()]

    def complete(self, streams, *, failed=False):
        for stream in streams:
            assert stream.admitted and not stream.result.done()
            result = (
                None
                if failed
                else GroupExport(
                    group_id=stream.group_id,
                    sessions=tuple(SessionExport((SampleExport("answer", sample),)) for sample in stream.samples),
                )
            )
            stream.result.set_result(result)
        self.changed()

    async def resume_generation(self, rollout_id):
        self.resumes.append(rollout_id)

    async def pause_generation(self):
        self.pauses += 1

    async def trim_memory(self):
        pass

    async def shutdown(self):
        for stream in tuple(self.streams.values()):
            await self.drop_group(stream)


class Harness:
    def __init__(self, monkeypatch, *, num_rollout=3, streaming=True):
        self.args = existing._runtime_args(
            fully_async=True,
            partial_rollout=False,
            max_staleness=2,
            colocate=False,
            rollout_batch_size=64,
            over_sampling_batch_size=80,
            n_samples_per_prompt=8,
            global_batch_size=512,
            num_iters_per_train_update=1,
            num_rollout=num_rollout,
            agentic_prelaunch=True,
            agentic_concurrency=96,
            agentic_runtime_groups=80,
            agentic_prepare_groups=16,
            use_dynamic_global_batch_size=False,
            use_dynamic_batch_size=streaming,
        )
        self.manager = SimpleNamespace(training_weight_updating=False)
        self.runtime = ControlledRuntime(self.args)
        self.sequence = 0
        self.writes = []
        self.partitions = set()
        self.allow_write = asyncio.Event()
        self.allow_write.set()
        self.write_started = asyncio.Event()
        self.allow_reward = asyncio.Event()
        self.allow_reward.set()
        self.tasks = []
        self.discarded = []
        source = SimpleNamespace(get_samples=SimpleNamespace(remote=self.fetch))
        self.prepare = PrepareDomain("steady", source, self.runtime)
        reward = SimpleNamespace(scored_session_count=0, set_progress_callback=lambda callback: None)
        self.pipeline = pipeline_module.AgenticResidentPipeline(
            self.args, self.prepare, self.runtime, reward, TransferDomain(args=self.args), self
        )
        monkeypatch.setattr(pipeline_module, "get_local_rollout_manager", lambda: self.manager)
        monkeypatch.setattr(pipeline_module, "_transfer_batch_to_data_system", self.write)
        monkeypatch.setattr(pipeline_module, "_run_group", self.run_group)
        monkeypatch.setattr(
            pipeline_module.replay_data_manager,
            "discard_async",
            AsyncMock(side_effect=lambda samples: self.discarded.extend(samples)),
        )

    async def fetch(self, count):
        groups = []
        for _ in range(count):
            group = self.sequence
            self.sequence += 1
            groups.append(
                [
                    Sample(
                        index=group * 8 + member,
                        group_index=group,
                        session_id=f"{group}/{member}",
                        prompt="prompt",
                        response="answer",
                        reward=1.0,
                        metadata={},
                    )
                    for member in range(8)
                ]
            )
        return groups

    async def run_group(self, runtime, _reward, stream, *_args, **_kwargs):
        group = await runtime.collect_group(stream)
        await self.allow_reward.wait()
        return group

    async def write(self, *, batch_samples, partition_id, is_last, on_put_started, **_kwargs):
        self.write_started.set()
        await self.allow_write.wait()
        on_put_started()
        self.writes.append((partition_id, tuple(tuple(group) for group in batch_samples), is_last))
        self.partitions.add(f"train_{partition_id}")

    async def async_get_partition_list(self):
        return sorted(self.partitions)

    def begin(self, rollout_id, *, final_backfill=False):
        self.prepare.open_step(prelaunch_next_step=rollout_id + 1 < self.args.num_rollout)
        context = pipeline_module._StepContext(rollout_id, final_backfill)
        self.pipeline._active_step = context
        task = asyncio.create_task(self.pipeline._run_rollout_step(context))
        self.tasks.append(task)
        return task, context

    async def close(self):
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.pipeline.shutdown()
        assert not self.runtime.streams
        assert not self.pipeline._active_group_tasks
        assert not self.pipeline._finalized_groups
        assert self.prepare.counts == (0, 0)

    def assert_unique_batches(self, count):
        groups = [group for _partition, batch, _last in self.writes for group in batch]
        assert all(len(group) == 8 for group in groups)
        indices = [sample.index for group in groups for sample in group]
        assert len(indices) == count * 512
        assert len(indices) == len(set(indices))
        assert Counter(partition for partition, _groups, last in self.writes if last) == Counter(range(count))
        for partition in range(count):
            assert sum(len(groups) for step, groups, _last in self.writes if step == partition) == 64
        assert max(Counter(group for group, _step in self.runtime.leased).values()) == 1


@pytest.mark.parametrize("streaming", [False, True])
async def test_steady_runtime_refills_after_39_completions_bursts_and_failures(monkeypatch, streaming):
    h = Harness(monkeypatch, streaming=streaming)
    try:
        task, context = h.begin(0)
        await until(lambda: h.runtime.runtime_group_count == 80 and h.prepare.ready_group_count == 16)
        h.runtime.complete(h.runtime.running()[:39])
        await until(lambda: len(context.output_groups) == 39 and h.runtime.runtime_group_count == 80)
        assert len(h.runtime.leased) == 119
        assert not task.done()
        h.runtime.complete(h.runtime.running()[:7])
        await until(lambda: len(context.output_groups) == 46 and h.runtime.runtime_group_count == 80)
        h.runtime.complete(h.runtime.running()[:3], failed=True)
        await until(lambda: len(h.runtime.leased) == 129 and h.runtime.runtime_group_count == 80)
        assert len(context.output_groups) == 46
        assert h.runtime.peak_runtime == 80 and h.runtime.peak_resident <= 96
        assert sum(h.prepare.counts) <= 16
        snapshot = h.pipeline.snapshot()
        assert snapshot["runtime_group_count"] == snapshot["runtime_group_target"] == 80
        assert snapshot["completed_buffer_group_count"] == 0
        assert snapshot["resident_group_limit"] == 96
        h.runtime.complete(h.runtime.running()[:18])
        output = await asyncio.wait_for(task, 8)
        assert len(output.samples) == 64 and h.runtime.pauses == 0
        h.assert_unique_batches(1)
    finally:
        await h.close()


async def test_steady_surplus_and_unfinished_groups_cross_batches_once(monkeypatch):
    h = Harness(monkeypatch, num_rollout=2)
    try:
        first, _ = h.begin(0)
        await until(lambda: h.runtime.runtime_group_count == 80 and h.prepare.ready_group_count == 16)
        h.runtime.complete(h.runtime.running())
        output = await asyncio.wait_for(first, 8)
        assert len(output.samples) == 64 and h.runtime.pauses == 0
        assert len(h.pipeline._finalized_groups) == 16
        original_tail = {stream.group_id: stream for stream in h.runtime.running()}
        second, context = h.begin(1)
        await until(lambda: len(context.output_groups) == 16 and h.runtime.runtime_group_count == 48)
        assert all(h.runtime.streams[key] is value for key, value in original_tail.items())
        h.runtime.complete(h.runtime.running())
        output = await asyncio.wait_for(second, 8)
        assert len(output.samples) == 64 and h.runtime.pauses == 1
        h.assert_unique_batches(2)
        assert len(h.runtime.leased) == 128
        assert not h.runtime.streams and not h.pipeline._finalized_groups
    finally:
        await h.close()


async def test_steady_transfer_backpressure_then_weight_update_keeps_pause(monkeypatch):
    h = Harness(monkeypatch)
    try:
        h.allow_write.clear()
        task, _ = h.begin(0)
        await until(lambda: h.runtime.runtime_group_count == 80)
        h.runtime.complete(h.runtime.running()[:64])
        await until(lambda: h.pipeline.snapshot()["transferring_group_count"] == 64)
        admitted_at_backpressure = len(h.runtime.leased)
        # Runtime completion can precede Reward/export collection, so some
        # bounded replacement work may already be admitted before debt is zero.
        assert 80 <= admitted_at_backpressure <= 144
        h.runtime.complete(h.runtime.running())
        await until(lambda: all(task.done() for task in h.pipeline._active_group_tasks))
        assert len(h.runtime.leased) == admitted_at_backpressure and not task.done()
        assert h.pipeline.snapshot()["completed_buffer_group_count"] <= 80
        h.manager.training_weight_updating = True
        h.allow_write.set()
        await asyncio.wait_for(task, 8)
        assert h.runtime.pauses == 1
        assert h.pipeline.snapshot()["refill_blocked_reason"] == "weight_update"
        second, context = h.begin(1)
        await until(lambda: len(context.output_groups) == admitted_at_backpressure - 64)
        assert h.runtime.runtime_group_count == 0
        h.manager.training_weight_updating = False
        h.pipeline._changed.set()
        await until(lambda: h.runtime.runtime_group_count == 80)
        assert not second.done()
    finally:
        await h.close()


async def test_steady_batch_boundary_obeys_max_staleness_two(monkeypatch):
    h = Harness(monkeypatch, num_rollout=4)
    try:
        h.partitions.update(("train_0", "train_1"))
        task, _ = h.begin(2)
        await until(lambda: h.runtime.runtime_group_count == 80)
        h.runtime.complete(h.runtime.running()[:64])
        await asyncio.wait_for(task, 8)
        assert h.runtime.pauses == 1
        assert h.pipeline.snapshot()["refill_blocked_reason"] == "staleness"
        assert h.pipeline._scheduler_metrics()["agentic/scheduler/refill_blocked/staleness"] == 1
    finally:
        await h.close()


async def test_steady_reward_backpressure_bounds_completed_ownership(monkeypatch):
    h = Harness(monkeypatch)
    try:
        h.allow_reward.clear()
        task, _ = h.begin(0)
        await until(lambda: h.runtime.runtime_group_count == 80)
        h.runtime.complete(h.runtime.running())
        await until(lambda: len(h.runtime.leased) == 144 and h.runtime.runtime_group_count == 64)
        h.runtime.complete(h.runtime.running())
        await until(lambda: h.runtime.runtime_group_count == 0)
        assert len(h.pipeline._active_group_tasks) == 144
        assert h.pipeline._runtime_group_gap(h.pipeline._active_step) == 0
        snapshot = h.pipeline.snapshot()
        assert snapshot["reward_inflight_group_count"] == 144
        assert snapshot["refill_blocked_reason"] == "completion_backpressure"
        assert sum(h.prepare.counts) <= 16
        h.allow_reward.set()
        await asyncio.wait_for(task, 8)
        assert len(h.pipeline._finalized_groups) == 80
        h.assert_unique_batches(1)
    finally:
        await h.close()


async def test_steady_final_backfill_replaces_failure_and_stops_at_debt(monkeypatch):
    h = Harness(monkeypatch, num_rollout=1)
    try:
        h.pipeline.transfer_domain.open_partition(0, 64, accepts_surplus=False)
        task, _ = h.begin(1, final_backfill=True)
        await until(lambda: h.runtime.runtime_group_count == 64)
        assert len(h.runtime.leased) == 64
        h.runtime.complete(h.runtime.running()[:1], failed=True)
        await until(lambda: len(h.runtime.leased) == 65 and h.runtime.runtime_group_count == 64)
        h.runtime.complete(h.runtime.running())
        output = await asyncio.wait_for(task, 8)
        assert len(output.samples) == 64
        assert len(h.runtime.leased) == 65
        assert h.runtime.pauses == 1 and h.pipeline.transfer_domain.total_debt == 0
        assert not h.runtime.streams and h.prepare.counts == (0, 0)
        h.assert_unique_batches(1)
    finally:
        await h.close()


async def test_runtime_counter_excludes_prepare_and_completed_groups():
    runtime = object.__new__(RuntimeDomain)
    prepared, _, _ = existing._runtime_result_stream("prepare")
    active, active_refs, _ = existing._runtime_result_stream("active")
    complete, complete_refs, _ = existing._runtime_result_stream("complete")
    active.runtime_admitted = complete.runtime_admitted = True
    complete_refs[0].completion.set_result(None)
    runtime._shards = [SimpleNamespace(group_streams={"p": prepared, "a": active, "c": complete})]
    runtime._progress_callback = lambda: None
    prepared.shard.handle.lease_group = SimpleNamespace(remote=AsyncMock())
    assert runtime.resident_group_count == 3 and runtime.runtime_group_count == 1
    await runtime.lease_group(prepared, 0)
    assert runtime.runtime_group_count == 2
    active_refs[0].completion.set_result(None)
    assert runtime.runtime_group_count == 1
    prepared.closing = True
    assert runtime.runtime_group_count == 0


async def test_steady_streaming_writes_do_not_block_refill_until_capacity_is_full(monkeypatch):
    h = Harness(monkeypatch)
    try:
        h.allow_write.clear()
        task, context = h.begin(0)
        await until(lambda: h.runtime.runtime_group_count == 80)
        h.runtime.complete(h.runtime.running()[:39])
        await until(lambda: len(context.output_groups) == 39 and h.runtime.runtime_group_count == 80)
        assert h.pipeline.snapshot()["transferring_group_count"] == 39
        assert len(h.runtime.leased) == 119
        assert not task.done()
        h.runtime.complete(h.runtime.running()[:25])
        await until(lambda: h.pipeline.snapshot()["transferring_group_count"] == 64)
        assert h.pipeline.snapshot()["refill_blocked_reason"] == "transfer_backpressure"
        admitted_at_backpressure = len(h.runtime.leased)
        assert 119 <= admitted_at_backpressure <= 144
        h.runtime.complete(h.runtime.running())
        await until(lambda: all(task.done() for task in h.pipeline._active_group_tasks))
        assert len(h.runtime.leased) == admitted_at_backpressure
        h.allow_write.set()
        await asyncio.wait_for(task, 8)
        h.assert_unique_batches(1)
    finally:
        await h.close()


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"agentic_runtime_groups": 0},
        {"agentic_prepare_groups": 0},
        {"agentic_concurrency": 80},
        {"fully_async": False},
        {"use_dynamic_global_batch_size": True},
    ],
)
async def test_steady_runtime_arguments_require_separate_resident_capacity(tmp_path, overrides):
    from relax.utils.arguments import _validate_agentic_rollout_args

    args = SimpleNamespace(
        use_agentic_rollout=True,
        agent_command="agent",
        agent_cwd=str(tmp_path),
        agent_timeout=1,
        agent_env=[],
        agentic_runtime_groups=80,
        agentic_prepare_groups=16,
        agentic_concurrency=None,
        agentic_eval_concurrency=None,
        fully_async=True,
        use_dynamic_global_batch_size=False,
        agentic_program_admission=False,
        agentic_session_lifecycle=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    if overrides:
        with pytest.raises(ValueError):
            _validate_agentic_rollout_args(args)
    else:
        _validate_agentic_rollout_args(args)
        assert args.agentic_concurrency == 96


@pytest.mark.parametrize("final_backfill", [False, True])
async def test_steady_final_drain_caps_runtime_when_two_partitions_have_debt(monkeypatch, final_backfill):
    h = Harness(monkeypatch, num_rollout=2)
    try:
        h.pipeline.transfer_domain.open_partition(0, 64, accepts_surplus=False)
        if final_backfill:
            h.pipeline.transfer_domain.open_partition(1, 64, accepts_surplus=False)
        task, context = h.begin(2 if final_backfill else 1, final_backfill=final_backfill)
        await until(lambda: h.runtime.runtime_group_count == 80)
        assert len(h.runtime.leased) == 80 and h.pipeline.transfer_domain.total_debt == 128
        h.runtime.complete(h.runtime.running())
        await until(lambda: len(context.output_groups) == 80 and h.runtime.runtime_group_count == 48)
        assert len(h.runtime.leased) == 128
        h.runtime.complete(h.runtime.running())
        output = await asyncio.wait_for(task, 8)
        assert len(output.samples) == 128
        assert len(h.runtime.leased) == 128 and h.runtime.peak_runtime == 80
        assert h.runtime.pauses == 1 and h.pipeline.transfer_domain.total_debt == 0
        assert not h.runtime.streams and h.prepare.counts == (0, 0)
        h.assert_unique_batches(2)
    finally:
        await h.close()
