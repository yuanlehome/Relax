# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for ``DeviceDirectBackend._update_rollout_engines`` recovery behavior.

These exercise the real method on a shell instance (``object.__new__``) with the
two collaborators it calls (``_healthcheck_rollout_engines`` /
``_remove_failed_engines``) stubbed, so we test the retry / prune / raise control
flow in isolation. The distributed cases additionally run the complete group
initializer across four real Gloo ranks, with only serving transport isolated.

Regression target: previously the method raised ``RuntimeError`` unconditionally
after exhausting retries -- even when healthy engines remained -- which failed the
actor weight sync and escalated to a full global restart (progress lost). It must
now only raise when *no* healthy engine is left, prune the unhealthy ones, and
invalidate the cached topology signature so the group is rebuilt cleanly.
"""

import concurrent.futures
import json
import multiprocessing as mp
import tempfile
import threading
import time
import traceback
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Set

import pytest


# DeviceDirectBackend's module imports megatron.core at module level.
pytest.importorskip("megatron.core")

from relax.distributed.checkpoint_service.backends import device_direct
from relax.distributed.checkpoint_service.backends.device_direct import DeviceDirectBackend


def _make_backend(engines: dict, health_fn: Callable[..., Set[int]]) -> DeviceDirectBackend:
    """Build a shell backend that only has the attributes the method
    touches."""
    backend = object.__new__(DeviceDirectBackend)
    backend.rollout_engines = dict(engines)
    backend.rollout_topology = {str(rank): {"rank": rank} for rank in engines}
    backend._rollout_topology_signature = "STALE_SIG"

    backend._healthcheck_rollout_engines = health_fn  # type: ignore[method-assign]

    def _remove(failed: Set[int]) -> None:
        for rank in set(failed):
            backend.rollout_engines.pop(rank, None)
            backend.rollout_topology.pop(str(rank), None)
            backend.rollout_topology.pop(rank, None)

    backend._remove_failed_engines = _remove  # type: ignore[method-assign]
    return backend


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(device_direct.time, "sleep", lambda *_a, **_k: None)


def _fixed(value: Set[int]) -> Callable[..., Set[int]]:
    return lambda *_a, **_k: set(value)


def _sequence(values):
    seq = list(values)

    def _fn(*_a, **_k):
        return set(seq.pop(0)) if seq else set()

    return _fn


def test_update_rollout_engines_all_healthy_returns_without_prune():
    backend = _make_backend({0: object(), 1: object()}, _fixed(set()))
    backend._update_rollout_engines(max_retries=3, retry_interval=0)
    assert set(backend.rollout_engines) == {0, 1}
    # No prune happened -> cached signature untouched.
    assert backend._rollout_topology_signature == "STALE_SIG"


def test_update_rollout_engines_dead_engine_seed_healthy_degrades_no_raise():
    # rank 1 (scale-out) always unhealthy, rank 0 (seed) healthy the whole time.
    backend = _make_backend({0: object(), 1: object()}, _fixed({1}))
    backend._update_rollout_engines(max_retries=3, retry_interval=0)
    # Dead engine pruned from both engines and topology, seed retained, no raise.
    assert set(backend.rollout_engines) == {0}
    assert "1" not in backend.rollout_topology
    # Signature bookkeeping is the caller's (init_process_group_for_rollout)
    # concern; this method does not touch it.
    assert backend._rollout_topology_signature == "STALE_SIG"


def test_update_rollout_engines_all_dead_raises():
    backend = _make_backend({0: object(), 1: object()}, _fixed({0, 1}))
    with pytest.raises(RuntimeError, match="No healthy rollout engines"):
        backend._update_rollout_engines(max_retries=2, retry_interval=0)
    assert backend.rollout_engines == {}


def test_update_rollout_engines_recovered_at_tail_keeps_all():
    # Unhealthy during the loop, but the final post-loop healthcheck sees it
    # recovered -> nothing pruned, nothing raised, signature untouched.
    health = _sequence([{1}, {1}, {1}, set()])
    backend = _make_backend({0: object(), 1: object()}, health)
    backend._update_rollout_engines(max_retries=3, retry_interval=0)
    assert set(backend.rollout_engines) == {0, 1}
    assert backend._rollout_topology_signature == "STALE_SIG"


def test_update_rollout_engines_transient_failure_recovers_in_loop():
    # Fails twice then recovers before retries are exhausted -> early return,
    # no prune (does not kill a merely-slow-loading engine).
    health = _sequence([{1}, {1}, set()])
    backend = _make_backend({0: object(), 1: object()}, health)
    backend._update_rollout_engines(max_retries=5, retry_interval=0)
    assert set(backend.rollout_engines) == {0, 1}
    assert backend._rollout_topology_signature == "STALE_SIG"


def test_update_rollout_engines_no_topology_raises():
    backend = _make_backend({}, _fixed(set()))
    backend.rollout_topology = {}
    with pytest.raises(RuntimeError, match="No rollout engines configured"):
        backend._update_rollout_engines(max_retries=2, retry_interval=0)


def test_topology_signature_distinguishes_pruned_from_full():
    # After a prune, ``init_process_group_for_rollout`` stores the signature of
    # the *pruned* topology (not the pre-prune full one). This asserts the two
    # signatures differ, so if the pruned engine is later re-listed the reuse
    # fast path sees a signature mismatch and rebuilds the group -- re-adding the
    # engine instead of orphaning it.
    full = {
        "0": {"ip": "host-a", "port": 100, "metadata": {"num_gpus_per_engine": 1}},
        "1": {"ip": "host-b", "port": 200, "metadata": {"num_gpus_per_engine": 1}},
    }
    pruned = {"0": full["0"]}
    sig_full = DeviceDirectBackend._rollout_topology_signature_of(full)
    sig_pruned = DeviceDirectBackend._rollout_topology_signature_of(pruned)
    assert sig_full != sig_pruned


class _RecoveryHTTPState:
    def __init__(self, case: str) -> None:
        self.case = case
        self.records = []
        self.lock = threading.Lock()
        self.mutating = threading.Event()


def _recovery_http_handler(state: _RecoveryHTTPState, engine_rank: int) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args) -> None:
            pass

        def reply(self, status: int, value: dict) -> None:
            content = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self) -> None:
            source = int(self.path.split("/")[1])
            # Reproduce the old slow PP: its health request completes only
            # after another PP has begun mutating the same serving schedulers.
            if state.case == "one_dead_slow_pp" and source == 2:
                if not state.mutating.wait(20):
                    return self.reply(503, {"error": "slow-PP test gate timed out"})
            with state.lock:
                late = state.mutating.is_set()
                state.records.append({"op": "health", "source": source, "engine": engine_rank, "late": late})
            failed = late or state.case == "all_unhealthy" or (state.case == "one_dead_slow_pp" and engine_rank == 2)
            self.reply(503 if failed else 200, {"healthy": not failed})

        def do_POST(self) -> None:
            source = int(self.path.split("/")[1])
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            state.mutating.set()
            with state.lock:
                state.records.append(
                    {
                        "op": self.path.rsplit("/", 1)[-1],
                        "source": source,
                        "engine": engine_rank,
                        "group": payload.get("group_name"),
                        "world_size": payload.get("world_size"),
                    }
                )
            self.reply(200, {"success": True})

    return Handler


class _ServingTransportGroup:
    """Marker for the mocked serving-side NCCL group, never a training Gloo
    group."""


def _gloo_recovery_worker(rank: int, world_size: int, rendezvous: str, topology: dict, case: str, queue: Any) -> None:
    import torch.distributed as dist

    from relax.utils.distributed_utils import get_gloo_group, init_gloo_group

    module = device_direct
    actor_cls = module.RolloutEngine.__ray_metadata__.modified_class
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

    class Proxy:
        def __init__(self, engine_rank: int, info: dict) -> None:
            self.actor = actor_cls(engine_rank, info)
            # Route real production HTTP methods through rank-labelled fake
            # service endpoints so the test can impose one PP's slow health.
            self.actor.base_url += f"/{rank}"

        def __getattr__(self, name: str) -> SimpleNamespace:
            method = getattr(self.actor, name)
            return SimpleNamespace(remote=lambda *args, **kwargs: executor.submit(method, *args, **kwargs))

    class RemoteFactory:
        def options(self, **_kwargs) -> "RemoteFactory":
            return self

        def remote(self, engine_rank: int, info: dict) -> Proxy:
            return Proxy(engine_rank, info)

    def ray_get(value: Any, timeout: float | None = None) -> Any:
        if isinstance(value, list):
            return [ray_get(item, timeout) for item in value]
        return value.result(timeout=timeout)

    def ray_wait(values: list, timeout: float | None = None, num_returns: int | None = None) -> tuple[list, list]:
        done, pending = concurrent.futures.wait(values, timeout=timeout)
        return list(done), list(pending)

    # Ray process scheduling and serving NCCL are the external boundaries.
    # Production HTTP health, retry/prune, topology planning and rebuilding run.
    module.RolloutEngine = RemoteFactory()
    module.with_control_plane_affinity = lambda *_args, **_kwargs: {}
    module.ray.get = ray_get
    module.ray.wait = ray_wait
    module.ray.kill = lambda *_args, **_kwargs: None
    module.ray._private.services.get_node_ip_address = lambda: "127.0.0.1"
    module.time = SimpleNamespace(sleep=lambda *_args: None)

    def create_serving_group(**_kwargs) -> _ServingTransportGroup:
        if case == "rebuild_error" and rank == 0:
            raise RuntimeError("injected serving NCCL init failure")
        return _ServingTransportGroup()

    module.init_process_group = create_serving_group
    real_destroy = dist.destroy_process_group

    def destroy(group: Any = None) -> None:
        if isinstance(group, _ServingTransportGroup):
            return
        return real_destroy(group)

    dist.destroy_process_group = destroy
    initialized = False
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{rendezvous}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=50),
        )
        init_gloo_group(distributed_timeout_minutes=1)
        initialized = True
        backend = object.__new__(module.DeviceDirectBackend)
        backend.role_info = object()
        backend.args = SimpleNamespace(rollout_num_gpus_per_engine=1, hybrid=False, offload_rollout=False)
        backend.backend_type = "nccl"
        backend._megatron = SimpleNamespace(
            mpu=SimpleNamespace(
                get_data_parallel_rank=lambda **_kwargs: 0,
                get_tensor_model_parallel_rank=lambda: rank % 2,
                get_pipeline_model_parallel_rank=lambda: rank // 2,
            )
        )
        backend.rollout_topology = dict(topology["nodes"]["rollout"])
        backend.rollout_engines = {}
        backend._model_update_groups = _ServingTransportGroup()
        backend._rollout_topology_signature = backend._rollout_topology_signature_of(backend.rollout_topology)
        if rank % 2 == 0:
            backend._create_rollout_engines(backend.rollout_topology)
        if case in ("asym_reuse", "rebuild_error") and rank == 2:
            backend._model_update_groups = None
        outcome = {"rank": rank, "error": None}
        try:
            backend.init_process_group_for_rollout(topology)
        except Exception as error:
            outcome["error"] = f"{type(error).__name__}: {error}"
        outcome["recipients"] = sorted(map(int, backend.rollout_engines)) if rank % 2 == 0 else None
        outcome["signature_matches"] = (
            backend._rollout_topology_signature == backend._rollout_topology_signature_of(backend.rollout_topology)
            if rank % 2 == 0
            else None
        )
        # Real agreement at the consumer proves non-source ranks exited too.
        outcomes = [None] * world_size
        dist.all_gather_object(outcomes, outcome, group=get_gloo_group())
        if rank == 0:
            queue.put({"outcomes": outcomes})
    except BaseException as error:
        queue.put({"worker_error": rank, "error": repr(error), "traceback": traceback.format_exc()})
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        if initialized:
            real_destroy()


@pytest.mark.parametrize("case", ["reuse", "one_dead_slow_pp", "asym_reuse", "rebuild_error", "all_unhealthy"])
def test_rollout_recovery_agrees_across_gloo_ranks(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    state = _RecoveryHTTPState(case)
    servers = []
    for engine_rank in range(3):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _recovery_http_handler(state, engine_rank))
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append(server)
    topology = {
        "nodes": {
            "rollout": {
                str(i): {
                    "ip": "127.0.0.1",
                    "port": server.server_port,
                    "metadata": {"num_gpus_per_engine": 1, "actor_id": f"engine-{i}"},
                }
                for i, server in enumerate(servers)
            }
        }
    }
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = []
    try:
        with tempfile.TemporaryDirectory(prefix="dcs-gloo-probe-") as temp:
            rendezvous = str(Path(temp) / "store")
            for rank in range(4):
                process = context.Process(
                    target=_gloo_recovery_worker,
                    args=(rank, 4, rendezvous, topology, case, queue),
                )
                process.start()
                processes.append(process)
            deadline = time.monotonic() + 120
            for process in processes:
                process.join(max(0, deadline - time.monotonic()))
            assert not any(process.is_alive() for process in processes), "Gloo probe timed out"
            assert all(process.exitcode == 0 for process in processes), [p.exitcode for p in processes]
            result = queue.get(timeout=3)
            assert "worker_error" not in result, result
            outcomes = result["outcomes"]
            errors = [item["error"] for item in outcomes]
            records = list(state.records)
            assert not any(item.get("late") for item in records), "health overlapped serving mutation"
            assert {item["source"] for item in records if item["op"] == "health"} == {0}
            if case == "all_unhealthy":
                assert all(errors) and len(set(errors)) == 1, outcomes
                assert "health check failed" in errors[0] and "No healthy rollout engines" in errors[0], errors
                assert not any(item["op"] != "health" for item in records), records
            elif case == "rebuild_error":
                assert all(errors) and len(set(errors)) == 1, outcomes
                assert "PP0 rebuild failed" in errors[0] and "after 3 attempts" in errors[0], errors
                assert not any(item["source"] == 2 and item["op"] != "health" for item in records)
            else:
                assert not any(errors), outcomes
                expected = [0, 1] if case == "one_dead_slow_pp" else [0, 1, 2]
                assert all(item["recipients"] == expected for item in outcomes if item["rank"] % 2 == 0)
                assert all(item["signature_matches"] for item in outcomes if item["rank"] % 2 == 0)
                mutations = [item for item in records if item["op"] != "health"]
                if case == "reuse":
                    assert not mutations, mutations
                else:
                    sequence = []
                    for item in mutations:
                        if not sequence or sequence[-1] != item["source"]:
                            sequence.append(item["source"])
                    assert sequence == [0, 2], sequence
                    sizes = {item["world_size"] for item in mutations if item["op"] == "init_weights_update_group"}
                    assert sizes == {len(expected) + 1}, sizes
    finally:
        state.mutating.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=3)
        for server in servers:
            server.shutdown()
            server.server_close()
        queue.close()
