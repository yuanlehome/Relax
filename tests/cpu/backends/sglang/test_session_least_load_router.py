# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Native HTTP PD placement and lifecycle; workers are controlled HTTP
peers."""

import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from tests.cpu.backends.sglang import test_pd_routing_replay_router as replay
from tests.cpu.backends.sglang.test_pd_failure_cleanup_router import Fault


router = replay.router
pytestmark = pytest.mark.parametrize("router", ["session"], indirect=True)


def control(router, session=None):
    url = router[0] + "/session_routes"
    kwargs = {}
    if session is not None:
        url += "/close"
        kwargs = {
            "data": json.dumps({"session_id": session}).encode(),
            "headers": {"Content-Type": "application/json"},
        }
    with urlopen(Request(url, **kwargs), timeout=3) as result:
        return json.load(result)


def generate(router, session, rid):
    try:
        with replay.generate(router, session=session, rid=rid) as result:
            return result.status, result.read(), dict(result.headers)
    except HTTPError as error:
        return error.code, error.read(), dict(error.headers)


def settled(router):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        state = control(router)
        if state["active_requests"] == 0:
            return state
        time.sleep(0.01)
    pytest.fail(f"active receipt leaked: {state}")


def test_idle_turns_balance_bindings_and_preserve_dp_through_tool_wait(router):
    owners = {}
    for turn in range(2):
        for i in range(20):
            assert generate(router, f"s{i}", f"s{i}-t{turn}")[0] == 200
            dp = router[1][1].received[-1]["routed_dp_rank"]
            if turn == 0:
                owners[i] = dp
            else:
                assert dp == owners[i]
    assert Counter(owners.values()) == {0: 10, 1: 10}
    state = settled(router)
    assert state["assignments"] == 20 and state["reuses"] == 20
    assert state["cleanup_unconfirmed"] == 0
    for i in range(20):
        ack = control(router, f"s{i}")
        assert ack["sealed"] and ack["binding_released"]
        assert ack["physical_cleanup_confirmed"] is False
    state = control(router)
    assert state["open_bindings"] == 0 and state["workers"] == []
    before = len(router[1][1].received)
    assert generate(router, "s0", "late-turn")[0] >= 400
    assert len(router[1][1].received) == before


def test_concurrent_arrivals_balance_pending_requests_and_close_race(router):
    gate = threading.Event()
    router[1][1].response_gate = gate
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(generate, router, f"s{i}", f"r{i}") for i in range(12)]
        try:
            deadline = time.monotonic() + 4
            while len(router[1][1].received) < 12 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert len(router[1][1].received) == 12
            assert Counter(r["routed_dp_rank"] for r in router[1][1].received) == {0: 6, 1: 6}
            state = control(router)
            assert state["active_requests"] == 12
            ack = control(router, "s0")
            assert ack["sealed"] and not ack["binding_released"] and ack["active_requests"] == 1
        finally:
            gate.set()
        assert all(f.result()[0] == 200 for f in futures)
    assert settled(router)["cleanup_unconfirmed"] == 0
    assert control(router, "s0")["binding_released"]
    assert generate(router, "s0", "late")[0] >= 400


@pytest.mark.parametrize("side", [0, 1])
def test_transport_uncertainty_is_not_released_by_close_ack(router, side):
    peer = Fault(router[1][1 - side])
    failed = Fault(router[1][side], disconnect=True)
    failed.wait_for = peer.registered
    code, _, headers = generate(router, "uncertain", "r1")
    assert code == 502 and headers["x-sglang-pd-cleanup"] == "unconfirmed"
    assert peer.terminal.is_set()
    state = settled(router)
    assert state["cleanup_unconfirmed"] == 1
    ack = control(router, "uncertain")
    assert ack["sealed"] and not ack["binding_released"]
    assert ack["cleanup_unconfirmed"] == 1
    attempts = [len(w.received) for w in router[1]]
    assert generate(router, "uncertain", "r2")[0] >= 400
    assert [len(w.received) for w in router[1]] == attempts


def test_close_before_first_request_and_missing_key_do_not_dispatch(router):
    assert control(router, "late-first")["binding_released"]
    assert generate(router, "late-first", "r1")[0] >= 400
    with pytest.raises(HTTPError) as error:
        replay.generate(router, session="", rid="missing-key")
    assert error.value.code == 400
    assert all(not w.received for w in router[1])


@pytest.mark.parametrize("complete", [False, True])
def test_stream_completion_proof_controls_binding_release(router, complete):
    body = replay.response(3, 1, [0] * 12)
    router[1][1].result = [f"data: {json.dumps(body)}\n\n".encode()] + ([b"data: [DONE]\n\n"] if complete else [])
    with replay.generate(router, session="stream", rid="stream-r1", stream=True) as result:
        result.read()
    state = settled(router)
    assert state["cleanup_unconfirmed"] == (0 if complete else 1)
    assert control(router, "stream")["binding_released"] is complete


def test_new_sessions_avoid_busy_dp_even_with_fewer_idle_bindings(router):
    decode = router[1][1]
    gate = threading.Event()
    decode.response_gates = {0: gate}
    with ThreadPoolExecutor(max_workers=1) as pool:
        busy = pool.submit(generate, router, "long", "long-r1")
        try:
            deadline = time.monotonic() + 3
            while not decode.received and time.monotonic() < deadline:
                time.sleep(0.01)
            assert decode.received[0]["routed_dp_rank"] == 0
            for i in range(6):
                assert generate(router, f"short-{i}", f"short-{i}-r1")[0] == 200
                assert decode.received[-1]["routed_dp_rank"] == 1
            assert not busy.done()
        finally:
            gate.set()
        assert busy.result()[0] == 200
    assert settled(router)["cleanup_unconfirmed"] == 0
    # Existing Session stays on DP1 even after DP0 becomes idle again.
    assert generate(router, "short-0", "short-0-r2")[0] == 200
    assert decode.received[-1]["routed_dp_rank"] == 1


def test_simultaneous_first_requests_for_one_session_create_one_binding(router):
    gate = threading.Event()
    router[1][1].response_gate = gate
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(generate, router, "same", f"unique-rid-{i}") for i in range(8)]
        try:
            deadline = time.monotonic() + 3
            while len(router[1][1].received) < 8 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert len(router[1][1].received) == 8
            assert len({r["routed_dp_rank"] for r in router[1][1].received}) == 1
            state = control(router)
            assert state["assignments"] == 1 and state["reuses"] == 7
        finally:
            gate.set()
        assert all(f.result()[0] == 200 for f in futures)
    assert settled(router)["active_requests"] == 0


def test_same_url_worker_recreation_does_not_migrate_existing_session(router):
    from urllib.parse import quote

    def workers():
        with urlopen(router[0] + "/workers", timeout=3) as response:
            return json.load(response)["workers"]

    def wait_workers(predicate):
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            state = workers()
            if predicate(state) and all(w["is_healthy"] for w in state):
                return state
            time.sleep(0.02)
        pytest.fail(f"worker registration transition did not complete: {state}")

    def register(generation):
        payload = {"url": router[1][1].url, "worker_type": "decode", "labels": {"test_generation": generation}}
        with urlopen(
            Request(
                router[0] + "/workers", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
            ),
            timeout=3,
        ) as response:
            receipt = json.load(response)
        wait_workers(
            lambda state: (
                len(state) == 4
                and all(
                    w.get("metadata", {}).get("test_generation") == generation
                    for w in state
                    if w["url"].startswith(router[1][1].url)
                )
            )
        )
        return receipt["worker_id"]

    # The registration receipt identifies the physical engine. Listed DP IDs
    # address @rank URLs and are not the deletion handle used by Relax.
    engine_id = register("first")
    assert generate(router, "old", "old-r1")[0] == 200
    with urlopen(Request(router[0] + "/workers/" + quote(engine_id, safe=""), method="DELETE"), timeout=3):
        pass
    wait_workers(lambda state: len(state) == 2)
    register("replacement")
    attempts = len(router[1][1].received)
    assert generate(router, "old", "old-r2")[0] >= 400
    assert len(router[1][1].received) == attempts
    assert generate(router, "new", "new-r1")[0] == 200
