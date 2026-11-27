# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Fault injection against the built native PD router, with retries enabled."""

import json
import socket
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from tests.cpu.backends.sglang import test_pd_routing_replay_router as replay


generate = replay.generate
router = replay.router


pytestmark = pytest.mark.parametrize("router", ["pd-retry"], indirect=True)


class Fault:
    """A backend whose RID registration, terminal response and abort ACK are
    independent."""

    def __init__(self, worker, *, status=None, disconnect=False, broken_body=False):
        self.worker = worker
        worker.fault = self
        self.status, self.disconnect, self.broken_body = status, disconnect, broken_body
        self.received = threading.Event()
        self.register = threading.Event()
        self.register.set()
        self.registered = threading.Event()
        self.abort_received = threading.Event()
        self.terminal = threading.Event()
        self.allow_terminal = threading.Event()
        self.allow_terminal.set()
        self.allow_ack = threading.Event()
        self.allow_ack.set()
        self.release = threading.Event()
        self.active = set()
        self.aborts = []
        self.lock = threading.Lock()
        self.wait_for = None
        self.drop_on_abort = False

    @staticmethod
    def disconnect_socket(handler):
        handler.close_connection = True
        handler.connection.shutdown(socket.SHUT_RDWR)
        handler.connection.close()

    def handle(self, handler, request):
        rid = request["rid"]
        if handler.path == "/abort_request":
            with self.lock:
                self.aborts.append(request)
                if rid in self.active:
                    self.release.set()
            self.abort_received.set()
            self.allow_ack.wait(timeout=6)
            handler.reply({"success": True})
            return

        self.worker.received.append(request)
        self.worker.wire_attempts.append((handler.client_address, request))
        self.received.set()
        self.register.wait(timeout=6)
        with self.lock:
            duplicate = rid in self.active
            if not duplicate:
                self.active.add(rid)
        self.registered.set()
        if duplicate:
            handler.reply({"error": {"message": f"Duplicate RID: {rid}"}}, 400)
            return
        if self.wait_for is not None:
            assert self.wait_for.wait(timeout=6), "paired request was never dispatched"
        if self.disconnect:
            self.disconnect_socket(handler)
            return
        if self.broken_body:
            handler.send_response(200)
            handler.send_header("Content-Length", "1000")
            handler.end_headers()
            handler.wfile.write(b"{")
            handler.wfile.flush()
            self.disconnect_socket(handler)
            return
        status = self.status
        if status is None:
            self.release.wait(timeout=9)
            if self.drop_on_abort:
                self.disconnect_socket(handler)
                return
            self.allow_terminal.wait(timeout=6)
            status = 499
        message = f"Request {rid} was aborted" if status == 499 else f"{self.worker.role} original failure"
        with self.lock:
            self.active.discard(rid)
        self.terminal.set()
        handler.reply({"error": {"message": message}}, status)


def request_outcome(router, *, rid="fault-rid", stream=False):
    try:
        with generate(router, rid=rid, stream=stream) as response:
            return response.status, response.read(), dict(response.headers)
    except HTTPError as error:
        return error.code, error.read(), dict(error.headers)


def background_request(router, **kwargs):
    outcome = {}

    def run():
        try:
            outcome["result"] = request_outcome(router, **kwargs)
        except Exception as error:
            outcome["error"] = repr(error)

    client = threading.Thread(target=run, daemon=True)
    client.start()
    return client, outcome


def join_request(client, outcome):
    client.join(timeout=12)
    assert not client.is_alive(), outcome
    assert "error" not in outcome, outcome
    return outcome["result"]


def assert_single_pair(workers):
    assert all(len(worker.received) == len(worker.wire_attempts) == 1 for worker in workers)
    p, d = (worker.received[0] for worker in workers)
    assert p["rid"] == d["rid"]
    assert p["bootstrap_room"] == d["bootstrap_room"]
    assert d["disagg_prefill_dp_rank"] == p["routed_dp_rank"]
    assert all(request["routed_dp_rank"] in (0, 1) for request in (p, d))


@pytest.mark.parametrize("side,kind", [(1, "400"), (0, "503"), (0, "transport"), (1, "transport"), (0, "body")])
@pytest.mark.parametrize("stream", [False, True])
def test_pd_failure_aborts_peer_without_replaying_rid(router, side, kind, stream):
    _, workers = router
    peer = Fault(workers[1 - side])
    failed = Fault(
        workers[side],
        status=int(kind) if kind.isdigit() else None,
        disconnect=kind == "transport",
        broken_body=kind == "body",
    )
    failed.wait_for = peer.registered
    if kind == "400":
        # Existing backend ownership remains protected; the rejected request
        # must not abort the older request on the rejecting D worker.
        failed.active.add("fault-rid")
    started = time.monotonic()
    code, body, headers = request_outcome(router, stream=stream)
    assert code == (int(kind) if kind.isdigit() else 502), body
    assert time.monotonic() - started < 4
    assert peer.abort_received.is_set() and peer.terminal.is_set()
    assert peer.active == set()
    assert all(abort == {"rid": "fault-rid", "abort_all": False} for abort in peer.aborts)
    assert_single_pair(workers)
    if kind == "400":
        assert b"Duplicate RID" in body
        assert failed.active == {"fault-rid"} and not failed.aborts
    elif kind in ("transport", "body"):
        assert failed.abort_received.is_set()
        assert headers.get("x-sglang-pd-cleanup") == "unconfirmed"
    else:
        assert b"prefill original failure" in body


@pytest.mark.parametrize("side", [0, 1])
def test_pd_abort_repeats_after_late_peer_registration(router, side):
    _, workers = router
    peer = Fault(workers[1 - side])
    peer.register.clear()
    failed = Fault(workers[side], status=503)
    failed.wait_for = peer.received
    client, outcome = background_request(router)
    try:
        assert peer.abort_received.wait(timeout=3), "ordinary failure did not trigger peer abort"
        assert client.is_alive(), "an ACK before registration is not terminal proof"
        assert not peer.registered.is_set()
        peer.register.set()
        code, body, _ = join_request(client, outcome)
        assert code == 503, body
        assert len(peer.aborts) >= 2 and peer.terminal.is_set()
        assert_single_pair(workers)
    finally:
        peer.register.set()
        peer.release.set()


@pytest.mark.parametrize("origin_status", [400, 499])
def test_pd_abort_ack_does_not_release_original_response(router, origin_status):
    _, (p, d) = router
    peer = Fault(p)
    peer.allow_terminal.clear()
    failed = Fault(d, status=origin_status)
    failed.wait_for = peer.registered
    client, outcome = background_request(router)
    try:
        assert peer.abort_received.wait(timeout=3)
        time.sleep(0.3)
        assert client.is_alive() and not peer.terminal.is_set()
        peer.allow_terminal.set()
        code, body, _ = join_request(client, outcome)
        assert code == origin_status, body
        assert peer.terminal.is_set()
        assert_single_pair((p, d))
    finally:
        peer.allow_terminal.set()
        peer.release.set()


def test_pd_terminal_response_joins_pending_abort_control(router):
    _, (p, d) = router
    peer = Fault(p)
    peer.allow_ack.clear()
    failed = Fault(d, status=499)
    failed.wait_for = peer.registered
    client, outcome = background_request(router)
    try:
        assert peer.terminal.wait(timeout=3)
        time.sleep(0.2)
        assert client.is_alive(), "a late abort could cancel the next use of this RID"
        peer.allow_ack.set()
        code, body, _ = join_request(client, outcome)
        assert code == 499 and b"Request fault-rid was aborted" in body
        assert_single_pair((p, d))
    finally:
        peer.allow_ack.set()
        peer.release.set()


@pytest.mark.parametrize("origin_status", [400, 503, 499])
def test_pd_peer_connection_loss_retains_original_error_and_forbids_replay(router, origin_status):
    _, (p, d) = router
    peer = Fault(p)
    peer.drop_on_abort = True
    failed = Fault(d, status=origin_status)
    failed.wait_for = peer.registered
    code, body, headers = request_outcome(router)
    if origin_status == 499:
        assert code == 424 and b"pd_cancellation_unconfirmed" in body
    else:
        assert code == origin_status and b"decode original failure" in body
        assert headers.get("x-sglang-pd-cleanup") == "unconfirmed"
    assert peer.abort_received.is_set() and not peer.terminal.is_set()
    assert_single_pair((p, d))


def test_pd_independent_error_wins_over_cancellation_race(router):
    _, (p, d) = router
    cancelled = Fault(p, status=499)
    failed = Fault(d, status=400)
    cancelled.wait_for = failed.registered
    failed.wait_for = cancelled.terminal
    code, body, _ = request_outcome(router)
    assert code == 400 and b"decode original failure" in body
    assert_single_pair((p, d))


def test_pd_session_affinity_survives_errors_and_changed_prompts(router):
    url, workers = router
    owners = {}
    for turn in range(2):
        for session in range(12):
            rid = f"session-{session}-turn-{turn}"
            payload = {
                "rid": rid,
                "input_ids": [1] * (3 + turn),
                "sampling_params": {"max_new_tokens": 1},
                "stream": False,
            }
            with urlopen(
                Request(
                    url + "/generate",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json", "X-SMG-Routing-Key": f"session-{session}"},
                ),
                timeout=6,
            ) as result:
                assert result.status == 200
            p, d = (worker.received[-1] for worker in workers)
            assert d["disagg_prefill_dp_rank"] == p["routed_dp_rank"]
            assert p["bootstrap_room"] == d["bootstrap_room"]
            owner = d["routed_dp_rank"]
            if turn == 0:
                owners[session] = owner
                workers[1].response_status = 400
                try:
                    payload["rid"] = f"failed-session-{session}"
                    with pytest.raises(HTTPError) as error:
                        urlopen(
                            Request(
                                url + "/generate",
                                data=json.dumps(payload).encode(),
                                headers={
                                    "Content-Type": "application/json",
                                    "X-SMG-Routing-Key": f"session-{session}",
                                },
                            ),
                            timeout=6,
                        )
                    assert error.value.code == 400
                finally:
                    workers[1].response_status = 200
            else:
                assert owner == owners[session]
    assert set(owners.values()) == {0, 1}


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("kind", ["503", "transport"])
def test_pd_completed_peer_does_not_make_failed_dispatch_replayable(router, side, kind):
    _, workers = router
    Fault(workers[side], status=503 if kind == "503" else None, disconnect=kind == "transport")
    code, body, _ = request_outcome(router)
    assert code == (503 if kind == "503" else 502), body
    assert_single_pair(workers)


@pytest.mark.parametrize("stream", [False, True])
def test_pd_decode_body_transport_failure_aborts_pending_prefill(router, stream):
    _, (p, d) = router
    peer = Fault(p)
    failed = Fault(d, broken_body=True)
    failed.wait_for = peer.registered
    code, body, headers = request_outcome(router, stream=stream)
    assert code == 502 and b"Decode server error" in body
    assert headers.get("x-sglang-pd-cleanup") == "unconfirmed"
    assert peer.terminal.is_set()
    assert_single_pair((p, d))


def test_pd_early_decode_stream_keeps_bytes_and_prefill_metadata(router):
    _, (p, d) = router
    p.response_gate = threading.Event()
    final = replay.response(3, 1, [0] * 12)
    wire = ("data: " + json.dumps(final, ensure_ascii=False) + "\n\ndata: [DONE]\n\n").encode()
    d.result = [bytes([byte]) for byte in wire]
    client, outcome = background_request(router, stream=True)
    try:
        assert d.response_finished.wait(timeout=3)
        assert client.is_alive()
        p.response_gate.set()
        code, body, _ = join_request(client, outcome)
        assert code == 200
        text = body.decode()
        assert "中文" in text and "[DONE]" in text
        data = next(line[6:] for line in text.splitlines() if line.startswith("data: {"))
        assert replay.decode(json.loads(data)["meta_info"]["routed_experts"]) == [1] * 12
        assert_single_pair((p, d))
    finally:
        p.response_gate.set()
