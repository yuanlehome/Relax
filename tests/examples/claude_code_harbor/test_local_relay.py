# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check real HTTP routing, streaming holds, capture and cancellation."""

from __future__ import annotations

import concurrent.futures
import http.client
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

import pytest

from examples.claude_code_harbor.relay import LocalModelRelay
from examples.claude_code_harbor.request_normalization import normalize_request


pytest.importorskip("torch")
pytest.importorskip("ray")


@pytest.fixture
def upstream() -> Iterator[dict[str, Any]]:
    state: dict[str, Any] = {"calls": [], "delay": 0, "status": 200, "started": threading.Event()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_POST(self) -> None:
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["calls"].append((self.path, self.headers.get("Authorization"), payload))
            state["started"].set()
            if "disconnected" in state:
                self.connection.settimeout(3)
                if self.connection.recv(1) == b"":
                    state["disconnected"].set()
                return
            if "release" in state:
                state["release"].wait(timeout=5)
            time.sleep(state["delay"])
            result = {
                "type": "message",
                "role": "assistant",
                "model": "proxy-model",
                "content": [{"type": "text", "text": "answer"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 2, "output_tokens": 1},
            }
            if state["status"] != 200:
                result = state.get(
                    "error_payload", {"type": "error", "error": {"type": "api_error", "message": "upstream failure"}}
                )
            body = json.dumps(result).encode()
            try:
                self.send_response(state["status"])
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["base_url"] = f"http://127.0.0.1:{server.server_port}/agentic_api"
    try:
        yield state
    finally:
        if "release" in state:
            state["release"].set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def send(
    relay: LocalModelRelay,
    *,
    stream: bool = False,
    credential: str = "session",
    payload: dict[str, Any] | None = None,
) -> bytes:
    body = json.dumps(
        payload
        if payload is not None
        else {
            "model": "proxy-model",
            "max_tokens": 8192,
            "stream": stream,
            "messages": [{"role": "user", "content": "task"}],
        }
    ).encode()
    request = urllib.request.Request(
        relay.base_url + "/v1/messages?beta=true",
        data=body,
        headers={"x-api-key": credential, "Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=5) as response:
        return response.read()


def test_local_relay_preserves_prefix_and_injects_bearer_credential(upstream: dict[str, Any]) -> None:
    relay = LocalModelRelay(upstream["base_url"], "session")
    relay.start()
    try:
        result = json.loads(send(relay))
        path, credential, payload = upstream["calls"][0]
        assert path == "/agentic_api/v1/messages"
        assert credential == "Bearer session"
        assert payload["stream"] is False
        assert result["content"][0]["text"] == "answer"
        assert relay.capture.export(1)[0]["messages"][-1] == {"role": "assistant", "content": "answer"}
    finally:
        relay.close()


def test_local_relay_normalizes_actual_requests_and_captured_lineage(upstream: dict[str, Any]) -> None:
    from examples.claude_code_harbor.contexts import canonical_context

    relay = LocalModelRelay(upstream["base_url"], "session", normalization_id="trial-rollout")
    reminder = "<total_tokens>200000</total_tokens>"
    first = {
        "model": "proxy-model",
        "max_tokens": 8192,
        "system": f"cc_version=2.1.150.aaa; cch=12345;\n{reminder}",
        "messages": [{"role": "user", "content": "task"}],
    }
    relay.start()
    try:
        result = json.loads(send(relay, payload=first))
        second = {
            **first,
            "system": f"cc_version=2.1.150.bbb; cch=abcde;\n{reminder}\n{reminder}",
            "messages": [
                *first["messages"],
                {"role": "assistant", "content": result["content"]},
                {"role": "user", "content": "next tool result"},
            ],
        }
        result = json.loads(send(relay, payload=second))
        first_forwarded = upstream["calls"][0][2]
        second_forwarded = upstream["calls"][1][2]
        assert first_forwarded == {**normalize_request(first, "trial-rollout"), "stream": False}
        assert second_forwarded == {**normalize_request(second, "trial-rollout"), "stream": False}
        assert first_forwarded["system"] == second_forwarded["system"]
        records = relay.capture.export(1)
        assert len(records) == 1
        expected = canonical_context(second_forwarded, result)
        assert {key: records[0][key] for key in expected} == expected
        # A real compaction replaces history and must remain a separate context.
        compacted = {**second, "messages": [{"role": "user", "content": "continue from compacted summary"}]}
        send(relay, payload=compacted)
        assert len(relay.capture.export(1)) == 2
    finally:
        relay.close()


def test_local_relay_sends_heartbeats_before_held_response(upstream: dict[str, Any]) -> None:
    upstream["delay"] = 0.15
    relay = LocalModelRelay(upstream["base_url"], "session", heartbeat_s=0.02)
    relay.start()
    try:
        stream = send(relay, stream=True).decode()
        assert stream.count("event: ping") >= 2
        assert stream.index("event: ping") < stream.index("event: message_start")
        assert "event: content_block_delta" in stream
        assert "event: message_stop" in stream
        assert len(relay.capture.export(1)) == 1
    finally:
        relay.close()


def test_local_relay_rejects_other_session_credentials(upstream: dict[str, Any]) -> None:
    relay = LocalModelRelay(upstream["base_url"], "session")
    relay.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            send(relay, credential="another-session")
        assert error.value.code == 401
        assert upstream["calls"] == []
    finally:
        relay.close()


def test_local_relay_does_not_capture_upstream_errors(upstream: dict[str, Any]) -> None:
    upstream["status"] = 500
    relay = LocalModelRelay(upstream["base_url"], "session")
    relay.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            send(relay)
        assert error.value.code == 500
        with pytest.raises(ValueError, match="without a captured"):
            relay.capture.export(0)
    finally:
        relay.close()


def test_local_relay_budget_keeps_committed_prefix_for_harbor(upstream: dict[str, Any]) -> None:
    relay = LocalModelRelay(upstream["base_url"], "session", max_requests=1)
    relay.start()
    try:
        send(relay)
        with pytest.raises(urllib.error.HTTPError) as error:
            send(relay)
        assert error.value.code == 400
        assert relay.terminal_reason == "max_steps"
        assert len(upstream["calls"]) == 1
        assert len(relay.capture.export(0)) == 1
    finally:
        relay.close()


@pytest.mark.parametrize("overflow", [True, False])
def test_local_relay_distinguishes_context_budget_from_other_bad_requests(
    upstream: dict[str, Any], overflow: bool
) -> None:
    upstream["status"] = 400
    upstream["error_payload"] = {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "This model's maximum context length is 262144 tokens." if overflow else "Invalid tool schema",
        },
    }
    relay = LocalModelRelay(upstream["base_url"], "session")
    relay.start()
    try:
        with pytest.raises(urllib.error.HTTPError):
            send(relay)
        assert relay.terminal_reason == ("context_overflow" if overflow else None)
    finally:
        relay.close()


def test_local_relay_close_interrupts_unbounded_upstream_read(upstream: dict[str, Any]) -> None:
    upstream["release"] = threading.Event()
    relay = LocalModelRelay(upstream["base_url"], "session")
    relay.start()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(send, relay)
        try:
            assert upstream["started"].wait(timeout=3)
            started = time.monotonic()
            relay.close()
            assert time.monotonic() - started < 2
            with pytest.raises((urllib.error.HTTPError, http.client.RemoteDisconnected)):
                pending.result(timeout=3)
        finally:
            upstream["release"].set()
            relay.close()


def test_local_relay_close_does_not_wait_for_blocked_connect(
    upstream: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = threading.Event()
    release = threading.Event()
    connect = http.client.HTTPConnection.connect
    upstream_port = urlsplit(upstream["base_url"]).port

    def blocked_connect(connection: http.client.HTTPConnection) -> None:
        if connection.port == upstream_port:
            entered.set()
            release.wait(timeout=5)
        connect(connection)

    monkeypatch.setattr(http.client.HTTPConnection, "connect", blocked_connect)
    relay = LocalModelRelay(upstream["base_url"], "session")
    relay.start()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(send, relay)
        try:
            assert entered.wait(timeout=3)
            started = time.monotonic()
            relay.close()
            assert time.monotonic() - started < 2
            with pytest.raises((urllib.error.HTTPError, http.client.RemoteDisconnected)):
                pending.result(timeout=3)
            assert upstream["calls"] == []
        finally:
            release.set()
            relay.close()


def test_local_relay_client_disconnect_cancels_held_upstream(upstream: dict[str, Any]) -> None:
    upstream["disconnected"] = threading.Event()
    relay = LocalModelRelay(upstream["base_url"], "session")
    relay.start()
    parsed = urlsplit(relay.base_url)
    client = socket.create_connection((parsed.hostname, parsed.port), timeout=3)
    body = json.dumps(
        {"model": "proxy-model", "max_tokens": 8192, "stream": True, "messages": [{"role": "user", "content": "task"}]}
    ).encode()
    try:
        client.sendall(
            b"POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nx-api-key: session\r\n"
            b"Content-Type: application/json\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
        )
        assert upstream["started"].wait(timeout=3)
        # Drain the initial SSE frame before closing, then prove that the
        # independent upstream TCP connection also closes while still held.
        client.recv(4096)
        client.shutdown(socket.SHUT_RDWR)
        client.close()
        assert upstream["disconnected"].wait(timeout=2)
        with pytest.raises(ValueError, match="without a captured"):
            relay.capture.export(0)
    finally:
        client.close()
        relay.close()
