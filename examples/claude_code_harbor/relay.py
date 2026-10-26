# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Session-local Anthropic transport for the source Claude Code SRT wrapper.

The source wrapper requires a loopback model endpoint. This relay preserves the
Relax route prefix and credential, captures complete successful contexts, and
keeps streaming clients alive while a resident request is held across steps.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import hmac
import http.client
import json
import select
import socket
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from examples.claude_code_harbor.contexts import ContextCapture
from examples.claude_code_harbor.request_normalization import normalize_request
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def _terminal_sse(response: dict[str, Any]) -> bytes:
    from relax.agentic.session.service import _anthropic_input_messages, _anthropic_sse_payload
    from relax.agentic.session.state import check_messages

    message = check_messages(_anthropic_input_messages([{"role": "assistant", "content": response["content"]}], None))[
        0
    ]
    usage = response["usage"]
    result = {
        "message": message,
        "finish_reason": {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
        }[response["stop_reason"]],
        "usage": {"prompt_tokens": usage["input_tokens"], "completion_tokens": usage["output_tokens"]},
    }
    return _anthropic_sse_payload(result, {"model": response.get("model")}).encode("utf-8")


class _ModelRequest:
    def __init__(self) -> None:
        self.request_id = uuid4().hex
        self.started = time.monotonic()
        self.phase = "admission"
        self.status: int | None = None
        self.failure_logged = False
        self.future: concurrent.futures.Future[tuple[int, dict[str, Any]]] = concurrent.futures.Future()
        self.cancelled = threading.Event()
        self._socket: socket.socket | None = None
        self._lock = threading.Lock()

    def bind(self, active: socket.socket) -> None:
        with self._lock:
            if self.cancelled.is_set():
                active.close()
                raise RuntimeError("Model request cancelled")
            self._socket = active

    def cancel(self) -> None:
        self.cancelled.set()
        with self._lock:
            active = self._socket
        if active is not None:
            try:
                active.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class LocalModelRelay:
    """One owned relay per Session; close it before the adapter exits."""

    def __init__(
        self,
        base_url: str,
        session_id: str,
        *,
        heartbeat_s: float = 15.0,
        max_requests: int = 200,
        normalization_id: str | None = None,
    ) -> None:
        upstream = urlsplit(base_url.rstrip("/") + "/v1/messages")
        if upstream.scheme not in {"http", "https"} or not upstream.hostname or upstream.username:
            raise ValueError("RELAX_BASE_URL must be an HTTP(S) URL without embedded credentials")
        if not session_id or heartbeat_s <= 0:
            raise ValueError("A Session credential and positive heartbeat interval are required")
        self.capture = ContextCapture()
        self._upstream = upstream
        self._session_id = session_id
        self._normalization_id = normalization_id or session_id
        self._heartbeat_s = heartbeat_s
        if max_requests < 1:
            raise ValueError("A positive model request budget is required")
        self._max_requests = max_requests
        self._request_count = 0
        self._activity_serial = 0
        self._last_activity = 0.0
        self._tool_cleanup_until = 0.0
        self.terminal_reason: str | None = None
        self._tasks: set[_ModelRequest] = set()
        self._lock = threading.Lock()
        self._closed = False
        self._close_lock = threading.Lock()
        self._shutdown_complete = False
        self._close_task: asyncio.Task[None] | None = None
        self._harvesting = False
        self._harvest_tasks: list[_ModelRequest] = []
        relay = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                # Never log Authorization headers, request histories or URLs.
                return

            def do_POST(self) -> None:
                stream_started = False
                task: _ModelRequest | None = None
                try:
                    supplied = self.headers.get("Authorization", "")
                    if supplied.lower().startswith("bearer "):
                        supplied = supplied[7:]
                    else:
                        supplied = self.headers.get("x-api-key", "")
                    if not hmac.compare_digest(supplied, relay._session_id):
                        self._json(401, {"type": "error", "error": {"type": "authentication_error"}})
                        return
                    if urlsplit(self.path).path != "/v1/messages":
                        self._json(404, {"type": "error", "error": {"type": "not_found_error"}})
                        return
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 64 * 1024 * 1024:
                        raise ValueError("Invalid request size")
                    payload = json.loads(self.rfile.read(size))
                    if not isinstance(payload, dict):
                        raise ValueError("Expected a JSON object")
                    streaming = payload.get("stream") is True
                    forwarded = {**payload, "stream": False}
                    task = relay._submit(forwarded)
                    if streaming:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        self.close_connection = True
                        stream_started = True
                        self.wfile.write(b'event: ping\ndata: {"type":"ping"}\n\n')
                        self.wfile.flush()
                    last_heartbeat = time.monotonic()
                    # A busy Gym process can assign sockets above FD_SETSIZE;
                    # poll has no select() fd limit and needs no extra fd.
                    client_poll = select.poll()
                    client_poll.register(self.connection, select.POLLIN)
                    while True:
                        try:
                            status, response = task.future.result(timeout=min(0.1, relay._heartbeat_s))
                            break
                        except concurrent.futures.TimeoutError:
                            if task.future.done():
                                raise
                            if task.cancelled.is_set():
                                raise ConnectionResetError("Model request cancelled")
                            events = client_poll.poll(0)
                            if events:
                                if events[0][1] & (select.POLLHUP | select.POLLERR | select.POLLNVAL):
                                    raise ConnectionResetError("Model client disconnected")
                                try:
                                    if not self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT):
                                        raise ConnectionResetError("Model client disconnected")
                                except BlockingIOError:
                                    pass
                            if streaming and time.monotonic() - last_heartbeat >= relay._heartbeat_s:
                                self.wfile.write(b'event: ping\ndata: {"type":"ping"}\n\n')
                                self.wfile.flush()
                                last_heartbeat = time.monotonic()
                    if streaming:
                        if status == 200:
                            self.wfile.write(_terminal_sse(response))
                        else:
                            self.wfile.write(b"event: error\ndata: " + json.dumps(response).encode() + b"\n\n")
                        self.wfile.flush()
                    else:
                        self._json(status, response)
                except (BrokenPipeError, ConnectionResetError) as error:
                    if task is not None and not task.failure_logged:
                        relay._log_failure(error, task, "client_transport")
                    self.close_connection = True
                except Exception as error:
                    if task is None or not task.failure_logged:
                        relay._log_failure(error, task, "client_handler")
                    # Error bodies intentionally exclude exception strings that
                    # could contain a routed URL or Session credential.
                    error = {"type": "error", "error": {"type": "api_error", "message": "Local model relay failed"}}
                    if stream_started:
                        try:
                            self.wfile.write(b"event: error\ndata: " + json.dumps(error).encode() + b"\n\n")
                            self.wfile.flush()
                        except OSError:
                            pass
                    else:
                        self._json(502, error)
                finally:
                    if task is not None:
                        # Source finalize waits for an already accepted model
                        # call even after the budget has stopped the CLI.
                        with relay._lock:
                            if not relay._harvesting:
                                task.cancel()

            def _json(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                if status >= 500 or status == 404:
                    self.send_header("x-should-retry", "false")
                self.end_headers()
                self.close_connection = True
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="cc-relay", daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}"

    def start(self) -> None:
        self._thread.start()

    def _submit(self, payload: dict[str, Any]) -> _ModelRequest:
        task = _ModelRequest()
        while True:
            with self._lock:
                if self._closed or self._harvesting:
                    raise RuntimeError("Relay closed")
                if time.monotonic() >= self._tool_cleanup_until:
                    self._tasks.add(task)
                    self._activity_serial += 1
                    self._last_activity = time.monotonic()
                    break
            time.sleep(0.02)

        def run() -> None:
            try:
                if task.cancelled.is_set():
                    raise RuntimeError("Model request cancelled")
                task.future.set_result(self._complete(payload, task))
            except Exception as error:
                self._log_failure(error, task, task.phase)
                task.future.set_exception(error)
            finally:
                with self._lock:
                    self._tasks.discard(task)

        # DNS resolution cannot be interrupted portably. A Session-owned daemon
        # thread cannot hold interpreter exit; cancellation rejects a late
        # connection before it sends credentials or a model request.
        threading.Thread(target=run, name="cc-model", daemon=True).start()
        return task

    def _log_failure(self, error: Exception, task: _ModelRequest | None, phase: str) -> None:
        # Exception messages, source lines and locals can contain credentials,
        # routed URLs or model histories. Keep safe stack locations and types.
        frames = [
            {"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
            for frame in traceback.extract_tb(error.__traceback__)
        ]
        diagnostic = {
            "event": "local_model_relay_failure",
            "rollout_hash": hashlib.sha256(self._normalization_id.encode()).hexdigest(),
            "request_id": task.request_id if task else None,
            "phase": phase,
            "exception_type": type(error).__name__,
            "cause_type": type(error.__cause__).__name__ if error.__cause__ else None,
            "errno": error.errno if isinstance(error, OSError) else None,
            "elapsed_s": round(time.monotonic() - task.started, 6) if task else None,
            "upstream_status": task.status if task else None,
            "cancelled": task.cancelled.is_set() if task else False,
            "stack": frames,
        }
        if task is not None:
            task.failure_logged = True
        logger.warning("Relay diagnostic %s", json.dumps(diagnostic, sort_keys=True))

    def model_activity(self, *, quiet_since: float | None = None) -> int | None:
        with self._lock:
            if quiet_since is not None and self._last_activity > quiet_since:
                return None
            return self._activity_serial

    def begin_tool_cleanup(self, expected_activity: int, lease_s: float = 10) -> float | None:
        """Atomically hold new admissions after all accepted calls
        completed."""
        with self._lock:
            if self._closed or self._harvesting or self._tasks or self._activity_serial != expected_activity:
                return None
            self._tool_cleanup_until = time.monotonic() + lease_s
            return self._tool_cleanup_until

    def end_tool_cleanup(self) -> None:
        with self._lock:
            self._tool_cleanup_until = 0.0

    def _complete(self, payload: dict[str, Any], task: _ModelRequest) -> tuple[int, dict[str, Any]]:
        task.phase = "normalize_request"
        payload = normalize_request(payload, self._normalization_id)
        with self._lock:
            if self.terminal_reason is not None or self._request_count >= self._max_requests:
                self.terminal_reason = self.terminal_reason or "max_steps"
                return 400, {
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "Agent model request budget exhausted"},
                }
            self._request_count += 1
        upstream = self._upstream
        cls = http.client.HTTPSConnection if upstream.scheme == "https" else http.client.HTTPConnection
        connection = cls(upstream.hostname, upstream.port, timeout=10)
        active_socket: socket.socket | None = None
        try:
            task.phase = "connect"
            connection.connect()
            active_socket = connection.sock
            assert active_socket is not None
            active_socket.settimeout(None)
            task.bind(active_socket)
            task.phase = "send_request"
            connection.request(
                "POST",
                upstream.path,
                body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Authorization": f"Bearer {self._session_id}", "Content-Type": "application/json"},
            )
            task.phase = "response_headers"
            result = connection.getresponse()
            task.status = result.status
            task.phase = "response_body"
            raw = result.read()
            task.phase = "decode_json"
            response = json.loads(raw)
            if not isinstance(response, dict):
                raise ValueError("Upstream response must be an object")
            if result.status == 200:
                if task.cancelled.is_set():
                    raise RuntimeError("Model request cancelled")
                task.phase = "capture_context"
                self.capture.record(payload, response)
            elif result.status == 400:
                error = response.get("error", {})
                # Relax's Anthropic projection omits the OpenAI error code;
                # match its precise context-length wording, not all HTTP 400s.
                if isinstance(error, dict) and (
                    error.get("code") == "context_length_exceeded"
                    or str(error.get("message", "")).startswith("This model's maximum context length")
                ):
                    with self._lock:
                        self.terminal_reason = "context_overflow"
            if result.status >= 500:
                self._log_failure(RuntimeError(), task, "upstream_http_error")
            return result.status, response
        finally:
            if active_socket is not None:
                active_socket.close()
            connection.close()

    def begin_harvest(self) -> None:
        """Fence new calls before stopping the CLI; retain accepted calls."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot harvest a closed relay")
            if not self._harvesting:
                self._harvesting = True
                self._harvest_tasks = list(self._tasks)

    async def finish_harvest(self) -> None:
        """Wait for canonical responses; transport failures stay failures."""
        with self._lock:
            if not self._harvesting:
                raise RuntimeError("Harvest has not started")
            pending = list(self._harvest_tasks)
        for task in pending:
            # Do not park an executor thread on an unbounded future: DNS or
            # connect may outlive cancellation, but the Session must not.
            while not task.future.done():
                await asyncio.sleep(0.1)
            status, _ = task.future.result()
            if status != 200:
                raise RuntimeError("Accepted model call failed during timeout harvest")
        self.capture.seal()

    async def aclose(self) -> None:
        """Share an owned shutdown operation without blocking the Gym loop."""
        if self._close_task is None:
            self._close_task = asyncio.create_task(asyncio.to_thread(self.close))
        await asyncio.shield(self._close_task)

    def close(self) -> None:
        with self._close_lock:
            if self._shutdown_complete:
                return
            with self._lock:
                self._closed = True
                active = list(self._tasks)
            for pending in active:
                pending.cancel()
            self.capture.seal()
            if self._thread.is_alive():
                self._server.shutdown()
                self._thread.join(timeout=2)
            self._server.server_close()
            self._shutdown_complete = True
