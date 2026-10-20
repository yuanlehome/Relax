# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Session-local Anthropic transport for the source Claude Code SRT wrapper.

The source wrapper requires a loopback model endpoint. This proxy preserves the
Relax route prefix and credential, captures complete successful contexts, and
keeps streaming clients alive while a resident request is held across steps.
"""

from __future__ import annotations

import concurrent.futures
import hmac
import http.client
import json
import select
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit


def _tool_arguments(value: Any) -> str:
    def canonical(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: canonical(part) for key, part in item.items()}
        if isinstance(item, list):
            return [canonical(part) for part in item]
        if isinstance(item, float) and item.is_integer() and abs(item) <= 2**53 - 1:
            return int(item)
        return item

    return json.dumps(canonical(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)


def _text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    return "".join(block["text"] for block in value if block.get("type") == "text")


def _chat_messages(
    raw_messages: list[dict[str, Any]], system: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    # shortcut: this MiMo code recipe is text-only; vision tasks need image block projection.
    messages = []
    if system is not None:
        text = "\n\n".join(block["text"] for block in system)
        if text:
            messages.append({"role": "system", "content": text})

    for raw in raw_messages:
        role = raw["role"]
        content = raw["content"]
        if role == "assistant":
            text_parts, thinking_parts, calls = [], [], []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block["text"])
                elif block.get("type") == "thinking" and block["thinking"]:
                    thinking_parts.append(block["thinking"])
                elif block.get("type") == "tool_use":
                    calls.append(
                        {
                            "id": block["id"],
                            "type": "function",
                            "function": {"name": block["name"], "arguments": _tool_arguments(block["input"])},
                        }
                    )
            message = {"role": "assistant", "content": "".join(text_parts)}
            if thinking_parts:
                message["reasoning_content"] = "\n\n".join(thinking_parts)
            if calls:
                message["tool_calls"] = calls
            messages.append(message)
            continue

        parts: list[str] = []

        def flush_user() -> None:
            if parts:
                messages.append({"role": "user", "content": "".join(parts)})
                parts.clear()

        for block in content:
            if block.get("type") == "text":
                parts.append(block["text"])
            elif block.get("type") == "tool_result":
                flush_user()
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": _text_content(block.get("content", "")),
                    }
                )
        flush_user()
    return messages


def canonical_context(request: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    tools = []
    for tool in request.get("tools") or []:
        function = {"name": tool["name"], "parameters": tool["input_schema"]}
        if tool.get("description") is not None:
            function["description"] = tool["description"]
        tools.append({"type": "function", "function": function})
    thinking_type = (request.get("thinking") or {}).get("type")
    return {
        "messages": [
            *_chat_messages(request["messages"], request.get("system")),
            *_chat_messages([{"role": "assistant", "content": response["content"]}]),
        ],
        "tools": tools,
        "chat_template_kwargs": (
            {"enable_thinking": thinking_type != "disabled"}
            if thinking_type in {"disabled", "enabled", "adaptive"}
            else {}
        ),
    }


def _is_prefix(parent: dict[str, Any], child: dict[str, Any]) -> bool:
    return parent["messages"] == child["messages"][: len(parent["messages"])]


class ContextCapture:
    """Keep the final complete context of each sequential CC history
    segment."""

    def __init__(self) -> None:
        self._segments: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._sealed = False

    def record(self, request: dict[str, Any], response: dict[str, Any]) -> None:
        context = canonical_context(request, response)
        with self._lock:
            if self._sealed:
                raise RuntimeError("Context capture is closed")
            # shortcut: foreground CC requests are sequential; concurrent calls need lineage capture.
            if self._segments and _is_prefix(self._segments[-1], context):
                self._segments[-1] = context
            else:
                self._segments.append(context)

    def seal(self) -> None:
        with self._lock:
            self._sealed = True

    def export(self, outcome: float) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "name": f"context_{index}",
                    **context,
                    "metadata": {"outcome_reward": outcome},
                }
                for index, context in enumerate(self._segments)
            ]


def _terminal_sse(response: dict[str, Any]) -> bytes:
    def event(name: str, payload: dict[str, Any]) -> str:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return f"event: {name}\ndata: {data}\n\n"

    usage = response["usage"]
    start = {
        "id": response["id"],
        "type": "message",
        "role": "assistant",
        "model": response.get("model"),
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": usage["input_tokens"], "output_tokens": 0},
    }
    frames = [event("message_start", {"type": "message_start", "message": start})]
    for index, block in enumerate(response["content"]):
        kind = block["type"]
        if kind == "text":
            initial = {"type": "text", "text": ""}
            deltas = [{"type": "text_delta", "text": block["text"]}]
        elif kind == "thinking":
            initial = {"type": "thinking", "thinking": "", "signature": ""}
            deltas = [
                {"type": "thinking_delta", "thinking": block["thinking"]},
                {"type": "signature_delta", "signature": block["signature"]},
            ]
        else:
            initial = {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
            deltas = [
                {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(block["input"], ensure_ascii=False, separators=(",", ":")),
                }
            ]
        frames.append(
            event("content_block_start", {"type": "content_block_start", "index": index, "content_block": initial})
        )
        frames.extend(
            event("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta})
            for delta in deltas
        )
        frames.append(event("content_block_stop", {"type": "content_block_stop", "index": index}))
    stop_reason = response["stop_reason"]
    frames.append(
        event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "end_turn" if stop_reason == "stop_sequence" else stop_reason,
                    "stop_sequence": None,
                },
                "usage": {"output_tokens": usage["output_tokens"]},
            },
        )
    )
    frames.append(event("message_stop", {"type": "message_stop"}))
    return "".join(frames).encode("utf-8")


class _ModelRequest:
    def __init__(self) -> None:
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


class LocalModelProxy:
    """One owned proxy per Session; close it before the adapter exits."""

    def __init__(self, base_url: str, session_id: str) -> None:
        upstream = urlsplit(base_url.rstrip("/") + "/v1/messages")
        self.capture = ContextCapture()
        self._upstream = upstream
        self._session_id = session_id
        self._tasks: set[_ModelRequest] = set()
        self._lock = threading.Lock()
        self._closed = False
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                stream_started = False
                task: _ModelRequest | None = None
                try:
                    supplied = self.headers.get("Authorization", "")
                    if supplied.lower().startswith("bearer "):
                        supplied = supplied[7:]
                    if not hmac.compare_digest(supplied, proxy._session_id):
                        self._json(401, {"type": "error", "error": {"type": "authentication_error"}})
                        return
                    payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    payload["stream"] = False
                    task = proxy._submit(payload)
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
                    while True:
                        try:
                            status, response = task.future.result(timeout=0.1)
                            break
                        except concurrent.futures.TimeoutError:
                            if task.cancelled.is_set():
                                raise ConnectionResetError("Model request cancelled")
                            readable, _, _ = select.select([self.connection], [], [], 0)
                            if readable and not self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT):
                                raise ConnectionResetError("Model client disconnected")
                            if time.monotonic() - last_heartbeat >= 15.0:
                                self.wfile.write(b'event: ping\ndata: {"type":"ping"}\n\n')
                                self.wfile.flush()
                                last_heartbeat = time.monotonic()
                    if status == 200:
                        self.wfile.write(_terminal_sse(response))
                    else:
                        self.wfile.write(b"event: error\ndata: " + json.dumps(response).encode() + b"\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True
                except Exception:
                    # Error bodies intentionally exclude exception strings that
                    # could contain a routed URL or Session credential.
                    error = {"type": "error", "error": {"type": "api_error", "message": "Local model proxy failed"}}
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
                        task.cancel()

            def _json(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="cc-proxy", daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}"

    def start(self) -> None:
        self._thread.start()

    def _submit(self, payload: dict[str, Any]) -> _ModelRequest:
        task = _ModelRequest()
        with self._lock:
            if self._closed:
                raise RuntimeError("Proxy closed")
            self._tasks.add(task)

        def run() -> None:
            try:
                if task.cancelled.is_set():
                    raise RuntimeError("Model request cancelled")
                task.future.set_result(self._complete(payload, task))
            except Exception as error:
                task.future.set_exception(error)
            finally:
                with self._lock:
                    self._tasks.discard(task)

        # A Session-owned daemon thread keeps interpreter exit independent from
        # DNS resolution; cancellation fences late connections before they send
        # credentials or a model request.
        threading.Thread(target=run, name="cc-model", daemon=True).start()
        return task

    def _complete(self, payload: dict[str, Any], task: _ModelRequest) -> tuple[int, dict[str, Any]]:
        upstream = self._upstream
        connection = http.client.HTTPConnection(upstream.hostname, upstream.port, timeout=10)
        active_socket: socket.socket | None = None
        try:
            connection.connect()
            active_socket = connection.sock
            active_socket.settimeout(None)
            task.bind(active_socket)
            connection.request(
                "POST",
                upstream.path,
                body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Authorization": f"Bearer {self._session_id}", "Content-Type": "application/json"},
            )
            result = connection.getresponse()
            response = json.loads(result.read())
            if result.status == 200:
                if task.cancelled.is_set():
                    raise RuntimeError("Model request cancelled")
                self.capture.record(payload, response)
            return result.status, response
        finally:
            if active_socket is not None:
                active_socket.close()
            connection.close()

    def close(self) -> None:
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
