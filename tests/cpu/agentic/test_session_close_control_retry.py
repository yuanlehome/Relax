# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Actual adapter + HTTP helper, with only the network transport controlled."""

import asyncio
from argparse import Namespace
from collections import Counter

import httpx
import pytest


SID = "agentic_session_shard_11.session-control-test"
SECRET = "NEVER_LOG_RESPONSE_BODY_TOKEN"


@pytest.fixture
def runtime(monkeypatch):
    import relax.agentic.pipeline.runtime as module

    lines = []

    class Logger:
        def info(self, message, *args, **kwargs):
            lines.append(message % args if args else str(message))

        warning = info

    monkeypatch.setattr(module, "logger", Logger())
    import relax.utils.http_utils as http_utils

    monkeypatch.setattr(http_utils, "logger", Logger())
    return module, lines


def make_backend(module, client, targets=None):
    backend = module.SGLangBackendAdapter.__new__(module.SGLangBackendAdapter)
    backend._session_lifecycle = True
    backend._session_endpoints = {} if targets is None else {SID: set(targets)}
    backend._managed_http = client
    backend._args = Namespace(sglang_router_ip="router", sglang_router_port=80)
    backend._requests = {"keep": object()}
    return backend


def test_all_success_deduplicates_workers_and_preserves_body_semantics(runtime):
    module, lines = runtime

    async def run():
        calls = Counter()

        async def transport(request):
            calls[(request.method, request.url.host)] += 1
            if request.method == "GET":
                return httpx.Response(
                    200,
                    json={"workers": [{"url": f"http://{host}:8000@{dp}"} for host in ("a", "b") for dp in (0, 1)]},
                )
            if request.url.host == "a":
                return httpx.Response(200, json={"success": False, "body": SECRET})
            return httpx.Response(200, content=b"")

        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            backend = make_backend(module, client)
            assert await backend.close_session(SID, timeout_s=0.1) is True
            assert calls == {("GET", "router"): 1, ("POST", "a"): 1, ("POST", "b"): 1}
            assert SID not in backend._session_endpoints
            assert list(backend._requests) == ["keep"]
            assert await backend.close_session(SID, timeout_s=0.1) is True
            assert calls == {("GET", "router"): 2, ("POST", "a"): 2, ("POST", "b"): 2}

    asyncio.run(run())
    assert SECRET not in "\n".join(lines)


def test_partial_ack_survives_timeout_and_only_pending_target_retries(runtime):
    module, lines = runtime

    async def run():
        calls = Counter()
        cancelled = []
        never = asyncio.Event()

        async def transport(request):
            host = request.url.host
            calls[host] += 1
            if host == "b" and calls[host] == 1:
                try:
                    await never.wait()
                finally:
                    cancelled.append(host)
            if host == "c" and calls[host] == 1:
                return httpx.Response(503, json={"error": SECRET})
            return httpx.Response(200, json={"body": SECRET})

        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            backend = make_backend(module, client, ["http://a:8000", "http://b:8000", "http://c:8000"])
            assert await backend.close_session(SID, timeout_s=0.03) is True
            assert calls == {"a": 1, "b": 2, "c": 2}
            assert cancelled == ["b"]
            assert SID not in backend._session_endpoints

    asyncio.run(run())
    text = "\n".join(lines)
    assert "http_success" in text and "TimeoutError" in text and "recovered" in text
    assert "'http_status': 503" in text
    assert SECRET not in text


def test_all_http_failures_are_bounded_and_safe(runtime):
    module, lines = runtime

    async def run():
        calls = Counter()

        async def transport(request):
            calls[request.url.host] += 1
            return httpx.Response(503, json={"error": SECRET})

        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            backend = make_backend(module, client, ["http://a:8000", "http://b:8000"])
            assert await backend.close_session(SID, timeout_s=0.1) is False
            assert calls == {"a": 3, "b": 3}
            assert SID not in backend._session_endpoints
            assert list(backend._requests) == ["keep"]

    asyncio.run(run())
    text = "\n".join(lines)
    assert "'http_status': 503" in text and "HTTPStatusError" in text and "attempt=3/3" in text
    assert SECRET not in text


def test_external_cancellation_awaits_all_children_without_retry(runtime):
    module, lines = runtime

    async def run():
        started = asyncio.Event()
        active, finished, calls = set(), set(), Counter()
        never = asyncio.Event()

        async def transport(request):
            host = request.url.host
            active.add(host)
            calls[host] += 1
            if len(active) == 2:
                started.set()
            try:
                await never.wait()
            finally:
                await asyncio.sleep(0)
                active.remove(host)
                finished.add(host)

        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            backend = make_backend(module, client, ["http://a:8000", "http://b:8000"])
            task = asyncio.create_task(backend.close_session(SID, timeout_s=10))
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not active and finished == {"a", "b"}
            assert calls == {"a": 1, "b": 1}
            assert SID not in backend._session_endpoints

    asyncio.run(run())


def test_discovery_failure_retries_without_changing_public_result(runtime):
    module, lines = runtime

    async def run():
        calls = Counter()

        async def transport(request):
            calls[request.url.path] += 1
            if request.url.path == "/workers":
                if calls[request.url.path] == 1:
                    return httpx.Response(404, json={"error": SECRET})
                return httpx.Response(200, json={"workers": [{"url": "http://a:8000@0"}]})
            if request.url.path == "/list_workers":
                return httpx.Response(404, json={"error": SECRET})
            return httpx.Response(200)

        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            backend = make_backend(module, client)
            assert await backend.close_session(SID, timeout_s=0.1) is True
            assert calls == {"/workers": 2, "/list_workers": 1, "/close_session": 1}

    asyncio.run(run())
    text = "\n".join(lines)
    assert "stage=discovery" in text and "HTTPStatusError" in text
    assert SECRET not in text


def test_discovery_deadline_and_empty_targets_stay_false(runtime):
    module, lines = runtime

    async def run():
        for empty in (False, True):
            calls = []
            finished = []

            async def transport(request):
                calls.append(request.url.path)
                if empty:
                    return httpx.Response(200, json={"workers": []})
                try:
                    await asyncio.Event().wait()
                finally:
                    finished.append(request.url.path)

            async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
                backend = make_backend(module, client)
                assert await backend.close_session(SID, timeout_s=0.01) is False
                assert calls == ["/workers"] * 3
                if not empty:
                    assert finished == calls
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: pytest.fail("unexpected HTTP"))
        ) as client:
            backend = make_backend(module, client, ["http://a:8000"])
            backend._session_lifecycle = False
            assert await backend.close_session(SID, timeout_s=0.01) is True
            assert SID not in backend._session_endpoints

    asyncio.run(run())
    text = "\n".join(lines)
    assert "stage=discovery" in text and "TimeoutError" in text and "RuntimeError" in text
