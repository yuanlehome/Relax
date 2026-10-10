# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
from pathlib import Path

import pytest

from examples.claude_code_harbor.scripts import fetch_oci_packages as packages


PAYLOAD = b"a pinned installation archive"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


def test_download_publishes_only_verified_package(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(packages.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(PAYLOAD))
    path = tmp_path / "package.tgz"
    packages.fetch(path, "https://packages.example/archive", len(PAYLOAD), DIGEST)
    assert path.read_bytes() == PAYLOAD
    assert list(tmp_path.iterdir()) == [path]


def test_verified_cache_never_accesses_network(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("A cached package must not be downloaded")

    monkeypatch.setattr(packages.urllib.request, "urlopen", unexpected)
    path = tmp_path / "package.tgz"
    path.write_bytes(PAYLOAD)
    packages.fetch(path, "https://packages.example/archive", len(PAYLOAD), DIGEST)


@pytest.mark.parametrize("payload", [b"truncated", b"x" * len(PAYLOAD), PAYLOAD + b"extra"])
def test_bad_download_is_not_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, payload: bytes) -> None:
    monkeypatch.setattr(packages.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(payload))
    with pytest.raises(ValueError):
        packages.fetch(tmp_path / "package.tgz", "https://packages.example/archive", len(PAYLOAD), DIGEST)
    assert not list(tmp_path.iterdir())


def test_corrupt_existing_cache_is_not_replaced(tmp_path: Path) -> None:
    path = tmp_path / "package.tgz"
    path.write_bytes(b"x" * len(PAYLOAD))
    with pytest.raises(ValueError, match="checksum mismatch"):
        packages.fetch(path, "https://packages.example/archive", len(PAYLOAD), DIGEST)
    assert path.read_bytes() == b"x" * len(PAYLOAD)


def test_failed_download_retries_without_leaking_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = []

    def unavailable(url: str, **kwargs: object) -> None:
        calls.append(url)
        raise urllib.error.URLError(url)

    monkeypatch.setattr(packages.urllib.request, "urlopen", unavailable)
    monkeypatch.setattr(packages.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError) as raised:
        packages.fetch(tmp_path / "package.tgz", "https://private.example/archive", len(PAYLOAD), DIGEST)
    assert "private.example" not in str(raised.value)
    assert len(calls) == 3
    assert not list(tmp_path.iterdir())


def test_offline_mode_does_not_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("Offline mode must not download missing packages")

    monkeypatch.setattr(packages, "fetch", unexpected)
    with pytest.raises(ValueError, match="Missing"):
        packages.prepare(tmp_path, check_only=True)
    assert not list(tmp_path.iterdir())


def test_cache_symlink_is_not_followed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"sources": {}, "packages": [{"path": "elsewhere/package.tgz", "bytes": 1, "sha256": ""}]})
    )
    monkeypatch.setattr(packages, "MANIFEST", manifest)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "elsewhere").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        packages.prepare(cache)


def test_concurrent_verified_download_does_not_replace_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "package.tgz"

    def concurrent(*args: object, **kwargs: object) -> io.BytesIO:
        path.write_bytes(PAYLOAD)
        return io.BytesIO(PAYLOAD)

    monkeypatch.setattr(packages.urllib.request, "urlopen", concurrent)
    packages.fetch(path, "https://packages.example/archive", len(PAYLOAD), DIGEST)
    assert path.read_bytes() == PAYLOAD
    assert list(tmp_path.iterdir()) == [path]
