# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build safety and isolation checks; the real installer is tested on Linux."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from examples.claude_code_harbor.scripts import build_gym_environment as build


@pytest.fixture
def image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sys, "version_info", (3, 12, 3))
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)


def test_existing_environment_is_never_overwritten(image: None, tmp_path: Path) -> None:
    prefix = tmp_path / "live"
    prefix.mkdir()
    marker = prefix / "important"
    marker.write_text("running environment")
    with pytest.raises(FileExistsError):
        build.prepare(prefix, build.REPO_ROOT)
    assert list(prefix.iterdir()) == [marker]
    assert marker.read_text() == "running environment"


def test_base_mismatch_fails_before_fetch_or_install(
    image: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(build, "packages", lambda *a, **k: {"ray": "wrong"})
    commands = []
    monkeypatch.setattr(build, "run", lambda command, **kw: commands.append(command))
    prefix = tmp_path / "new"
    with pytest.raises(RuntimeError, match="Incompatible package"):
        build.prepare(prefix, build.REPO_ROOT)
    assert not commands
    assert not (prefix / "build.json").exists()


def test_builder_rejects_an_activated_venv(image: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "prefix", "/some/venv")
    with pytest.raises(RuntimeError, match="system Python"):
        build.prepare(tmp_path / "new", build.REPO_ROOT)
    assert not (tmp_path / "new").exists()


def test_install_environment_cannot_redirect_pip(monkeypatch: pytest.MonkeyPatch) -> None:
    unsafe = (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "PIP_TARGET",
        "PIP_PREFIX",
        "PIP_USER",
        "PIP_PYTHON",
        "PIP_ROOT",
        "PIP_REQUIREMENT",
        "PIP_CONSTRAINT",
        "PIP_EDITABLE",
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_CONFIG_COUNT",
    )
    for name in unsafe:
        monkeypatch.setenv(name, "/foreign/location")
    monkeypatch.setenv("PIP_INDEX_URL", "https://packages.example/simple")
    env = build.clean_environment()
    assert all(name not in env for name in unsafe)
    assert env["PIP_INDEX_URL"] == "https://packages.example/simple"
    assert env["PIP_REQUIRE_VIRTUALENV"] == "true"
    assert env["PIP_CONFIG_FILE"] == build.os.devnull


def test_check_rejects_source_drift_without_reinstalling(
    image: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prefix = tmp_path / "built"
    prefix.mkdir()
    requirements = build.SCRIPT_DIR / "gym-requirements.txt"
    base = build.SCRIPT_DIR / "gym-base-requirements.txt"
    (prefix / "build.json").write_text(
        json.dumps(
            {
                "nemo_commit": build.NEMO_COMMIT,
                "requirements_sha256": hashlib.sha256(requirements.read_bytes() + base.read_bytes()).hexdigest(),
            }
        )
    )
    monkeypatch.setattr(build, "packages", lambda *a, **k: build.read_pins(base))
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> str:
        commands.append(command)
        return build.NEMO_COMMIT if command[1] == "rev-parse" else " M nemo_gym/server_utils.py"

    monkeypatch.setattr(build, "run", run)
    with pytest.raises(RuntimeError, match="unmodified pinned commit"):
        build.prepare(prefix, build.REPO_ROOT, check=True)
    assert all(command[0] == "git" for command in commands)


def test_version_contract_separates_image_and_overlay() -> None:
    base = build.read_pins(build.SCRIPT_DIR / "gym-base-requirements.txt")
    overlay = build.read_pins(build.SCRIPT_DIR / "gym-requirements.txt")
    assert not set(base).intersection(overlay)
    assert {"torch", "ray", "openai", "anthropic"} <= set(base)
    assert not {"torch", "ray", "sglang", "triton", "nemo-gym"}.intersection(overlay)
