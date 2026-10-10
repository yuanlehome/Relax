# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from examples.claude_code_harbor.asset_paths import asset_path
from examples.claude_code_harbor.scripts.prepare_data import image_mapping, parse_arguments, prepare


@pytest.fixture
def original(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[dict]]:
    cloud = tmp_path / "new cloud"
    source = cloud / "owner/datasets/raw"
    old = Path("/unmounted/old-cloud")
    for key, value in {
        "CLOUD_ROOT": str(cloud),
        "SOURCE_ROOT": "owner/datasets/raw",
        "DATA_ROOT": "prepared",
        "BBS_WHEEL_PATH": "wheels/bbs.whl",
        "DRESSAGE_WHEEL_PATH": "wheels/dressage.whl",
    }.items():
        monkeypatch.setenv(key, value)
    for key in ("ORIGINAL_CLOUD_ROOT", "IMAGE_MAP", "CLAUDE_CODE_DATA_ROOT"):
        monkeypatch.delenv(key, raising=False)
    for name, content in {
        "shared/mimo_git_runtime.py": "before\n",
        "shared/runtime.tgz": "payload",
        "wheels/bbs.whl": "bbs",
        "wheels/dressage.whl": "dressage",
    }.items():
        path = cloud / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    rows = []
    for i in range(2):
        relative = Path(f"owner/datasets/raw/tasks/task-{i}/tests")
        tests = cloud / relative
        tests.mkdir(parents=True)
        for name in ("test.sh", "test_patch.diff", "git_targets.json"):
            (tests / name).write_text(f"fixture-{i}-{name}")
        rows.append(
            {
                "prompt": "fixture",
                "metadata": {
                    "harbor_tests_dir": str(old / relative),
                    "sandbox_image": f"registry.example/media/task:{i}-obd",
                    "sandbox_extra_params": {
                        "nex_uploads": [
                            {"src": str(old / "shared/mimo_git_runtime.py"), "dst": "/tmp/mimo_git_runtime.py"},
                            {"src": str(old / "shared/runtime.tgz"), "dst": "/tmp/runtime.tgz"},
                            {"src": str(old / relative / "git_targets.json"), "dst": "/tmp/mimo_git_targets.json"},
                            {"src_env": "BBS_WHEEL_PATH", "dst": "/tmp/bbs.whl"},
                            {"src_env": "DRESSAGE_WHEEL_PATH", "dst": "/tmp/dressage.whl"},
                        ]
                    },
                    "backend_options": {"workdir": "/testbed"},
                },
            }
        )
    data = "".join(json.dumps(row) + "\n" for row in rows)
    (source / "data.jsonl").write_text(data)
    (source / "manifest.json").write_text(json.dumps({"data_sha256": hashlib.sha256(data.encode()).hexdigest()}))
    (cloud / "runtime.patch").write_text("--- a/runtime\n+++ b/runtime\n@@ -1 +1 @@\n-before\n+after\n")
    return cloud, rows


def prepare_fixture() -> dict:
    return prepare(
        parse_arguments(
            [
                "--runtime-patch",
                "runtime.patch",
                "--expected-rows",
                "2",
                "--expected-uploads",
                "6",
            ]
        )
    )


def test_preparation_relocates_unmounted_inputs_and_survives_another_move(
    original: tuple[Path, list[dict]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cloud, original_rows = original
    assert prepare_fixture()["status"] == "PASS"
    prepared = cloud / "prepared"
    data = (prepared / "aligned-150037-oci/data.jsonl").read_bytes()
    assert b"/unmounted" not in data and str(cloud).encode() not in data
    mapping = json.loads((prepared / "ordinary-image-map.json").read_text())
    assert mapping == {
        f"registry.example/media/task:{i}-obd": f"registry.example/media-local/task:{i}" for i in range(2)
    }
    moved = tmp_path / "another mount/prepared"
    moved.parent.mkdir()
    shutil.move(prepared, moved)
    monkeypatch.setenv("CLAUDE_CODE_DATA_ROOT", str(moved))
    monkeypatch.chdir(tmp_path)
    rows = [json.loads(line) for line in data.splitlines()]
    for before, row in zip(original_rows, rows, strict=True):
        expected = copy.deepcopy(before)
        meta = row["metadata"]
        tests = asset_path(meta["harbor_tests_dir"])
        assert tests.is_dir()
        for upload in meta["sandbox_extra_params"]["nex_uploads"]:
            assert not Path(upload["src"]).is_absolute()
            assert asset_path(upload["src"]).is_file()
        expected["metadata"]["harbor_tests_dir"] = meta["harbor_tests_dir"]
        for old, new in zip(
            expected["metadata"]["sandbox_extra_params"]["nex_uploads"],
            meta["sandbox_extra_params"]["nex_uploads"],
            strict=True,
        ):
            old.pop("src_env", None)
            old["src"] = new["src"]
        assert expected == row  # Only host paths change; dst/workdir/task content are preserved.
    assert (moved / "compat/mimo_git_runtime.py").read_text() == "after\n"
    assert (moved / "aligned-150037-oci/data.jsonl").read_bytes() == data
    manifest = json.loads((moved / "aligned-150037-oci/source_manifest.json").read_text())
    assert (
        manifest["ordinary_image_map_sha256"]
        == hashlib.sha256((moved / "ordinary-image-map.json").read_bytes()).hexdigest()
    )


def test_explicit_old_mount_handles_changed_dataset_layout(
    original: tuple[Path, list[dict]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cloud, _ = original
    shutil.move(cloud / "owner/datasets/raw", cloud / "renamed-dataset")
    monkeypatch.setenv("SOURCE_ROOT", "renamed-dataset")
    with pytest.raises(ValueError, match="ORIGINAL_CLOUD_ROOT"):
        prepare_fixture()
    monkeypatch.setenv("ORIGINAL_CLOUD_ROOT", "/unmounted/old-cloud")
    assert prepare_fixture()["status"] == "PASS"


def test_missing_original_asset_fails_without_creating_output(original: tuple[Path, list[dict]]) -> None:
    cloud, _ = original
    (cloud / "shared/runtime.tgz").unlink()
    with pytest.raises(FileNotFoundError, match="Original task upload"):
        prepare_fixture()
    assert not (cloud / "prepared").exists()


def test_preparation_rejects_symlink_escape(original: tuple[Path, list[dict]], tmp_path: Path) -> None:
    cloud, _ = original
    shutil.move(cloud / "shared", tmp_path / "outside")
    (cloud / "shared").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        prepare_fixture()


def test_unknown_image_rule_requires_explicit_mapping(tmp_path: Path) -> None:
    rows = [{"metadata": {"sandbox_image": "registry.example/custom/task:obd"}}]
    with pytest.raises(ValueError, match="explicit"):
        image_mapping(rows, None)
    supplied = tmp_path / "mapping.json"
    supplied.write_text(json.dumps({"registry.example/custom/task:obd": "registry.example/ordinary/task:1"}))
    assert image_mapping(rows, supplied)["registry.example/custom/task:obd"] == "registry.example/ordinary/task:1"
    supplied.write_text("{}")
    with pytest.raises(ValueError, match="exactly cover"):
        image_mapping(rows, supplied)


@pytest.mark.parametrize("value", ["../outside", "tests/../../outside", "/unmounted/old-cloud/secret"])
def test_runtime_paths_cannot_escape_prepared_root(tmp_path: Path, value: str) -> None:
    with pytest.raises(ValueError, match="traversal|outside"):
        asset_path(value, tmp_path)


def test_runtime_relative_paths_require_explicit_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLAUDE_CODE_DATA_ROOT", raising=False)
    with pytest.raises(ValueError, match="require"):
        asset_path("assets/file")


def test_runtime_rejects_symlink_escape(tmp_path: Path) -> None:
    (tmp_path / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        asset_path("escape/file", tmp_path)


def test_shell_roots_are_resolved_before_cwd_changes(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[3] / "examples/claude_code_harbor/scripts/cloud_paths.sh"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; harbor_resolve_paths DATA_ROOT MODEL_DIR; cd /; printf "%s\\n" "$DATA_ROOT" "$MODEL_DIR"',
            "fixture",
            str(script),
        ],
        env={
            **os.environ,
            "CLOUD_ROOT": str(tmp_path / "cloud with spaces"),
            "DATA_ROOT": "prepared",
            "MODEL_DIR": "models",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == [
        str(tmp_path / "cloud with spaces" / name) for name in ("prepared", "models")
    ]
