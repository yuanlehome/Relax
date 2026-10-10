# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "verify_inputs", ROOT / "repro/claude-code-harbor-r5/scripts/verify_inputs.py"
)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path]:
    root, model = tmp_path / "data", tmp_path / "model"
    for name in ("aligned-150037-oci", "compat", "tests"):
        (root / name).mkdir(parents=True)
    model.mkdir()
    for name in ("test.sh", "test_patch.diff", "git_targets.json"):
        (root / "tests" / name).write_text("fixture")
    (root / "compat/mimo_git_runtime.py").write_text("# fixture\n")
    row = {
        "metadata": {
            "sandbox_image": "source-image",
            "harbor_tests_dir": str(root / "tests"),
            "sandbox_extra_params": {"nex_uploads": [{"src": str(root / "compat/mimo_git_runtime.py")}]},
        }
    }
    (root / "aligned-150037-oci/data.jsonl").write_text(json.dumps(row) + "\n")
    (root / "aligned-150037-oci/source_manifest.json").write_text(json.dumps({"rows": 1}))
    (root / "ordinary-image-map.json").write_text(json.dumps({"source-image": "registry.example/task:version"}))
    records = [
        {
            "path": str(p.relative_to(root)),
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in root.rglob("*")
        if p.is_file()
    ]
    (root / "file-manifest.json").write_text(json.dumps({"records": records}))
    for name in ("config.json", "tokenizer.json", "weights.safetensors"):
        (model / name).write_text("fixture")
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "weights.safetensors"}}))
    return root, model


def test_prepared_inputs_need_no_historical_environment_bundle(inputs: tuple[Path, Path]) -> None:
    result = checker.verify(*inputs, expected_rows=1)
    assert result["status"] == "PASS" and result["model_shards"] == 1


def test_relative_prepared_inputs_verify_after_mount_change(inputs: tuple[Path, Path], tmp_path: Path) -> None:
    root, model = inputs
    path = root / "aligned-150037-oci/data.jsonl"
    row = json.loads(path.read_text())
    row["metadata"]["harbor_tests_dir"] = "tests"
    row["metadata"]["sandbox_extra_params"]["nex_uploads"][0]["src"] = "compat/mimo_git_runtime.py"
    path.write_text(json.dumps(row) + "\n")
    manifest = json.loads((root / "file-manifest.json").read_text())
    record = next(r for r in manifest["records"] if r["path"] == "aligned-150037-oci/data.jsonl")
    record.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (root / "file-manifest.json").write_text(json.dumps(manifest))
    moved = tmp_path / "different-mount"
    shutil.move(root, moved)
    assert checker.verify(moved, model, expected_rows=1)["status"] == "PASS"


@pytest.mark.parametrize("missing", ["tests/test.sh", "compat/mimo_git_runtime.py"])
def test_missing_task_asset_fails_before_submission(inputs: tuple[Path, Path], missing: str) -> None:
    root, model = inputs
    (root / missing).unlink()
    with pytest.raises(ValueError, match="Required input"):
        checker.verify(root, model, expected_rows=1)


def test_same_size_runtime_change_is_detected(inputs: tuple[Path, Path]) -> None:
    root, model = inputs
    path = root / "compat/mimo_git_runtime.py"
    path.write_text(path.read_text().replace("fixture", "changed"))
    with pytest.raises(ValueError, match="Git runtime changed"):
        checker.verify(root, model, expected_rows=1)


def test_missing_model_shard_is_detected(inputs: tuple[Path, Path]) -> None:
    root, model = inputs
    (model / "weights.safetensors").unlink()
    with pytest.raises(ValueError, match="Required input"):
        checker.verify(root, model, expected_rows=1)


def test_model_shard_cannot_escape_model_root(inputs: tuple[Path, Path]) -> None:
    root, model = inputs
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "../data/file-manifest.json"}})
    )
    with pytest.raises(ValueError, match="outside"):
        checker.verify(root, model, expected_rows=1)
