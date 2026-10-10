# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Verify persisted r5 inputs, model shard completeness and file checksums."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def read_metadata(bundle: Path, name: str) -> dict:
    compressed = bundle / (name + ".gz")
    if compressed.exists():
        with gzip.open(compressed, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads((bundle / name).read_text())


def verify(bundle: Path, root: Path, model: Path, full: bool) -> dict[str, int | str]:
    manifest = read_metadata(bundle, "bundle-manifest.json")
    roots = {"bundle": bundle, "repo": bundle.parents[1], "data": root, "model": model}

    def check_file(record: dict[str, str | int]) -> int:
        scope = roots[record["root"]]
        path = scope / record["path"]
        if not path.resolve().is_relative_to(scope) or not path.is_file() or path.is_symlink():
            raise ValueError("A recorded file is absent or points outside its configured root")
        if record["root"] == "data" and record["path"] == "aligned-150037-oci/data.jsonl":
            # This file is intentionally regenerated if the cloud mount changes;
            # its JSON values are checked against the preserved template below.
            return 1
        if path.stat().st_size != record["bytes"]:
            raise ValueError("A persisted file has changed size")
        if full:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != record["sha256"]:
                raise ValueError("A persisted file checksum differs")
        return 1

    data_manifest = json.loads((root / "file-manifest.json").read_text())
    records = manifest["records"] + [{**record, "root": "data"} for record in data_manifest["records"]]
    with ThreadPoolExecutor(max_workers=16) as pool:
        checked = sum(pool.map(check_file, records))
    rows = [json.loads(line) for line in (root / "aligned-150037-oci/data.jsonl").open() if line.strip()]
    expected = copy.deepcopy([json.loads(line) for line in (root / "tasks.r5.relative.jsonl").open() if line.strip()])
    for row in [*rows, *expected]:
        meta = row["metadata"]
        meta["harbor_tests_dir"] = str((root / meta["harbor_tests_dir"]).resolve())
        for upload in meta["sandbox_extra_params"]["nex_uploads"]:
            upload["src"] = str((root / upload["src"]).resolve())
    if rows != expected:
        raise ValueError("Task content differs from the preserved r5 template")
    original_path = root / "source/data.jsonl"
    if hashlib.sha256(original_path.read_bytes()).hexdigest() != manifest["data_reference"]["source_data_sha256"]:
        raise ValueError("Original input differs from the recorded 150037 dataset")
    original = [json.loads(line) for line in original_path.open() if line.strip()]
    for before, after in zip(original, copy.deepcopy(rows), strict=True):
        before_meta, after_meta = before["metadata"], after["metadata"]
        before_meta["harbor_tests_dir"] = after_meta["harbor_tests_dir"] = "<relocated>"
        for source_upload, target_upload in zip(
            before_meta["sandbox_extra_params"]["nex_uploads"],
            after_meta["sandbox_extra_params"]["nex_uploads"],
            strict=True,
        ):
            source_upload.pop("src_env", None)
            source_upload["src"] = target_upload["src"] = "<relocated>"
        if before != after:
            raise ValueError("Conversion changed a value outside host paths and wheel environment references")
    if json.loads((root / "ordinary-image-map.json").read_text()) != read_metadata(
        bundle, "input-config/ordinary-image-map.json"
    ):
        raise ValueError("OCI reference map differs from the recorded map")
    uploads, tests = set(), set()
    for row in rows:
        meta = row["metadata"]
        test = Path(meta["harbor_tests_dir"])
        if not test.resolve().is_relative_to(root):
            raise ValueError("A test directory points outside the prepared cloud data")
        tests.add(test)
        for name in ("test.sh", "test_patch.diff", "git_targets.json"):
            if not (test / name).is_file():
                raise FileNotFoundError("Verifier input missing")
        for upload in meta["sandbox_extra_params"]["nex_uploads"]:
            path = Path(upload["src"])
            if not path.resolve().is_relative_to(root) or not path.is_file() or path.is_symlink():
                raise ValueError("An upload points outside the prepared cloud data")
            uploads.add(path)
    if (len(rows), len(uploads), len(tests)) != (2683, 2708, 2683):
        raise ValueError("Task/asset counts differ from r5")
    shards = set(json.loads((model / "model.safetensors.index.json").read_text())["weight_map"].values())
    if len(shards) != 26 or any(not (model / name).is_file() for name in shards):
        raise ValueError("Initial model shards are incomplete")
    return {
        "status": "PASS",
        "files": checked,
        "checksum_mode": "sha256" if full else "size",
        "rows": len(rows),
        "unique_uploads": len(uploads),
        "test_dirs": len(tests),
        "model_shards": len(shards),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root", type=Path, required=True, help="Prepared cloud task and asset root")
    parser.add_argument("--model-dir", type=Path, required=True, help="HF model directory, including model name")
    parser.add_argument("--full", action="store_true", help="Read all persisted files and verify SHA-256")
    arguments = parser.parse_args()
    cloud = Path(os.environ.get("CLOUD_ROOT", "."))
    print(
        json.dumps(
            verify(
                (cloud / arguments.bundle).resolve(),
                (cloud / arguments.data_root).resolve(),
                (cloud / arguments.model_dir).resolve(),
                arguments.full,
            )
        )
    )
