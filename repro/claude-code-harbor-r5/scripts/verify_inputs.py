# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Check freshly prepared task assets and model files before Ray submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def contained_file(root: Path, relative: str) -> Path:
    path = root / relative
    if (
        ".." in Path(relative).parts
        or not path.resolve().is_relative_to(root.resolve())
        or path.is_symlink()
        or not path.is_file()
    ):
        raise ValueError("Required input is absent, a symlink, or outside its configured root")
    return path


def verify(root: Path, model: Path, *, expected_rows: int = 2683) -> dict[str, str | int]:
    records = json.loads((root / "file-manifest.json").read_text())["records"]
    indexed = {record["path"]: record for record in records}
    for record in records:
        path = contained_file(root, record["path"])
        if path.stat().st_size != record["bytes"]:
            raise ValueError("Prepared asset size differs; re-run data preparation at a fresh path")
    # Protect the routing and compatibility code as well as file presence. Full
    # archive checksum verification remains available separately.
    for relative in ("aligned-150037-oci/data.jsonl", "compat/mimo_git_runtime.py", "ordinary-image-map.json"):
        path = contained_file(root, relative)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != indexed[relative]["sha256"]:
            raise ValueError("Prepared JSONL or Git runtime changed; re-run data preparation")
    image_map = json.loads(contained_file(root, "ordinary-image-map.json").read_text())
    count = 0
    with (root / "aligned-150037-oci/data.jsonl").open() as source:
        for line in source:
            if not line.strip():
                continue
            meta = json.loads(line)["metadata"]
            image = image_map.get(meta["sandbox_image"])
            if not isinstance(image, str) or not image or image.endswith("-obd"):
                raise ValueError("Missing ordinary OCI image mapping")
            for upload in meta["sandbox_extra_params"]["nex_uploads"]:
                path = Path(upload["src"])
                contained_file(root, str(path))
            for name in ("test.sh", "test_patch.diff", "git_targets.json"):
                contained_file(root, str(Path(meta["harbor_tests_dir"]) / name))
            count += 1
    expected = json.loads((root / "aligned-150037-oci/source_manifest.json").read_text())["rows"]
    if count != expected or count != expected_rows:
        raise ValueError("Prepared task count differs from its source manifest")
    for name in ("config.json", "tokenizer.json"):
        if not contained_file(model, name).stat().st_size:
            raise ValueError("Model configuration or tokenizer is empty")
    shards = set(json.loads(contained_file(model, "model.safetensors.index.json").read_text())["weight_map"].values())
    if not shards or any(not contained_file(model, name).stat().st_size for name in shards):
        raise ValueError("Model shard set is incomplete")
    return {"status": "PASS", "rows": count, "asset_files": len(records), "model_shards": len(shards)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    cloud = Path(os.environ.get("CLOUD_ROOT", "."))
    print(json.dumps(verify((cloud / args.data_root).resolve(), (cloud / args.model_dir).resolve())))
