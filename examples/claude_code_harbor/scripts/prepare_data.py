# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Prepare Harbor training inputs directly from the original Dressage
dataset."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def apply_exact_patch(text: str, patch: str) -> str:
    """Apply the checked-in unified diff without fuzzy context matching."""
    original = text.splitlines(keepends=True)
    changes = patch.splitlines(keepends=True)
    result: list[str] = []
    cursor = 0
    index = 0
    hunks = 0
    while index < len(changes):
        match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", changes[index])
        if not match:
            index += 1
            continue
        start = int(match[1]) - 1
        old_count, new_count = int(match[2] or 1), int(match[4] or 1)
        if start < cursor:
            raise ValueError("Overlapping runtime patch hunks")
        result.extend(original[cursor:start])
        cursor = start
        consumed = produced = 0
        index += 1
        while index < len(changes) and not changes[index].startswith(("@@ ", "--- ", "+++ ")):
            line = changes[index]
            if line[:1] in (" ", "-"):
                if cursor >= len(original) or original[cursor] != line[1:]:
                    raise ValueError("Runtime patch context differs from the original 150037 runtime")
                cursor += 1
                consumed += 1
            if line[:1] in (" ", "+"):
                result.append(line[1:])
                produced += 1
            if line[:1] not in (" ", "+", "-"):
                raise ValueError("Unsupported runtime patch line")
            index += 1
        if (consumed, produced) != (old_count, new_count):
            raise ValueError("Runtime patch hunk counts differ")
        hunks += 1
    if not hunks:
        raise ValueError("Runtime patch contains no hunks")
    result.extend(original[cursor:])
    return "".join(result)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(arguments: argparse.Namespace) -> dict[str, int | str]:
    source = arguments.source_root.resolve()
    destination = arguments.output_root.resolve()
    published = (arguments.published_root or destination).resolve()
    if destination.exists():
        raise FileExistsError("Output exists; choose a new output directory to preserve existing inputs")
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Source and output directories must be separate")
    source_jsonl = source / "data.jsonl"
    rows = [json.loads(line) for line in source_jsonl.open(encoding="utf-8") if line.strip()]
    source_manifest = json.loads((source / "manifest.json").read_text())
    source_digest = sha256(source_jsonl)
    if source_manifest.get("data_sha256") != source_digest or len(rows) != arguments.expected_rows:
        raise ValueError("Original dataset digest or row count differs from its manifest")
    open_map = gzip.open if arguments.image_map.suffix == ".gz" else open
    with open_map(arguments.image_map, "rt", encoding="utf-8") as stream:
        image_map = json.load(stream)
    if set(image_map) != {row["metadata"]["sandbox_image"] for row in rows}:
        raise ValueError("OCI image map does not exactly cover the original task images")
    if any(not isinstance(value, str) or not value or value.endswith("-obd") for value in image_map.values()):
        raise ValueError("An ordinary OCI image reference is missing")
    env_paths = {
        "BBS_WHEEL_PATH": arguments.bbs_wheel.resolve(),
        "DRESSAGE_WHEEL_PATH": arguments.dressage_wheel.resolve(),
    }
    copies: dict[Path, Path] = {}
    rendered = copy.deepcopy(rows)
    tests: set[Path] = set()
    uploads: set[Path] = set()
    runtimes: set[Path] = set()

    def asset(path: Path) -> Path:
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError("Original task upload or verifier file is missing or is a symlink")
        path = path.resolve(strict=True)
        if path.is_relative_to(source / "tasks"):
            relative = Path("source/tasks") / path.relative_to(source / "tasks")
        elif path in env_paths.values():
            relative = Path("wheels") / path.name
        else:
            relative = Path("assets") / hashlib.sha256(str(path).encode()).hexdigest()[:16] / path.name
        copies[path] = relative
        return relative

    for row in rendered:
        metadata = row["metadata"]
        test_dir = Path(metadata["harbor_tests_dir"]).resolve(strict=True)
        if not test_dir.is_relative_to(source / "tasks"):
            raise ValueError("Verifier directory is outside the original dataset tasks")
        for name in ("test.sh", "test_patch.diff", "git_targets.json"):
            asset(test_dir / name)
        tests.add(test_dir)
        metadata["harbor_tests_dir"] = str(published / "source/tasks" / test_dir.relative_to(source / "tasks"))
        for upload in metadata["sandbox_extra_params"]["nex_uploads"]:
            if "src_env" in upload:
                name = upload.pop("src_env")
                if name not in env_paths or upload.get("src"):
                    raise ValueError("Unrecognized or ambiguous upload environment reference")
                path = env_paths[name]
            else:
                path = Path(upload["src"]).absolute()
            relative = asset(path)
            uploads.add(path)
            if upload["dst"] == "/tmp/mimo_git_runtime.py":
                runtimes.add(path)
                relative = Path("compat/mimo_git_runtime.py")
            upload["src"] = str(published / relative)
    if (len(uploads), len(tests), len(runtimes)) != (2708, arguments.expected_rows, 1):
        raise ValueError("Original task upload, verifier or runtime counts differ from r5")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + ".preparing-", dir=destination.parent))
    try:

        def copy_file(item: tuple[Path, Path]) -> dict[str, int | str]:
            original, relative = item
            target = staging / relative
            if not target.resolve().is_relative_to(staging):
                raise ValueError("An asset copy would escape the preparation directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
            return {
                "source": str(original),
                "path": str(relative),
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            }

        with ThreadPoolExecutor(max_workers=16) as pool:
            source_files = list(pool.map(copy_file, copies.items()))
        shutil.copy2(source_jsonl, staging / "source/data.jsonl")
        shutil.copy2(source / "manifest.json", staging / "source/manifest.json")
        runtime = next(iter(runtimes))
        (staging / "compat").mkdir()
        patched = apply_exact_patch(runtime.read_text(), arguments.runtime_patch.read_text())
        (staging / "compat/mimo_git_runtime.py").write_text(patched)
        shutil.copymode(runtime, staging / "compat/mimo_git_runtime.py")
        aligned = staging / "aligned-150037-oci"
        aligned.mkdir()
        shutil.copy2(arguments.runtime_patch, aligned / "mimo_git_reflog.patch")
        write_json(staging / "ordinary-image-map.json", image_map)
        for path, values in ((aligned / "data.jsonl", rendered),):
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values))
        portable = copy.deepcopy(rendered)
        for row in portable:
            meta = row["metadata"]
            meta["harbor_tests_dir"] = str(Path(meta["harbor_tests_dir"]).relative_to(published))
            for upload in meta["sandbox_extra_params"]["nex_uploads"]:
                upload["src"] = str(Path(upload["src"]).relative_to(published))
        (staging / "tasks.r5.relative.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in portable)
        )
        write_json(
            aligned / "source_manifest.json",
            {
                "source_dataset": str(source),
                "source_data_sha256": source_digest,
                "prepared_root": str(published),
                "rows": len(rows),
                "unique_uploads": len(uploads),
                "test_dirs": len(tests),
                "source_runtime_sha256": sha256(runtime),
                "patched_runtime_sha256": sha256(staging / "compat/mimo_git_runtime.py"),
                "runtime_patch_sha256": sha256(arguments.runtime_patch),
                "ordinary_image_map_sha256": sha256(arguments.image_map),
                "changes": [
                    "host paths relocated",
                    "two src_env wheel paths materialized",
                    "only the Git runtime upload replaced with the exact compatibility patch",
                ],
                "source_files": source_files,
            },
        )
        records = [
            {"path": str(path.relative_to(staging)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        ]
        write_json(staging / "file-manifest.json", {"status": "PASS", "records": records})
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return {
        "status": "PASS",
        "rows": len(rows),
        "unique_uploads": len(uploads),
        "test_dirs": len(tests),
        "files": len(records),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--published-root", type=Path, help="Final cloud path when staging on a transfer host")
    parser.add_argument("--bbs-wheel", type=Path, required=True)
    parser.add_argument("--dressage-wheel", type=Path, required=True)
    parser.add_argument("--image-map", type=Path, required=True, help="Verified ordinary OCI reference map")
    parser.add_argument(
        "--runtime-patch", type=Path, default=Path(__file__).resolve().parents[1] / "mimo_git_reflog.patch"
    )
    parser.add_argument("--expected-rows", type=int, default=2683)
    print(json.dumps(prepare(parser.parse_args())))
