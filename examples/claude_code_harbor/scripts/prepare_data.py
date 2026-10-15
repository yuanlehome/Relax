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


def cloud_path(path: Path, cloud: Path | None) -> Path:
    """CLI/env paths are relative to CLOUD_ROOT, or cwd when it is unset."""
    return ((cloud / path) if cloud and not path.is_absolute() else path).absolute()


def original_dataset_root(rows: list[dict]) -> Path:
    """Locate the recorded dataset root without accessing the old mount."""
    roots = set()
    for row in rows:
        tests = Path(row["metadata"]["harbor_tests_dir"])
        if ".." in tests.parts or tests.name != "tests" or tests.parent.parent.name != "tasks":
            raise ValueError("Expected original verifier layout: DATASET/tasks/TASK/tests")
        roots.add(tests.parent.parent.parent)
    if len(roots) != 1:
        raise ValueError("Original verifier paths do not share one dataset root")
    return roots.pop()


def image_mapping(rows: list[dict], supplied: Path | None) -> dict[str, str]:
    images = {row["metadata"]["sandbox_image"] for row in rows}
    if supplied:
        open_map = gzip.open if supplied.suffix == ".gz" else open
        with open_map(supplied, "rt", encoding="utf-8") as stream:
            mapping = json.load(stream)
    else:
        # The verified 150037 registry naming convention, not a general OCI rule.
        if any(image.count("/media/") != 1 or not image.endswith("-obd") for image in images):
            raise ValueError("Unknown source image convention; supply an explicit --image-map")
        mapping = {
            image: image.replace("/media/", "/media-local/", 1).removesuffix("-obd") for image in sorted(images)
        }
    if set(mapping) != images:
        raise ValueError("OCI image map does not exactly cover the original task images")
    if any(not isinstance(value, str) or not value or value.endswith("-obd") for value in mapping.values()):
        raise ValueError("An ordinary OCI image reference is missing")
    return mapping


def discover_wheels(
    source: Path,
    assets_manifest: Path | None = None,
    *,
    keys: tuple[str, ...] = ("BBS_WHEEL_PATH", "DRESSAGE_WHEEL_PATH"),
) -> dict[str, str]:
    """Resolve checksum-pinned original wheels from the source project's
    layout.

    The dataset uses src_env placeholders for these two uploads. The companion
    manifest records their original locations and hashes; no prior prepared run
    is needed at launch time.
    """
    source = source.resolve()
    lock = json.loads((assets_manifest or Path(__file__).with_name("source-assets.json")).read_text())
    relative = Path(lock["dataset_project_path"])
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Invalid source asset manifest dataset path")
    digest = sha256(source / "data.jsonl")
    if digest != lock["dataset_sha256"] or digest != json.loads((source / "manifest.json").read_text()).get(
        "data_sha256"
    ):
        raise ValueError("Dataset differs from the pinned source assets; set explicit wheel paths for another dataset")
    if source.parts[-len(relative.parts) :] != relative.parts:
        raise ValueError("Source project layout changed; set BBS_WHEEL_PATH and DRESSAGE_WHEEL_PATH")
    project = source.parents[len(relative.parts) - 1]
    result = {}
    for key in keys:
        record = lock["wheels"][key]
        relative_wheel = Path(record["project_path"])
        path = project / relative_wheel
        if relative_wheel.is_absolute() or ".." in relative_wheel.parts or not path.resolve().is_relative_to(project):
            raise ValueError("Source wheel path escapes its project")
        if not path.is_file() or path.is_symlink():
            raise ValueError("Original source wheel is missing; mount the dataset's referenced runtime assets")
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError("Original source wheel differs from its pinned size or checksum")
        result[key] = str(path)
    return result


def prepare(arguments: argparse.Namespace) -> dict[str, int | str]:
    cloud = arguments.cloud_root.resolve() if arguments.cloud_root else None
    source = cloud_path(arguments.source_root, cloud).resolve()
    destination = cloud_path(arguments.output_root, cloud).resolve()
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
    image_map = image_mapping(rows, cloud_path(arguments.image_map, cloud) if arguments.image_map else None)
    original_source = original_dataset_root(rows)
    original_cloud = arguments.original_cloud_root
    if original_cloud and (not cloud or not original_cloud.is_absolute()):
        raise ValueError("ORIGINAL_CLOUD_ROOT must be absolute and requires CLOUD_ROOT")
    if cloud and not original_cloud and source.is_relative_to(cloud):
        # Preserve the cloud-relative tree while changing only its mount point.
        suffix = source.relative_to(cloud).parts
        if suffix and original_source.parts[-len(suffix) :] == suffix:
            original_cloud = original_source.parents[len(suffix) - 1]

    def relocate(value: str) -> Path:
        path = Path(value)
        if ".." in path.parts:
            raise ValueError("Original asset reference contains parent traversal")
        if path.is_relative_to(original_source):
            base, relative = source, path.relative_to(original_source)
        elif not path.is_absolute():
            base, relative = source, path
        elif original_cloud and path.is_relative_to(original_cloud):
            base, relative = cloud, path.relative_to(original_cloud)
        elif cloud:
            raise ValueError("Cannot relocate original upload; set ORIGINAL_CLOUD_ROOT for the old mount")
        else:
            base, relative = path.parent, Path(path.name)
        result = base / relative
        if not result.resolve().is_relative_to(base.resolve()):
            raise ValueError("Original asset escapes its configured root")
        return result

    env_paths = {
        "BBS_WHEEL_PATH": cloud_path(arguments.bbs_wheel, cloud),
        "DRESSAGE_WHEEL_PATH": cloud_path(arguments.dressage_wheel, cloud),
    }
    copies: dict[Path, Path] = {}
    destinations: dict[Path, Path] = {}
    rendered = copy.deepcopy(rows)
    tests: set[Path] = set()
    uploads: set[Path] = set()
    runtimes: set[Path] = set()

    def asset(path: Path) -> Path:
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError("Original task upload or verifier file is missing or is a symlink")
        if path.is_relative_to(source / "tasks"):
            relative = Path("source/tasks") / path.relative_to(source / "tasks")
        elif path in env_paths.values():
            relative = Path("wheels") / path.name
        else:
            identity = path.relative_to(cloud) if cloud and path.is_relative_to(cloud) else path
            relative = Path("assets") / hashlib.sha256(str(identity).encode()).hexdigest()[:16] / path.name
        if relative in destinations and destinations[relative] != path:
            raise ValueError("Distinct source assets would overwrite one prepared path")
        copies[path] = relative
        destinations[relative] = path
        return relative

    for row in rendered:
        metadata = row["metadata"]
        test_dir = relocate(metadata["harbor_tests_dir"])
        if not test_dir.is_relative_to(source / "tasks"):
            raise ValueError("Verifier directory is outside the original dataset tasks")
        for name in ("test.sh", "test_patch.diff", "git_targets.json"):
            asset(test_dir / name)
        tests.add(test_dir)
        metadata["harbor_tests_dir"] = str(Path("source/tasks") / test_dir.relative_to(source / "tasks"))
        for upload in metadata["sandbox_extra_params"]["nex_uploads"]:
            if "src_env" in upload:
                name = upload.pop("src_env")
                if name not in env_paths or upload.get("src"):
                    raise ValueError("Unrecognized or ambiguous upload environment reference")
                path = env_paths[name]
            else:
                path = relocate(upload["src"])
            relative = asset(path)
            uploads.add(path)
            if upload["dst"] == "/tmp/mimo_git_runtime.py":
                runtimes.add(path)
                relative = Path("compat/mimo_git_runtime.py")
            upload["src"] = str(relative)
    if (len(uploads), len(tests), len(runtimes)) != (arguments.expected_uploads, arguments.expected_rows, 1):
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
                "source": str(original.relative_to(cloud) if cloud and original.is_relative_to(cloud) else original),
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
        runtime_patch = cloud_path(arguments.runtime_patch, cloud)
        patched = apply_exact_patch(runtime.read_text(), runtime_patch.read_text())
        (staging / "compat/mimo_git_runtime.py").write_text(patched)
        shutil.copymode(runtime, staging / "compat/mimo_git_runtime.py")
        aligned = staging / "aligned-150037-oci"
        aligned.mkdir()
        shutil.copy2(runtime_patch, aligned / "mimo_git_reflog.patch")
        write_json(staging / "ordinary-image-map.json", image_map)
        for path, values in ((aligned / "data.jsonl", rendered),):
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values))
        shutil.copy2(aligned / "data.jsonl", staging / "tasks.r5.relative.jsonl")
        write_json(
            aligned / "source_manifest.json",
            {
                "source_dataset": str(source.relative_to(cloud) if cloud and source.is_relative_to(cloud) else source),
                "source_data_sha256": source_digest,
                "prepared_root": ".",
                "path_format": "relative-v1",
                "rows": len(rows),
                "unique_uploads": len(uploads),
                "test_dirs": len(tests),
                "source_runtime_sha256": sha256(runtime),
                "patched_runtime_sha256": sha256(staging / "compat/mimo_git_runtime.py"),
                "runtime_patch_sha256": sha256(runtime_patch),
                "ordinary_image_map_sha256": sha256(staging / "ordinary-image-map.json"),
                "image_map_source": "explicit" if arguments.image_map else "150037-media-local-without-obd",
                "changes": [
                    "host paths stored relative to CLAUDE_CODE_DATA_ROOT",
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


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, env in (
        ("cloud-root", "CLOUD_ROOT"),
        ("original-cloud-root", "ORIGINAL_CLOUD_ROOT"),
        ("source-root", "SOURCE_ROOT"),
        ("output-root", "DATA_ROOT"),
        ("bbs-wheel", "BBS_WHEEL_PATH"),
        ("dressage-wheel", "DRESSAGE_WHEEL_PATH"),
        ("image-map", "IMAGE_MAP"),
    ):
        parser.add_argument("--" + flag, type=Path, default=os.environ.get(env), help=f"Defaults to ${env}")
    parser.add_argument("--published-root", type=Path, help="Legacy option; output paths are now relative")
    parser.add_argument(
        "--runtime-patch", type=Path, default=Path(__file__).resolve().parents[1] / "mimo_git_reflog.patch"
    )
    parser.add_argument("--expected-rows", type=int, default=2683)
    parser.add_argument("--expected-uploads", type=int, default=2708)
    args = parser.parse_args(argv)
    for name in ("source_root", "output_root", "bbs_wheel", "dressage_wheel"):
        if getattr(args, name) is None:
            parser.error(f"Set --{name.replace('_', '-')} or its environment variable")
    return args


if __name__ == "__main__":
    print(json.dumps(prepare(parse_arguments())))
