# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Prepare Harbor tasks and Relax prompts from the aligned MiMo asset
bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def materialize(
    source: Path,
    asset_root: Path,
    image_map: Path,
    output: Path,
) -> None:
    """Publish Harbor tasks and Relax prompts from the aligned MiMo bundle."""
    asset_root, source, image_map, output = (path.resolve() for path in (asset_root, source, image_map, output))
    mapping = json.loads(image_map.read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        tasks = staging / "tasks"
        tasks.mkdir()

        def write_task(row: dict) -> str:
            metadata = row["metadata"]
            name = metadata["harbor_task"]
            task = tasks / name
            task.mkdir(parents=True)
            (task / "tests").mkdir()
            tests = asset_root / metadata["harbor_tests_dir"]
            for test_name in ("test.sh", "test_patch.diff", "git_targets.json"):
                shutil.copyfile(tests / test_name, task / "tests" / test_name)
            (task / "instruction.md").write_text(row["prompt"][0]["content"].strip() + "\n", encoding="utf-8")
            (task / "task.toml").write_text(
                'version = "1.0"\n\n[environment]\n'
                f"docker_image = {json.dumps(mapping[metadata['sandbox_image']])}\n"
                f"workdir = {json.dumps(metadata['backend_options']['workdir'])}\n"
                '\n[verifier]\nenvironment_mode = "shared"\n',
                encoding="utf-8",
            )
            return (
                json.dumps(
                    {
                        "input": [{"role": "user", "content": "dummy"}],
                        "metadata": {"harbor": {"task": f"tasks/{name}"}},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        with source.open(encoding="utf-8") as rows, (staging / "prompts.jsonl").open("w", encoding="utf-8") as prompts:
            with ThreadPoolExecutor(max_workers=16) as pool:
                prompts.writelines(pool.map(write_task, map(json.loads, rows)))
        staging.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--image-map", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.source, args.asset_root, args.image_map, args.output_root)


if __name__ == "__main__":
    main()
