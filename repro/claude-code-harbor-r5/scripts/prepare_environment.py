# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Build Gym from pinned sources; the historical environment archive is
optional."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--repo", type=Path, required=True, help="Relax checkout used by the reproduction")
    parser.add_argument("--check", action="store_true", help="Validate an existing build without installing")
    arguments = parser.parse_args()
    subprocess.run(
        [
            sys.executable,
            str(arguments.repo.resolve() / "examples/claude_code_harbor/scripts/build_gym_environment.py"),
            "--repo",
            str(arguments.repo.resolve()),
            "--prefix",
            str(arguments.bundle.resolve() / "runtime/gym"),
            *(["--check"] if arguments.check else []),
        ],
        check=True,
    )
