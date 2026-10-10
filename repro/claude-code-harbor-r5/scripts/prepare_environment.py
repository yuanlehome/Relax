# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Rebuild the separate Gym Python overlay from preserved packages, offline."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def prepare(bundle: Path, repo: Path) -> dict[str, str]:
    if platform.system() != "Linux" or sys.version_info[:3] != (3, 12, 3):
        raise RuntimeError("Use Linux CPython 3.12.3 in the recorded Relax training image")
    frozen = bundle / "runtime/nemo-site-packages"
    destination = bundle / "runtime/nemo-venv"
    if not frozen.is_dir():
        raise FileNotFoundError("The preserved Gym package overlay is missing")
    observed = json.loads((bundle / "runtime/observed-python-packages.json").read_text())
    probe = (
        "import json,importlib.metadata as m; "
        "print(json.dumps(sorted([(d.metadata['Name'],d.version) for d in m.distributions() "
        "if d.metadata.get('Name')])))"
    )
    packages = json.loads(subprocess.check_output([sys.executable, "-c", probe], text=True))
    expected = sorted(observed["base"]["packages"])
    actual = sorted(packages)
    if actual != expected:
        raise RuntimeError("Base-image Python packages differ from r5; use the recorded training image")
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", "--copies", str(destination)],
        check=True,
    )
    site = destination / "lib/python3.12/site-packages"
    shutil.copytree(frozen, site, dirs_exist_ok=True)
    # The original editable finder embeds an external source path. The rebuilt
    # environment imports the frozen source through an explicit local .pth file.
    for path in site.glob("__editable__*nemo_gym*"):
        if path.is_file():
            path.unlink()
    (site / "nemo_gym_cloud_source.pth").write_text(str(bundle / "runtime/nemo-gym") + "\n")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repo), str(bundle / "runtime/nemo-gym"), str(bundle / "runtime/Megatron-LM")]
    )
    check = (
        "from pathlib import Path;import nemo_gym,fastapi,uvicorn; "
        "from examples.claude_code_harbor.gym_service import create_service; "
        "assert str(Path(nemo_gym.__file__).resolve()).startswith(" + repr(str(bundle)) + "); "
        "print('Gym imports PASS')"
    )
    subprocess.run([str(destination / "bin/python"), "-c", check], env=environment, check=True)
    return {"status": "PASS", "method": "same-image package overlay; no pip install or service startup"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--repo", type=Path, required=True, help="Relax repository used by the reproduction")
    arguments = parser.parse_args()
    print(json.dumps(prepare(arguments.bundle.resolve(), arguments.repo.resolve())))
