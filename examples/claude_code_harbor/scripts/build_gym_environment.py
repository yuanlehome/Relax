# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Build the Harbor Gym overlay from pinned source and package versions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


NEMO_COMMIT = "a85670eb167ba9b48cc53a36a070eed815e6c40d"
NEMO_REPOSITORY = "https://github.com/NVIDIA-NeMo/Gym.git"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]


def read_pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, version = line.split("==")
            pins[name] = version
    return pins


def clean_environment() -> dict[str, str]:
    env = dict(os.environ)
    network_options = {
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
        "PIP_TRUSTED_HOST",
        "PIP_CERT",
        "PIP_CLIENT_CERT",
        "PIP_TIMEOUT",
        "PIP_RETRIES",
        "PIP_CACHE_DIR",
        "PIP_NO_CACHE_DIR",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "GIT_ASKPASS",
        "GIT_SSL_CAINFO",
        "GIT_SSL_CAPATH",
        "GIT_PROXY_COMMAND",
    }
    for key in list(env):
        if key.startswith(("PIP_", "GIT_")) and key not in network_options:
            env.pop(key)
    for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONOPTIMIZE", "VIRTUAL_ENV"):
        env.pop(key, None)
    env.update(
        PYTHONNOUSERSITE="1", PIP_CONFIG_FILE=os.devnull, PIP_REQUIRE_VIRTUALENV="true", GIT_TERMINAL_PROMPT="0"
    )
    return env


def run(command: list[str], *, env: dict[str, str], log: Path, stage: str, cwd: Path | None = None) -> str:
    with log.open("ab") as output:
        result = subprocess.run(
            command, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=output, check=False, timeout=900
        )
        output.write(result.stdout)
    if result.returncode:
        # Installer output may contain authenticated package-index URLs. Keep it
        # in the private build directory instead of echoing it into job logs.
        raise RuntimeError(f"{stage} failed; inspect the private build log: {log}")
    return result.stdout.decode().strip()


def packages(python: Path, *, env: dict[str, str], log: Path) -> dict[str, str]:
    code = (
        "import importlib.metadata as m,json; "
        "names={d.metadata['Name'] for d in m.distributions() if d.metadata.get('Name')}; "
        "print(json.dumps({name:m.version(name) for name in sorted(names)}))"
    )
    return json.loads(run([str(python), "-c", code], env=env, log=log, stage="Package inventory"))


def require_pins(actual: dict[str, str], expected: dict[str, str]) -> None:
    normalized = {name.lower().replace("_", "-"): version for name, version in actual.items()}
    mismatched = [
        name for name, version in expected.items() if normalized.get(name.lower().replace("_", "-")) != version
    ]
    if mismatched:
        raise RuntimeError("Incompatible package versions: " + ", ".join(sorted(mismatched)))


def smoke(prefix: Path, repo: Path, env: dict[str, str], log: Path) -> None:
    smoke_env = dict(env, PYTHONPATH=os.pathsep.join([str(repo), str(prefix / "nemo-gym")]))
    run(
        [str(prefix / "nemo-venv/bin/python"), str(SCRIPT_DIR / "check_gym_environment.py"), str(prefix)],
        env=smoke_env,
        log=log,
        stage="Harbor service smoke",
        cwd=repo,
    )


def prepare(prefix: Path, repo: Path, source_url: str = NEMO_REPOSITORY, *, check: bool = False) -> dict[str, str]:
    if platform.system() != "Linux" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use the Relax Linux training image with CPython 3.12")
    if sys.prefix != sys.base_prefix:
        raise RuntimeError("Run the builder with the training image's system Python, outside any venv")
    if not (repo / "examples/claude_code_harbor/gym_service.py").is_file():
        raise ValueError("--repo must point to a Relax checkout containing the Harbor adapter")
    overlay = SCRIPT_DIR / "gym-requirements.txt"
    base = SCRIPT_DIR / "gym-base-requirements.txt"
    signature = hashlib.sha256(overlay.read_bytes() + base.read_bytes()).hexdigest()
    env = clean_environment()
    if check:
        manifest = json.loads((prefix / "build.json").read_text())
        if manifest["nemo_commit"] != NEMO_COMMIT or manifest["requirements_sha256"] != signature:
            raise RuntimeError("The built environment does not match this recipe; choose a fresh --prefix")
    else:
        # Reserve the final path atomically. Venv shebangs contain this path, so
        # never build elsewhere and move it, nor overwrite a live environment.
        prefix.mkdir(parents=True, mode=0o700, exist_ok=False)
    log = prefix / "build.log"
    log.touch(mode=0o600, exist_ok=True)
    python = Path(sys.executable)
    before = packages(python, env=env, log=log)
    require_pins(before, read_pins(base))
    source = prefix / "nemo-gym"
    venv_python = prefix / "nemo-venv/bin/python"
    if not check:
        run(["git", "init", "--quiet", "--template=", str(source)], env=env, log=log, stage="Create source checkout")
        run(["git", "remote", "add", "origin", source_url], env=env, log=log, stage="Set source mirror", cwd=source)
        run(
            ["git", "sparse-checkout", "set", "nemo_gym", "responses_api_agents/claude_code_agent"],
            env=env,
            log=log,
            stage="Select Harbor source",
            cwd=source,
        )
        run(
            ["git", "fetch", "--quiet", "--filter=blob:none", "--depth=1", "origin", NEMO_COMMIT],
            env=env,
            log=log,
            stage="Fetch NeMo",
            cwd=source,
        )
        run(
            ["git", "checkout", "--quiet", "--detach", NEMO_COMMIT],
            env=env,
            log=log,
            stage="Checkout NeMo",
            cwd=source,
        )
        run(
            [str(python), "-m", "venv", "--system-site-packages", "--copies", str(prefix / "nemo-venv")],
            env=env,
            log=log,
            stage="Create Gym venv",
        )
        run(
            [
                str(venv_python),
                "-m",
                "pip",
                "--disable-pip-version-check",
                "--no-input",
                "install",
                "--ignore-installed",
                "--no-deps",
                "-r",
                str(overlay),
            ],
            env=env,
            log=log,
            stage="Install pinned Gym overlay",
        )
        # Harbor imports NeMo directly. Avoid installing the upstream kitchen-
        # sink extras (MLflow, separate Ray, host Claude Code, other recipes).
        site = prefix / "nemo-venv/lib/python3.12/site-packages"
        (site / "relax_harbor_nemo.pth").write_text(str(source) + "\n")
    revision = run(["git", "rev-parse", "HEAD"], env=env, log=log, stage="Verify source revision", cwd=source)
    dirty = run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        env=env,
        log=log,
        stage="Verify source tree",
        cwd=source,
    )
    if revision != NEMO_COMMIT or dirty:
        raise RuntimeError("NeMo source must be the unmodified pinned commit")
    effective = packages(venv_python, env=env, log=log)
    require_pins(effective, {**read_pins(base), **read_pins(overlay)})
    smoke(prefix, repo, env, log)
    if packages(python, env=env, log=log) != before:
        raise RuntimeError("Training-image package inventory changed during the build")
    if not check:
        manifest = {
            "nemo_commit": NEMO_COMMIT,
            "requirements_sha256": signature,
            "python": platform.python_version(),
            "base_packages": before,
            "effective_packages": effective,
        }
        (prefix / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"status": "PASS", "runtime": str(prefix), "mode": "check" if check else "build"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, type=Path, help="Fresh, permanent Gym runtime directory")
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--source-url", default=NEMO_REPOSITORY, help="Git mirror; the required commit stays fixed")
    parser.add_argument("--check", action="store_true", help="Validate an existing build without installing packages")
    args = parser.parse_args()
    result = prepare(args.prefix.absolute(), args.repo.resolve(), args.source_url, check=args.check)
    sys.stdout.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
