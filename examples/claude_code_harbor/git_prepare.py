# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Repair narrow image Git defects while preserving the trusted preparation
gate."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable


_BASELINE_CHECK = ("diff", "--quiet", "HEAD", "--")
_MAX_GIT_METADATA_BYTES = 64 * 1024 * 1024


def _offline_git(cwd: Path, *arguments: str, timeout: float = 30.0, check: bool = True) -> subprocess.CompletedProcess:
    """Read/clean a private object store without fetching missing promised
    objects."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_NO_LAZY_FETCH="1", GIT_ALLOW_PROTOCOL="")
    command = [
        "git",
        "--no-pager",
        "-c",
        f"safe.directory={cwd}",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "diff.external=",
        "-c",
        "gc.auto=0",
        "-c",
        "gc.recentObjectsHook=/bin/true",
        "-c",
        "protocol.allow=never",
        "-C",
        str(cwd),
        *arguments,
    ]
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr, env=env, timeout=timeout, check=False)
        if os.fstat(stdout.fileno()).st_size > _MAX_GIT_METADATA_BYTES or os.fstat(stderr.fileno()).st_size > 65536:
            raise ValueError("Offline Git metadata exceeds recovery budget")
        stdout.seek(0)
        stderr.seek(0)
        completed = subprocess.CompletedProcess(command, result.returncode, stdout.read(), stderr.read())
    if check and completed.returncode:
        raise RuntimeError("Offline Git command failed")
    return completed


def _required_objects(git: Callable, root: Path) -> set[str]:
    output = git(root, "rev-list", "--objects", "--missing=print", "--no-object-names", "--all", "HEAD").stdout
    rows = output.decode("ascii").splitlines()
    if not rows or any(not re.fullmatch(r"[0-9a-f]{40}", row) for row in rows):
        raise ValueError("Required object closure is incomplete or unsupported")
    return set(rows)


def _recover_promisor_objects(
    cwd: str | Path, reachable: set[str], visible: dict[str, str], record: dict[str, Any]
) -> tuple[set[str], dict[str, str]] | None:
    """Demote stale promisor markers only after a complete local baseline is
    proved.

    Called at the original final inventory gate, after reset/ref stripping and
    the baseline checks. Never delete objects directly or add a network source.
    """
    root = Path(cwd).resolve()
    gd = root / ".git"
    objects, packdir = gd / "objects", gd / "objects/pack"
    record["stage"] = "private_paths"
    for path in (gd, objects, packdir):
        if not path.is_dir() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Object store is not private")
    markers = sorted(packdir.glob("*.promisor"))
    if not markers:
        return None
    if len(markers) > 64:
        raise ValueError("Too many promisor packs")
    for relative in (
        "commondir",
        "worktrees",
        "shallow",
        "info/grafts",
        "objects/info/alternates",
        "objects/info/http-alternates",
        "index.lock",
        "gc.pid",
    ):
        path = gd / relative
        if path.exists() or path.is_symlink():
            raise ValueError("Shared, shallow or alternate object stores are unsupported")
    for path in (gd / "info", objects / "info"):
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError("Git metadata directories must be private")
    for path in packdir.iterdir():
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("Pack directory contains nonprivate files")
    for directory in objects.iterdir():
        if directory.name in {"pack", "info"}:
            continue
        if not re.fullmatch(r"[0-9a-f]{2}", directory.name) or directory.is_symlink() or not directory.is_dir():
            raise ValueError("Unsupported loose object directory")
        for path in directory.iterdir():
            info = path.lstat()
            if not re.fullmatch(r"[0-9a-f]{38}", path.name) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("Loose object files must be private")
    for marker in markers:
        if not re.fullmatch(r"pack-[0-9a-f]{40}\.promisor", marker.name):
            raise ValueError("Unsupported promisor marker name")
        for path in (marker, marker.with_suffix(".pack"), marker.with_suffix(".idx")):
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("Promisor pack files must be private regular files")
    deadline = time.monotonic() + 90.0

    def git(path: Path, *arguments: str, **kwargs: Any) -> subprocess.CompletedProcess:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Promisor recovery deadline exceeded")
        return _offline_git(path, *arguments, timeout=min(30.0, remaining), **kwargs)

    record["stage"] = "repository_identity"
    if Path(os.fsdecode(git(root, "rev-parse", "--show-toplevel").stdout).strip()).resolve() != root:
        raise ValueError("Unexpected worktree root")
    for option in ("--absolute-git-dir", "--git-common-dir"):
        path = Path(os.fsdecode(git(root, "rev-parse", option).stdout).strip())
        if (root / path).resolve() != gd:
            raise ValueError("Unexpected Git store")
    if (
        git(root, "remote").stdout.strip()
        or git(root, "for-each-ref", "--format=%(refname)", "refs/replace/").stdout.strip()
    ):
        raise ValueError("Remote or replacement refs remain")
    if git(root, "config", "--get", "extensions.partialClone", check=False).returncode == 0:
        raise ValueError("Partial-clone configuration remains")
    if git(root, "config", "--bool", "--get", "extensions.preciousObjects", check=False).stdout.strip() == b"true":
        raise ValueError("Precious objects require separate handling")
    hooks = git(root, "config", "--get-all", "gc.recentObjectsHook", check=False).stdout.splitlines()
    if any(hook != b"/bin/true" for hook in hooks):
        raise ValueError("Custom recent-object hook is unsupported")
    identity = git(root, "rev-parse", "HEAD", "HEAD^{tree}").stdout
    refs = git(root, "for-each-ref", "--format=%(refname) %(objectname)").stdout
    if (
        git(root, "diff", "--cached", "--quiet", "HEAD", "--", check=False).returncode
        or git(root, *_BASELINE_CHECK, check=False).returncode
    ):
        raise ValueError("Baseline or index is not clean")
    if git(root, "rev-list", "--all", "--not", "HEAD").stdout.strip():
        raise ValueError("Future refs remain")
    record["stage"] = "local_closure"
    required = _required_objects(git, root)
    if required != reachable:
        raise ValueError("Local required closure differs from the original inventory")
    extras = set(visible) - reachable
    promisor_objects: set[str] = set()
    for marker in markers:
        output = git(root, "verify-pack", "-v", str(marker.with_suffix(".idx"))).stdout
        for line in output.splitlines():
            oid = line.split(b" ", 1)[0]
            if re.fullmatch(rb"[0-9a-f]{40}", oid):
                promisor_objects.add(oid.decode("ascii"))
    if not extras or not extras.issubset(promisor_objects):
        raise ValueError("Unreachable objects are not confined to promisor packs")
    record.update(original_unreachable=len(extras), required_objects=len(required), marker_count=len(markers))
    record["stage"] = "demote_markers"
    for marker in markers:
        marker.unlink()
    record["stage"] = "offline_gc"
    git(root, "gc", "--prune=now")
    record["stage"] = "recheck"
    if (
        git(root, "rev-parse", "HEAD", "HEAD^{tree}").stdout != identity
        or git(root, "for-each-ref", "--format=%(refname) %(objectname)").stdout != refs
    ):
        raise RuntimeError("Baseline identity or refs changed during recovery")
    if (
        git(root, "diff", "--cached", "--quiet", "HEAD", "--", check=False).returncode
        or git(root, *_BASELINE_CHECK, check=False).returncode
    ):
        raise RuntimeError("Baseline or index changed during recovery")
    after_required = _required_objects(git, root)
    after_visible: dict[str, str] = {}
    for line in git(
        root, "cat-file", "--batch-all-objects", "--batch-check=%(objectname) %(objecttype)"
    ).stdout.splitlines():
        fields = line.decode("ascii").split()
        if (
            len(fields) != 2
            or not re.fullmatch(r"[0-9a-f]{40}", fields[0])
            or fields[1] not in {"blob", "tree", "commit", "tag"}
        ):
            raise RuntimeError("Unexpected object inventory")
        after_visible[fields[0]] = fields[1]
    if after_required != required or set(after_visible) != required:
        raise RuntimeError("Required closure changed or unreachable objects remain")
    record.update(stage="complete", remaining_unreachable=0)
    return after_required, after_visible


def instrument_inventory(original: Callable) -> Callable:
    def observed(cwd: str | Path) -> tuple[set[str], dict[str, str]]:
        reachable, visible = original(cwd)
        if not set(visible) - reachable:
            return reachable, visible
        record: dict[str, Any] = {}
        recovered = None
        try:
            recovered = _recover_promisor_objects(cwd, reachable, visible, record)
        except Exception as error:
            record.update(status="declined_or_failed", error_type=type(error).__name__)
        if recovered is not None:
            record["status"] = "recovered"
        if record.get("status"):
            try:
                sys.stderr.write("RELAX_GIT_PROMISOR_RECOVERY=" + json.dumps(record, sort_keys=True) + "\n")
                sys.stderr.flush()
            except Exception:
                pass
        # A refusal or failed repair preserves the original nonempty extras,
        # so the original runtime gate still rejects this preparation attempt.
        return recovered if recovered is not None else (reachable, visible)

    return observed


def _raw_changes(git: Callable, cwd: Path) -> list[tuple[list[str], str]]:
    raw = git(
        cwd, "diff", "--no-ext-diff", "--no-textconv", "--no-renames", "--no-abbrev", "--raw", "-z", "HEAD", "--"
    ).stdout
    if len(raw) > 65536:
        raise ValueError("Baseline difference exceeds recovery metadata limit")
    fields = raw.split(b"\0")
    if fields.pop() != b"" or len(fields) % 2:
        raise ValueError("Unexpected raw Git difference")
    result = []
    for index in range(0, len(fields), 2):
        metadata = fields[index].decode("ascii").split()
        if len(metadata) != 5 or not metadata[0].startswith(":"):
            raise ValueError("Unexpected raw Git metadata")
        metadata[0] = metadata[0][1:]
        result.append((metadata, os.fsdecode(fields[index + 1])))
    return result


def _private_child(root: Path, name: str) -> Path:
    relative = Path(name)
    if not relative.parts or relative.is_absolute() or any(part in ("..", ".git") for part in relative.parts):
        raise ValueError("Unsafe baseline path")
    target = root / relative
    if not target.resolve().is_relative_to(root):
        raise ValueError("Baseline path escapes private worktree")
    current = target
    while current != root:
        if current.is_symlink():
            raise ValueError("Baseline path contains a symlink")
        current = current.parent
    return target


def _restore_missing_submodule_files(git: Callable, cwd: str | Path) -> list[dict[str, Any]]:
    """Restore only deleted regular files in private, already-pinned
    submodules."""
    root = Path(cwd).resolve()
    changes = _raw_changes(git, root)
    if not changes or len(changes) > 8:
        return []
    if git(root, "diff", "--cached", "--quiet", "HEAD", "--", check=False).returncode:
        return []
    parent_identity = git(root, "rev-parse", "HEAD", "HEAD^{tree}").stdout
    plans = []
    for metadata, name in changes:
        old_mode, new_mode, pinned, current_oid, status = metadata
        if (old_mode, new_mode, status) != ("160000", "160000", "M") or pinned != current_oid:
            return []
        child = _private_child(root, name)
        if Path(os.fsdecode(git(child, "rev-parse", "--show-toplevel").stdout).strip()).resolve() != child:
            return []
        for option in ("--absolute-git-dir", "--git-common-dir"):
            location = Path(os.fsdecode(git(child, "rev-parse", option).stdout).strip())
            if not (child / location).resolve().is_relative_to(root):
                return []
        common = (child / location).resolve()
        objects = common / "objects"
        if not objects.resolve().is_relative_to(root):
            return []
        alternates = objects / "info/alternates"
        if alternates.is_symlink() or (alternates.exists() and alternates.read_bytes().strip()):
            return []
        # Do not run task-provided filters or let a missing blob trigger lazy fetch.
        if git(child, "config", "--get", "extensions.partialClone", check=False).returncode == 0:
            return []
        if git(child, "config", "--get-regexp", r"^remote\..*\.promisor$", check=False).returncode == 0:
            return []
        if git(child, "for-each-ref", "--format=%(refname)", "refs/replace/").stdout.strip():
            return []
        if git(child, "rev-parse", "HEAD").stdout.decode().strip() != pinned:
            return []
        if git(child, "diff", "--cached", "--quiet", pinned, "--", check=False).returncode:
            return []
        deleted = _raw_changes(git, child)
        if not deleted:
            return []
        paths = []
        for item, filename in deleted:
            if item[0] not in ("100644", "100755") or item[1] != "000000" or item[4] != "D":
                return []
            target = _private_child(child, filename)
            try:
                target.lstat()
            except FileNotFoundError:
                pass
            else:
                return []
            git(child, "cat-file", "-e", item[2] + "^{blob}")
            paths.append(filename)
        for source in ((), ("--cached",)):
            attributes = git(child, "--literal-pathspecs", "check-attr", *source, "-z", "filter", "--", *paths).stdout
            values = attributes.split(b"\0")
            if values.pop() != b"" or len(values) != 3 * len(paths):
                return []
            if any(value not in (b"unspecified", b"unset") for value in values[2::3]):
                return []
        plans.append({"path": name, "commit": pinned, "files": paths})
        if sum(len(plan["files"]) for plan in plans) > 256:
            return []
    # Validate every candidate before modifying any file. Literal pathspecs avoid
    # treating a tracked filename containing '*' or ':(' as a checkout pattern.
    for plan in plans:
        child = root / plan["path"]
        git(child, "--literal-pathspecs", "checkout", plan["commit"], "--", *plan["files"])
        if git(child, "rev-parse", "HEAD").stdout.decode().strip() != plan["commit"]:
            raise RuntimeError("Submodule HEAD changed during baseline restoration")
        if git(child, "diff", "--cached", "--quiet", plan["commit"], "--", check=False).returncode:
            raise RuntimeError("Submodule index changed during baseline restoration")
    if git(root, "rev-parse", "HEAD", "HEAD^{tree}").stdout != parent_identity:
        raise RuntimeError("Parent baseline changed during restoration")
    if git(root, "diff", "--cached", "--quiet", "HEAD", "--", check=False).returncode:
        raise RuntimeError("Parent index changed during restoration")
    return plans


def _git_metadata(cwd: str | Path, arguments: tuple[str, ...], timeout: float) -> dict[str, Any]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env["GIT_OPTIONAL_LOCKS"] = "0"
    command = [
        "git",
        "--no-pager",
        "-c",
        f"safe.directory={cwd}",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "diff.external=",
        "-C",
        str(cwd),
        *arguments,
    ]
    # Capture metadata only, with bounded memory, output and runtime. No patches,
    # full config, environment, remote URLs or repository file contents are read.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, env=env, timeout=timeout, check=False)
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            return {"timeout": True}
        stdout.seek(0)
        stderr.seek(0, os.SEEK_END)
        stderr.seek(max(0, stderr.tell() - 1024))
        return {
            "returncode": returncode,
            "stdout_head": stdout.read(2048).decode(errors="replace"),
            "stderr_tail": stderr.read(1024).decode(errors="replace"),
        }


def _record_failure(cwd: str | Path, result: subprocess.CompletedProcess, elapsed: float) -> None:
    deadline = time.monotonic() + 6.0
    record = {
        "returncode": result.returncode,
        "kind": "tracked_difference" if result.returncode == 1 else "git_command_error",
        "elapsed_seconds": round(elapsed, 6),
        "stderr_tail": (result.stderr or b"")[-2048:].decode(errors="replace"),
        "metadata": {},
    }
    commands = {
        "head": ("rev-parse", "HEAD", "HEAD^{tree}"),
        "raw_diff": ("diff", "--no-ext-diff", "--no-textconv", "--no-renames", "--raw", "HEAD", "--"),
        "status": ("status", "--porcelain=v2", "--untracked-files=no"),
    }
    for name, arguments in commands.items():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            record["metadata"][name] = _git_metadata(cwd, arguments, min(2.0, remaining))
        except Exception as error:
            record["metadata"][name] = {"diagnostic_error": type(error).__name__}
    sys.stderr.write("RELAX_GIT_BASELINE_DIAGNOSTIC=" + json.dumps(record, sort_keys=True) + "\n")
    sys.stderr.flush()


def instrument_git(original: Callable, *, restore_missing: bool = False) -> Callable:
    def observed(cwd: str | Path, *arguments: str, **kwargs: Any) -> subprocess.CompletedProcess:
        result = original(cwd, *arguments, **kwargs)
        if restore_missing and arguments == _BASELINE_CHECK and result.returncode == 1:
            try:
                restored = _restore_missing_submodule_files(original, cwd)
            except Exception:
                # Unsupported or unsuccessful restoration must still fail the gate.
                restored = []
            if restored:
                result = original(cwd, *arguments, **kwargs)
                try:
                    sys.stderr.write(
                        "RELAX_GIT_SUBMODULE_BASELINE_RESTORED="
                        + json.dumps(
                            {
                                "submodules": restored,
                                "original_returncode": 1,
                                "recheck_returncode": result.returncode,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    sys.stderr.flush()
                except Exception:
                    pass
        return result

    return observed


def main() -> None:
    runtime_path = Path(sys.argv.pop(1))
    spec = importlib.util.spec_from_file_location("_relax_mimo_git_runtime", runtime_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load MiMo Git runtime")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    original_git = runtime.git
    preparing = len(sys.argv) > 1 and sys.argv[1] == "prepare"
    runtime.git = instrument_git(original_git, restore_missing=preparing)
    if preparing:
        original_inventory = runtime.inventory
        runtime.inventory = instrument_inventory(original_inventory)
    try:
        runtime.main()
    finally:
        runtime.git = original_git
        if preparing:
            runtime.inventory = original_inventory


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        sys.stderr.write(f"MIMO_GIT_PREPARE_FAILED: {error}\n")
        raise SystemExit(1) from error
