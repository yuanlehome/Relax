# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Restore missing loop device nodes in a Pod's private /dev after startup
fails."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path


def restore_missing_loop_device(
    diagnostic: str,
    *,
    device_root: Path = Path("/dev"),
    sysfs_root: Path = Path("/sys/class/block"),
) -> dict[str, str | int] | None:
    """Restore only a kernel-verified missing node; never attach or detach a
    device."""
    match = re.search(r"could not open /dev/(loop([0-9]+)): no such file or directory(?:\r?\n|$)", diagnostic)
    if match is None:
        return None
    name, number = match.groups()
    minor = int(number)
    expected = f"7:{minor}"
    kernel_device = (sysfs_root / name / "dev").read_text().strip()
    if kernel_device != expected:
        raise RuntimeError(
            f"Refusing loop-device recovery: {name} has sysfs device {kernel_device!r}, expected {expected}"
        )
    path = device_root / name
    created = False
    try:
        node = path.lstat()
    except FileNotFoundError:
        try:
            os.mknod(path, stat.S_IFBLK | 0o600, os.makedev(7, minor))
            created = True
        except FileExistsError:
            # Another Session may restore the same node concurrently.
            pass
        node = path.lstat()
    if not stat.S_ISBLK(node.st_mode) or node.st_rdev != os.makedev(7, minor):
        raise RuntimeError(f"Refusing loop-device recovery: {path} is not the expected block device {expected}")
    return {
        "path": str(path),
        "major": 7,
        "minor": minor,
        "sysfs_device": kernel_device,
        "action": "created" if created else "already_valid",
    }
