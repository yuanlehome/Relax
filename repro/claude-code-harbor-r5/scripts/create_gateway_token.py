# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Create a new host-only credential for a new reproduction fleet."""

import argparse
import os
import secrets
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    private = args.bundle.resolve() / "private"
    private.mkdir(mode=0o700, exist_ok=True)
    target = private / "gateway.token"
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.stat().st_mode & 0o077:
            raise RuntimeError("Existing credential must be a regular private file with mode 0600")
        print("Existing reproduction Gateway credential retained")
    else:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(secrets.token_urlsafe(48) + "\n")
        print("New reproduction Gateway credential created; content omitted")
