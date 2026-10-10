# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Fetch the pinned Docker/FUSE archives without executing or installing
them."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


MANIFEST = Path(__file__).with_name("oci-packages.json")


def verify(path: Path, size: int, digest: str) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
        raise ValueError(f"Missing, unsafe or wrong-sized OCI package: {path.name}")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != digest:
        raise ValueError(f"OCI package checksum mismatch: {path.name}")


def fetch(path: Path, url: str, size: int, digest: str) -> None:
    if path.exists() or path.is_symlink():
        verify(path, size, digest)
        return
    # The cache receives a final filename only after the complete hash matches.
    # An existing file is never overwritten, including concurrent downloads.
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".oci-download-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            for attempt in range(3):
                stream.seek(0)
                stream.truncate()
                received = 0
                deadline = time.monotonic() + 600
                try:
                    with urllib.request.urlopen(url, timeout=30) as response:
                        while chunk := response.read(1024 * 1024):
                            received += len(chunk)
                            if received > size:
                                raise ValueError(f"Oversized OCI package: {path.name}")
                            if time.monotonic() > deadline:
                                raise TimeoutError("OCI package download deadline exceeded")
                            stream.write(chunk)
                    break
                except (urllib.error.URLError, TimeoutError):
                    if attempt == 2:
                        # Network errors can contain authenticated mirror URLs.
                        raise RuntimeError(f"Could not download OCI package: {path.name}") from None
                    time.sleep(attempt + 1)
            stream.flush()
            verify(temporary, size, digest)
            try:
                os.link(temporary, path)
            except FileExistsError:
                verify(path, size, digest)
        finally:
            temporary.unlink(missing_ok=True)


def prepare(assets: Path, *, check_only: bool = False, sources: dict[str, str] | None = None) -> int:
    manifest = json.loads(MANIFEST.read_text())
    locations = {**manifest["sources"], **(sources or {})}
    for url in locations.values():
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("OCI mirror must be an HTTP(S) base URL without query or fragment")
    assets = assets.resolve()
    for package in manifest["packages"]:
        path = assets / package["path"]
        if not path.parent.resolve().is_relative_to(assets) or path.is_symlink():
            raise ValueError("OCI package path escapes its cache or is a symlink")
        size, digest = package["bytes"], package["sha256"]
        if check_only:
            verify(path, size, digest)
        else:
            url = locations[package["source"]].rstrip("/") + "/" + package["remote_path"]
            fetch(path, url, size, digest)
    return len(manifest["packages"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true", help="Verify local files; never access the network")
    parser.add_argument("--docker-base-url", help="Mirror of Docker's static stable x86_64 directory")
    parser.add_argument("--ubuntu-base-url", help="Mirror root containing Ubuntu's pool directory")
    args = parser.parse_args()
    sources = {name: url for name, url in (("docker", args.docker_base_url), ("ubuntu", args.ubuntu_base_url)) if url}
    try:
        count = prepare(args.assets_dir, check_only=args.check_only, sources=sources)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps({"status": "PASS", "packages": count, "mode": "check" if args.check_only else "fetch"}))


if __name__ == "__main__":
    main()
