#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Install the r5 OCI tools into a private directory using preserved packages.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: install_oci_runtime.sh --assets-dir DIR --prefix DIR

Run on Linux x86_64. The prefix must not exist. No sudo, package-manager
installation, daemon startup, or training operation is performed.

Required files under --assets-dir:
  docker-28.5.2.tgz
  fuse-debs/fuse-overlayfs_1.13-1_amd64.deb
  fuse-debs/fuse3_3.14.0-5build1_amd64.deb
  fuse-debs/libfuse3-3_3.14.0-5build1_amd64.deb

These are external installation packages, not Git inputs. Use the preserved
runtime/oci-runtime directory as --assets-dir, or obtain the same packages
from your package mirror. The script does not copy an installed environment.

Output layout:
  PREFIX/docker/    Docker, dockerd, containerd, runc
  PREFIX/fuse/      Extracted fuse-overlayfs and libfuse packages

NeMo Gym and the Relax training image are existing prerequisites. Each task's
Claude Code/SRT installation is handled by install_task_runtime.sh inside its
own OCI container. Task images and verifier files come from prepared data.
EOF
}

assets_dir=
prefix=
while [ "$#" -gt 0 ]; do
    case "$1" in
        --assets-dir|--prefix)
            [ "$#" -ge 2 ] || { usage >&2; exit 2; }
            case "$1" in
                --assets-dir) assets_dir="$2" ;;
                --prefix) prefix="$2" ;;
            esac
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[ -n "$assets_dir" ] && [ -n "$prefix" ] || { usage >&2; exit 2; }
[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || {
    echo 'Linux x86_64 is required.' >&2; exit 2;
}
for tool in python3 dpkg-deb mktemp mv; do
    command -v "$tool" >/dev/null || { echo "Required tool missing: $tool" >&2; exit 2; }
done
assets_dir="$(cd -- "$assets_dir" && pwd)"
prefix="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$prefix")"
[ ! -e "$prefix" ] && [ ! -L "$prefix" ] || {
    echo 'Installation prefix already exists; choose a new directory.' >&2; exit 2;
}
docker_archive="$assets_dir/docker-28.5.2.tgz"
fuse_packages=(
    "$assets_dir/fuse-debs/fuse-overlayfs_1.13-1_amd64.deb"
    "$assets_dir/fuse-debs/fuse3_3.14.0-5build1_amd64.deb"
    "$assets_dir/fuse-debs/libfuse3-3_3.14.0-5build1_amd64.deb"
)
for package in "$docker_archive" "${fuse_packages[@]}"; do
    [ -s "$package" ] || { echo "Required installation package missing: $package" >&2; exit 2; }
done

mkdir -p -- "$(dirname -- "$prefix")"
staging="$(mktemp -d "${prefix}.install.XXXXXX")"
trap 'rm -rf -- "$staging"' EXIT
python3 - "$docker_archive" "$staging" <<'PY'
import sys
import tarfile
from pathlib import PurePosixPath

with tarfile.open(sys.argv[1], "r:gz") as archive:
    members = archive.getmembers()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "docker":
            raise ValueError("Docker archive has an unexpected path")
        if not (member.isfile() or member.isdir()):
            raise ValueError("Docker archive must contain only files and directories")
    archive.extractall(sys.argv[2], members=members, filter="data")
PY
mkdir -- "$staging/fuse"
for package in "${fuse_packages[@]}"; do
    [ "$(dpkg-deb -f "$package" Architecture)" = amd64 ] || {
        echo 'The fuse packages must target amd64.' >&2; exit 2;
    }
    dpkg-deb -x "$package" "$staging/fuse"
done
for binary in docker dockerd docker-init containerd containerd-shim-runc-v2 runc; do
    test -x "$staging/docker/$binary"
done
"$staging/docker/docker" --version | grep -F 'Docker version 28.5.2,' >/dev/null
LD_LIBRARY_PATH="$staging/fuse/usr/lib/x86_64-linux-gnu:$staging/fuse/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$staging/fuse/usr/bin/fuse-overlayfs" --version >/dev/null
mv -T --no-clobber -- "$staging" "$prefix"
[ ! -d "$staging" ] || { echo 'Prefix appeared during installation; nothing replaced.' >&2; exit 2; }
echo "OCI tools installed under $prefix"
