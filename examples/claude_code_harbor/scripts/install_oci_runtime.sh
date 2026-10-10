#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Download and install the pinned r5 OCI tools into a private directory.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage: install_oci_runtime.sh --prefix DIR [--cache-dir DIR]
       install_oci_runtime.sh --prefix DIR --assets-dir DIR

Without --assets-dir, download pinned packages from Docker/Ubuntu official
sources and verify SHA-256 before extraction. --cache-dir defaults to
PREFIX.packages; verified cached files are reused. Optional mirror flags:
  --docker-base-url URL  Static stable x86_64 directory
  --ubuntu-base-url URL  Ubuntu archive root (containing pool/)
With --assets-dir, verify local packages only; no network access is made.

Run on Linux x86_64. The prefix must not exist. No sudo, package-manager
installation, daemon startup, or training operation is performed.

Required files under --assets-dir:
  docker-28.5.2.tgz
  fuse-debs/fuse-overlayfs_1.13-1_amd64.deb
  fuse-debs/fuse3_3.14.0-5build1_amd64.deb
  fuse-debs/libfuse3-3_3.14.0-5build1_amd64.deb

Packages and generated runtime directories are not Git inputs. No saved r5
installation packages or installed environment are required for online mode.

Output layout:
  PREFIX/docker/    Docker, dockerd, containerd, runc
  PREFIX/fuse/      Extracted fuse-overlayfs and libfuse packages

NeMo Gym and the Relax training image are existing prerequisites. Each task's
Claude Code/SRT installation is handled by install_task_runtime.sh inside its
own OCI container. Task images and verifier files come from prepared data.
EOF
}

assets_dir=
cache_dir=
prefix=
mirror_args=()
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
        --cache-dir|--docker-base-url|--ubuntu-base-url)
            [ "$#" -ge 2 ] || { usage >&2; exit 2; }
            if [ "$1" = --cache-dir ]; then
                cache_dir="$2"
            else
                mirror_args+=("$1" "$2")
            fi
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[ -n "$prefix" ] || { usage >&2; exit 2; }
if [ -n "$assets_dir" ] && { [ -n "$cache_dir" ] || [ "${#mirror_args[@]}" -gt 0 ]; }; then
    echo '--assets-dir is offline mode; do not combine it with download options.' >&2; exit 2
fi
[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || {
    echo 'Linux x86_64 is required.' >&2; exit 2;
}
for tool in python3 dpkg-deb mktemp mv; do
    command -v "$tool" >/dev/null || { echo "Required tool missing: $tool" >&2; exit 2; }
done
source "${SCRIPT_DIR}/cloud_paths.sh"
harbor_resolve_paths prefix assets_dir cache_dir
[ ! -e "$prefix" ] && [ ! -L "$prefix" ] || {
    echo 'Installation prefix already exists; choose a new directory.' >&2; exit 2;
}
if [ -n "$assets_dir" ]; then
    python3 "$SCRIPT_DIR/fetch_oci_packages.py" --assets-dir "$assets_dir" --check-only
else
    assets_dir="${cache_dir:-${prefix}.packages}"
    python3 "$SCRIPT_DIR/fetch_oci_packages.py" --assets-dir "$assets_dir" "${mirror_args[@]}"
fi
assets_dir="$(cd -- "$assets_dir" && pwd)"
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
