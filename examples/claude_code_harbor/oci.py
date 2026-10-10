# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Private, persistent OCI task environments for the NeMo CC harness."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


class _ImageCache:
    """One service process owns admission and unused images on its private
    daemon."""

    instances: ClassVar[dict[tuple[str, ...], _ImageCache]] = {}

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.pulls = asyncio.Semaphore(4)
        self.references: dict[str, int] = {}
        self.unused: OrderedDict[str, None] = OrderedDict()
        self.limit = int(os.environ.get("CLAUDE_CODE_MAX_CACHED_IMAGES", "4"))
        if self.limit < 0:
            raise ValueError("CLAUDE_CODE_MAX_CACHED_IMAGES must be nonnegative")


class DockerSandbox:
    """Only the named container and its explicit staging mounts are owned
    here."""

    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata
        self.name = "relax-harbor-" + uuid4().hex
        self.docker = [os.environ["CLAUDE_CODE_DOCKER_BINARY"], "--host=" + os.environ["CLAUDE_CODE_DOCKER_HOST"]]
        key = tuple(self.docker)
        if key not in _ImageCache.instances:
            _ImageCache.instances[key] = _ImageCache()
        self.cache = _ImageCache.instances[key]
        self.image: str | None = None
        self.root = Path(tempfile.mkdtemp(prefix=self.name + "-"))
        self.process: asyncio.subprocess.Process | None = None
        self.stderr: Any = None
        self.closed = False

    async def command(self, *args: str, timeout: float = 120) -> str:
        proc = await asyncio.create_subprocess_exec(
            *self.docker, *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout)
        except BaseException:
            if proc.returncode is None:
                proc.kill()
            await proc.communicate()
            raise
        if proc.returncode:
            raise RuntimeError(f"Owned OCI operation {args[0]} failed: {stderr.decode(errors='replace')[-2000:]}")
        return stdout.decode()

    async def start(self) -> None:
        metadata = self.metadata
        image = metadata["sandbox_image"]
        mapping_file = os.environ.get("CLAUDE_CODE_IMAGE_MAP")
        if mapping_file:
            image = json.loads(Path(mapping_file).read_text())[image]
        if image.endswith("-obd"):
            raise ValueError(
                "Docker requires the source ordinary OCI reference; OverlayBD needs its native snapshotter"
            )
        for name in ("control", "assets", "logs", "verifier"):
            (self.root / name).mkdir()
        shutil.copy2(Path(__file__).with_name("sandbox_worker.py"), self.root / "control/worker.py")
        shutil.copy2(Path(__file__).with_name("claude_settings.json"), self.root / "control/settings.json")
        commands = ["set -euo pipefail"]
        targets: Path | None = None
        for index, upload in enumerate(metadata["sandbox_extra_params"]["nex_uploads"]):
            source, destination = Path(upload["src"]), Path(upload["dst"])
            if (
                not source.is_file()
                or source.is_symlink()
                or not destination.is_absolute()
                or ".." in destination.parts
            ):
                raise ValueError("Task uploads require regular sources and absolute container destinations")
            mode = int(str(upload.get("mode", 644)), 8)
            if mode < 0 or mode > 0o777:
                raise ValueError("Task uploads must not set special permissions")
            shutil.copyfile(source, self.root / "assets" / str(index))
            commands.append(f"install -D -m {mode:o} /run/relax-assets/{index} {shlex.quote(str(destination))}")
            if str(destination) == "/tmp/mimo_git_targets.json":
                targets = source
        matches = re.findall(
            r"--config-sha256\s+([0-9a-f]{64})",
            "\n".join(item["cmd"] for item in metadata["sandbox_extra_params"]["nex_post_upload_cmds"]),
        )
        if len(matches) != 1 or targets is None or hashlib.sha256(targets.read_bytes()).hexdigest() != matches[0]:
            raise ValueError("Source anti-leak configuration checksum is missing or mismatched")
        self.git_config_sha256 = matches[0]
        commands.append(r"""
PYGIT=""
for candidate in /opt/python/bin/python3.12 python3.13 python3.12 python3.11 python3.10 python3.9 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys;raise SystemExit(0 if sys.version_info >= (3,9) else 1)' >/dev/null 2>&1; then
    PYGIT="$candidate"; break
  fi
done
if [ -z "$PYGIT" ]; then
  mkdir -p /opt
  cat /tmp/py-chunks/py-chunk-* | tar xzf - -C /opt
  PYGIT=/opt/python/bin/python3.12
fi
exec "$PYGIT" /run/relax-control/worker.py
""")
        (self.root / "control/boot.sh").write_text("\n".join(commands))
        memory = re.fullmatch(r"(\d+)Mi", str(metadata["sandbox_memory"]))
        if memory is None:
            raise ValueError("The source memory limit must use Mi units")
        argv = [
            "create",
            "--name",
            self.name,
            "--label",
            "ai.relax.recipe=claude-code-harbor",
            "--init",
            "--interactive",
            "--network",
            "host",
            "--user",
            "0:0",
            "--cpus",
            str(metadata["sandbox_cpu"]),
            "--memory",
            str(int(memory[1]) * 1024 * 1024),
            "--cap-add",
            "SYS_ADMIN",
            "--security-opt",
            "seccomp=unconfined",
            "--add-host",
            "srtllm.internal:127.0.0.1",
            "--env",
            "RELAX_HARBOR_PRIVATE_CONTAINER=1",
        ]
        for name, destination, mode in (
            ("control", "/run/relax-control", "ro"),
            ("assets", "/run/relax-assets", "ro"),
            ("verifier", "/run/relax-verifier", "ro"),
            ("logs", "/logs", "rw"),
        ):
            argv.extend(["--volume", f"{self.root / name}:{destination}:{mode}"])
        argv.extend(["--entrypoint", "/bin/bash", image, "/run/relax-control/boot.sh"])
        # Finish the bounded create operation before cancellation can remove its
        # name; otherwise a late daemon reply could leave an orphan container.
        # Reserve before pulling so eviction cannot race a not-yet-created
        # container. Bound concurrent downloads without blocking cleanup.
        async with self.cache.lock:
            self.image = image
            self.cache.references[image] = self.cache.references.get(image, 0) + 1
            self.cache.unused.pop(image, None)
        async with self.cache.pulls:
            await self.command("pull", image, timeout=1800)
            creation = asyncio.create_task(self.command(*argv))
            try:
                await asyncio.shield(creation)
            except asyncio.CancelledError:
                await asyncio.shield(creation)
                raise
        self.stderr = (self.root / "container-stderr.log").open("wb")
        self.process = await asyncio.create_subprocess_exec(
            *self.docker,
            "start",
            "--attach",
            "--interactive",
            self.name,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=self.stderr,
        )
        await self.receive(1800)
        await self.request(
            "initialize",
            2700,
            workdir=metadata["backend_options"]["workdir"],
            git_config_sha256=self.git_config_sha256,
        )
        for path in (self.root / "assets").iterdir():
            path.unlink()

    async def receive(self, timeout: float | None) -> dict[str, Any]:
        assert self.process is not None and self.process.stdout is not None
        line = await asyncio.wait_for(self.process.stdout.readline(), timeout)
        if not line:
            raise RuntimeError("Private OCI controller exited")
        result = json.loads(line)
        if result.get("ok") is not True:
            raise RuntimeError(f"Private OCI controller failed: {result.get('error_type')}: {result.get('error')}")
        return result

    async def request(self, operation: str, wait_timeout: float | None, **kwargs: Any) -> dict[str, Any]:
        assert self.process is not None and self.process.stdin is not None
        self.process.stdin.write((json.dumps({"operation": operation, **kwargs}) + "\n").encode())
        await self.process.stdin.drain()
        return await self.receive(wait_timeout)

    def stop_agent(self) -> None:
        (self.root / "control/stop-agent").touch()

    def read_log(self, relative: str) -> str:
        path = self.root / "logs" / relative
        path.resolve(strict=True).relative_to((self.root / "logs").resolve(strict=True))
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise ValueError("OCI logs must be regular files inside the owned log directory")
            return source.read().decode(errors="replace")

    async def verify(self) -> dict[str, Any]:
        source = Path(self.metadata["harbor_tests_dir"])
        for path in source.iterdir():
            if not path.is_file() or path.is_symlink():
                raise ValueError("Harbor tests must be regular files")
            shutil.copy2(path, self.root / "verifier" / path.name)
        timeout = float(self.metadata["verifier_timeout"])
        return await self.request("verify", timeout + 30, timeout=timeout)

    async def stop(self) -> None:
        if self.closed:
            return
        # Removing the owned container kills its complete private PID tree,
        # even when a CC tool escaped its parent's process group.
        result = await self.command("container", "ls", "--all", "--quiet", "--filter", "name=^/" + self.name + "$")
        if result.strip():
            await self.command("rm", "--force", self.name)
        if self.process is not None:
            await asyncio.wait_for(self.process.wait(), 10)
        if self.stderr is not None:
            self.stderr.close()
        async with self.cache.lock:
            if self.image is not None:
                image, self.image = self.image, None
                self.cache.references[image] -= 1
                if self.cache.references[image] == 0:
                    self.cache.unused[image] = None
            while len(self.cache.unused) > self.cache.limit:
                image, _ = self.cache.unused.popitem(last=False)
                try:
                    await self.command("image", "rm", image)
                except RuntimeError:
                    self.cache.unused[image] = None
                    logger.warning("Unused recipe image could not be evicted: %s", image)
                    break
                else:
                    self.cache.references.pop(image, None)

    async def close(self) -> None:
        if self.closed:
            return
        await self.stop()
        shutil.rmtree(self.root)
        self.closed = True
