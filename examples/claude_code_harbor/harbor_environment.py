# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bounded Apptainer environment for Harbor trials inside a training
container."""

from __future__ import annotations

import asyncio
import os
import secrets
import shutil
import signal
import sys
from collections import deque
from pathlib import Path
from typing import Any

import httpx
from harbor.environments.capabilities import EnvironmentResourceCapabilities
from harbor.environments.singularity import SingularityEnvironment

from examples.claude_code_harbor.sif_cache import SifCache


class BoundedApptainerEnvironment(SingularityEnvironment):
    """Adapt Harbor's Singularity backend to a bounded Relax Session.

    Harbor's default launch omits cgroup limits and preserves converted images.
    This variant adds CPU/memory/PID limits, a leased SIF, the prepared MiMo
    runtime and an authenticated server, then cleans up within Relax's TERM
    window.
    """

    def __init__(
        self,
        *args,
        pids_limit: int,
        **kwargs,
    ) -> None:
        self._pids_limit = pids_limit
        self._sif_cache_root = Path(os.environ["HARBOR_SIF_CACHE_ROOT"])
        self._image_context = None
        self._image_lease = None
        self._stop_task = None
        self._exec_token = secrets.token_hex(32)
        self._server_output_tail: deque[str] = deque(maxlen=64)
        super().__init__(*args, singularity_image_cache_dir=self._sif_cache_root, **kwargs)

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities(cpu_limit=True, memory_limit=True, fractional_cpus=True)

    def _resolve_workdir(self) -> str:
        return self.task_env_config.workdir

    async def start(self, force_build: bool) -> None:
        try:
            cache = SifCache(self._sif_cache_root)
            self._image_context = cache.acquire(self._docker_image)
            self._image_lease = await self._image_context.__aenter__()
            self._sif_path = self._image_lease.path
            await self._start_server()
            await self._upload_environment_dir_after_start()
        except BaseException:
            await self.stop(delete=True)
            raise

    async def _start_server(self) -> None:
        self._staging_dir = Path(self.trial_paths.trial_dir) / "apptainer-staging"
        self._staging_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        sandbox_tmp = self._staging_dir / "tmp"
        sandbox_tmp.mkdir(mode=0o1777)
        sandbox_tmp.chmod(0o1777)

        # Installed Harbor may live outside the Relax checkout. Resolve assets from
        # the owning Harbor module rather than assuming a repository layout.
        import harbor.environments.singularity.singularity as singularity_module

        source_dir = Path(singularity_module.__file__).parent
        installed = self._sif_cache_root / "runtime"
        shutil.copy(source_dir / "server.py", self._staging_dir / "_hbexec.py")
        shutil.copy(Path(__file__).with_name("sif_cache.py"), self._staging_dir / "execution_server.py")
        (self._staging_dir / "server-token").write_text(self._exec_token)
        recipe = Path(__file__).parent
        system_prompt = self._staging_dir / "system_prompt.txt"
        system_prompt_target = "/run/claude-code-system-prompt.txt"
        shutil.copy(recipe / "bootstrap.sh", self._staging_dir / "bootstrap.sh")
        system_prompt.write_text((recipe / "system_prompt.txt").read_text().rstrip("\n"))
        shutil.copy(self.environment_dir.parent / "tests/git_targets.json", self._staging_dir / "git_targets.json")
        protected = '"/opt/mimo-verifier-state"'
        claude_exec = 'exec "$SRT_BIN" --settings "$SETTINGS" -- "$REAL" "$@"'
        source_wrapper = (installed / "tmp/claude_srt_wrapper.sh").read_text()
        if claude_exec not in source_wrapper:
            raise RuntimeError("Claude SRT wrapper does not match the expected command")
        claude_command = claude_exec[len("exec ") :].replace(
            '"$@"', f'--system-prompt-file {system_prompt_target} "$@"'
        )
        normalized_claude_exec = (
            f"if {claude_command}; then\n"
            "  claude_exit_code=0\n"
            "else\n"
            "  claude_exit_code=$?\n"
            "fi\n"
            # Preserve Claude's exit status for diagnostics while allowing Harbor to run the verifier.
            "printf '%s\\n' \"$claude_exit_code\" > /logs/agent/claude-exit-code.txt\n"
            "exit 0"
        )
        wrapper = source_wrapper.replace(
            protected, '"/staging", "/run/agent-runtime", "/run/harbor-image.lock", "/opt/python", ' + protected
        ).replace(claude_exec, normalized_claude_exec)
        (self._staging_dir / "claude_wrapper.sh").write_text(wrapper)

        bind_mounts = ["-B", f"{self._staging_dir}:/staging"]
        bind_mounts.extend(["-B", f"{system_prompt}:{system_prompt_target}:ro"])
        bind_mounts.extend(["-B", f"{sandbox_tmp}:/tmp"])
        bind_mounts.extend(["-B", f"{self._image_lease.lock_path}:/run/harbor-image.lock:ro"])
        bind_mounts.extend(["-B", f"{installed}:/run/agent-runtime:ro"])
        bind_mounts.extend(["-B", f"{installed / 'opt/python'}:/opt/python:ro"])
        bind_mounts.extend(["-B", f"{installed / 'opt/srt-bins'}:/opt/srt-bins:ro"])
        bind_mounts.extend(["-B", f"{installed / 'usr/local/lib/node_modules'}:/usr/local/lib/node_modules:ro"])
        for mount in self._mounts:
            if mount.get("type") == "bind":
                mode = ":ro" if mount.get("read_only") else ""
                bind_mounts.extend(["-B", f"{mount['source']}:{mount['target']}{mode}"])

        no_mount_args = ["--no-mount", "home", "--no-mount", "tmp", "--no-mount", "bind-paths"]
        resource_args = [
            "--pids-limit",
            str(self._pids_limit),
            "--cpus",
            str(self._effective_cpus),
            "--memory",
            f"{self._effective_memory_mb}M",
            "--memory-swap",
            f"{self._effective_memory_mb}M",
        ]

        max_port_retries = 3
        last_error = None
        for port_attempt in range(max_port_retries):
            self._server_output_tail.clear()
            reserved_socket, self._server_port = self._reserve_port()
            bootstrap_cmd = [
                "bash",
                "/staging/bootstrap.sh",
                "/staging/execution_server.py",
                "serve",
                "--port",
                str(self._server_port),
                "--workdir",
                self._workdir,
            ]
            command = [
                "apptainer",
                "exec",
                *resource_args,
                *no_mount_args,
                "--pwd",
                self._workdir,
                "--writable-tmpfs",
                "--contain",
                "--pid",
                "--cleanenv",
                "--no-home",
                *bind_mounts,
                str(self._sif_path),
                *bootstrap_cmd,
            ]

            self.logger.debug(
                f"Starting Apptainer execution server on port {self._server_port} "
                f"(attempt {port_attempt + 1}/{max_port_retries})"
            )
            reserved_socket.close()
            supervisor = [
                sys.executable,
                str(Path(__file__).with_name("sif_cache.py")),
                "run",
                str(os.getpid()),
                *command,
            ]
            self._server_process = await asyncio.create_subprocess_exec(
                *supervisor,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            self._stream_task = asyncio.create_task(self._stream_server_output())
            self._http_client = httpx.AsyncClient(
                timeout=30.0, headers={"Authorization": f"Bearer {self._exec_token}"}
            )

            server_ready = False
            for _ in range(60):
                try:
                    response = await self._http_client.get(f"http://127.0.0.1:{self._server_port}/health")
                    if response.status_code == 200:
                        if self._server_process.returncode is not None:
                            await self._stream_task
                            last_error = RuntimeError(
                                f"Port collision on {self._server_port}: health check succeeded "
                                "but our server process died"
                            )
                            self.logger.warning(
                                f"Health check reached another server on port {self._server_port}; retrying"
                            )
                            break
                        server_ready = True
                        break
                except httpx.RequestError:
                    pass

                if self._server_process.returncode is not None:
                    await self._stream_task
                    diagnostic = "\n".join(self._server_output_tail)
                    last_error = RuntimeError(
                        f"Bounded Apptainer server exited during startup ({self._server_process.returncode})\n"
                        f"{diagnostic}"
                    )
                    self.logger.warning(
                        f"Execution server failed on port {self._server_port}; retrying with a new port"
                    )
                    break
                await asyncio.sleep(1)

            if server_ready:
                return

            if self._server_process.returncode is None:
                os.killpg(self._server_process.pid, signal.SIGKILL)
                await self._server_process.wait()
            if not self._stream_task.done():
                self._stream_task.cancel()
                await asyncio.gather(self._stream_task, return_exceptions=True)

            await self._http_client.aclose()
            self._http_client = None
            self._server_process = None
            self._stream_task = None

        raise last_error or RuntimeError("Bounded Apptainer server failed to start after retries")

    async def _stream_server_output(self) -> None:
        if not self._server_process or not self._server_process.stdout:
            return
        async for line in self._server_process.stdout:
            decoded = line.decode(errors="replace").rstrip()
            if decoded:
                self._server_output_tail.append(decoded)
                self.logger.debug(f"[server] {decoded}")

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> Any:
        # Harbor phase timeouts own termination; an HTTP default must span them.
        return await super().exec(
            command, cwd=cwd, env=env, timeout_sec=timeout_sec if timeout_sec is not None else 2_147_483, user=user
        )

    async def prepare_verification(self) -> None:
        """Clear verifier directories before Harbor injects trusted tests."""
        from pathlib import PurePosixPath

        result = await self.empty_dirs([PurePosixPath("/tests"), PurePosixPath("/logs/verifier")], chmod=False)
        if result.return_code != 0:
            raise RuntimeError("Failed to clear the shared verifier directories")

    async def stop(self, delete: bool) -> None:
        if self._stop_task is None:
            self._stop_task = asyncio.create_task(self._stop())
        try:
            await asyncio.shield(self._stop_task)
        except asyncio.CancelledError:
            await self._stop_task
            raise

    async def _stop(self) -> None:
        """Fit inner cleanup inside Relax's outer five-second TERM window."""
        if self._server_process:
            try:
                os.killpg(self._server_process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self._server_process.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
            try:
                os.killpg(self._server_process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if self._server_process.returncode is None:
                await self._server_process.wait()
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            await asyncio.gather(self._stream_task, return_exceptions=True)
        try:
            if self._http_client:
                await self._http_client.aclose()
                self._http_client = None
        finally:
            try:
                if self._staging_dir and self._staging_dir.exists():
                    shutil.rmtree(self._staging_dir)
            finally:
                if self._image_lease is not None:
                    await self._image_context.__aexit__(None, None, None)
                    self._image_lease = None
