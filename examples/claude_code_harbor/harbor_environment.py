# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bounded Apptainer environment for Harbor trials inside a training
container."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import signal
import sys
import time
from itertools import islice
from pathlib import Path
from typing import Any

import httpx
from harbor.environments.capabilities import EnvironmentResourceCapabilities
from harbor.environments.singularity import SingularityEnvironment

from examples.claude_code_harbor.loop_devices import restore_missing_loop_device
from examples.claude_code_harbor.resource_release import ResourceReleaseGuard
from examples.claude_code_harbor.sif_cache import SifCache, _resource_release_termination_profile


def _shutdown_process_snapshot(pid: int, *, task_owner_pid: int | None = None) -> dict[str, Any]:
    """Bounded failure diagnostics; never read process arguments or
    environment."""
    result: dict[str, Any] = {"pid": pid}
    base = Path(f"/proc/{pid}") if task_owner_pid is None else Path(f"/proc/{task_owner_pid}/task/{pid}")
    try:
        with (base / "stat").open(errors="replace") as source:
            fields = source.read(4096).rsplit(")", 1)[1].split()
        result.update(state=fields[0], ppid=int(fields[1]), pgid=int(fields[2]), starttime=int(fields[19]))
        with (base / "wchan").open(errors="replace") as source:
            result["wchan"] = source.read(64).strip()
    except (OSError, IndexError, ValueError) as error:
        result["read_error"] = type(error).__name__
    return result


def _shutdown_wait_snapshot(supervisor_pid: int, children: list[dict[str, Any]]) -> dict[str, Any]:
    """Inspect one owned wait target on failure; this is never release
    evidence."""
    result: dict[str, Any] = {"started_monotonic": time.monotonic()}

    def read(path: Path, limit: int) -> str | None:
        try:
            with path.open(errors="replace") as source:
                value = source.read(limit + 1)
            if len(value) > limit:
                result.setdefault("truncated_reads", []).append(path.name)
            return value[:limit]
        except OSError as error:
            result.setdefault("read_errors", []).append({"file": path.name, "error": type(error).__name__})
            return None

    def status(pid: int, *, task_owner_pid: int | None = None) -> dict[str, str] | None:
        base = Path(f"/proc/{pid}") if task_owner_pid is None else Path(f"/proc/{task_owner_pid}/task/{pid}")
        value = read(base / "status", 8192)
        if value is None:
            return None
        allowed = {
            "State",
            "Tgid",
            "Pid",
            "PPid",
            "Threads",
            "TracerPid",
            "NSpid",
            "NStgid",
            "SigPnd",
            "ShdPnd",
            "SigBlk",
        }
        return {
            key: data.strip()[:80]
            for line in value.splitlines()
            if ":" in line
            for key, data in [line.split(":", 1)]
            if key in allowed
        }

    before = _shutdown_process_snapshot(supervisor_pid)
    result["supervisor"] = {**before, "status": status(supervisor_pid)}
    value = read(Path(f"/proc/{supervisor_pid}/syscall"), 256)
    result["wait_syscall"] = {"state": "unavailable"}
    if value is not None:
        try:
            fields = value.split()
            number = int(fields[0])
            # Linux syscall numbers below are specific to the deployed x86-64 hosts.
            if os.uname().machine == "x86_64" and number in (61, 247):
                args = [int(field, 0) for field in fields[1:7]]
                result["wait_syscall"] = (
                    {"name": "wait4", "target": args[0], "options": args[2]}
                    if number == 61
                    else {"name": "waitid", "idtype": args[0], "target": args[1], "options": args[3]}
                )
            else:
                result["wait_syscall"] = {"state": "not_wait_or_unsupported_arch"}
        except (ValueError, IndexError):
            result["wait_syscall"] = {"state": "running_or_unparsed"}
    owned = [
        child
        for child in children
        if child.get("ppid") == supervisor_pid and child.get("pgid") == child.get("pid") and "starttime" in child
    ]
    # One detailed child keeps the existing 16 KiB IPC diagnostic envelope bounded.
    wait_target = result["wait_syscall"].get("target") if result["wait_syscall"].get("name") == "wait4" else None
    child = next((child for child in owned if child["pid"] == wait_target), None)
    result["child_selection"] = "wait4_target" if child is not None else "owned_zombie_or_first_fallback"
    if child is None:
        child = next((child for child in owned if child.get("state") == "Z"), owned[0] if owned else None)
    if child is not None:
        pid = child["pid"]
        detail: dict[str, Any] = {**child, "status": status(pid), "threads": []}
        result["child"] = detail
        try:
            with os.scandir(f"/proc/{pid}/task") as entries:
                tids = [entry.name for entry in islice(entries, 9)]
            detail["thread_list_truncated"] = len(tids) > 8
            stack_budget = 2
            for tid in tids[:8]:
                if not tid.isdecimal():
                    continue
                row = _shutdown_process_snapshot(int(tid), task_owner_pid=pid)
                if row.get("state") in ("R", "S", "D", "t", "T", "I") and stack_budget:
                    base = Path(f"/proc/{pid}/task/{tid}")
                    row["status"] = status(int(tid), task_owner_pid=pid)
                    value = read(base / "syscall", 256)
                    row["syscall"] = {"state": "unavailable"}
                    if value is not None:
                        # Arguments can contain user pointers; retain only the
                        # syscall number, never its arguments or registers.
                        try:
                            number = int(value.split()[0])
                            row["syscall"] = {"number": number} if number >= 0 else {"state": "not_in_syscall"}
                        except (ValueError, IndexError):
                            row["syscall"] = {"state": "running_or_unparsed"}
                    row["kernel_stack"] = read(base / "stack", 512)
                    stack_budget -= 1
                detail["threads"].append(row)
        except OSError as error:
            detail["thread_read_error"] = type(error).__name__
        after = _shutdown_process_snapshot(pid)
        detail["identity_stable"] = after.get("starttime") == child["starttime"]
        detail["after"] = after
    after = _shutdown_process_snapshot(supervisor_pid)
    result["supervisor_identity_stable"] = "starttime" in before and after.get("starttime") == before["starttime"]
    result["finished_monotonic"] = time.monotonic()
    # Preserve room for the existing child identities and redacted server tail.
    if len(json.dumps(result, ensure_ascii=False)) > 6000:
        result["snapshot_truncated"] = True
        for row in result.get("child", {}).get("threads", []):
            row.pop("kernel_stack", None)
        threads = result.get("child", {}).get("threads", [])
        while threads and len(json.dumps(result, ensure_ascii=False)) > 6000:
            threads.pop()
            result["child"]["thread_list_truncated"] = True
    return result


def _build_claude_wrapper(source_wrapper: str, system_prompt_target: str) -> str:
    protected = '"/opt/mimo-verifier-state"'
    real_assignment = 'REAL="${SRT_WRAP_REAL:-/usr/local/bin/claude.real}"'
    claude_exec = 'exec "$SRT_BIN" --settings "$SETTINGS" -- "$REAL" "$@"'
    if (
        source_wrapper.count(real_assignment) != 1
        or source_wrapper.count(claude_exec) != 1
        or source_wrapper.count('"denyWrite": [') != 1
    ):
        raise RuntimeError("Claude SRT wrapper does not match the expected command")
    # Harbor probes the version with API variables already set. This metadata-only
    # invocation must not create a network namespace or inject the agent prompt.
    source_wrapper = source_wrapper.replace(
        real_assignment,
        real_assignment + '\nif [[ $# == 1 && ( "$1" == --version || "$1" == -v ) ]]; then\n  exec "$REAL" "$@"\nfi',
        1,
    )
    claude_command = claude_exec[len("exec ") :].replace('"$@"', f'--system-prompt-file {system_prompt_target} "$@"')
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
    return (
        source_wrapper.replace(
            protected, '"/staging", "/run/agent-runtime", "/run/harbor-image.lock", "/opt/python", ' + protected
        )
        .replace('"denyWrite": [', '"denyWrite": ["/run/relax-node-cpus.cjs", ', 1)
        .replace(claude_exec, normalized_claude_exec)
    )


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
        self._resource_guard = ResourceReleaseGuard()
        self._pids_limit = pids_limit
        self._sif_cache_root = Path(os.environ["HARBOR_SIF_CACHE_ROOT"])
        self._image_context = None
        self._image_lease = None
        self._start_task = None
        self._stop_task = None
        self._exec_token = secrets.token_hex(32)
        self._server_log_path: Path | None = None
        self._same_root_identity = False
        super().__init__(*args, singularity_image_cache_dir=self._sif_cache_root, **kwargs)

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities(cpu_limit=True, memory_limit=True, fractional_cpus=True)

    def _resolve_workdir(self) -> str:
        return self.task_env_config.workdir

    async def start(self, force_build: bool) -> None:
        if self._stop_task is not None:
            raise RuntimeError("Cannot start an environment after stop was requested")
        if self._start_task is not None:
            raise RuntimeError("Environment startup already requested")
        self._start_task = asyncio.create_task(self._start())
        try:
            await asyncio.shield(self._start_task)
        except BaseException:
            await self.stop(delete=True)
            raise

    async def _start(self) -> None:
        self._same_root_identity = False
        started = asyncio.get_running_loop().time()
        phase = "sif_acquire"
        try:
            cache = SifCache(self._sif_cache_root)
            self._image_context = cache.acquire(self._docker_image)
            self._image_lease = await self._image_context.__aenter__()
            self._sif_path = self._image_lease.path
            if self._stop_task is not None:
                raise asyncio.CancelledError
            phase = "execution_server"
            await self._start_server()
            if self._stop_task is not None:
                raise asyncio.CancelledError
            phase = "environment_upload"
            await self._upload_environment_dir_after_start()
        except BaseException as error:
            elapsed = asyncio.get_running_loop().time() - started
            self.logger.warning(
                f"Apptainer startup failed in {phase} after {elapsed:.1f}s ({type(error).__name__}); "
                f"server_log={self._server_log_path}\n{self._server_diagnostic()}"
            )
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
        if self._resource_guard.required:
            shutil.copy(recipe / "resource_release.py", self._staging_dir / "resource_release.py")
        shutil.copy(recipe / "git_prepare.py", self._staging_dir / "git_prepare.py")
        node_cpu_helper = self._staging_dir / "node_cpu_limit.cjs"
        shutil.copy(recipe / "node_cpu_limit.cjs", node_cpu_helper)
        system_prompt.write_text((recipe / "system_prompt.txt").read_text().rstrip("\n"))
        shutil.copy(self.environment_dir.parent / "tests/git_targets.json", self._staging_dir / "git_targets.json")
        source_wrapper = (installed / "tmp/claude_srt_wrapper.sh").read_text()
        wrapper = _build_claude_wrapper(source_wrapper, system_prompt_target)
        (self._staging_dir / "claude_wrapper.sh").write_text(wrapper)

        bind_mounts = ["-B", f"{self._staging_dir}:/staging"]
        bind_mounts.extend(["-B", f"{system_prompt}:{system_prompt_target}:ro"])
        bind_mounts.extend(["-B", f"{node_cpu_helper}:/run/relax-node-cpus.cjs:ro"])
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
            self._same_root_identity = False
            # Bind inside the execution server and keep that socket open. A host
            # reservation released before Git preparation leaves a long race.
            self._server_port = 0
            self._server_port_file = self._staging_dir / f"server-port-{port_attempt}"
            bootstrap_cmd = [
                "bash",
                "/staging/bootstrap.sh",
                "/staging/execution_server.py",
                "serve",
                "--port",
                str(self._server_port),
                "--port-file",
                f"/staging/{self._server_port_file.name}",
                "--workdir",
                self._workdir,
            ]
            if self._resource_guard.required:
                self._resource_nonce = secrets.token_hex(24)
                self._resource_identity_file = self._staging_dir / f"task-cgroup-{port_attempt}.json"
                self._resource_accepted_file = self._staging_dir / f"task-cgroup-{port_attempt}.accepted"
                bootstrap_cmd = [
                    "/opt/python/bin/python3.12",
                    "-I",
                    "/staging/resource_release.py",
                    f"/staging/{self._resource_identity_file.name}",
                    f"/staging/{self._resource_accepted_file.name}",
                    self._resource_nonce,
                    *bootstrap_cmd,
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
            supervisor = [
                sys.executable,
                str(Path(__file__).with_name("sif_cache.py")),
                "run",
                str(os.getpid()),
                *command,
            ]
            # A surviving descendant must not hold an asyncio PIPE open and make
            # Process.wait() hang after the supervisor has already exited.
            self._server_log_path = Path(self.trial_paths.trial_dir) / f"apptainer-server-{port_attempt}.log"
            self._resource_guard.before_start()
            with self._server_log_path.open("wb") as server_log:
                self._server_process = await asyncio.create_subprocess_exec(
                    *supervisor,
                    stdout=server_log,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                    env=self._resource_guard.supervisor_environment(),
                )
            if self._resource_guard.required:
                self._resource_guard.supervisor_pid = self._server_process.pid
            self._http_client = httpx.AsyncClient(
                timeout=30.0, headers={"Authorization": f"Bearer {self._exec_token}"}
            )

            try:
                await self._wait_for_server()
                return
            except RuntimeError as error:
                last_error = error
                self.logger.warning(f"{error}; retrying with a new port")

            await self._stop_server()
            if port_attempt + 1 < max_port_retries:
                try:
                    recovery = restore_missing_loop_device(str(last_error))
                except (OSError, RuntimeError, ValueError) as error:
                    raise RuntimeError(f"{last_error}\nLoop-device recovery failed: {error}") from error
                if recovery is not None:
                    self.logger.warning(f"Apptainer loop-device recovery before retry: {recovery}")

        raise last_error or RuntimeError("Bounded Apptainer server failed to start after retries")

    def _capture_task_resource(self) -> None:
        guard = self._resource_guard
        if not guard.required or guard.identity is not None:
            return
        try:
            identity = json.loads(self._resource_identity_file.read_text())
        except FileNotFoundError:
            return
        if identity.get("nonce") != self._resource_nonce:
            raise RuntimeError("Task cgroup startup identity differs")
        guard.capture(
            identity["cgroup"],
            cpus=self._effective_cpus,
            memory_bytes=self._effective_memory_mb * 2**20,
            pids=self._pids_limit,
        )
        self._resource_accepted_file.write_text(self._resource_nonce)

    def _server_diagnostic(self) -> str:
        if self._server_log_path is None or not self._server_log_path.exists():
            return ""
        with self._server_log_path.open("rb") as log:
            log.seek(max(0, log.seek(0, os.SEEK_END) - 4096))
            return log.read().decode(errors="replace")

    async def _wait_for_server(self) -> None:
        self._same_root_identity = False
        # Harbor owns the total environment-start budget, including SIF acquisition
        # and Git preparation. A live container may legitimately need over 60s.
        while self._server_process.returncode is None:
            self._capture_task_resource()
            if self._server_port == 0:
                try:
                    port = int(self._server_port_file.read_text())
                except FileNotFoundError:
                    await asyncio.sleep(1)
                    continue
                except ValueError as error:
                    raise RuntimeError("Execution server published an invalid port") from error
                if not 1 <= port <= 65535:
                    raise RuntimeError(f"Execution server published an invalid port: {port}")
                self._server_port = port
            try:
                response = await self._http_client.get(f"http://127.0.0.1:{self._server_port}/health")
                if response.status_code == 200 and self._server_process.returncode is None:
                    headers = getattr(response, "headers", {})
                    self._same_root_identity = all(
                        headers.get(f"x-relax-exec-{field}") == "0" for field in ("uid", "euid", "gid", "egid")
                    ) and headers.get("x-relax-exec-groups") in {"", "0"}
                    self.logger.debug(
                        f"Execution server identity permits direct root commands: {self._same_root_identity}"
                    )
                    return
            except httpx.RequestError:
                pass
            await asyncio.sleep(1)
        raise RuntimeError(
            f"Bounded Apptainer server exited during startup ({self._server_process.returncode}); "
            f"server_log={self._server_log_path}\n{self._server_diagnostic()}"
        )

    def _resolve_user(self, user: str | int | None) -> str | int | None:
        resolved = super()._resolve_user(user)
        # Trusted verifier housekeeping can still run if an agent damages passwd.
        # This uses the authenticated server's existing identity, not a user switch.
        if self._same_root_identity and type(resolved) is str and resolved == "root":
            return None
        return resolved

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
        cancelled = False
        while not self._stop_task.done():
            try:
                await asyncio.shield(self._stop_task)
            except asyncio.CancelledError:
                cancelled = True
        self._stop_task.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _stop_server(self) -> None:
        """Allow the supervisor to reap its separate sandbox process group."""
        process = self._server_process
        cleanup = {
            "returncode_before_stop": process.returncode if process is not None else None,
            "term_timeout": False,
            "reap_grace_timeout": False,
            "kill_timeout": False,
        }
        if self._resource_guard.required:
            # Retain syscall errors even when the original exception exits this
            # method before the later proof checks. The dictionary is diagnostic.
            self._resource_guard.supervisor_cleanup = cleanup

        def signal_group(pid: int, sig: int) -> None:
            record = {"target_pgid": pid, "signal": int(sig), "monotonic": time.monotonic()}
            try:
                os.killpg(pid, sig)
            except OSError as error:
                record.update(
                    outcome="not_found" if isinstance(error, ProcessLookupError) else "error", errno=error.errno
                )
                raise
            else:
                record["outcome"] = "syscall_returned"
            finally:
                # A successful signal syscall is not an exit/reap proof.
                records = cleanup.setdefault("group_signal_attempts", [])
                if len(records) < 8:
                    records.append(record)
                else:
                    cleanup["group_signal_attempts_truncated"] = True

        if process and self._resource_guard.required and self._resource_guard.identity is None:
            try:
                self._capture_task_resource()
            except Exception as error:
                self.logger.warning(f"Task identity capture failed before cleanup: {error}")
        if self._server_process:
            try:
                signal_group(self._server_process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(
                    self._server_process.wait(), timeout=11.0 if _resource_release_termination_profile() else 3.0
                )
            except asyncio.TimeoutError:
                cleanup["term_timeout"] = True
                self.logger.warning(f"Apptainer supervisor {self._server_process.pid} did not stop after TERM")
                cleanup["supervisor_at_term_timeout"] = _shutdown_process_snapshot(self._server_process.pid)
                # The supervisor launches Apptainer in a separate process group.
                # Kill its verified direct-child groups before killing the owner.
                children = Path(f"/proc/{self._server_process.pid}/task/{self._server_process.pid}/children")
                try:
                    child_pids = children.read_text().split()
                except FileNotFoundError:
                    child_pids = []
                cleanup["children_at_term_timeout"] = [_shutdown_process_snapshot(int(pid)) for pid in child_pids[:16]]
                try:
                    cleanup["wait_at_term_timeout"] = _shutdown_wait_snapshot(
                        self._server_process.pid, cleanup["children_at_term_timeout"]
                    )
                except Exception as error:
                    cleanup["wait_snapshot_error"] = type(error).__name__
                for child_pid in child_pids:
                    try:
                        fields = Path(f"/proc/{child_pid}/stat").read_text().rsplit(")", 1)[1].split()
                        if int(fields[1]) == self._server_process.pid and int(fields[2]) == int(child_pid):
                            signal_group(int(child_pid), signal.SIGKILL)
                    except (FileNotFoundError, ProcessLookupError):
                        pass
                # Killing the child group must not immediately kill its only
                # subreaper. Let it join children and publish the original
                # nonce-bound proof; an exit alone still proves nothing.
                try:
                    await asyncio.wait_for(self._server_process.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    cleanup["reap_grace_timeout"] = True
                    try:
                        signal_group(self._server_process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        await asyncio.wait_for(self._server_process.wait(), timeout=0.5)
                    except asyncio.TimeoutError:
                        cleanup["kill_timeout"] = True
                        self.logger.warning(f"Apptainer supervisor {self._server_process.pid} still has not exited")
            self._server_process = None
        if process is not None and self._resource_guard.required:
            cleanup["returncode"] = process.returncode
            cleanup["server_log_path"] = str(self._server_log_path) if self._server_log_path is not None else None
            if not self._resource_guard.supervisor_reaped():
                # Retain failure evidence before the Session's temporary IO is
                # removed. This is not a resource-release proof or a sampler.
                from examples.claude_code_harbor.diagnostics import _redact_text

                try:
                    tail = self._server_diagnostic()
                    for secret in (
                        getattr(self, "_exec_token", None),
                        getattr(self, "_resource_nonce", None),
                        self._resource_guard.attempt_nonce,
                        os.environ.get("RELAX_RESOURCE_RELEASE_NONCE"),
                    ):
                        if secret:
                            tail = tail.replace(secret, "<redacted>")
                    cleanup["server_log_tail"] = _redact_text(tail)[-4096:]
                except OSError as error:
                    cleanup["server_log_error"] = type(error).__name__
            self._resource_guard.supervisor_cleanup = cleanup
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        await self._resource_guard.wait_after_stop(
            supervisor_exited=process is not None and process.returncode is not None
        )

    async def _stop(self) -> None:
        """Join startup and sandbox cleanup inside the selected launcher
        policy."""
        try:
            # No startup continuation may acquire a lease or launch a process
            # after the resource proof is published by permanent stop.
            if self._start_task is not None:
                if not self._start_task.done():
                    self._start_task.cancel()
                await asyncio.gather(self._start_task, return_exceptions=True)
            await self._stop_server()
        finally:
            try:
                # Task release is settled by the original supervisor/cgroup
                # proof above. Publish it before potentially slow host file GC;
                # Relax still requires the managed process group to be closed.
                self._resource_guard.finish()
            finally:
                try:
                    if self._staging_dir and self._staging_dir.exists():
                        shutil.rmtree(self._staging_dir)
                finally:
                    if self._image_lease is not None:
                        await self._image_context.__aexit__(None, None, None)
                        self._image_lease = None
