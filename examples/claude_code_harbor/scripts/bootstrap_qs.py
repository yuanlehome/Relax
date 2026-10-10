# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Prepare and run the 128-GPU Harbor recipe on a fresh, dedicated QS RayJob.

Run as the QS entrypoint, not independently on every worker. All paths and
input assets come from environment variables documented in ENVIRONMENT.md. Keep
this driver alive so its ordinary Ray actors retain the node-local Gym
processes.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path


def configuration(values: dict[str, str]) -> dict[str, str]:
    env = dict(values)
    cloud = Path(env["CLOUD_ROOT"]).resolve()
    for key in (
        "REPO",
        "SOURCE_ROOT",
        "DATA_ROOT",
        "BUNDLE_DIR",
        "MODEL_DIR",
        "SAVE_DIR",
        "BBS_WHEEL_PATH",
        "DRESSAGE_WHEEL_PATH",
    ):
        env[key] = str((cloud / env[key]).resolve())
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", env["EXP_NAME"]):
        raise ValueError("EXP_NAME must be a plain directory name")
    env["MEGATRON"] = str((cloud / env.get("MEGATRON", "/root/Megatron-LM")).resolve())
    env["PYTHONPATH"] = os.pathsep.join((env["REPO"], env["MEGATRON"], env.get("PYTHONPATH", "")))
    env["OCI_ROOT"] = env.get("OCI_ROOT", f"/tmp/harbor-{env['EXP_NAME']}")
    env["RAY_NO_WAIT"] = ""
    env["RELAX_SUBMISSION_ID"] = f"harbor-{env['EXP_NAME']}"
    env.pop("RELAX_ENTRYPOINT_MODE", None)
    # Never resume old checkpoints or replace a half-built environment implicitly.
    for path in (Path(env["DATA_ROOT"]), Path(env["BUNDLE_DIR"]), Path(env["SAVE_DIR"]) / env["EXP_NAME"]):
        if path.exists():
            raise ValueError("Reproduction output already exists; choose fresh DATA_ROOT/BUNDLE_DIR/EXP_NAME")
    return env


def gpu_nodes(nodes: list[dict], expected: int = 16, per_node: int = 8) -> list[dict]:
    workers = [node for node in nodes if node["Alive"] and node["Resources"].get("GPU", 0) > 0]
    if len(workers) > expected or any(node["Resources"]["GPU"] != per_node for node in workers):
        raise ValueError("Ray GPU topology differs from the selected 128-GPU recipe")
    return sorted(workers, key=lambda node: node["NodeManagerAddress"]) if len(workers) == expected else []


def service_plan(probes: list[dict]) -> dict[str, str | None]:
    native = sorted(probe["ip"] for probe in probes if probe["gpu"] and probe["oci"])
    if not native:
        raise RuntimeError("No GPU worker supports user namespaces and accessible FUSE")
    result = {}
    index = 0
    for probe in sorted(probes, key=lambda probe: probe["ip"]):
        if probe["ip"] in native:
            result[probe["ip"]] = None
        else:
            result[probe["ip"]] = native[index % len(native)]
            index += 1
    return result


def wait_ready(host: str, port: int, timeout: int = 90) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with opener.open(f"http://{host}:{port}/readyz", timeout=3) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(1)
    raise RuntimeError("Node-local Gym readiness timed out")


def private_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w")


class HarborNode:
    """Own service descendants without consuming training GPUs or placement
    groups."""

    def __init__(self, env: dict[str, str], ip: str, gpu: bool) -> None:
        self.env = {**os.environ, **env, "NODE_IP": ip}
        self.ip = ip
        self.gpu = gpu
        self.processes: list[subprocess.Popen] = []
        self.state = Path(env["BUNDLE_DIR"]) / "service_state" / ip

    def probe(self) -> dict:
        for path in (
            Path(self.env["REPO"]) / "repro/claude-code-harbor-r5/scripts/start_node.sh",
            Path(self.env["DATA_ROOT"]) / "ordinary-image-map.json",
            Path(self.env["MODEL_DIR"]) / "Qwen3.6-35B-A3B/config.json",
        ):
            if not path.is_file():
                raise RuntimeError("A worker lacks a required shared input or image Megatron checkout")
        if not (Path(self.env["MEGATRON"]) / "megatron").is_dir():
            raise RuntimeError("Worker image lacks the Megatron checkout")
        port = int(self.env.get("GYM_PORT", "27980"))
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
        if self.gpu:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=30
            )
            names = result.stdout.strip().splitlines()
            if result.returncode or len(names) != 8 or any("H800" not in name for name in names):
                raise RuntimeError("Worker hardware differs from 8 x H800")
        result = subprocess.run(["unshare", "-Ur", "true"], capture_output=True, timeout=10)
        oci = result.returncode == 0 and os.access("/dev/fuse", os.R_OK | os.W_OK)
        return {"ip": self.ip, "gpu": self.gpu, "oci": oci}

    def forward(self, host: str, port: int, upstream: str, upstream_port: int, label: str) -> None:
        with private_log(self.state / f"{label}.log") as output:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "examples.claude_code_harbor.scripts.forward_gym",
                    "--host",
                    host,
                    "--port",
                    str(port),
                    "--upstream-host",
                    upstream,
                    "--upstream-port",
                    str(upstream_port),
                ],
                env=self.env,
                cwd=self.env["REPO"],
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        self.processes.append(process)
        wait_ready(host, port)
        if process.poll() is not None:
            raise RuntimeError("Gym forwarding process exited")

    def start(self, upstream: str | None) -> dict:
        port = int(self.env.get("GYM_PORT", "27980"))
        uplink = int(self.env.get("GYM_UPLINK_PORT", "27981"))
        if upstream is None:
            script = Path(self.env["REPO"]) / "repro/claude-code-harbor-r5/scripts/start_node.sh"
            with private_log(self.state / "start.log") as output:
                result = subprocess.run(
                    ["bash", str(script)], env=self.env, stdout=output, stderr=subprocess.STDOUT, timeout=900
                )
            if result.returncode:
                raise RuntimeError("Native Gym startup failed; inspect the private node start.log")
            self.forward(self.ip, uplink, "127.0.0.1", port, "uplink")
        else:
            self.forward("127.0.0.1", port, upstream, uplink, "forwarder")
        return {"native": upstream is None, "ready": True}

    def health(self) -> bool:
        if any(process.poll() is not None for process in self.processes):
            return False
        wait_ready("127.0.0.1", int(self.env.get("GYM_PORT", "27980")), timeout=5)
        return True


def prepare(env: dict[str, str], report) -> None:
    repo = Path(env["REPO"])
    scripts = repo / "examples/claude_code_harbor/scripts"
    helpers = repo / "repro/claude-code-harbor-r5/scripts"
    bundle = Path(env["BUNDLE_DIR"])
    # Use the image's configured package index without exposing credentials in logs.
    for key, setting in (("PIP_INDEX_URL", "global.index-url"), ("PIP_TRUSTED_HOST", "global.trusted-host")):
        if not env.get(key):
            result = subprocess.run(
                [sys.executable, "-m", "pip", "config", "get", setting], capture_output=True, text=True, timeout=20
            )
            if result.returncode == 0 and result.stdout.strip():
                env[key] = result.stdout.strip()
    commands = (
        ("prepare_data", [sys.executable, str(scripts / "prepare_data.py")]),
        (
            "build_gym",
            [
                "bash",
                str(scripts / "install_gym_runtime.sh"),
                "--repo",
                str(repo),
                "--prefix",
                str(bundle / "runtime/gym"),
            ],
        ),
        (
            "install_oci",
            ["bash", str(scripts / "install_oci_runtime.sh"), "--prefix", str(bundle / "runtime/oci-runtime")],
        ),
        ("gateway_token", [sys.executable, str(helpers / "create_gateway_token.py"), "--bundle", str(bundle)]),
        (
            "verify_inputs",
            [
                sys.executable,
                str(helpers / "verify_inputs.py"),
                "--data-root",
                env["DATA_ROOT"],
                "--model-dir",
                str(Path(env["MODEL_DIR"]) / "Qwen3.6-35B-A3B"),
            ],
        ),
    )
    for stage, command in commands:
        report(stage)
        with private_log(bundle / "bootstrap_logs" / f"{stage}.log") as output:
            result = subprocess.run(command, cwd=repo, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=7200)
        if result.returncode:
            raise RuntimeError(f"{stage} failed (exit {result.returncode}); inspect its private bootstrap log")


def run() -> int:
    import ray
    from ray.job_submission import JobSubmissionClient
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from relax.utils.logging_utils import get_logger

    logger = get_logger(__name__)
    env = configuration(dict(os.environ))
    bundle = Path(env["BUNDLE_DIR"])
    bundle.mkdir(parents=True, mode=0o700)
    status: dict = {"started_at": time.time(), "experiment": env["EXP_NAME"]}

    def report(stage: str, **details) -> None:
        status.update(stage=stage, updated_at=time.time(), **details)
        temporary = bundle / "bootstrap-status.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(bundle / "bootstrap-status.json")
        logger.info("Harbor bootstrap stage=%s", stage)

    try:
        prepare(env, report)
        report("wait_for_gpu_nodes")
        ray.init(address="auto", log_to_driver=False)
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            nodes = [node for node in ray.nodes() if node["Alive"]]
            workers = gpu_nodes(nodes)
            if workers:
                break
            time.sleep(10)
        else:
            raise RuntimeError("Timed out waiting for 16 x 8 GPU workers")
        heads = [node for node in nodes if node["Resources"].get("node:__internal_head__", 0)]
        if len(heads) != 1:
            raise RuntimeError("Cannot identify a unique QS Ray head")
        head = heads[0]
        env["RAY_ADDRESS"] = ray.get_runtime_context().gcs_address
        env["RAY_DASHBOARD_ADDRESS"] = f"http://{head['NodeManagerAddress']}:8265"
        # CPU agent processes can be scheduled on the head or any worker.
        env["NEMO_GYM_CALLBACK_ALLOWED_HOSTS"] = ",".join(node["NodeManagerAddress"] for node in nodes)
        keys = (
            "CLOUD_ROOT",
            "REPO",
            "DATA_ROOT",
            "BUNDLE_DIR",
            "MODEL_DIR",
            "MEGATRON",
            "OCI_ROOT",
            "PYTHONPATH",
            "NEMO_GYM_CALLBACK_ALLOWED_HOSTS",
            "GYM_PORT",
            "GYM_UPLINK_PORT",
        )
        service_env = {key: env[key] for key in keys if key in env}
        node_type = ray.remote(num_cpus=0, num_gpus=0, max_restarts=0, max_task_retries=0)(HarborNode)
        actors = {
            node["NodeManagerAddress"]: node_type.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
            ).remote(service_env, node["NodeManagerAddress"], bool(node["Resources"].get("GPU", 0)))
            for node in nodes
        }
        report("probe_node_services", gpu_nodes=len(workers), gpus=128)
        probes = ray.get([actor.probe.remote() for actor in actors.values()], timeout=180)
        plan = service_plan(probes)
        report("start_native_gyms", native_gyms=sum(upstream is None for upstream in plan.values()))
        ray.get([actors[ip].start.remote(None) for ip, upstream in plan.items() if upstream is None], timeout=1200)
        report("start_forwarders", forwarders=sum(upstream is not None for upstream in plan.values()))
        ray.get(
            [actors[ip].start.remote(upstream) for ip, upstream in plan.items() if upstream is not None], timeout=180
        )
        wait_ready("127.0.0.1", int(env.get("GYM_PORT", "27980")))
        report("submit_training")
        script = Path(env["REPO"]) / "repro/claude-code-harbor-r5/scripts/launch.sh"
        with private_log(bundle / "bootstrap_logs/training.log") as output:
            process = subprocess.Popen(
                ["bash", str(script)], cwd=env["REPO"], env=env, stdout=output, stderr=subprocess.STDOUT
            )
        client = JobSubmissionClient(env["RAY_DASHBOARD_ADDRESS"])
        # The blocking recipe owns the child job. Do not exit early and orphan it.
        while process.poll() is None:
            time.sleep(30)
            try:
                healthy = all(ray.get([actor.health.remote() for actor in actors.values()], timeout=30))
                report("training", services_healthy=healthy)
            except Exception:
                report("training", services_healthy=False)
        terminal = str(client.get_job_status(env["RELAX_SUBMISSION_ID"]))
        report("finished", training_exit_code=process.returncode, ray_job_status=terminal)
        return 0 if process.returncode == 0 and terminal == "SUCCEEDED" else 1
    except Exception as exc:
        # Diagnostics may include URLs or source payloads: keep details in private logs.
        with private_log(bundle / "bootstrap_logs/failure.log") as output:
            traceback.print_exc(file=output)
        report("failed", error_type=type(exc).__name__)
        logger.error("Harbor bootstrap failed: %s; inspect private stage logs", type(exc).__name__)
        return 1


if __name__ == "__main__":
    os.umask(0o077)
    raise SystemExit(run())
