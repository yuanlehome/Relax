# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Prepare and run the 128-GPU Harbor recipe on a fresh, dedicated QS RayJob.

Run as the QS entrypoint, not independently on every worker. All paths and
input assets come from environment variables documented in ENVIRONMENT.md. Keep
this driver alive so its ordinary Ray actors retain the node-local Gym
processes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from examples.claude_code_harbor.scripts.oci_paths import default_oci_root, validate_oci_root


def input_configuration(values: dict[str, str], source: str, model: str) -> dict[str, str]:
    """Expand the two-input CLI into the existing environment contract."""
    env = dict(values)
    source_path, model_path = Path(source).resolve(), Path(model).resolve()
    if model_path.name == "Qwen3.6-35B-A3B":
        model_path = model_path.parent
    env.update(SOURCE_ROOT=str(source_path), MODEL_DIR=str(model_path))
    if not env.get("CLOUD_ROOT"):
        assets = json.loads(Path(__file__).with_name("source-assets.json").read_text())
        suffix = Path(assets["dataset_cloud_path"]).parts
        if source_path.parts[-len(suffix) :] != suffix:
            raise ValueError("Source layout differs from the original cloud tree; set CLOUD_ROOT explicitly")
        env["CLOUD_ROOT"] = str(source_path.parents[len(suffix) - 1])
    env.setdefault("REPO", str(Path(__file__).resolve().parents[3]))
    env.setdefault("SAVE_DIR", str(Path.cwd() / "harbor-checkpoints"))
    save = (Path(env["CLOUD_ROOT"]) / env["SAVE_DIR"]).resolve()
    if "SAVE_DIR" not in values and not save.is_relative_to(Path(env["CLOUD_ROOT"]).resolve()):
        raise ValueError("Run from a writable shared cloud directory, or set SAVE_DIR on shared storage")
    run_root = save.parent / "harbor-runs" / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    env.setdefault("DATA_ROOT", str(run_root / "prepared"))
    env.setdefault("BUNDLE_DIR", str(run_root / "bundle"))
    env.setdefault("EXP_NAME", "qwen36-35b-a3b-harbor-128h800-hybrid")
    return env


def configuration(values: dict[str, str]) -> dict[str, str]:
    env = dict(values)
    if env.get("HARBOR_REUSE_DATA", "0") not in ("0", "1"):
        raise ValueError("HARBOR_REUSE_DATA must be 0 or 1")
    required = ["CLOUD_ROOT", "REPO", "DATA_ROOT", "BUNDLE_DIR", "MODEL_DIR", "SAVE_DIR", "EXP_NAME"]
    if env.get("HARBOR_REUSE_DATA") != "1":
        required.append("SOURCE_ROOT")
    missing = [key for key in required if not env.get(key, "").strip()]
    if missing:
        raise ValueError("Missing required settings: " + ", ".join(missing))
    cloud = Path(env["CLOUD_ROOT"]).resolve()
    env["CLOUD_ROOT"] = str(cloud)
    for key in (
        "REPO",
        "SOURCE_ROOT",
        "DATA_ROOT",
        "BUNDLE_DIR",
        "MODEL_DIR",
        "SAVE_DIR",
        "BBS_WHEEL_PATH",
        "DRESSAGE_WHEEL_PATH",
        "GYM_RUNTIME_DIR",
        "OCI_PACKAGES_DIR",
    ):
        if env.get(key):
            env[key] = str((cloud / env[key]).resolve())
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", env["EXP_NAME"]):
        raise ValueError("EXP_NAME must be a plain directory name")
    env["MEGATRON"] = str((cloud / env.get("MEGATRON", "/root/Megatron-LM")).resolve())
    env["PYTHONPATH"] = os.pathsep.join((env["REPO"], env["MEGATRON"], env.get("PYTHONPATH", "")))
    env["OCI_ROOT"] = validate_oci_root(cloud / env["OCI_ROOT"]) if env.get("OCI_ROOT") else default_oci_root()
    env["RAY_NO_WAIT"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["RELAX_SUBMISSION_ID"] = f"harbor-{env['EXP_NAME']}-{uuid.uuid4().hex[:12]}"
    env.pop("RELAX_ENTRYPOINT_MODE", None)
    # Never resume old checkpoints or replace a half-built environment implicitly.
    outputs = [Path(env["BUNDLE_DIR"]), Path(env["SAVE_DIR"]) / env["EXP_NAME"]]
    if env.get("HARBOR_REUSE_DATA") == "1":
        if not Path(env["DATA_ROOT"]).is_dir():
            raise ValueError("HARBOR_REUSE_DATA requires an existing prepared DATA_ROOT")
    else:
        outputs.append(Path(env["DATA_ROOT"]))
    for path in outputs:
        if path.exists() or path.is_symlink():
            raise ValueError("Reproduction output already exists; choose fresh output roots / EXP_NAME")
    env["HARBOR_REUSE_GYM"] = "1" if env.get("GYM_RUNTIME_DIR") else "0"
    if env["HARBOR_REUSE_GYM"] == "1" and not Path(env["GYM_RUNTIME_DIR"]).is_dir():
        raise ValueError("GYM_RUNTIME_DIR must point to an existing completed Gym build")
    inputs = [Path(env[key]) for key in ("SOURCE_ROOT", "MODEL_DIR") if env.get(key)]
    if env.get("HARBOR_REUSE_DATA") == "1":
        inputs.append(Path(env["DATA_ROOT"]))
    if env["HARBOR_REUSE_GYM"] == "1":
        inputs.append(Path(env["GYM_RUNTIME_DIR"]))
    if any(output.is_relative_to(source) or source.is_relative_to(output) for output in outputs for source in inputs):
        raise ValueError("Output directories must be separate from original and reused inputs")
    if any(
        a.is_relative_to(b) or b.is_relative_to(a) for index, a in enumerate(outputs) for b in outputs[index + 1 :]
    ):
        raise ValueError("DATA_ROOT, BUNDLE_DIR and checkpoint outputs must not overlap")
    if not env.get("GYM_RUNTIME_DIR"):
        env["GYM_RUNTIME_DIR"] = str(Path(env["BUNDLE_DIR"]) / "runtime/gym")
    for key in ("GYM_PORT", "GYM_UPLINK_PORT"):
        if key in env and (not env[key].isdigit() or not 1024 <= int(env[key]) <= 65535):
            raise ValueError(f"{key} must be a port between 1024 and 65535")
    if env.get("GYM_PORT", "27980") == env.get("GYM_UPLINK_PORT", "27981"):
        raise ValueError("GYM_PORT and GYM_UPLINK_PORT must differ")
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

    def verify(self, gateway_host: str | None, native_ip: str, label: str) -> dict:
        """Exercise an existing Gym with a node-local callback and private
        evidence."""
        port = int(
            self.env.get("GYM_UPLINK_PORT" if gateway_host else "GYM_PORT", "27981" if gateway_host else "27980")
        )
        host = gateway_host or "127.0.0.1"
        output = self.state / "verification" / label
        command = [
            str(Path(self.env["GYM_RUNTIME_DIR"]) / "nemo-venv/bin/python"),
            "-m",
            "examples.claude_code_harbor.scripts.verify_oci",
            "--dataset",
            str(Path(self.env["DATA_ROOT"]) / "aligned-150037-oci/data.jsonl"),
            "--output",
            str(output),
            "--gateway-url",
            f"http://{host}:{port}",
            "--token-file",
            str(Path(self.env["BUNDLE_DIR"]) / "private/gateway.token"),
            "--callback-host",
            self.ip,
            "--artifact-root",
            str(Path(self.env["BUNDLE_DIR"]) / "service_state" / native_ip / "artifacts"),
        ]
        with private_log(output / "verify.log") as log:
            result = subprocess.run(
                command, env=self.env, cwd=self.env["REPO"], stdout=log, stderr=subprocess.STDOUT, timeout=3900
            )
        status_file = output / "status.json"
        if result.returncode or not status_file.is_file():
            raise RuntimeError("Controlled OCI verification failed; inspect the private verification log")
        evidence = json.loads(status_file.read_text())
        if (
            evidence.get("status") != "PASS"
            or evidence.get("auto_compaction_verified") is not True
            or evidence.get("protected_verifier_read_denied") is not True
            or evidence.get("model_calls", 0) < 2
            or evidence.get("contexts", 0) < 2
        ):
            raise RuntimeError("Controlled OCI verification did not produce PASS evidence")
        wait_ready(host, port)
        return {
            "passed": True,
            "model_calls": int(evidence["model_calls"]),
            "contexts": int(evidence["contexts"]),
            "callback_cross_node": self.ip != native_ip,
        }


def prepare(env: dict[str, str], report: Callable[..., None]) -> None:
    repo = Path(env["REPO"])
    scripts = repo / "examples/claude_code_harbor/scripts"
    helpers = repo / "repro/claude-code-harbor-r5/scripts"
    bundle = Path(env["BUNDLE_DIR"])
    if env.get("HARBOR_REUSE_DATA") != "1" and any(
        not env.get(key) for key in ("BBS_WHEEL_PATH", "DRESSAGE_WHEEL_PATH")
    ):
        from examples.claude_code_harbor.scripts.prepare_data import discover_wheels

        report("locate_source_wheels")
        missing_wheels = tuple(key for key in ("BBS_WHEEL_PATH", "DRESSAGE_WHEEL_PATH") if not env.get(key))
        discovered = discover_wheels(Path(env["SOURCE_ROOT"]), keys=missing_wheels)
        for key, value in discovered.items():
            if not env.get(key):
                env[key] = value
    # Use the image's configured package index without exposing credentials in logs.
    for key, setting in (("PIP_INDEX_URL", "global.index-url"), ("PIP_TRUSTED_HOST", "global.trusted-host")):
        if not env.get(key):
            result = subprocess.run(
                [sys.executable, "-m", "pip", "config", "get", setting], capture_output=True, text=True, timeout=20
            )
            if result.returncode == 0 and result.stdout.strip():
                env[key] = result.stdout.strip()
    commands: list[tuple[str, list[str]]] = []
    if env.get("HARBOR_REUSE_DATA") != "1":
        commands.append(("prepare_data", [sys.executable, str(scripts / "prepare_data.py")]))
    # Reject invalid data/model inputs before spending time building the runtimes.
    commands.append(
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
        )
    )
    gym = [
        sys.executable,
        str(scripts / "build_gym_environment.py"),
        "--repo",
        str(repo),
        "--prefix",
        env["GYM_RUNTIME_DIR"],
    ]
    reuse_gym = env["HARBOR_REUSE_GYM"] == "1"
    if reuse_gym:
        gym.extend(("--check", "--check-log", str(bundle / "bootstrap_logs/gym-check.log")))
    commands.append(("check_gym" if reuse_gym else "build_gym", gym))
    oci = ["bash", str(scripts / "install_oci_runtime.sh"), "--prefix", str(bundle / "runtime/oci-runtime")]
    if env.get("OCI_PACKAGES_DIR"):
        oci.extend(("--assets-dir", env["OCI_PACKAGES_DIR"]))
    commands.extend(
        (
            ("install_oci", oci),
            ("gateway_token", [sys.executable, str(helpers / "create_gateway_token.py"), "--bundle", str(bundle)]),
        )
    )
    for stage, command in commands:
        report(stage)
        with private_log(bundle / "bootstrap_logs" / f"{stage}.log") as output:
            result = subprocess.run(command, cwd=repo, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=7200)
        if result.returncode:
            raise RuntimeError(f"{stage} failed (exit {result.returncode}); inspect its private bootstrap log")


def require_idle_cluster(ray: Any, client: Any, env: dict[str, str]) -> None:
    """Refuse unrelated work before invoking the recipe's cluster cleanup."""
    own_job = ray.get_runtime_context().get_job_id()
    own_submission = env.get("RAY_JOB_SUBMISSION_ID")
    for job in client.list_jobs():
        if str(job.status) in {"SUCCEEDED", "FAILED", "STOPPED"}:
            continue
        if job.job_id == own_job or (own_submission and job.submission_id == own_submission):
            continue
        if job.entrypoint == "sleep infinity":
            continue
        raise RuntimeError("Dedicated cluster required: another Ray job is active")
    if ray.available_resources().get("GPU", 0) != 128:
        raise RuntimeError("Dedicated cluster required: not all 128 GPUs are idle")
    if any(group["state"] != "REMOVED" for group in ray.util.placement_group_table().values()):
        raise RuntimeError("Dedicated cluster required: placement groups already exist")
    if ray.serve.status().applications:
        raise RuntimeError("Dedicated cluster required: Serve applications already exist")


def wait_for_training(
    process: subprocess.Popen,
    client: Any,
    submission_id: str,
    health: Callable[[], bool],
    report: Callable[..., None],
    interval: int = 30,
) -> int:
    """Hold ordinary actors even if the blocking CLI loses its log
    connection."""
    seen = False
    absent_after_exit = 0
    while True:
        exit_code = process.poll()
        try:
            jobs = [job for job in client.list_jobs() if job.submission_id == submission_id]
            state = str(jobs[0].status) if jobs else "NOT_FOUND"
        except Exception:
            state = "UNKNOWN"
        seen = seen or state not in {"UNKNOWN", "NOT_FOUND"}
        if state in {"SUCCEEDED", "FAILED", "STOPPED"}:
            report("finished", training_exit_code=exit_code, ray_job_status=state)
            return 0 if state == "SUCCEEDED" else 1
        # Allow for a submit request still reaching the server after CLI failure.
        absent_after_exit = absent_after_exit + 1 if exit_code is not None and state == "NOT_FOUND" else 0
        if not seen and absent_after_exit >= 5:
            raise RuntimeError("Training submission was not registered; inspect the private training log")
        try:
            healthy = health()
        except Exception:
            healthy = False
        report(
            "training_status_unknown" if state == "UNKNOWN" else "training",
            services_healthy=healthy,
            ray_job_status=state,
            training_exit_code=exit_code,
        )
        time.sleep(interval)


def retain_services(health: Callable[[], bool], report: Callable[..., None], interval: int = 60) -> None:
    """Keep the run frame and its ordinary Ray actors alive after training
    stops."""
    while True:
        try:
            healthy = health()
        except Exception:
            healthy = False
        report("services_retained", service_driver_alive=True, services_healthy=healthy)
        time.sleep(interval)


def run(env: dict[str, str], *, prepare_only: bool = False) -> int:

    from relax.utils.logging_utils import get_logger

    logger = get_logger(__name__)
    bundle = Path(env["BUNDLE_DIR"])
    bundle.mkdir(parents=True, mode=0o700)
    status: dict = {"started_at": time.time(), "experiment": env["EXP_NAME"]}
    actors: dict = {}
    services_started = False

    def report(stage: str, **details) -> None:
        status.update(stage=stage, updated_at=time.time(), **details)
        temporary = bundle / "bootstrap-status.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(bundle / "bootstrap-status.json")
        logger.info("Harbor bootstrap stage=%s", stage)

    try:
        prepare(env, report)
        report("prepared", data_prepared=True, environment_prepared=True)
        if prepare_only:
            report("prepared", training_submitted=False)
            return 0
        import ray
        import ray.serve
        from ray.job_submission import JobSubmissionClient
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        report("wait_for_gpu_nodes")
        ray.init(address=env.get("RAY_ADDRESS") or "auto", log_to_driver=False)
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
        if ray.get_runtime_context().get_node_id() != head["NodeID"]:
            raise RuntimeError("Run the bootstrap on the Ray head so its local Gym endpoint is verified")
        env["RAY_ADDRESS"] = ray.get_runtime_context().gcs_address
        env["RAY_DASHBOARD_ADDRESS"] = env.get("RAY_DASHBOARD_ADDRESS") or (
            f"http://{head['NodeManagerAddress']}:{env.get('RAY_DASHBOARD_PORT', '8265')}"
        )
        client = JobSubmissionClient(env["RAY_DASHBOARD_ADDRESS"])
        report("check_cluster_idle")
        require_idle_cluster(ray, client, env)
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
            "GYM_RUNTIME_DIR",
            "PYTHONDONTWRITEBYTECODE",
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
        native = [ip for ip, upstream in plan.items() if upstream is None]
        report("start_first_native_gym", native_gyms=len(native))
        services_started = True
        ray.get(actors[native[0]].start.remote(None), timeout=1200)
        report("verify_first_native_oci_callback")
        verified = {
            native[0]: ray.get(
                actors[head["NodeManagerAddress"]].verify.remote(native[0], native[0], "first-native"), timeout=4000
            )
        }
        report("first_native_verified", controlled_oci=verified)
        report("start_native_gyms")
        ray.get([actors[ip].start.remote(None) for ip in native[1:]], timeout=1200)
        report("start_forwarders", forwarders=sum(upstream is not None for upstream in plan.values()))
        ray.get(
            [actors[ip].start.remote(upstream) for ip, upstream in plan.items() if upstream is not None], timeout=180
        )
        wait_ready("127.0.0.1", int(env.get("GYM_PORT", "27980")))
        for ip, upstream in plan.items():
            if ip == native[0]:
                continue
            report("verify_node_oci_callback", verifying_node=ip)
            caller = actors[head["NodeManagerAddress"]] if upstream is None else actors[ip]
            verified[ip] = ray.get(
                caller.verify.remote(ip if upstream is None else None, upstream or ip, ip), timeout=4000
            )
            report("node_verified", controlled_oci=verified, verified_service_nodes=len(verified))
        report("services_verified", verified_service_nodes=len(verified))
        require_idle_cluster(ray, client, env)
        report("submit_training", training_submission_id=env["RELAX_SUBMISSION_ID"])
        script = Path(env["REPO"]) / "repro/claude-code-harbor-r5/scripts/launch.sh"
        with private_log(bundle / "bootstrap_logs/training.log") as output:
            process = subprocess.Popen(
                ["bash", str(script)], cwd=env["REPO"], env=env, stdout=output, stderr=subprocess.STDOUT
            )
        health = lambda: all(ray.get([actor.health.remote() for actor in actors.values()], timeout=30))
        training_exit = wait_for_training(
            process,
            client,
            env["RELAX_SUBMISSION_ID"],
            health,
            report,
        )
        report("training_terminal", bootstrap_training_result=training_exit)
        retain_services(health, report)
    except Exception as exc:
        # Diagnostics may include URLs or source payloads: keep details in private logs.
        with private_log(bundle / "bootstrap_logs/failure.log") as output:
            traceback.print_exc(file=output)
        report("failed", failed_stage=status.get("stage"), error_type=type(exc).__name__)
        logger.error("Harbor bootstrap failed: %s; inspect private stage logs", type(exc).__name__)
        if actors and services_started:
            retain_services(
                lambda: all(ray.get([actor.health.remote() for actor in actors.values()], timeout=30)), report
            )
        return 1


def main(argv: list[str] | None = None) -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", help="Original dataset directory; pair with --model-dir")
    parser.add_argument("--model-dir", help="Qwen3.6-35B-A3B directory or its parent; pair with --source-root")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-config", action="store_true", help="Validate settings only; no writes or cluster calls")
    mode.add_argument(
        "--prepare-only", action="store_true", help="Prepare shared inputs/runtimes; no services or training"
    )
    args = parser.parse_args(argv)
    if bool(args.source_root) != bool(args.model_dir):
        parser.error("Pass both --source-root and --model-dir")
    try:
        values = dict(os.environ)
        if args.source_root:
            values = input_configuration(values, args.source_root, args.model_dir)
        env = configuration(values)
    except ValueError as exc:
        parser.error(str(exc))
    if args.check_config:
        parser.exit(message="Harbor configuration valid; no environment or training was started.\n")
    return run(env, prepare_only=args.prepare_only)


if __name__ == "__main__":
    raise SystemExit(main())
