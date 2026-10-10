# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from pathlib import Path

import pytest

from examples.claude_code_harbor.scripts.bootstrap_qs import configuration, gpu_nodes, service_plan


def test_qs_bootstrap_resolves_cloud_paths_and_blocks_resume(tmp_path: Path):
    env = {
        "CLOUD_ROOT": str(tmp_path),
        "EXP_NAME": "harbor-repro",
        "RAY_NO_WAIT": "1",
        "RELAX_ENTRYPOINT_MODE": "spmd",
        **{
            key: key.lower()
            for key in (
                "REPO",
                "SOURCE_ROOT",
                "DATA_ROOT",
                "BUNDLE_DIR",
                "MODEL_DIR",
                "SAVE_DIR",
                "BBS_WHEEL_PATH",
                "DRESSAGE_WHEEL_PATH",
            )
        },
    }
    result = configuration(env)
    assert result["REPO"] == str(tmp_path / "repo")
    assert result["RAY_NO_WAIT"] == ""
    assert "RELAX_ENTRYPOINT_MODE" not in result
    (tmp_path / "save_dir/harbor-repro").mkdir(parents=True)
    with pytest.raises(ValueError, match="output already exists"):
        configuration(env)


def test_qs_bootstrap_waits_for_all_gpu_workers_and_rejects_wrong_size():
    workers = [{"Alive": True, "Resources": {"GPU": 8}, "NodeManagerAddress": str(index)} for index in range(16)]
    head = {"Alive": True, "Resources": {}, "NodeManagerAddress": "head"}
    assert gpu_nodes(workers[:-1] + [head]) == []
    assert len(gpu_nodes(workers + [head])) == 16
    with pytest.raises(ValueError, match="topology"):
        gpu_nodes(workers + [workers[0]])
    workers[0]["Resources"]["GPU"] = 4
    with pytest.raises(ValueError, match="topology"):
        gpu_nodes(workers)


def test_qs_bootstrap_routes_cpu_head_and_restricted_workers_to_native():
    probes = [
        {"ip": "a", "gpu": True, "oci": True},
        {"ip": "b", "gpu": True, "oci": True},
        {"ip": "c", "gpu": True, "oci": False},
        {"ip": "d", "gpu": True, "oci": False},
        {"ip": "head", "gpu": False, "oci": True},
    ]
    assert service_plan(probes) == {"a": None, "b": None, "c": "a", "d": "b", "head": "a"}
    with pytest.raises(RuntimeError, match="No GPU worker"):
        service_plan([probes[-1]])
