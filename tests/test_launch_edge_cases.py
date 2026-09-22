"""Regression coverage for launch selection, overrides, and explicit sweeps."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from fxr.launch import cli
from fxr.launch.cluster import cluster_submit_params
from fxr.launch.exp_config import build_submission_configs


@pytest.fixture(autouse=True)
def restore_device_environment():
    """Restore device policy after helpers assign process environment values."""
    original = {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "FXR_DEVICE")}
    yield
    for key, value in original.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def build(tmp_path):
    """Build submissions from a small complete base without launching jobs."""
    base = tmp_path / "base.yml"
    base.write_text(yaml.safe_dump({
        "experiment": {"seed": 40}, "optim": {"lr": 0.1},
        "dataloader": {"batch_size": 8}, "log": {"root": "?"},
    }))

    def make(experiment=None, **kwargs):
        return build_submission_configs(
            group="test", base_cfgs=[str(base)], experiment_cfg=experiment or {},
            cluster_overrides=kwargs.pop("cluster_overrides", {}),
            scratch_root="/scratch", add_date=False, **kwargs,
        )

    return make


def test_literal_lists_and_explicit_cartesian_axes(build):
    experiment = {
        "model": {"filters": [16, 32]},
        "sweep": {"optim.lr": [0.01, 0.02], "train.epochs": [1, 2, 3]},
    }
    original = deepcopy(experiment)
    cells = build(experiment)
    assert len(cells) == 6
    assert all(c["model"]["filters"] == [16, 32] for c in cells)
    assert len({(c["optim"]["lr"], c["train"]["epochs"]) for c in cells}) == 6
    cells[0]["model"]["filters"].append(64)
    assert cells[1]["model"]["filters"] == [16, 32]
    assert experiment == original


@pytest.mark.parametrize("source", ["cli", "yaml"])
def test_explicit_sweep_beats_experiment_literal(build, source):
    experiment = {"optim": {"lr": 0.5}}
    kwargs = {}
    if source == "cli":
        kwargs["sweep_overrides"] = {"optim.lr": [0.01, 0.02]}
    else:
        experiment["sweep"] = {"optim": {"lr": [0.01, 0.02]}}
    assert [c["optim"]["lr"] for c in build(experiment, **kwargs)] == [0.01, 0.02]


def test_cli_sweep_replaces_yaml_axis(build):
    cells = build({"sweep": {"optim.lr": [0.1, 0.2, 0.3]}},
                  sweep_overrides={"optim.lr": [0.01, 0.02]})
    assert [c["optim"]["lr"] for c in cells] == [0.01, 0.02]


@pytest.mark.parametrize("source", ["cli", "yaml_convenience", "yaml_sweep"])
def test_seed_range_uses_experiment_start_seed(build, source):
    experiment = {"experiment": {"seed": 100}}
    kwargs = {}
    if source == "cli":
        kwargs["sweep_overrides"] = {"experiment.seed_range": 3}
    elif source == "yaml_sweep":
        experiment["sweep"] = {"experiment.seed_range": 3}
    else:
        experiment["experiment"]["seed_range"] = 3
    assert [c["experiment"]["seed"] for c in build(experiment, **kwargs)] == [100, 101, 102]


def test_cli_seed_range_beats_yaml_seed_range(build):
    cells = build({"experiment": {"seed": 100, "seed_range": 2}},
                  sweep_overrides={"experiment.seed_range": 3})
    assert [c["experiment"]["seed"] for c in cells] == [100, 101, 102]


@pytest.mark.parametrize("override", [{"experiment.seed": 7}, {"experiment": {"seed": 7}}])
def test_literal_seed_override_collapses_seed_sweep(build, override):
    cells = build({"experiment": {"seed": 100, "seed_range": 3}}, set_overrides=override)
    assert [c["experiment"]["seed"] for c in cells] == [7]


@pytest.mark.parametrize("override", [{"dataloader.batch_size": 2}, {"dataloader": {"batch_size": 2}}])
def test_set_wins_over_cluster_and_sweep_without_duplicate_jobs(build, override):
    cells = build({"sweep": {"dataloader.batch_size": [4, 16]}},
                  cluster_overrides={"dataloader": {"batch_size": 8}}, set_overrides=override)
    assert len(cells) == 1
    assert cells[0]["dataloader"]["batch_size"] == 2


@pytest.mark.parametrize("source", ["experiment", "cli", "cli_mapping", "sweep"])
def test_explicit_log_roots_are_preserved(build, source):
    experiment, kwargs = {}, {}
    expected = ["/chosen"]
    if source == "experiment":
        experiment["log"] = {"root": "/chosen"}
    elif source == "cli":
        kwargs["set_overrides"] = {"log.root": "/chosen"}
    elif source == "cli_mapping":
        kwargs["set_overrides"] = {"log": {"root": "/chosen"}}
    else:
        experiment["sweep"] = {"log.root": ["/one", "/two"]}
        expected = ["/one", "/two"]
    assert [c["log"]["root"] for c in build(experiment, **kwargs)] == expected


@pytest.mark.parametrize("axes", [{"optim.lr": []}, {"optim.lr": 0.1}, {"experiment.seed_range": 0},
                                  {"experiment.seed_range": True}, {"experiment.seed_range": 1.5}])
def test_yaml_and_cli_axes_share_validation(build, axes):
    with pytest.raises(ValueError, match="positive integer|non-empty YAML list"):
        build({"sweep": axes})
    raw = [f"{key}={yaml.safe_dump(value, default_flow_style=True).strip()}" for key, value in axes.items()]
    with pytest.raises(ValueError, match="positive integer|non-empty YAML list"):
        cli._parse_sweep_overrides(raw)


@pytest.mark.parametrize("invalid", [None, False, [], "optim.lr"])
def test_yaml_sweep_requires_a_mapping(build, invalid):
    with pytest.raises(ValueError, match="sweep must be a mapping"):
        build({"sweep": invalid})


@pytest.mark.parametrize("inherited,index,expected", [
    ("4,7", 0, "4"), ("4,7", 1, "7"),
    ("GPU-first,GPU-second", 1, "GPU-second"),
    ("MIG-first", 0, "MIG-first"), (None, 1, "1"),
])
def test_gpu_selection_preserves_scheduler_assignment(monkeypatch, inherited, index, expected):
    if inherited is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", inherited)
    monkeypatch.setattr(cli, "_visible_gpu_count", lambda: 1 if inherited == "MIG-first" else 2)
    cli._configure_local_device_environment(argparse.Namespace(submitit=False, device="auto", gpu=index))
    assert os.environ["CUDA_VISIBLE_DEVICES"] == expected


def test_invalid_gpu_selection_preserves_environment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    monkeypatch.setattr(cli, "_visible_gpu_count", lambda: 1)
    with pytest.raises(ValueError, match="out of range"):
        cli._configure_local_device_environment(argparse.Namespace(submitit=False, device="auto", gpu=1))
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4"


def test_no_gpu_auto_fallback_and_explicit_requests(monkeypatch):
    monkeypatch.setattr(cli, "_visible_gpu_count", lambda: 0)
    assert cli._select_visible_gpu(0, "auto") is None
    for index, policy in [(0, "cuda"), (1, "auto")]:
        with pytest.raises(ValueError, match="no CUDA"):
            cli._select_visible_gpu(index, policy)


def test_device_discovery_is_isolated_even_without_cuda(tmp_path, monkeypatch):
    marker = tmp_path / "discovery.pid"
    (tmp_path / "torch.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "class cuda:\n"
        "    @staticmethod\n"
        "    def device_count():\n"
        "        Path(os.environ['FXR_TEST_DISCOVERY_PID']).write_text(str(os.getpid()))\n"
        "        return 2\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("FXR_TEST_DISCOVERY_PID", str(marker))
    assert cli._visible_gpu_count() == 2
    assert int(marker.read_text()) != os.getpid()


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired("probe", 60), subprocess.CalledProcessError(1, "probe")])
def test_device_discovery_failure_has_a_launch_error(monkeypatch, failure):
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(cli.subprocess, "run", fail)
    with pytest.raises(ValueError, match="inspect visible CUDA"):
        cli._visible_gpu_count()


@pytest.mark.parametrize("fallback", [False, True])
def test_real_cuda_selection_survives_nvml_failure(fallback):
    # Both the launcher and its discovery subprocess start before any CUDA call.
    code = '''
import argparse, json, os, subprocess
from fxr.launch import cli
real_run = subprocess.run
def run_probe(command, **kwargs):
    if os.environ["FXR_TEST_NVML_FAILURE"] == "1":
        command = list(command)
        command[-1] = "import torch; torch.cuda._device_count_nvml = lambda: -1; " + command[-1]
    return real_run(command, **kwargs)
cli.subprocess.run = run_probe
try:
    cli._configure_local_device_environment(argparse.Namespace(submitit=False, device="cuda", gpu=1))
except ValueError as exc:
    print(json.dumps({"skip": str(exc)}))
else:
    import torch
    print(json.dumps({"runtime_count": torch._C._cuda_getDeviceCount(),
                      "selected": os.environ["CUDA_VISIBLE_DEVICES"]}))
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
               FXR_TEST_NVML_FAILURE="1" if fallback else "0")
    run = subprocess.run([sys.executable, "-c", code], env=env, text=True,
                         capture_output=True, timeout=90, check=True)
    result = json.loads(run.stdout.strip().splitlines()[-1])
    if "skip" in result:
        if "no CUDA" in result["skip"] or "out of range" in result["skip"]:
            pytest.skip(result["skip"])
        pytest.fail(result["skip"])
    assert result["runtime_count"] == 1, result


def test_cluster_root_expansion(monkeypatch):
    monkeypatch.setenv("FXR_TEST_SCRATCH", "/scratch/test")
    params = cluster_submit_params({
        "scratch_root": "$FXR_TEST_SCRATCH/flexray", "ct_data_path": "~/ct",
        "xray_data_path": "$FXR_TEST_SCRATCH/xray", "generated_data_path": "$FXR_TEST_SCRATCH/generated",
    })
    assert params.scratch_root == "/scratch/test/flexray"
    assert params.ct_data_path == str(Path.home() / "ct")
    assert params.xray_data_path == "/scratch/test/xray"
    assert params.generated_data_path == "/scratch/test/generated"
