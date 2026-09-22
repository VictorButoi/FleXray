"""Tests for the cluster-config + exp-config submission flow (``fxr-submit``)."""

from __future__ import annotations

import datetime
from importlib.resources import files

import pytest

from fxr.launch.cluster import (
    cluster_submit_params,
    load_cluster_config,
    split_cluster_config,
)
from fxr.launch.exp_config import build_submission_configs, load_exp_config
from fxr.launch.submit_cli import main

# A minimal training base with a placeholder the cluster is expected to fill.
_BASE_YML = (
    "experiment: {_class: tests.test_launch.FakeExperiment, seed: 40}\n"
    "optim: {lr: 1.0e-4}\n"
    "dataloader: {batch_size: '?'}\n"
    "log: {root: '?'}\n"
)

_CLUSTER = {
    "mode": "slurm",
    "scratch_root": "/scratch/FleXray",
    "ct_data_path": "/data/ct",
    "xray_data_path": "/data/xray",
    "generated_data_path": "/data/generated",
    "add_date": True,
    "slurm_args": {
        "slurm_partition": "gpu",
        "gpus_per_node": 1,
        "cpus_per_task": None,  # dropped by cluster_submit_params
    },
    "training_requeue": {"enabled": True, "max_num_timeout": 5, "slurm_signal_delay_s": 120},
    "dataloader": {"batch_size": 8},
}


def _write_base(tmp_path) -> str:
    path = tmp_path / "base.yml"
    path.write_text(_BASE_YML, encoding="utf-8")
    return str(path)


def test_split_cluster_config_partitions_keys():
    submit_cfg, overrides = split_cluster_config(_CLUSTER)
    assert set(submit_cfg) == {"mode", "scratch_root", "ct_data_path", "xray_data_path", "generated_data_path", "add_date", "slurm_args", "training_requeue"}
    assert overrides == {"dataloader": {"batch_size": 8}}


def test_cluster_submit_params_strips_none_and_folds_requeue():
    submit_cfg, _ = split_cluster_config(_CLUSTER)
    params = cluster_submit_params(submit_cfg)
    assert params.mode == "slurm"
    assert params.scratch_root == "/scratch/FleXray"
    assert params.ct_data_path == "/data/ct"
    assert params.xray_data_path == "/data/xray"
    assert params.generated_data_path == "/data/generated"
    assert params.slurm_max_num_timeout == 5
    # None entries dropped; requeue signal delay folded in.
    assert params.slurm_kwargs == {
        "slurm_partition": "gpu",
        "gpus_per_node": 1,
        "slurm_signal_delay_s": 120,
    }


def test_cluster_submit_params_requeue_disabled():
    params = cluster_submit_params({"scratch_root": "/s", "training_requeue": {"enabled": False}})
    assert params.slurm_max_num_timeout is None
    assert "slurm_signal_delay_s" not in params.slurm_kwargs


def test_every_packaged_cluster_resolves_to_submit_parameters():
    cluster_root = files("fxr.configs").joinpath("cluster")
    names = tuple(
        sorted(
            path.name.removesuffix(".yml")
            for path in cluster_root.iterdir()
            if path.name.endswith(".yml")
        )
    )

    assert names
    for name in names:
        cfg = load_cluster_config(name=name)
        submit_cfg, _ = split_cluster_config(cfg)
        params = cluster_submit_params(submit_cfg)

        assert params.mode in {"local", "slurm"}
        assert params.scratch_root


def test_load_exp_config_reads_spec(tmp_path):
    spec = tmp_path / "sweep.yml"
    spec.write_text(
        "base_cfgs: [base]\nexperiment_cfg: {group: my_group, optim: {lr: [1, 2]}}\n",
        encoding="utf-8",
    )
    group, base_cfgs, experiment_cfg = load_exp_config(str(spec))
    assert group == "my_group"
    assert base_cfgs == ["base"]
    assert experiment_cfg["optim"]["lr"] == [1, 2]


def test_load_exp_config_rejects_missing_group(tmp_path):
    spec = tmp_path / "sweep.yml"
    spec.write_text("base_cfgs: [base]\nexperiment_cfg: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="group"):
        load_exp_config(str(spec))


def test_build_submission_configs_job_count_and_log_root(tmp_path):
    base = _write_base(tmp_path)
    _, overrides = split_cluster_config(_CLUSTER)
    configs = build_submission_configs(
        group="exp",
        base_cfgs=[base, base],
        experiment_cfg={"group": "exp", "sweep": {"optim.lr": [1e-4, 3e-4]}},
        cluster_overrides=overrides,
        scratch_root="/scratch/FleXray",
        add_date=True,
    )
    # 2 bases x 2 lr values.
    assert len(configs) == 4
    date = datetime.datetime.now().strftime("%m_%d_%y")
    assert configs[0]["log"]["root"] == f"/scratch/FleXray/training/{date}_exp"
    # Cluster filled the placeholder batch_size.
    assert configs[0]["dataloader"]["batch_size"] == 8


def test_build_submission_configs_cluster_overrides_win(tmp_path):
    base = _write_base(tmp_path)
    configs = build_submission_configs(
        group="exp",
        base_cfgs=[base],
        experiment_cfg={"group": "exp", "dataloader": {"batch_size": 99}},
        cluster_overrides={"dataloader": {"batch_size": 8}},
        scratch_root="/scratch",
        add_date=False,
    )
    assert len(configs) == 1
    assert configs[0]["dataloader"]["batch_size"] == 8  # cluster merged last
    assert configs[0]["log"]["root"] == "/scratch/training/exp"  # no date prefix


def test_build_submission_configs_validates_missing(tmp_path):
    base = _write_base(tmp_path)
    with pytest.raises(ValueError, match="missing required values"):
        build_submission_configs(
            group="exp",
            base_cfgs=[base],
            experiment_cfg={"group": "exp"},
            cluster_overrides={},  # batch_size placeholder stays "?"
            scratch_root="/scratch",
            add_date=False,
        )


def _write_cluster_and_spec(tmp_path):
    cluster = tmp_path / "cluster.yml"
    cluster.write_text(
        "mode: slurm\n"
        "scratch_root: /scratch/FleXray\n"
        "ct_data_path: /data/ct\n"
        "xray_data_path: /data/xray\n"
        "generated_data_path: /data/generated\n"
        "add_date: false\n"
        "slurm_args: {slurm_partition: gpu, gpus_per_node: 1}\n"
        "training_requeue: {enabled: true, max_num_timeout: 5, slurm_signal_delay_s: 120}\n"
        "dataloader: {batch_size: 8}\n",
        encoding="utf-8",
    )
    spec = tmp_path / "sweep.yml"
    spec.write_text(
        f"base_cfgs: ['{_write_base(tmp_path)}']\n"
        "experiment_cfg: {group: exp, sweep: {optim.lr: [1.0e-4, 3.0e-4]}}\n",
        encoding="utf-8",
    )
    return str(cluster), str(spec)


def test_submit_cli_dry_run_does_not_submit(tmp_path, monkeypatch, capsys):
    called = {"submitted": False}
    monkeypatch.setattr(
        "fxr.launch.submit_cli.submit_configs",
        lambda *a, **k: called.__setitem__("submitted", True),
    )
    cluster, spec = _write_cluster_and_spec(tmp_path)
    main(["--cluster-cfg", cluster, "--exp-config", spec, "--dry-run"])
    out = capsys.readouterr().out
    assert "Generated configs: 2" in out
    assert "exp" in out
    assert called["submitted"] is False


def test_submit_cli_rejects_smoke_data_without_dry_run(tmp_path, monkeypatch):
    called = {"submitted": False}
    monkeypatch.setattr(
        "fxr.launch.submit_cli.submit_configs",
        lambda *a, **k: called.__setitem__("submitted", True),
    )
    cluster, spec = _write_cluster_and_spec(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--cluster-cfg",
                cluster,
                "--exp-config",
                spec,
                "--smoke-data",
            ]
        )

    assert exc_info.value.code == 2
    assert called["submitted"] is False


def test_submit_cli_forwards_submit_params(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "fxr.launch.submit_cli.submit_configs",
        lambda configs, **kwargs: captured.update(configs=configs, **kwargs),
    )
    cluster, spec = _write_cluster_and_spec(tmp_path)
    main(["--cluster-cfg", cluster, "--exp-config", spec])
    assert len(captured["configs"]) == 2
    assert captured["folder"] == "/scratch/FleXray/training/exp/submitit"
    assert captured["slurm_kwargs"] == {
        "slurm_partition": "gpu",
        "gpus_per_node": 1,
        "slurm_signal_delay_s": 120,
    }
    assert captured["slurm_max_num_timeout"] == 5
    assert captured["ct_data_path"] == "/data/ct"
    assert captured["xray_data_path"] == "/data/xray"
    assert captured["generated_data_path"] == "/data/generated"
    assert captured["local"] is False
