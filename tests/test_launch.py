"""Tests for the training launch harness."""

from __future__ import annotations

import json
import os
from importlib.resources import files
from pathlib import Path

import yaml

import pytest
import torch

from fxr.config import config_digest
from fxr.datasets import load_dataset_layouts
from fxr.launch import build_submission_configs, expand_configs
from fxr.launch.cluster import (
    cluster_submit_params,
    load_cluster_config,
    split_cluster_config,
)
from fxr.launch.cli import load_training_base, main
from fxr.launch.readiness import validate_training_config

BASE = {
    "experiment": {"_class": "tests.test_launch.FakeExperiment", "seed": 40},
    "optim": {"lr": 1e-4},
    "log": {"root": "?"},
}


class FakeExperiment:
    """Test double persisting enough run state to exercise launch semantics."""

    instances: list["FakeExperiment"] = []

    def __init__(self, source: dict | str | Path, load_data: bool = True) -> None:
        """Open an existing path or retain a new config for the test."""
        self.loaded_existing = isinstance(source, (str, Path))
        if self.loaded_existing:
            self.path = Path(source)
            self.config = yaml.safe_load(
                (self.path / "config.yml").read_text(encoding="utf-8")
            )
        else:
            self.path = None
            self.config = dict(source)
        self.load_data = load_data
        self.ran = False
        self.instances.append(self)

    @classmethod
    def from_config(
        cls,
        config: dict,
        *,
        uuid: str | None = None,
        load_data: bool = True,
    ) -> "FakeExperiment":
        """Create and persist a fake new run under ``log.root``."""
        instance = cls(config, load_data)
        instance.path = Path(config["log"]["root"]) / (uuid or "fake-run")
        instance.path.mkdir(parents=True, exist_ok=False)
        (instance.path / "config.yml").write_text(
            yaml.safe_dump(config),
            encoding="utf-8",
        )
        return instance

    def run(self) -> None:
        """Record that the fake training loop was invoked."""
        self.ran = True


def _readiness_xray_config(tmp_path):
    return {
        "experiment": {"_class": "fxr.experiment.FleXrayTrainExperiment"},
        "protocol": {"name": "all_structures_flexray_v4"},
        "train": {"epochs": 1, "eval_freq": 1},
        "dataloader": {"batch_size": 1, "num_workers": 0},
        "data": {"Xray": {"HipRay": {}}},
        "log": {"root": str(tmp_path)},
    }


def _readiness_ct_config(tmp_path):
    cfg = _readiness_xray_config(tmp_path)
    cfg["data"] = {"CT": {"RSNAFrac": {"version": 3.0}}}
    return cfg


def _mkdir_dataset_path(root, *parts):
    path = root.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _packaged_config_names(folder: str) -> tuple[str, ...]:
    """Discover every packaged YAML config in one config folder."""

    root = files("fxr.configs").joinpath(folder)
    names = tuple(
        sorted(
            path.name.removesuffix(".yml")
            for path in root.iterdir()
            if path.name.endswith(".yml")
        )
    )
    assert names, f"FleXray must package at least one {folder} config."
    return names


@pytest.fixture(autouse=True)
def _reset_fakes():
    """Reset launch doubles and process-local GPU selection after each test."""
    original_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    original_fxr_device = os.environ.get("FXR_DEVICE")
    FakeExperiment.instances.clear()
    yield
    FakeExperiment.instances.clear()
    if original_cuda_devices is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_devices
    if original_fxr_device is None:
        os.environ.pop("FXR_DEVICE", None)
    else:
        os.environ["FXR_DEVICE"] = original_fxr_device


def test_expand_configs_product_count():
    configs = expand_configs(BASE, {"optim.lr": [1e-4, 3e-4], "train.epochs": [10, 20]})
    assert len(configs) == 4
    lrs = sorted({cfg["optim"]["lr"] for cfg in configs})
    assert lrs == [1e-4, 3e-4]


def test_expand_configs_seed_range():
    configs = expand_configs(BASE, {"experiment.seed_range": 3})
    seeds = sorted(cfg["experiment"]["seed"] for cfg in configs)
    assert seeds == [40, 41, 42]


def test_expand_configs_scalar_passthrough_and_merge_precedence():
    configs = expand_configs(BASE, {"optim.lr": 5e-4})
    assert len(configs) == 1
    assert configs[0]["optim"]["lr"] == 5e-4
    assert configs[0]["experiment"]["_class"] == "tests.test_launch.FakeExperiment"


def test_expand_configs_returns_independent_cells_without_mutating_inputs():
    base = {"nested": {"values": ["base"]}}
    overrides = {"optim.lr": [1.0, 2.0]}

    configs = expand_configs(base, overrides)
    configs[0]["nested"]["values"].append("first-cell-only")
    configs[0]["optim"]["lr"] = 3.0

    assert base == {"nested": {"values": ["base"]}}
    assert overrides == {"optim.lr": [1.0, 2.0]}
    assert configs[1]["nested"]["values"] == ["base"]
    assert configs[1]["optim"]["lr"] == 2.0


def test_cli_set_applies_typed_override(tmp_path, monkeypatch):
    config_path = tmp_path / "cfg.yml"
    config_path.write_text(
        "experiment: {_class: tests.test_launch.FakeExperiment, seed: 1}\n"
        "log: {root: '?'}\n",
        encoding="utf-8",
    )
    main([str(config_path), "--set", f"log.root={tmp_path}", "--set", "optim.lr=2"])
    assert len(FakeExperiment.instances) == 1
    instance = FakeExperiment.instances[0]
    assert instance.ran is True
    assert instance.config["optim"]["lr"] == 2
    assert isinstance(instance.config["optim"]["lr"], int)


def test_cli_persists_explicit_unverified_label_order_opt_in(tmp_path, capsys):
    config_path = tmp_path / "cfg.yml"
    config_path.write_text(
        "experiment: {_class: tests.test_launch.FakeExperiment}\n"
        f"log: {{root: {tmp_path}}}\n",
        encoding="utf-8",
    )

    main(
        [
            str(config_path),
            "--init-from",
            "weights.safetensors",
            "--allow-unverified-label-order",
        ]
    )

    initialization = FakeExperiment.instances[0].config["initialization"]
    assert initialization["kind"] == "pretrained"
    assert initialization["allow_unverified_label_order"] is True

    with pytest.raises(SystemExit) as exc_info:
        main([str(config_path), "--allow-unverified-label-order"])
    assert exc_info.value.code == 2
    assert "requires --init-from" in capsys.readouterr().err


def test_cli_missing_value_is_a_friendly_parser_error(tmp_path, capsys):
    config_path = tmp_path / "cfg.yml"
    config_path.write_text(
        "experiment: {_class: tests.test_launch.FakeExperiment}\nlog: {root: '?'}\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        main([str(config_path)])

    assert exc_info.value.code == 2
    assert "missing required values" in capsys.readouterr().err


def test_cli_sets_cuda_visible_devices(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("fxr.launch.cli._visible_gpu_count", lambda: 8)
    config_path = tmp_path / "cfg.yml"
    config_path.write_text(
        "experiment: {_class: tests.test_launch.FakeExperiment}\n"
        f"log: {{root: {tmp_path}}}\n",
        encoding="utf-8",
    )
    main([str(config_path), "--gpu", "3"])
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    assert os.environ["FXR_DEVICE"] == "auto"


def test_cli_cpu_device_hides_cuda(tmp_path, monkeypatch):
    observed_environment = {}

    def inspect_environment(import_name):
        observed_environment["import_name"] = import_name
        observed_environment["cuda"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        observed_environment["device"] = os.environ.get("FXR_DEVICE")

    monkeypatch.setattr(
        "fxr.launch.cli.require_training_extra", inspect_environment
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    config_path = tmp_path / "cfg.yml"
    config_path.write_text(
        "experiment: {_class: tests.test_launch.FakeExperiment}\n"
        f"log: {{root: {tmp_path}}}\n",
        encoding="utf-8",
    )

    main([str(config_path), "--device", "cpu", "--gpu", "3"])

    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert os.environ["FXR_DEVICE"] == "cpu"
    assert observed_environment == {
        "import_name": "kornia",
        "cuda": "",
        "device": "cpu",
    }


def test_cli_explicit_cuda_reports_preflight_failure(tmp_path, monkeypatch, capsys):
    from fxr.experiment import _device as device_module

    config_path = tmp_path / "cfg.yml"
    config_path.write_text(
        "experiment: {_class: tests.test_launch.FakeExperiment}\n"
        f"log: {{root: {tmp_path}}}\n",
        encoding="utf-8",
    )

    def reject_cuda(requested):
        raise RuntimeError(
            "--device cuda was requested, but this wheel omits sm_70; "
            "install CUDA 12.6"
        )

    monkeypatch.setattr(device_module, "resolve_training_device", reject_cuda)
    monkeypatch.setattr("fxr.launch.cli._visible_gpu_count", lambda: 1)
    with pytest.raises(SystemExit) as exc_info:
        main([str(config_path), "--device", "cuda"])

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "omits sm_70" in error
    assert "CUDA 12.6" in error


def test_cli_dry_run_skips_training(tmp_path, capsys):
    config_path = tmp_path / "cfg.yml"
    config_path.write_text(
        "experiment: {_class: tests.test_launch.FakeExperiment}\n"
        f"log: {{root: {tmp_path}}}\n",
        encoding="utf-8",
    )
    main([str(config_path), "--dry-run"])
    instance = FakeExperiment.instances[0]
    assert instance.ran is False
    assert instance.load_data is False
    assert instance.path is not None
    assert not instance.path.exists()
    assert "experiment" in capsys.readouterr().out


def test_cli_set_lists_are_literal_and_sweep_is_explicit(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "scheduler-owned")
    monkeypatch.setenv("FXR_DEVICE", "inherited")

    def fake_submit_configs(configs, *, folder, slurm_kwargs, local):
        captured["configs"] = configs
        captured["folder"] = folder
        captured["slurm_kwargs"] = slurm_kwargs

    monkeypatch.setattr("fxr.launch.submit.submit_configs", fake_submit_configs)
    config_path = tmp_path / "cfg.yml"
    config_path.write_text(
        "experiment: {_class: tests.test_launch.FakeExperiment}\n"
        f"log: {{root: {tmp_path}}}\n",
        encoding="utf-8",
    )
    main(
        [
            str(config_path),
            "--submitit",
            "--set",
            "model.filters=[2,4]",
            "--sweep",
            "optim.lr=[1,2,3]",
            "--partition",
            "gpu",
        ]
    )

    assert [config["optim"]["lr"] for config in captured["configs"]] == [1, 2, 3]
    assert all(config["model"]["filters"] == [2, 4] for config in captured["configs"])
    assert captured["folder"] == str(tmp_path / "submitit")
    assert captured["slurm_kwargs"] == {"slurm_partition": "gpu"}
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "scheduler-owned"
    assert os.environ["FXR_DEVICE"] == "inherited"


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (["--submitit", "--sweep", "optim.lr=2"], "non-empty YAML list"),
        (["--submitit", "--gpus", "2"], "exactly one GPU"),
        (["--submitit", "--device", "cpu"], "local launch option"),
    ],
)
def test_cli_invalid_launch_options_are_friendly_errors(
    tmp_path, capsys, extra_args, message
):
    config_path = tmp_path / "cfg.yml"
    config_path.write_text(
        "experiment: {_class: tests.test_launch.FakeExperiment}\n"
        f"log: {{root: {tmp_path}}}\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        main([str(config_path), *extra_args])

    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err


def test_load_training_base_resolves_base_inheritance(tmp_path):
    parent = tmp_path / "parent.yml"
    parent.write_text(
        "train: {epochs: 100, eval_freq: 5}\n"
        "dataloader: {batch_size: 8, num_workers: 4}\n",
        encoding="utf-8",
    )
    child = tmp_path / "child.yml"
    child.write_text(
        "_base_: parent.yml\n" "dataloader: {proportions: {MOOSE: 3}}\n",
        encoding="utf-8",
    )

    cfg = load_training_base(str(child))

    assert "_base_" not in cfg
    # Inherited scalars survive; the child deep-merges its overrides.
    assert cfg["train"] == {"epochs": 100, "eval_freq": 5}
    assert cfg["dataloader"]["batch_size"] == 8
    assert cfg["dataloader"]["proportions"] == {"MOOSE": 3}


def test_load_training_base_rejects_inheritance_cycles(tmp_path):
    first = tmp_path / "first.yml"
    second = tmp_path / "second.yml"
    first.write_text("_base_: second.yml\n", encoding="utf-8")
    second.write_text("_base_: first.yml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="inheritance cycle"):
        load_training_base(str(first))


@pytest.mark.parametrize("config_name", _packaged_config_names("training"))
def test_packaged_training_configs_are_internally_consistent(
    config_name: str,
) -> None:
    config = load_training_base(config_name)
    configured_sources = {
        name
        for datasets in config["data"].values()
        for name in datasets
    }
    layout_resource = files("fxr.configs").joinpath(
        "dataset_layouts", "training.yml"
    )
    registered_sources = {
        layout.dataset_name
        for layout in load_dataset_layouts(layout_resource, replace=True)
    }
    eval_sources = set(
        config.get("callbacks", {})
        .get("epoch", {})
        .get("eval_sets", {})
        .get("data", {})
        .get("Xray", {})
    )

    assert configured_sources
    assert configured_sources <= registered_sources
    assert eval_sources <= registered_sources
    assert set(config["dataloader"]["proportions"]) == configured_sources
    assert set(config["loss_func"]["dataset_losses"]) == configured_sources
    assert set(config["drr_model"]["datasets"]) == set(config["data"]["CT"])
    assert set(config["loss_func"]["dataset_losses"].values()) <= set(
        config["loss_func"]["losses"]
    )


def test_submission_config_prunes_inactive_source_but_keeps_eval_coverage(tmp_path):
    base = load_training_base("base")
    train_xray = set(base["data"]["Xray"])
    eval_xray = set(base["callbacks"]["epoch"]["eval_sets"]["data"]["Xray"])
    omitted_name = sorted(train_xray & eval_xray)[0]
    expected_sources = {
        name
        for datasets in base["data"].values()
        for name in datasets
    } - {omitted_name}

    configs = build_submission_configs(
        group="flexray_default",
        base_cfgs=["base"],
        experiment_cfg={
            "group": "flexray_default",
            "dataloader": {"proportions": {omitted_name: 0}},
        },
        cluster_overrides={
            "dataloader": {"batch_size": 1, "num_workers": 0},
            "log": {"wandb": {"mode": "disabled"}},
        },
        scratch_root=str(tmp_path),
        add_date=False,
    )

    assert len(configs) == 1
    config = configs[0]
    configured_sources = {
        name
        for datasets in config["data"].values()
        for name in datasets
    }
    assert configured_sources == expected_sources
    assert set(config["loss_func"]["dataset_losses"]) == expected_sources
    assert set(config["drr_model"]["datasets"]) == set(config["data"]["CT"])
    assert omitted_name in config["callbacks"]["epoch"]["eval_sets"]["data"]["Xray"]
    assert config["log"]["root"] == str(tmp_path / "training" / "flexray_default")


@pytest.mark.parametrize("config_name", _packaged_config_names("training"))
@pytest.mark.parametrize("cluster_name", _packaged_config_names("cluster"))
def test_packaged_clusters_make_training_configs_launchable(
    tmp_path, config_name: str, cluster_name: str
) -> None:
    cluster_cfg = load_cluster_config(name=cluster_name)
    submit_cfg, cluster_overrides = split_cluster_config(cluster_cfg)

    configs = build_submission_configs(
        group="flexray_default",
        base_cfgs=[config_name],
        experiment_cfg={"group": "flexray_default"},
        cluster_overrides=cluster_overrides,
        scratch_root=str(tmp_path),
        add_date=False,
    )

    config = configs[0]
    assert (
        config["dataloader"]["batch_size"]
        == cluster_overrides["dataloader"]["batch_size"]
    )
    assert (
        config["dataloader"]["num_workers"]
        == cluster_overrides["dataloader"]["num_workers"]
    )
    assert cluster_submit_params(submit_cfg).scratch_root


def test_readiness_reports_missing_datapath(tmp_path, monkeypatch):
    monkeypatch.delenv("XRAY_DATAPATH", raising=False)
    config = _readiness_xray_config(tmp_path)

    with pytest.raises(ValueError, match="XRAY_DATAPATH"):
        validate_training_config(config)


def test_readiness_reports_missing_dataset_path(tmp_path):
    config = _readiness_xray_config(tmp_path)

    with pytest.raises(ValueError, match="does not exist"):
        validate_training_config(config, xray_data_root=tmp_path)


def test_readiness_reports_missing_ct_drr_profile(tmp_path):
    config = _readiness_ct_config(tmp_path)
    _mkdir_dataset_path(tmp_path, "RSNAFrac", "thunder_dbs", "3.0")

    with pytest.raises(ValueError, match="drr_model is required"):
        validate_training_config(config, ct_data_root=tmp_path)


def test_readiness_reports_routed_loss_coverage(tmp_path):
    config = _readiness_xray_config(tmp_path)
    config["loss_func"] = {
        "_class": "DatasetRoutedLoss",
        "losses": {"standard": {"_class": "SoftDiceLoss"}},
        "dataset_losses": {},
    }
    _mkdir_dataset_path(tmp_path, "HipRay")

    with pytest.raises(ValueError, match="DatasetRoutedLoss"):
        validate_training_config(config, xray_data_root=tmp_path)


def test_readiness_validates_callback_local_eval_paths(tmp_path):
    config = _readiness_xray_config(tmp_path)
    _mkdir_dataset_path(tmp_path, "HipRay")
    config["callbacks"] = {
        "epoch": {
            "eval_sets": {
                "_class": "fxr.experiment.EvalSetMetricLogger",
                "data": {"Xray": {"JRST": {}}},
                "split": "val",
            }
        }
    }

    with pytest.raises(ValueError, match="JRST.*does not exist"):
        validate_training_config(config, xray_data_root=tmp_path)


def test_readiness_smokes_callback_requested_split(tmp_path, monkeypatch):
    config = _readiness_xray_config(tmp_path)
    config["callbacks"] = {
        "epoch": {
            "eval_sets": {
                "_class": "fxr.experiment.EvalSetMetricLogger",
                "data": {"Xray": {"JRST": {}}},
                "split": "holdout",
            }
        }
    }
    _mkdir_dataset_path(tmp_path, "HipRay")
    _mkdir_dataset_path(tmp_path, "JRST")
    opened: list[tuple[str, str, str]] = []

    class SmokeDataset:
        """One-sample X-ray dataset used to record smoke-open requests."""

        def __len__(self):
            """Return one available sample."""
            return 1

        def __getitem__(self, index):
            """Return one valid dense-label X-ray sample."""
            del index
            return {
                "image": torch.zeros(1, 8, 8),
                "label": torch.zeros(8, 8, dtype=torch.long),
            }

        def close(self):
            """Release the no-op test dataset."""

    def fake_build(dataset_name, *, split, modality, cfg):
        del cfg
        opened.append((dataset_name, split, modality))
        return SmokeDataset()

    monkeypatch.setattr(
        "fxr.launch.readiness.build_training_dataset",
        fake_build,
    )

    validate_training_config(config, smoke_data=True, xray_data_root=tmp_path)

    assert opened == [
        ("JRST", "holdout", "xray"),
        ("HipRay", "train", "xray"),
    ]


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("train", "epochs", 0, "train.epochs"),
        ("train", "eval_freq", -1, "train.eval_freq"),
        ("dataloader", "batch_size", 0, "dataloader.batch_size"),
        ("dataloader", "num_workers", -1, "dataloader.num_workers"),
        ("model_weights", "save_freq", -1, "log.model_weights.save_freq"),
    ],
)
def test_readiness_rejects_invalid_runtime_scalars(
    tmp_path, section, key, value, message
):
    config = _readiness_xray_config(tmp_path)
    _mkdir_dataset_path(tmp_path, "HipRay")
    if section == "model_weights":
        config["log"]["model_weights"] = {key: value}
    else:
        config[section][key] = value

    with pytest.raises(ValueError, match=message):
        validate_training_config(config, xray_data_root=tmp_path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"batch_size": 0}, "dataloader.Xray.batch_size"),
        ({"batch_size": 1.9}, "must be an integer"),
        ({"pin_memory": "false"}, "pin_memory must be a bool"),
        (
            {"num_workers": 1, "prefetch_factor": 0},
            "prefetch_factor must be an integer",
        ),
        ({"mystery": 1}, "Unexpected dataloader.Xray keys"),
        (
            {"num_workers": 0, "prefetch_factor": 2},
            "prefetch_factor requires num_workers",
        ),
        (
            {"num_workers": 0, "persistent_workers": True},
            "persistent_workers requires num_workers",
        ),
    ],
)
def test_readiness_validates_xray_loader_overrides(
    tmp_path, overrides, message
):
    config = _readiness_xray_config(tmp_path)
    config["dataloader"]["Xray"] = overrides
    _mkdir_dataset_path(tmp_path, "HipRay")

    with pytest.raises(ValueError, match=message):
        validate_training_config(config, xray_data_root=tmp_path)


def test_readiness_smoke_data_checks_ct_affine_and_centroids(tmp_path, monkeypatch):
    config = _readiness_ct_config(tmp_path)
    config["drr_model"] = {
        "default": {
            "preset": "frontal",
            "intrinsics_cfg": {"height": 8, "width": 8, "sdd": 1000.0, "delx": 2.0},
            "isocenter_cfg": {"sample_scheme": "volume_center"},
        },
        "datasets": {
            "RSNAFrac": {
                "isocenter_cfg": {"sample_scheme": "random_label", "replacement": True}
            }
        },
    }
    _mkdir_dataset_path(tmp_path, "RSNAFrac", "thunder_dbs", "3.0")

    class SmokeDataset:
        """Dataset test double returning one CT sample missing render metadata."""

        def __len__(self):
            """Return the number of samples in the test double."""
            return 1

        def __getitem__(self, index):
            """Return one CT sample without affine or centroid metadata."""
            return {
                "image": torch.zeros(1, 2, 2, 2),
                "label": torch.zeros(1, 2, 2, 2, dtype=torch.long),
                "metadata": {},
            }

    monkeypatch.setattr(
        "fxr.launch.readiness.build_training_dataset",
        lambda *args, **kwargs: SmokeDataset(),
    )

    with pytest.raises(ValueError) as exc_info:
        validate_training_config(config, smoke_data=True, ct_data_root=tmp_path)

    message = str(exc_info.value)
    assert "missing affine" in message
    assert "fg_centroids_ijk" in message


def test_cli_local_explicit_sweep_is_a_friendly_error(tmp_path, capsys):
    config_path = tmp_path / "cfg.yml"
    config_path.write_text(
        "experiment: {_class: tests.test_launch.FakeExperiment}\n"
        f"log: {{root: {tmp_path}}}\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        main([str(config_path), "--sweep", "optim.lr=[1,2]"])

    assert exc_info.value.code == 2
    assert "use --submitit" in capsys.readouterr().err


def test_run_config_uses_explicit_identity_and_only_resumes_on_request(tmp_path):
    from fxr.launch.run import resume_run, run_config

    config = {
        "experiment": {"_class": "tests.test_launch.FakeExperiment"},
        "log": {"root": str(tmp_path)},
    }
    created = run_config(config, train=False, run_id="stable-run-id")

    assert created.path == tmp_path / "stable-run-id"
    assert created.loaded_existing is False
    with pytest.raises(FileExistsError, match="--resume"):
        run_config(config, train=False, run_id="stable-run-id")

    changed_config = {**config, "optim": {"lr": 2.0}}
    with pytest.raises(ValueError, match="does not match the immutable config"):
        run_config(
            changed_config,
            train=False,
            run_id="stable-run-id",
            resume_existing=True,
        )

    with pytest.raises(FileNotFoundError, match="Only checkpointed runs"):
        resume_run(created.path)
    checkpoint_dir = created.path / "checkpoints"
    checkpoint_dir.mkdir()
    torch.save({"model": {}}, checkpoint_dir / "last.pt")

    continued = resume_run(created.path)
    assert continued.path == created.path
    assert continued.loaded_existing is True
    assert continued.ran is True


@pytest.mark.parametrize(
    "run_id",
    [
        "../escape",
        "/tmp/escape",
        "two-parts",
        "too-many-parts-here",
        "left--right",
        "..\\-unsafe-id",
        "left-\n-right",
    ],
)
def test_run_config_rejects_unsafe_or_malformed_explicit_ids(tmp_path, run_id):
    from fxr.launch.run import run_config

    config = {
        "experiment": {"_class": "tests.test_launch.FakeExperiment"},
        "log": {"root": str(tmp_path)},
    }

    with pytest.raises(ValueError, match="run_id"):
        run_config(config, train=False, run_id=run_id)

    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "group",
    ["..", ".", "../escape", "nested/group", "nested\\group", "bad\x00name"],
)
def test_submission_group_cannot_escape_scratch_root(tmp_path, group):
    from fxr.launch.exp_config import _group_log_root

    with pytest.raises(ValueError, match="safe directory|control"):
        _group_log_root(str(tmp_path), group, add_date=False)


def test_base_experiment_validates_uuid_before_joining_log_root(tmp_path):
    from fxr.experiment import BaseExperiment

    with pytest.raises(ValueError, match="run_id"):
        BaseExperiment.from_config(
            {"log": {"root": str(tmp_path)}},
            uuid="../outside",
        )

    assert not list(tmp_path.iterdir())
    experiment = BaseExperiment.from_config(
        {"log": {"root": str(tmp_path)}},
        uuid="human-readable-id",
    )
    assert experiment.path == tmp_path / "human-readable-id"


def test_base_experiment_persists_identity_and_loads_legacy_metadata(tmp_path):
    from fxr.experiment import BaseExperiment

    experiment = BaseExperiment.from_config(
        {"log": {"root": str(tmp_path)}},
        uuid="human-readable-id",
    )
    metadata_path = experiment.path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["run_id"] == experiment.path.name
    assert metadata["config_digest"] == config_digest(experiment.config)

    legacy_metadata = dict(metadata)
    legacy_metadata.pop("run_id")
    legacy_metadata.pop("config_digest")
    metadata_path.write_text(json.dumps(legacy_metadata), encoding="utf-8")
    assert BaseExperiment(experiment.path).path == experiment.path

    metadata["config_digest"] = "wrong"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="config digest"):
        BaseExperiment(experiment.path)


def test_base_experiment_detects_modified_generated_run_config(tmp_path):
    from fxr.experiment import BaseExperiment

    experiment = BaseExperiment.from_config({"log": {"root": str(tmp_path)}})
    config_path = experiment.path / "config.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["modified"] = True
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="config digest"):
        BaseExperiment(experiment.path)


def test_cli_resume_rejects_config_mutation(tmp_path, capsys):
    from fxr.launch.run import run_config

    config = {
        "experiment": {"_class": "tests.test_launch.FakeExperiment"},
        "log": {"root": str(tmp_path)},
    }
    created = run_config(config, train=False, run_id="existing-run-id")

    with pytest.raises(SystemExit) as exc_info:
        main(["--resume", str(created.path), "--set", "optim.lr=2"])

    assert exc_info.value.code == 2
    assert "immutable config" in capsys.readouterr().err


def test_submitit_requeue_keeps_the_same_run_identity(tmp_path, monkeypatch):
    from fxr.launch.submit import _RunConfigJob

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "scheduler-owned")
    monkeypatch.setenv("FXR_DEVICE", "cpu")

    calls = []
    monkeypatch.setattr(
        "fxr.launch.submit.run_config",
        lambda config, **kwargs: calls.append((config, kwargs)),
    )
    job = _RunConfigJob(
        {
            "experiment": {"_class": "tests.test_launch.FakeExperiment"},
            "log": {"root": str(tmp_path)},
            "data": {"Xray": {"HipRay": {}, "JRST": {}}},
            "dataloader": {"proportions": {"HipRay": 1, "JRST": 0}},
        }
    )

    job()
    job()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "scheduler-owned"
    assert os.environ["FXR_DEVICE"] == "auto"
    assert set(job.config["data"]["Xray"]) == {"HipRay"}
    assert job.run_id.endswith(config_digest(job.config))
    assert calls[0][1]["run_id"] == calls[1][1]["run_id"]
    assert calls[0][1]["resume_existing"] is True
    assert calls[1][1]["resume_existing"] is True
    with pytest.raises(ValueError, match="run_id"):
        _RunConfigJob(job.config, run_id="../unsafe-run-id")


def test_cli_replace_head_flags_persist_into_initialization_config(tmp_path, capsys):
    config_path = tmp_path / "cfg.yml"
    config_path.write_text(
        "experiment: {_class: tests.test_launch.FakeExperiment}\n"
        f"log: {{root: {tmp_path}}}\n",
        encoding="utf-8",
    )

    main([str(config_path), "--init-from", "weights.safetensors", "--replace-head", "--freeze-backbone"])

    initialization = FakeExperiment.instances[-1].config["initialization"]
    assert initialization["replace_head"] is True
    assert initialization["freeze_backbone"] is True

    for flags, message in (
        (["--replace-head"], "requires --init-from"),
        (["--init-from", "w.safetensors", "--freeze-backbone"], "requires --replace-head"),
    ):
        with pytest.raises(SystemExit) as exc_info:
            main([str(config_path), *flags])
        assert exc_info.value.code == 2
        assert message in capsys.readouterr().err
