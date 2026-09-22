"""Keep training view counts stable while separating offline render defaults."""

from copy import deepcopy

import pytest
import torch
import yaml

from fxr.config import merge_configs
from fxr.experiment.drr_forward import DrrForwardPipeline
from fxr.launch.cli import _build_parser, _resolve_new_configs, load_training_base
from fxr.launch.exp_config import build_submission_configs
from fxr.launch.render import OfflineRenderRequest, _render_config, render_dataset
from fxr.launch.render_cli import build_parser
from fxr.launch.run import resume_run, run_config
from fxr.models.camera.config import resolve_ct_profiles, resolve_num_views
from tests.test_camera import _pipeline_config, _synthetic_ct
from tests.test_render_cli import _pack_ct, _render_config as offline_config


@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_legacy_packaged_recipe_uses_training_batch_size(batch_size):
    config = merge_configs(load_training_base("base"), {"dataloader": {"batch_size": batch_size}})
    profiles = resolve_ct_profiles(config)
    assert {resolve_num_views(profile, config) for profile in profiles.values()} == {batch_size}


@pytest.mark.parametrize("profile,config,expected", [
    ({}, {}, 1), ({}, {"dataloader": {"batch_size": 8}}, 8),
    ({"num_views": 3}, {"dataloader": {"batch_size": 8}}, 3),
    ({"num_views": 3}, {"dataloader": {"batch_size": "?"}}, 3),
])
def test_view_resolver_preserves_existing_two_argument_api(profile, config, expected):
    assert resolve_num_views(profile, config) == expected


def test_training_runtime_renders_legacy_number_of_views():
    config = _pipeline_config()
    config["drr_model"]["default"].pop("num_views")
    config["drr_model"]["default"]["num_samples"] = 16
    config["dataloader"]["batch_size"] = 8
    pipeline = DrrForwardPipeline.from_config(config, device=torch.device("cpu"))
    volume, label, affine = _synthetic_ct()
    result = pipeline.render(volume=volume, label=label, affine=affine, dataset_name="ToyCT")
    assert result.images.shape == (8, 1, 8, 8)


def _launch_config(tmp_path):
    """Return a legacy CT config using a persistence-only experiment double."""
    config = _pipeline_config()
    config["drr_model"]["default"].pop("num_views")
    config["experiment"] = {"_class": "tests.test_launch.FakeExperiment"}
    config["log"] = {"root": str(tmp_path / "runs")}
    config["dataloader"]["batch_size"] = 8
    return config


def test_new_cli_run_persists_explicit_views_after_batch_override(tmp_path):
    config = _launch_config(tmp_path)
    source = tmp_path / "base.yml"
    source.write_text(yaml.safe_dump(config))
    args = _build_parser().parse_args([str(source), "--set", "dataloader.batch_size=4"])
    [resolved] = _resolve_new_configs(args)
    run = run_config(resolved, train=False)
    saved = yaml.safe_load((run.path / "config.yml").read_text())
    assert saved["drr_model"]["datasets"]["ToyCT"]["num_views"] == 4
    assert yaml.safe_load(source.read_text()) == config
    changed_batch = merge_configs(saved, {"dataloader": {"batch_size": 16}})
    profiles = resolve_ct_profiles(changed_batch)
    assert resolve_num_views(profiles["ToyCT"], changed_batch) == 4


@pytest.mark.parametrize("explicit", [None, 3])
def test_submit_records_views_after_final_cluster_and_cli_overrides(tmp_path, explicit):
    config = _launch_config(tmp_path)
    if explicit is not None:
        config["drr_model"]["default"]["num_views"] = explicit
    source = tmp_path / "base.yml"
    source.write_text(yaml.safe_dump(config))
    cells = build_submission_configs(
        group="views", base_cfgs=[str(source)], experiment_cfg={},
        cluster_overrides={"dataloader": {"batch_size": 8}},
        set_overrides={"dataloader.batch_size": 4}, scratch_root=str(tmp_path), add_date=False,
    )
    assert cells[0]["drr_model"]["datasets"]["ToyCT"]["num_views"] == (explicit or 4)


def test_legacy_resume_keeps_config_bytes_and_view_counts(tmp_path):
    config = _launch_config(tmp_path)
    run = run_config(config, train=False)
    (run.path / "checkpoints").mkdir()
    (run.path / "checkpoints" / "last.pt").touch()
    saved_path = run.path / "config.yml"
    before = saved_path.read_bytes()
    resumed = resume_run(run.path, train=False)
    profiles = resolve_ct_profiles(resumed.config)
    assert resolve_num_views(profiles["ToyCT"], resumed.config) == 8
    assert saved_path.read_bytes() == before
    assert "num_views" not in resumed.config["drr_model"]["default"]


def test_unvalidated_submission_keeps_missing_batch_size_for_later_validation(tmp_path):
    config = _launch_config(tmp_path)
    config["dataloader"]["batch_size"] = "?"
    source = tmp_path / "base.yml"
    source.write_text(yaml.safe_dump(config))
    [resolved] = build_submission_configs(
        group="views", base_cfgs=[str(source)], experiment_cfg={},
        cluster_overrides={}, scratch_root=str(tmp_path), add_date=False, validate=False,
    )
    assert resolved["dataloader"]["batch_size"] == "?"


@pytest.mark.parametrize("batch_size", ["?", 8])
def test_offline_render_defaults_to_one_view_independent_of_batch_size(tmp_path, batch_size):
    packed = _pack_ct(tmp_path)
    config = offline_config()
    config["drr_model"]["default"].pop("num_views")
    config["dataloader"]["batch_size"] = batch_size
    original = deepcopy(config)
    report = render_dataset(OfflineRenderRequest(
        dataset_path=packed, dataset_name="ToyCT", config=config,
        profile="MOOSE", output=tmp_path / "renders", renders=2,
    ))
    assert report.num_samples == 2
    assert config == original


def test_offline_explicit_dataset_views_override_default(tmp_path):
    config = offline_config(num_views=2)
    config["drr_model"]["datasets"]["MOOSE"]["num_views"] = 3
    request = OfflineRenderRequest(dataset_path=tmp_path, dataset_name="ToyCT",
                                   config=config, profile="MOOSE", output=tmp_path)
    resolved = _render_config(request)
    profile = resolve_ct_profiles(resolved)["ToyCT"]
    assert resolve_num_views(profile, resolved) == 3


def test_render_cli_resolves_relative_dataset_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["relative-data", "--dataset-name", "A", "--output", "out"])
    assert args.dataset == tmp_path / "relative-data"
