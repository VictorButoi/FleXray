"""Tests for the CT->DRR segmentation rendering runtime."""

from __future__ import annotations

import pytest
import torch

import fxr.models.camera.runtime as camera_runtime
from fxr.drr import IsocenterConfig, compute_isocenter
from fxr.experiment.drr_forward import DrrForwardPipeline
from fxr.models.camera import (
    DRRRenderConfig,
    ScalarRangeSampler,
    SegmentationDRRRuntime,
    resolve_ct_profiles,
)
from fxr.models.camera.config import _parse_attenuation, resolve_num_views
from fxr.protocols import (
    resolve_run_attenuated_label_ids,
    resolve_run_output_label_names,
    resolve_run_render_label_collapse_index,
    resolve_run_render_label_names,
)

_PROTOCOL = "all_structures_flexray_v4"
_CPU = torch.device("cpu")


def _render_config(**overrides) -> DRRRenderConfig:
    """Build a minimal three-channel render config for synthetic CT rendering."""
    defaults = dict(
        height=8,
        width=8,
        sdd_bounds=(1000.0, 1000.0),
        delx_bounds=(2.0, 2.0),
        dely_bounds=None,
        x0_bounds=(0.0, 0.0),
        y0_bounds=(0.0, 0.0),
        orientation="AP",
        isocenter_cfg=IsocenterConfig(sample_scheme="volume_center"),
        ortho_prob=0.0,
        num_views=2,
        n_samples=64,
        preset="frontal",
        camera_displacement=800.0,
        sample_params={},
        attenuation_range=None,
        do_per_label_attenuation=False,
        attenuated_label_ids=[1, 2],
        attenuation_dist=None,
        attenuation_prob=1.0,
        input_num_label_channels=3,
        output_num_label_channels=3,
        render_soft_labels=False,
        seg_threshold=0.0,
        label_smoothing_sigma=0.0,
        label_smoothing_kernel_size=None,
    )
    defaults.update(overrides)
    return DRRRenderConfig(**defaults)


def _synthetic_ct(seed: int = 0):
    """Return a synthetic CT volume, model-id label, and affine."""
    generator = torch.Generator().manual_seed(seed)
    volume = torch.rand(1, 1, 16, 16, 16, generator=generator) * 500 - 200
    label = torch.randint(0, 3, (1, 1, 16, 16, 16), generator=generator)
    affine = torch.eye(4).unsqueeze(0)
    return volume, label, affine


# --------------------------------------------------------------- config layer
def test_scalar_range_sampler_parses_scalar_and_range():
    assert ScalarRangeSampler.parse(900, "sdd") == (900.0, 900.0)
    assert ScalarRangeSampler.parse([900, 1100], "sdd") == (900.0, 1100.0)
    assert ScalarRangeSampler.sample((5.0, 5.0)) == 5.0
    sample = ScalarRangeSampler.sample((1.0, 2.0))
    assert 1.0 <= sample <= 2.0


def test_scalar_range_sampler_rejects_inverted_range():
    with pytest.raises(ValueError, match="inverted range"):
        ScalarRangeSampler.parse([2.0, 1.0], "sdd")


def _profile_config(datasets: dict) -> dict:
    return {
        "data": {"CT": {name: {} for name in datasets}},
        "dataloader": {"batch_size": 4},
        "drr_model": {
            "default": {
                "preset": "frontal",
                "num_views": 3,
                "intrinsics_cfg": {"height": 16, "width": 16, "sdd": 1000.0, "delx": 2.0},
                "isocenter_cfg": {"sample_scheme": "volume_center"},
            },
            "datasets": datasets,
        },
    }


def test_resolve_ct_profiles_merges_default_and_overrides():
    config = _profile_config({"A": {}, "B": {"num_views": 3, "preset": "lateral"}})
    profiles = resolve_ct_profiles(config)

    assert set(profiles) == {"A", "B"}
    assert profiles["A"]["preset"] == "frontal"
    assert profiles["B"]["preset"] == "lateral"
    assert resolve_num_views(profiles["A"]) == 3


def test_resolve_ct_profiles_requires_matching_dataset_keys():
    config = _profile_config({"A": {}})
    config["data"]["CT"]["B"] = {}
    with pytest.raises(ValueError, match="missing"):
        resolve_ct_profiles(config)


def test_resolve_ct_profiles_rejects_mismatched_geometry():
    config = _profile_config({"A": {}, "B": {"intrinsics_cfg": {"height": 32, "width": 32}}})
    with pytest.raises(ValueError, match="height/width"):
        resolve_ct_profiles(config)


def test_resolve_ct_profiles_returns_none_without_drr_model():
    assert resolve_ct_profiles({"data": {"CT": {"A": {}}}}) is None


# ------------------------------------------------------------- runtime render
def test_runtime_render_shapes_and_background_complement():
    runtime = SegmentationDRRRuntime(_render_config(), device=_CPU)
    volume, label, affine = _synthetic_ct()

    result = runtime.render(volume=volume, label=label, affine=affine)

    assert result.images.shape == (2, 1, 8, 8)
    assert result.labels.shape == (2, 3, 8, 8)
    assert result.images.dtype == torch.float32
    background = result.labels[:, 0]
    foreground = result.labels[:, 1:].amax(dim=1)
    assert torch.equal(background, (foreground <= 0).float())


def test_runtime_render_hard_labels_are_binary():
    runtime = SegmentationDRRRuntime(_render_config(), device=_CPU)
    volume, label, affine = _synthetic_ct(seed=3)

    result = runtime.render(volume=volume, label=label, affine=affine)

    assert torch.equal(result.labels, result.labels.round())


def test_runtime_render_soft_labels_stay_in_unit_range():
    config = _render_config(
        render_soft_labels=True,
        label_smoothing_sigma=1.0,
        label_smoothing_kernel_size=(3, 3),
    )
    runtime = SegmentationDRRRuntime(config, device=_CPU)
    volume, label, affine = _synthetic_ct(seed=5)

    result = runtime.render(volume=volume, label=label, affine=affine)

    assert float(result.labels.min()) >= 0.0
    assert float(result.labels.max()) <= 1.0


def test_attenuation_cfg_requires_prob_and_rejects_unknown_keys():
    valid = {"prob": 0.5, "range": [0.5, 2.0], "scope": "global"}

    assert _parse_attenuation(valid, attenuated_label_ids=[1])["prob"] == 0.5
    assert _parse_attenuation(None, attenuated_label_ids=[1])["prob"] == 0.0
    with pytest.raises(ValueError, match="prob"):
        _parse_attenuation({"range": [0.5, 2.0], "scope": "global"}, attenuated_label_ids=[1])
    with pytest.raises(ValueError, match="unknown"):
        _parse_attenuation({**valid, "sigma": 1.0}, attenuated_label_ids=[1])
    with pytest.raises(ValueError, match="prob"):
        _parse_attenuation({**valid, "prob": 1.5}, attenuated_label_ids=[1])


@pytest.mark.parametrize("prob, expect_attenuated", [(0.0, False), (1.0, True)])
def test_attenuation_prob_gates_render_multipliers(monkeypatch, prob, expect_attenuated):
    real_render_drr = camera_runtime.render_drr
    seen: dict = {}

    def spy(**kwargs):
        seen.update(kwargs)
        return real_render_drr(**kwargs)

    monkeypatch.setattr(camera_runtime, "render_drr", spy)
    config = _render_config(
        attenuation_range=(0.5, 2.0),
        do_per_label_attenuation=True,
        attenuation_dist={"type": "lognormal", "mode": 1.0, "sigma": 1.0},
        attenuation_prob=prob,
    )
    runtime = SegmentationDRRRuntime(config, device=_CPU)
    volume, label, affine = _synthetic_ct()

    runtime.render(volume=volume, label=label, affine=affine)

    assert (seen["attenuation"] is not None) is expect_attenuated
    assert seen["do_per_label_attenuation"] is expect_attenuated
    assert (seen["attenuation_dist"] is not None) is expect_attenuated


def test_runtime_random_label_falls_back_to_volume_center_on_empty_centroids():
    config = _render_config(
        isocenter_cfg=IsocenterConfig(sample_scheme="random_label", replacement=True)
    )
    runtime = SegmentationDRRRuntime(config, device=_CPU)
    volume, label, affine = _synthetic_ct()
    expected = compute_isocenter(
        volume, label, affine[0], IsocenterConfig(sample_scheme="volume_center"), num_views=2
    )

    resolved = runtime._resolve_isocenter(volume, label, affine[0], None, torch.zeros(0, 3))

    assert torch.equal(resolved, expected)


def test_runtime_random_label_isocenter_requires_centroids():
    config = _render_config(
        isocenter_cfg=IsocenterConfig(sample_scheme="random_label", replacement=True)
    )
    runtime = SegmentationDRRRuntime(config, device=_CPU)
    volume, label, affine = _synthetic_ct()

    with pytest.raises(ValueError, match="fg_centroids_ijk"):
        runtime.render(volume=volume, label=label, affine=affine)


# ---------------------------------------------------------------- pipeline
def _pipeline_config() -> dict:
    return {
        "protocol": {"name": _PROTOCOL},
        "data": {"CT": {"ToyCT": {}}},
        "dataloader": {"batch_size": 2},
        "drr_model": {
            "default": {
                "preset": "frontal",
                "num_views": 2,
                "intrinsics_cfg": {"height": 8, "width": 8, "sdd": 1000.0, "delx": 2.0},
                "isocenter_cfg": {"sample_scheme": "volume_center"},
                "seg_cfg": {"soft_labels": False, "threshold": 0.0},
            },
            "datasets": {"ToyCT": {}},
        },
    }


def test_pipeline_renders_protocol_channel_labels():
    pipeline = DrrForwardPipeline.from_config(_pipeline_config(), device=_CPU)
    assert pipeline is not None
    assert pipeline.get_render_size() == (8, 8)

    num_classes = len(resolve_run_output_label_names(_pipeline_config()))
    volume, label, affine = _synthetic_ct()
    result = pipeline.render(volume=volume, label=label, affine=affine, dataset_name="ToyCT")

    assert result.images.shape == (2, 1, 8, 8)
    assert result.labels.shape == (2, num_classes, 8, 8)


def test_pipeline_rejects_unknown_dataset():
    pipeline = DrrForwardPipeline.from_config(_pipeline_config(), device=_CPU)
    volume, label, affine = _synthetic_ct()
    with pytest.raises(ValueError, match="No CT DRR profile"):
        pipeline.render(volume=volume, label=label, affine=affine, dataset_name="Missing")


def test_pipeline_is_none_without_drr_model():
    assert DrrForwardPipeline.from_config({"data": {"CT": {}}}, device=_CPU) is None


# ---------------------------------------------------------------- resolvers
def test_render_label_resolvers_match_output_labels():
    config = {"protocol": {"name": _PROTOCOL}}
    output = resolve_run_output_label_names(config)

    assert output[0] == "background"
    assert len(output) == 61
    assert "lumbar_spine" not in output
    assert "thoracolumbar_spine" not in output
    assert resolve_run_render_label_names(config) == output
    assert resolve_run_attenuated_label_ids(config) == list(range(1, len(output)))
    assert resolve_run_render_label_collapse_index(config) is None


def test_render_label_resolvers_none_without_protocol():
    assert resolve_run_output_label_names({}) is None
    assert resolve_run_attenuated_label_ids({}) is None
