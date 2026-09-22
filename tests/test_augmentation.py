from __future__ import annotations

import ast
from importlib.resources import files
from pathlib import Path

import pytest
import torch

import fxr
import fxr.augmentation as public_augmentation
from fxr.augmentation import (
    RandomIntensityScale,
    RandomLabelZoom,
    RandomLetterDrop,
    SegmentationAugmentationPipeline,
    apply_segmentation_augmentation,
    build_segmentation_augmentation_pipeline,
    load_named_augmentation_preset,
)


def _write_preset(root: Path, name: str, text: str, *, suffix: str = ".yml") -> Path:
    path = root / "augmentations" / f"{name}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _pipeline_names(pipeline) -> list[str]:
    if pipeline is None:
        return []
    return [type(op).__name__ for op in pipeline.children()]


def _label_aware_names(pipeline: SegmentationAugmentationPipeline) -> list[str]:
    return [type(op).__name__ for op in pipeline.label_aware_ops]


def _packaged_preset_names() -> tuple[str, ...]:
    """Discover every packaged augmentation preset."""

    root = files("fxr.configs").joinpath("augmentations")
    names = tuple(
        sorted(
            path.name.removesuffix(".yml")
            for path in root.iterdir()
            if path.name.endswith(".yml")
        )
    )
    assert names, "FleXray must package at least one augmentation preset."
    return names


def test_public_augmentation_api_exports_planned_names() -> None:
    assert "augmentation" in fxr.__all__
    expected = {
        "DEFAULT_NORMALIZATION_EPS",
        "DEFAULT_NORMALIZATION_PERCENTILES",
        "MinMaxNormalize",
        "PercentileMinMaxNormalize",
        "RandomAspectCrop",
        "RandomClaheOrGamma",
        "RandomIntensityScale",
        "RandomLabelZoom",
        "RandomLetterDrop",
        "SegmentationAugmentationPipeline",
        "Standardize",
        "apply_segmentation_augmentation",
        "build_input_normalizer",
        "build_segmentation_augmentation_pipeline",
        "load_named_augmentation_preset",
        "resolve_augmentation_presets",
        "resolve_input_normalization_config",
        "restore_hard_background",
        "snapshot_augmentation_presets",
    }
    assert set(public_augmentation.__all__) == expected
    assert RandomLabelZoom.__name__ == "RandomLabelZoom"
    assert RandomLetterDrop.__name__ == "RandomLetterDrop"
    assert RandomIntensityScale.__name__ == "RandomIntensityScale"
    assert not hasattr(public_augmentation, "RandomCropMask")


def test_augmentation_initializer_is_reexport_only() -> None:
    tree = ast.parse(Path(public_augmentation.__file__).read_text(encoding="utf-8"))
    disallowed = (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef)

    assert not any(isinstance(node, disallowed) for node in tree.body)


@pytest.mark.parametrize("preset_name", _packaged_preset_names())
def test_packaged_presets_reference_supported_transform_modules(
    preset_name: str,
) -> None:
    preset = load_named_augmentation_preset(preset_name)
    module_names = {
        transform_config["module"] for transform_config in preset.values()
    }

    assert preset
    assert module_names <= {"fxr.augmentation", "kornia.augmentation"}


def test_load_named_augmentation_preset_preserves_order_and_validates(
    tmp_path: Path,
) -> None:
    _write_preset(
        tmp_path,
        "ordered",
        """\
RandomGamma:
  module: kornia.augmentation
  p: 0.1
RandomIntensityScale:
  module: fxr.augmentation
  factors: [0.0, 0.0]
  prob: 1.0
""",
    )

    preset = load_named_augmentation_preset("ordered", config_root=tmp_path)

    assert list(preset) == ["RandomGamma", "RandomIntensityScale"]
    assert preset["RandomIntensityScale"]["module"] == "fxr.augmentation"

    for bad_name in [
        "../ordered",
        "nested/ordered",
        "nested\\ordered",
        "ordered.txt",
        "",
    ]:
        with pytest.raises(ValueError):
            load_named_augmentation_preset(bad_name, config_root=tmp_path)

    with pytest.raises(FileNotFoundError, match="missing"):
        load_named_augmentation_preset("missing", config_root=tmp_path)

    _write_preset(tmp_path, "not_mapping", "- RandomGamma\n")
    with pytest.raises(TypeError, match="must be a mapping"):
        load_named_augmentation_preset("not_mapping", config_root=tmp_path)

    _write_preset(tmp_path, "bad_transform", "RandomGamma: null\n")
    with pytest.raises(ValueError, match="RandomGamma"):
        load_named_augmentation_preset("bad_transform", config_root=tmp_path)


def test_build_segmentation_augmentation_pipeline_splits_label_aware_ops() -> None:
    preset = {
        "RandomGamma": {
            "module": "kornia.augmentation",
            "gamma": (1.0, 1.0),
            "gain": (1.0, 1.0),
            "p": 0.0,
        },
        "RandomLabelZoom": {
            "module": "fxr.augmentation",
            "zoom_range": (1.5, 2.0),
            "p": 0.5,
        },
        "RandomIntensityScale": {
            "module": "fxr.augmentation",
            "factors": (0.0, 0.0),
            "prob": 1.0,
        },
    }

    pipeline = build_segmentation_augmentation_pipeline(preset)

    assert isinstance(pipeline, SegmentationAugmentationPipeline)
    assert _label_aware_names(pipeline) == ["RandomLabelZoom"]
    assert _pipeline_names(pipeline.kornia_pipeline) == [
        "RandomGamma",
        "RandomIntensityScale",
    ]


@pytest.mark.parametrize("preset_name", _packaged_preset_names())
def test_packaged_presets_build_each_configured_transform_once(
    preset_name: str,
) -> None:
    preset = load_named_augmentation_preset(preset_name)
    pipeline = build_segmentation_augmentation_pipeline(preset)
    built_names = _label_aware_names(pipeline) + _pipeline_names(
        pipeline.kornia_pipeline
    )

    assert len(built_names) == len(preset)
    assert set(built_names) == set(preset)


def test_apply_segmentation_augmentation_accepts_4d_and_rejects_bad_shapes() -> None:
    pipeline = build_segmentation_augmentation_pipeline({})
    image = torch.zeros(2, 1, 8, 9)
    label = torch.zeros(2, 3, 8, 9)

    out_image, out_label = apply_segmentation_augmentation(pipeline, image, label)

    assert out_image.shape == image.shape
    assert out_label.shape == label.shape

    with pytest.raises(ValueError, match="image must be a 4D"):
        apply_segmentation_augmentation(pipeline, image[0], label)
    with pytest.raises(ValueError, match="label must be a 4D"):
        apply_segmentation_augmentation(pipeline, image, label[0])
    with pytest.raises(ValueError, match="aligned batch"):
        apply_segmentation_augmentation(pipeline, image, torch.zeros(1, 3, 8, 9))


def test_apply_segmentation_augmentation_returns_aligned_augmented_shapes() -> None:
    pipeline = build_segmentation_augmentation_pipeline(
        {
            "RandomLabelZoom": {
                "module": "fxr.augmentation",
                "zoom_range": (2.0, 2.0),
                "p": 1.0,
            },
            "RandomAffine": {
                "module": "kornia.augmentation",
                "degrees": 0.0,
                "p": 0.0,
            },
        }
    ).train()
    image = torch.arange(64, dtype=torch.float32).view(1, 1, 8, 8)
    label = torch.zeros(1, 2, 8, 8)
    label[:, 1, 5:7, 5:7] = 1.0

    out_image, out_label = apply_segmentation_augmentation(pipeline, image, label)

    assert out_image.shape == image.shape
    assert out_label.shape == label.shape
    assert out_label[:, 1].sum() > 0


def test_random_intensity_scale_scales_training_batches_and_validates() -> None:
    transform = RandomIntensityScale(factors=(0.5, 0.5), prob=1.0).train()
    x = torch.ones(2, 1, 3, 4)

    assert torch.allclose(transform(x), torch.full_like(x, 1.5))

    transform.eval()
    assert torch.equal(transform(x), x)

    with pytest.raises(ValueError, match="prob"):
        RandomIntensityScale(prob=1.5)
    with pytest.raises(ValueError, match="4D"):
        transform(torch.ones(1, 3, 4))


def test_random_letter_drop_stamps_marker_and_validates() -> None:
    torch.manual_seed(7)
    transform = RandomLetterDrop(
        p=1.0,
        min_size=0.5,
        max_size=0.5,
        intensity_range=(1.0, 1.0),
    ).train()
    x = torch.zeros(1, 2, 20, 20)

    out = transform(x)

    assert out.shape == x.shape
    assert out.max().item() == pytest.approx(1.0)
    assert out.sum().item() > 0

    transform.eval()
    assert torch.equal(transform(x), x)

    with pytest.raises(ValueError, match="min_size"):
        RandomLetterDrop(min_size=0.2, max_size=0.1)


def test_random_label_zoom_zooms_foreground_and_validates() -> None:
    torch.manual_seed(3)
    transform = RandomLabelZoom(zoom_range=(2.0, 2.0), p=1.0).train()
    image = torch.arange(64, dtype=torch.float32).view(1, 1, 8, 8)
    label = torch.zeros(1, 2, 8, 8)
    label[:, 1, 5:7, 5:7] = 1.0

    out_image, out_label = transform(image, label)

    assert out_image.shape == image.shape
    assert out_label.shape == label.shape
    assert not torch.equal(out_image, image)
    assert out_label[:, 1].sum() > 0

    transform.eval()
    eval_image, eval_label = transform(image, label)
    assert torch.equal(eval_image, image)
    assert torch.equal(eval_label, label)

    with pytest.raises(ValueError, match="zoom_range"):
        RandomLabelZoom(zoom_range=(0.9, 1.0))
    with pytest.raises(ValueError, match="aligned batch"):
        transform.train()(torch.zeros(1, 1, 8, 8), torch.zeros(2, 2, 8, 8))


def test_random_clahe_or_gamma_is_exclusive_and_validates_probabilities() -> None:
    from fxr.augmentation import RandomClaheOrGamma

    image = torch.rand(4, 1, 32, 32)
    untouched = RandomClaheOrGamma(clahe_prob=0.0, gamma_prob=0.0).train()(image)
    assert torch.equal(untouched, image)

    gamma_only = RandomClaheOrGamma(
        clahe_prob=0.0, gamma_prob=1.0, gamma=(2.0, 2.0), gain=(1.0, 1.0)
    ).train()
    assert torch.allclose(gamma_only(image), image**2, atol=1e-5)
    assert torch.equal(gamma_only.eval()(image), image)

    with pytest.raises(AssertionError, match="<= 1"):
        RandomClaheOrGamma(clahe_prob=0.6, gamma_prob=0.6)


def test_random_aspect_crop_zeroes_one_axis_and_matches_parity() -> None:
    from fxr.augmentation import RandomAspectCrop

    torch.manual_seed(0)
    image = torch.ones(8, 1, 32, 32)
    label = torch.ones(8, 2, 32, 32)
    crop = RandomAspectCrop(height_range=[8, 16], width_range=[8, 16], p=1.0).train()

    out_image = crop(image)
    params = crop._params
    out_label = crop(label, params=params, data_keys=["mask"])

    for index in range(8):
        rows = out_image[index, 0].sum(dim=1) > 0
        columns = out_image[index, 0].sum(dim=0) > 0
        kept_height, kept_width = int(rows.sum()), int(columns.sum())
        if bool(params["is_landscape"][index]):
            assert 8 <= kept_height <= 16 and kept_height % 2 == 0 and kept_width == 32
        else:
            assert 8 <= kept_width <= 16 and kept_width % 2 == 0 and kept_height == 32
        top, bottom = int(rows.nonzero()[0]), 32 - int(rows.nonzero()[-1]) - 1
        assert top == bottom
    assert torch.equal(out_label[:, 0], out_image[:, 0])

    small = torch.ones(2, 1, 8, 8)
    with pytest.raises(ValueError, match="exceeds input axis"):
        RandomAspectCrop(height_range=[64, 224], width_range=[64, 224], p=1.0).train()(small)
    with pytest.raises(ValueError, match="parity"):
        RandomAspectCrop(height_range=[7, 7], width_range=[8, 8], p=1.0).train()(small)

    partial = build_segmentation_augmentation_pipeline(
        {"RandomAspectCrop": {"height_range": [8, 16], "width_range": [8, 16], "p": 0.5}}
    ).train()
    torch.manual_seed(1)
    out_image, out_label = apply_segmentation_augmentation(partial, image, label)
    assert torch.equal(out_label[:, 1], out_image[:, 0])
    assert 0 < int((out_image.sum(dim=(1, 2, 3)) == 32 * 32).sum()) < 8


def test_restore_hard_background_after_elastic_and_affine() -> None:
    from fxr.augmentation import restore_hard_background

    pipeline = build_segmentation_augmentation_pipeline(
        {
            "RandomElasticTransform": {
                "module": "kornia.augmentation",
                "kernel_size": [15, 15],
                "alpha": [4.0, 4.0],
                "sigma": [6.0, 6.0],
                "p": 1.0,
            },
            "RandomAffine": {"module": "kornia.augmentation", "degrees": 30, "p": 1.0},
        }
    ).train()
    torch.manual_seed(0)
    image = torch.rand(2, 1, 32, 32)
    label = torch.zeros(2, 3, 32, 32)
    label[:, 1, 8:20, 8:20] = 1.0
    label[:, 2, 16:28, 4:12] = 1.0
    label[:, 0] = label[:, 1:].amax(dim=1) <= 0

    _, augmented = apply_segmentation_augmentation(pipeline, image, label)

    assert torch.equal(augmented, augmented.round())
    assert torch.equal(augmented[:, 0], (augmented[:, 1:].amax(dim=1) <= 0).float())

    soft = torch.tensor([[[[0.2, 0.7]], [[0.6, 0.4]]]])
    assert torch.equal(
        restore_hard_background(soft), torch.tensor([[[[0.0, 1.0]], [[1.0, 0.0]]]])
    )
    single = torch.rand(1, 1, 2, 2)
    assert restore_hard_background(single) is single


def test_presets_route_by_modality_and_default_to_base_names() -> None:
    from fxr.augmentation import resolve_augmentation_presets

    assert resolve_augmentation_presets(None) == {"ct": "CT_base", "xray": "Xray_base"}
    assert resolve_augmentation_presets({"augmentation": {"presets": {"CT": "base_light"}}}) == {
        "ct": "base_light",
        "xray": "Xray_base",
    }
    with pytest.raises(ValueError, match="unknown modality"):
        resolve_augmentation_presets({"augmentation": {"presets": {"MRI": "CT_base"}}})
    with pytest.raises(ValueError, match="unknown key"):
        resolve_augmentation_presets({"augmentation": {"preset": "CT_base"}})


def test_snapshot_augmentation_presets_writes_once(tmp_path: Path) -> None:
    from fxr.augmentation import snapshot_augmentation_presets

    presets = {"ct": {"RandomInvert": {"module": "kornia.augmentation", "p": 0.5}}}
    snapshot_augmentation_presets(tmp_path, presets)
    path = tmp_path / "augmentations" / "ct.yml"
    first = path.read_text()
    assert load_named_augmentation_preset("ct", config_root=tmp_path) == presets["ct"]

    snapshot_augmentation_presets(tmp_path, {"ct": {}})
    assert path.read_text() == first
