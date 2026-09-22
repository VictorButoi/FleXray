from __future__ import annotations

import gc
import weakref
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

import fxr.datasets.packaging as dataset_packaging
from fxr.datasets import (
    CTTrainingDataset,
    DatasetPackageReport,
    SplitThunderDBStorageBackend,
    TrainingLabelRemap,
    XrayTrainingDataset,
    pack_dataset,
    validate_dataset_manifest,
    validate_packed_dataset,
)
from fxr.datasets.cli import main as dataset_main

thunderpack = pytest.importorskip("thunderpack")

_XRAY_SHAPE = (64, 64)


def _write_manifest(root: Path, body: dict[str, object]) -> Path:
    """Write one YAML packaging manifest for a test.

    Args:
        root: Test directory that owns the manifest.
        body: YAML-compatible manifest body.

    Returns:
        Written manifest path.
    """

    path = root / "dataset.yml"
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


def _sample(
    sample_id: str,
    subject_id: str,
    split: str,
    *,
    image: str,
    label: str,
    **payloads: str,
) -> dict[str, object]:
    """Build a manifest sample mapping with explicit split ownership.

    Args:
        sample_id: Stable sample identifier.
        subject_id: Subject identifier used for leakage checks.
        split: Explicit split name.
        image: Relative image payload path.
        label: Relative label payload path.
        **payloads: Optional CT affine and spacing paths.

    Returns:
        YAML-compatible sample mapping.
    """

    return {
        "sample_id": sample_id,
        "subject_id": subject_id,
        "split": split,
        "image": image,
        "label": label,
        **payloads,
    }


def _write_ct_case_manifest(
    root: Path,
    *,
    label: np.ndarray,
    affine: np.ndarray,
) -> Path:
    """Write a minimal CT manifest with caller-selected failure payloads."""

    image = np.ones(label.shape[-3:], dtype=np.float32)
    payloads = {
        "image": image,
        "label": label,
        "affine": affine,
        "spacing": np.ones(3, dtype=np.float32),
    }
    for name, array in payloads.items():
        np.save(root / f"{name}.npy", array)
    return _write_manifest(
        root,
        {
            "schema_version": 1,
            "dataset_name": "CtContract",
            "dataset_type": "ct-seg",
            "stored_labels": {0: "background", 1: "foreground"},
            "samples": [
                _sample(
                    "volume",
                    "patient",
                    "train",
                    image="image.npy",
                    label="label.npy",
                    affine="affine.npy",
                    spacing="spacing.npy",
                )
            ],
        },
    )


def test_pack_xray_dense_images_emits_canonical_layout_and_runtime_samples(
    tmp_path: Path,
) -> None:
    image_a = (np.arange(np.prod(_XRAY_SHAPE)) % 256).astype(np.uint8)
    image_a = image_a.reshape(_XRAY_SHAPE)
    image_b = np.flipud(image_a).copy()
    label_a = np.zeros(_XRAY_SHAPE, dtype=np.uint8)
    label_a[1:3, 2:4] = 2
    label_b = np.ones(_XRAY_SHAPE, dtype=np.uint8)
    Image.fromarray(image_a).save(tmp_path / "image_a.png")
    Image.fromarray(image_b).save(tmp_path / "image_b.tiff")
    Image.fromarray(label_a).save(tmp_path / "label_a.png")
    Image.fromarray(label_b).save(tmp_path / "label_b.tiff")
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_name": "ClinicXray",
            "dataset_type": "xray-seg",
            "stored_labels": {0: "background", 1: "organ", 2: "implant"},
            "samples": [
                _sample(
                    "study_a",
                    "patient_a",
                    "train",
                    image="image_a.png",
                    label="label_a.png",
                ),
                _sample(
                    "study_b",
                    "patient_b",
                    "val",
                    image="image_b.tiff",
                    label="label_b.tiff",
                ),
            ],
        },
    )

    report = pack_dataset(manifest, tmp_path / "packed")

    assert report == DatasetPackageReport(
        path=(tmp_path / "packed").resolve(),
        dataset_name="ClinicXray",
        dataset_type="xray-seg",
        label_encoding="dense",
        num_subjects=2,
        num_samples=2,
        split_counts={"train": 1, "val": 1},
    )
    with thunderpack.ThunderDB.open(str(tmp_path / "packed"), "r") as db:
        assert db["_subjects"] == ["patient_a", "patient_b"]
        assert db["_samples"] == ["study_a", "study_b"]
        assert db["_splits"] == {"train": ["study_a"], "val": ["study_b"]}
        assert db["_metadata"]["study_a"]["subject_id"] == "patient_a"
        assert db["_attrs"]["payload_keys"] == {
            "image": "img",
            "label": "seg",
        }
        assert db["_attrs"]["stored_labels"] == {
            "0": "background",
            "1": "organ",
            "2": "implant",
        }
        backend = SplitThunderDBStorageBackend(
            db,
            dataset_name="ClinicXray",
            modality="xray",
            split="train",
            subject_grouping="subject",
        )
        dataset = XrayTrainingDataset(
            backend=backend,
            dataset_name="ClinicXray",
            label_mode="native",
        )
        sample = dataset[0]
        expected_image = (
            torch.as_tensor(image_a, dtype=torch.float32)[None]
            / float(np.iinfo(image_a.dtype).max)
        )
        torch.testing.assert_close(sample["image"], expected_image)
        torch.testing.assert_close(sample["label"], torch.as_tensor(label_a).long())
        assert sample["metadata"]["subject_id"] == "patient_a"


def test_pack_rgb_png_canonicalizes_grayscale_intensity_for_runtime(
    tmp_path: Path,
) -> None:
    rgb_tile = np.array(
        [
            [[0, 0, 0], [255, 255, 255]],
            [[255, 0, 0], [0, 255, 0]],
        ],
        dtype=np.uint8,
    )
    rgb = np.tile(rgb_tile, (32, 32, 1))
    label = np.zeros(_XRAY_SHAPE, dtype=np.uint8)
    Image.fromarray(rgb).save(tmp_path / "image.png")
    Image.fromarray(label).save(tmp_path / "label.png")
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_name": "RgbClinicXray",
            "dataset_type": "xray-seg",
            "stored_labels": {0: "background"},
            "samples": [
                _sample(
                    "study",
                    "patient",
                    "train",
                    image="image.png",
                    label="label.png",
                )
            ],
        },
    )

    plan = dataset_packaging._load_package_plan(manifest)
    assert plan.samples[0].image.shape == (1, *_XRAY_SHAPE)
    assert plan.samples[0].image.dtype == np.dtype(np.float32).str

    pack_dataset(manifest, tmp_path / "packed")

    with thunderpack.ThunderDB.open(str(tmp_path / "packed"), "r") as db:
        stored = np.asarray(db["study"]["img"])
        assert stored.shape == (1, *_XRAY_SHAPE)
        assert stored.dtype == np.float32
        assert stored.flags.c_contiguous
        assert np.all((stored >= 0) & (stored <= 1))
        expected_grayscale = (
            np.asarray(Image.fromarray(rgb).convert("F"), dtype=np.float32)
            / np.iinfo(rgb.dtype).max
        )
        np.testing.assert_allclose(stored[0], expected_grayscale, atol=1e-6)
        backend = SplitThunderDBStorageBackend(
            db,
            dataset_name="RgbClinicXray",
            modality="xray",
            split="train",
        )
        runtime = XrayTrainingDataset(
            backend=backend,
            dataset_name="RgbClinicXray",
            label_mode="native",
        )[0]
        assert runtime["image"].shape == (1, *_XRAY_SHAPE)
        assert runtime["image"].dtype == torch.float32
        torch.testing.assert_close(runtime["image"], torch.from_numpy(stored.copy()))


def test_manifest_validation_rejects_multichannel_numpy_xray_image(
    tmp_path: Path,
) -> None:
    np.save(tmp_path / "image.npy", np.ones((2, 3, 4), dtype=np.float32))
    np.save(tmp_path / "label.npy", np.zeros((3, 4), dtype=np.uint8))
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_name": "MultiChannelXray",
            "dataset_type": "xray-seg",
            "stored_labels": {0: "background"},
            "samples": [
                _sample(
                    "study",
                    "patient",
                    "train",
                    image="image.npy",
                    label="label.npy",
                )
            ],
        },
    )

    with pytest.raises(ValueError, match="2D or one-channel CHW"):
        validate_dataset_manifest(manifest)


def test_pack_xray_scales_integer_dtype_range_and_accepts_small_shapes(
    tmp_path: Path,
) -> None:
    image = np.array(
        [[0, 1, 1024], [2048, 4094, 4095]],
        dtype=np.uint16,
    )
    label = np.zeros(image.shape, dtype=np.uint8)
    np.save(tmp_path / "image.npy", image)
    np.save(tmp_path / "label.npy", label)
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_name": "TinyXray",
            "dataset_type": "xray-seg",
            "stored_labels": {0: "background"},
            "samples": [
                _sample(
                    "study",
                    "patient",
                    "train",
                    image="image.npy",
                    label="label.npy",
                )
            ],
        },
    )

    report = pack_dataset(manifest, tmp_path / "packed")

    assert report.split_counts == {"train": 1}
    with thunderpack.ThunderDB.open(str(tmp_path / "packed"), "r") as db:
        stored = np.asarray(db["study"]["img"])
    expected = image.astype(np.float32)[None] / np.iinfo(image.dtype).max
    np.testing.assert_allclose(stored, expected)
    assert stored.dtype == np.float32
    assert stored.shape == (1, *image.shape)


def test_manifest_validation_rejects_unscaled_float_xray(
    tmp_path: Path,
) -> None:
    image = np.array([[0.0, 1.01], [0.5, 0.75]], dtype=np.float32)
    np.save(tmp_path / "image.npy", image)
    np.save(tmp_path / "label.npy", np.zeros(image.shape, dtype=np.uint8))
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_name": "UnscaledFloatXray",
            "dataset_type": "xray-seg",
            "stored_labels": {0: "background"},
            "samples": [
                _sample(
                    "study",
                    "patient",
                    "train",
                    image="image.npy",
                    label="label.npy",
                )
            ],
        },
    )

    with pytest.raises(ValueError, match=r"must already be in \[0, 1\]"):
        validate_dataset_manifest(manifest)


def test_manifest_validation_rejects_variable_xray_shapes(tmp_path: Path) -> None:
    shapes = ((64, 64), (64, 65))
    samples = []
    for index, shape in enumerate(shapes):
        image_path = tmp_path / f"image_{index}.npy"
        label_path = tmp_path / f"label_{index}.npy"
        np.save(image_path, np.ones(shape, dtype=np.float32))
        np.save(label_path, np.zeros(shape, dtype=np.uint8))
        samples.append(
            _sample(
                f"study_{index}",
                f"patient_{index}",
                "train" if index == 0 else "val",
                image=image_path.name,
                label=label_path.name,
            )
        )
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_name": "VariableXray",
            "dataset_type": "xray-seg",
            "stored_labels": {0: "background"},
            "samples": samples,
        },
    )

    with pytest.raises(ValueError, match="share one spatial shape"):
        validate_dataset_manifest(manifest)


def test_manifest_validation_rejects_multichannel_ct_image(tmp_path: Path) -> None:
    arrays = {
        "image": np.ones((2, 3, 4, 5), dtype=np.float32),
        "label": np.zeros((3, 4, 5), dtype=np.uint8),
        "affine": np.eye(4, dtype=np.float32),
        "spacing": np.ones(3, dtype=np.float32),
    }
    for name, array in arrays.items():
        np.save(tmp_path / f"{name}.npy", array)
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_name": "MultiChannelCT",
            "dataset_type": "ct-seg",
            "stored_labels": {0: "background"},
            "samples": [
                _sample(
                    "volume",
                    "patient",
                    "train",
                    image="image.npy",
                    label="label.npy",
                    affine="affine.npy",
                    spacing="spacing.npy",
                )
            ],
        },
    )

    with pytest.raises(ValueError, match="one-channel 4D volume"):
        validate_dataset_manifest(manifest)


def test_manifest_validation_rejects_multichannel_ct_label(
    tmp_path: Path,
) -> None:
    arrays = {
        "image": np.ones((3, 4, 5), dtype=np.float32),
        "label": np.zeros((2, 3, 4, 5), dtype=np.uint8),
        "affine": np.eye(4, dtype=np.float32),
        "spacing": np.ones(3, dtype=np.float32),
    }
    for name, array in arrays.items():
        np.save(tmp_path / f"{name}.npy", array)
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_name": "MultiChannelCtLabel",
            "dataset_type": "ct-seg",
            "stored_labels": {0: "background"},
            "samples": [
                _sample(
                    "volume",
                    "patient",
                    "train",
                    image="image.npy",
                    label="label.npy",
                    affine="affine.npy",
                    spacing="spacing.npy",
                )
            ],
        },
    )

    with pytest.raises(ValueError, match="dense 3D map or one-channel 4D map"):
        validate_dataset_manifest(manifest)


def test_packaging_keeps_only_one_samples_payload_arrays_live(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples: list[dict[str, object]] = []
    payload_paths: set[Path] = set()
    for index in range(4):
        image_path = tmp_path / f"image_{index}.npy"
        label_path = tmp_path / f"label_{index}.npy"
        np.save(image_path, np.full(_XRAY_SHAPE, index, dtype=np.uint8))
        np.save(label_path, np.zeros(_XRAY_SHAPE, dtype=np.uint8))
        payload_paths.update({image_path.resolve(), label_path.resolve()})
        samples.append(
            _sample(
                f"study_{index}",
                f"patient_{index}",
                "train",
                image=image_path.name,
                label=label_path.name,
            )
        )
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_name": "StreamingXray",
            "dataset_type": "xray-seg",
            "stored_labels": {0: "background"},
            "samples": samples,
        },
    )

    live_payloads = 0
    peak_live_payloads = 0
    load_counts: dict[Path, int] = {}
    real_image_loader = dataset_packaging._load_xray_image
    real_label_loader = dataset_packaging._load_xray_label

    def track(array: np.ndarray, path: Path) -> np.ndarray:
        """Track the lifetime of an array returned to the packager."""

        nonlocal live_payloads, peak_live_payloads
        live_payloads += 1
        peak_live_payloads = max(peak_live_payloads, live_payloads)
        load_counts[path] = load_counts.get(path, 0) + 1

        def release() -> None:
            """Record that one tracked payload array left live scope."""

            nonlocal live_payloads
            live_payloads -= 1

        weakref.finalize(array, release)
        return array

    def load_image(path: Path, *, context: str) -> np.ndarray:
        """Load an image while observing its live-array lifetime."""

        return track(real_image_loader(path, context=context), path)

    def load_label(path: Path, *, context: str) -> np.ndarray:
        """Load a label while observing its live-array lifetime."""

        return track(real_label_loader(path, context=context), path)

    monkeypatch.setattr(dataset_packaging, "_load_xray_image", load_image)
    monkeypatch.setattr(dataset_packaging, "_load_xray_label", load_label)

    report = pack_dataset(manifest, tmp_path / "packed")
    gc.collect()

    assert report.num_samples == len(samples)
    assert peak_live_payloads <= 2
    assert live_payloads == 0
    assert set(load_counts) == payload_paths
    assert all(1 <= count <= 2 for count in load_counts.values())


def test_pack_xray_named_channels_preserves_partial_labels_and_projects_by_name(
    tmp_path: Path,
) -> None:
    image = np.ones((1, *_XRAY_SHAPE), dtype=np.float32)
    label = np.zeros((2, *_XRAY_SHAPE), dtype=np.float32)
    label[0, 0, 1] = 1
    label[1, 2, 3] = 1
    np.save(tmp_path / "image.npy", image)
    np.save(tmp_path / "label.npy", label)
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_name": "PartialClinicXray",
            "dataset_type": "xray-seg",
            "protocol_name": "clinic-protocol",
            "label_names": ["lungs", "heart"],
            "samples": [
                _sample(
                    "study",
                    "patient",
                    "train",
                    image="image.npy",
                    label="label.npy",
                )
            ],
        },
    )

    report = pack_dataset(manifest, tmp_path / "packed")
    assert report.label_encoding == "channels"
    with thunderpack.ThunderDB.open(str(tmp_path / "packed"), "r") as db:
        backend = SplitThunderDBStorageBackend(
            db,
            dataset_name="PartialClinicXray",
            modality="xray",
            split="train",
        )
        assert backend.records[0].label_names == ("lungs", "heart")
        native = XrayTrainingDataset(
            backend=backend,
            dataset_name="PartialClinicXray",
            label_mode="native",
        )[0]
        assert native["label"].dtype == torch.float32
        torch.testing.assert_close(native["label"], torch.as_tensor(label))

        remap = TrainingLabelRemap(
            dataset_name="PartialClinicXray",
            protocol_name="clinic-protocol",
            label_lut=torch.zeros(1, dtype=torch.long),
            label_names=("background", "heart", "lungs", "clavicles"),
            native_id_to_model_id={},
            native_id_to_model_label={},
        )
        projected = XrayTrainingDataset(
            backend=backend,
            dataset_name="PartialClinicXray",
            label_mode="model",
            label_remap=remap,
        )[0]["label"]
        assert projected.shape == (4, *_XRAY_SHAPE)
        torch.testing.assert_close(projected[1], torch.as_tensor(label[1]))
        torch.testing.assert_close(projected[2], torch.as_tensor(label[0]))
        assert projected[0].count_nonzero() == 0
        assert projected[3].count_nonzero() == 0


def test_pack_ct_seg_round_trips_affine_spacing_and_subject_samples(
    tmp_path: Path,
) -> None:
    image = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    label = np.zeros((2, 3, 4), dtype=np.uint8)
    label[1, 2, 3] = 1
    affine = np.eye(4, dtype=np.float32)
    affine[:3, 3] = [10, 20, 30]
    spacing = np.array([0.7, 0.8, 1.5], dtype=np.float32)
    for name, array in {
        "image": image,
        "label": label,
        "affine": affine,
        "spacing": spacing,
    }.items():
        np.save(tmp_path / f"{name}.npy", array)
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_name": "ClinicCT",
            "dataset_type": "ct-seg",
            "stored_labels": {0: "background", 1: "anatomy"},
            "samples": [
                _sample(
                    "volume",
                    "patient",
                    "train",
                    image="image.npy",
                    label="label.npy",
                    affine="affine.npy",
                    spacing="spacing.npy",
                )
            ],
        },
    )

    report = pack_dataset(manifest, tmp_path / "packed")

    assert report.dataset_type == "ct-seg"
    assert report.label_encoding == "dense"
    with thunderpack.ThunderDB.open(str(tmp_path / "packed"), "r") as db:
        backend = SplitThunderDBStorageBackend(
            db,
            dataset_name="ClinicCT",
            modality="ct",
            split="train",
            subject_grouping="subject",
        )
        sample = CTTrainingDataset(
            backend=backend,
            dataset_name="ClinicCT",
            label_mode="native",
        )[0]
        torch.testing.assert_close(sample["image"], torch.as_tensor(image)[None])
        torch.testing.assert_close(
            sample["label"], torch.as_tensor(label, dtype=torch.long)[None]
        )
        np.testing.assert_allclose(sample["metadata"]["affine"], affine)
        np.testing.assert_allclose(sample["metadata"]["spacing"], spacing)


def test_manifest_validation_rejects_singular_ct_affine(tmp_path: Path) -> None:
    label = np.zeros((3, 4, 5), dtype=np.uint8)
    label[1, 2, 3] = 1
    singular_affine = np.eye(4, dtype=np.float32)
    singular_affine[2] = singular_affine[1]
    manifest = _write_ct_case_manifest(
        tmp_path,
        label=label,
        affine=singular_affine,
    )

    with pytest.raises(ValueError, match="affine must be invertible"):
        validate_dataset_manifest(manifest)


def test_pack_rejects_background_only_ct_without_creating_output(
    tmp_path: Path,
) -> None:
    manifest = _write_ct_case_manifest(
        tmp_path,
        label=np.zeros((3, 4, 5), dtype=np.uint8),
        affine=np.eye(4, dtype=np.float32),
    )
    output = tmp_path / "packed"

    with pytest.raises(ValueError, match="at least one foreground"):
        pack_dataset(manifest, output)

    assert not output.exists()


def test_manifest_validation_rejects_subject_split_leakage(tmp_path: Path) -> None:
    np.save(tmp_path / "image.npy", np.ones(_XRAY_SHAPE, dtype=np.float32))
    np.save(tmp_path / "label.npy", np.zeros(_XRAY_SHAPE, dtype=np.uint8))
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_name": "Leaky",
            "dataset_type": "xray-seg",
            "stored_labels": {0: "background"},
            "samples": [
                _sample(
                    "a",
                    "patient",
                    "train",
                    image="image.npy",
                    label="label.npy",
                ),
                _sample(
                    "b",
                    "patient",
                    "val",
                    image="image.npy",
                    label="label.npy",
                ),
            ],
        },
    )

    with pytest.raises(ValueError, match="subject-disjoint"):
        validate_dataset_manifest(manifest)


def test_manifest_validation_rejects_xray_spatial_shape_mismatch(
    tmp_path: Path,
) -> None:
    np.save(tmp_path / "image.npy", np.ones((64, 65), dtype=np.float32))
    np.save(tmp_path / "label.npy", np.zeros((65, 64), dtype=np.uint8))
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_name": "Mismatched",
            "dataset_type": "xray-seg",
            "stored_labels": {0: "background"},
            "samples": [
                _sample(
                    "sample",
                    "patient",
                    "train",
                    image="image.npy",
                    label="label.npy",
                )
            ],
        },
    )

    with pytest.raises(ValueError, match="spatial shapes differ"):
        validate_dataset_manifest(manifest)


def test_manifest_validation_requires_names_matching_channel_masks(
    tmp_path: Path,
) -> None:
    np.save(tmp_path / "image.npy", np.ones(_XRAY_SHAPE, dtype=np.float32))
    np.save(
        tmp_path / "label.npy",
        np.zeros((2, *_XRAY_SHAPE), dtype=np.uint8),
    )
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_name": "UnnamedChannels",
            "dataset_type": "xray-seg",
            "label_names": ["only_one"],
            "samples": [
                _sample(
                    "sample",
                    "patient",
                    "train",
                    image="image.npy",
                    label="label.npy",
                )
            ],
        },
    )

    with pytest.raises(ValueError, match="2 label channels but 1 label_names"):
        validate_dataset_manifest(manifest)


def test_pack_refuses_existing_output_without_mutating_it(tmp_path: Path) -> None:
    np.save(tmp_path / "image.npy", np.ones(_XRAY_SHAPE, dtype=np.float32))
    np.save(tmp_path / "label.npy", np.zeros(_XRAY_SHAPE, dtype=np.uint8))
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_name": "SafeOutput",
            "dataset_type": "xray-seg",
            "stored_labels": {0: "background"},
            "samples": [
                _sample(
                    "sample",
                    "patient",
                    "train",
                    image="image.npy",
                    label="label.npy",
                )
            ],
        },
    )
    output = tmp_path / "packed"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("unchanged", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        pack_dataset(manifest, output)

    assert sentinel.read_text(encoding="utf-8") == "unchanged"


def test_pack_rejects_output_containing_payload_files(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    output = tmp_path / "packed"
    output.mkdir()
    image_path = output / "image.npy"
    label_path = output / "label.npy"
    np.save(image_path, np.ones(_XRAY_SHAPE, dtype=np.float32))
    np.save(label_path, np.zeros(_XRAY_SHAPE, dtype=np.uint8))
    manifest = _write_manifest(
        source_root,
        {
            "schema_version": 1,
            "dataset_name": "ContainedPayloads",
            "dataset_type": "xray-seg",
            "stored_labels": {0: "background"},
            "samples": [
                _sample(
                    "sample",
                    "patient",
                    "train",
                    image=str(image_path),
                    label=str(label_path),
                )
            ],
        },
    )

    with pytest.raises(ValueError, match="contain.*payload") as exc_info:
        pack_dataset(manifest, output, overwrite=True)

    message = str(exc_info.value)
    assert str(image_path.resolve()) in message
    assert str(label_path.resolve()) in message
    np.testing.assert_array_equal(np.load(image_path), np.ones(_XRAY_SHAPE))
    np.testing.assert_array_equal(np.load(label_path), np.zeros(_XRAY_SHAPE))


def test_pack_overwrite_preserves_non_package_directory(tmp_path: Path) -> None:
    np.save(tmp_path / "image.npy", np.ones(_XRAY_SHAPE, dtype=np.float32))
    np.save(tmp_path / "label.npy", np.zeros(_XRAY_SHAPE, dtype=np.uint8))
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_name": "SafeOverwrite",
            "dataset_type": "xray-seg",
            "stored_labels": {0: "background"},
            "samples": [
                _sample(
                    "sample",
                    "patient",
                    "train",
                    image="image.npy",
                    label="label.npy",
                )
            ],
        },
    )
    output = tmp_path / "unrelated"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("do not replace", encoding="utf-8")

    with pytest.raises(ValueError, match="not a canonical FleXray package"):
        pack_dataset(manifest, output, overwrite=True)

    assert sentinel.read_text(encoding="utf-8") == "do not replace"
    assert list(output.iterdir()) == [sentinel]
    assert not list(tmp_path.glob(".unrelated.fxr-pack-*"))


def test_pack_overwrite_replaces_valid_canonical_package(tmp_path: Path) -> None:
    image_path = tmp_path / "image.npy"
    np.save(image_path, np.ones(_XRAY_SHAPE, dtype=np.float32))
    np.save(tmp_path / "label.npy", np.zeros(_XRAY_SHAPE, dtype=np.uint8))
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_name": "CanonicalOverwrite",
            "dataset_type": "xray-seg",
            "stored_labels": {0: "background"},
            "samples": [
                _sample(
                    "sample",
                    "patient",
                    "train",
                    image="image.npy",
                    label="label.npy",
                )
            ],
        },
    )
    output = tmp_path / "packed"
    pack_dataset(manifest, output)
    replacement = np.arange(
        np.prod(_XRAY_SHAPE), dtype=np.uint16
    ).reshape(_XRAY_SHAPE)
    np.save(image_path, replacement)

    report = pack_dataset(manifest, output, overwrite=True)

    assert report.path == output.resolve()
    with thunderpack.ThunderDB.open(str(output), "r") as db:
        np.testing.assert_array_equal(
            db["sample"]["img"],
            (replacement.astype(np.float32) / np.iinfo(replacement.dtype).max)[None],
        )
    assert validate_packed_dataset(output).dataset_name == "CanonicalOverwrite"
    assert not list(tmp_path.glob(".packed.fxr-backup-*"))
    assert not list(tmp_path.glob(".packed.fxr-pack-*"))


def test_dataset_cli_validates_packs_and_checks_real_database(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    np.save(tmp_path / "image.npy", np.ones(_XRAY_SHAPE, dtype=np.float32))
    np.save(tmp_path / "label.npy", np.zeros(_XRAY_SHAPE, dtype=np.uint8))
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_name": "CommandLineRay",
            "dataset_type": "xray-seg",
            "stored_labels": {0: "background"},
            "samples": [
                _sample(
                    "sample",
                    "patient",
                    "train",
                    image="image.npy",
                    label="label.npy",
                )
            ],
        },
    )
    output = tmp_path / "packed"

    assert dataset_main(["validate", str(manifest)]) == 0
    assert dataset_main(["pack", str(manifest), str(output)]) == 0
    assert dataset_main(["check", str(output)]) == 0

    stdout = capsys.readouterr().out
    assert stdout.count("CommandLineRay: xray-seg") == 3
    assert stdout.count("1 subjects, 1 samples (train=1)") == 3
    checked = validate_packed_dataset(output)
    assert checked.num_samples == 1
