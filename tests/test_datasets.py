from __future__ import annotations

import ast
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import fxr.datasets as public_datasets
import fxr.datasets.storage as dataset_storage
from fxr.datasets import (
    CTTrainingDataset,
    CompositeSourceDataset,
    DatasetRecord,
    HomogeneousSourceBatchSampler,
    ManifestStorageBackend,
    SplitThunderDBStorageBackend,
    MixedDataLoader,
    SequentialDataLoader,
    ThunderDBStorageBackend,
    XrayTrainingDataset,
    compile_training_label_remap_by_name,
)
from fxr.experiment import resolve_batch_inputs
from fxr.experiment._collation import _ct_safe_collate
from fxr.experiment.drr_forward import DrrForwardPipeline
from fxr.protocols import load_protocol_by_name

PROTOCOL_NAME = "all_structures_flexray_v4"
_UNSAFE_PAYLOAD_EFFECTS: list[str] = []


def _record_unsafe_payload_effect() -> None:
    """Record execution if an unsafe pickle reducer is invoked."""

    _UNSAFE_PAYLOAD_EFFECTS.append("executed")


class _UnsafeManifestPayload:
    """Pickle payload whose reducer would execute a visible side effect."""

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        """Return the side-effect function used only by the security test."""

        return (_record_unsafe_payload_effect, ())


class _CloseCountingDatabase(dict[str, object]):
    """Dictionary database recording owned-reader cleanup calls."""

    def __init__(self, values: dict[str, object]) -> None:
        """Copy ``values`` and initialize the close counter."""

        super().__init__(values)
        self.close_calls = 0

    def close(self) -> None:
        """Record one close request."""

        self.close_calls += 1


MOOSE_LABEL_NAMES = (
    "background",
    "skull",
    "clavicle_left",
    "clavicle_right",
    "carpal_left",
    "carpal_right",
    "rib_left_1",
    "rib_right_1",
    "rib_left_12",
    "rib_right_12",
    "hip_left",
    "hip_right",
    "kidney_left",
    "kidney_right",
    "lung_upper_lobe_left",
    "lung_upper_lobe_right",
    "heart_myocardium",
    "pancreas",
    "rib_left_13",
)


def _save(path: Path, array: np.ndarray) -> str:
    np.save(path, array)
    return path.name


def _manifest(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    import yaml

    path = tmp_path / "manifest.yml"
    path.write_text(yaml.safe_dump({"records": records}), encoding="utf-8")
    return path


def _ct_manifest(
    tmp_path: Path,
    *,
    image: np.ndarray,
    label: np.ndarray | None = None,
    affine: np.ndarray | None = None,
    spacing: np.ndarray | None = None,
    data_id: str = "ct_001",
) -> Path:
    image_key = _save(tmp_path / f"{data_id}_image.npy", image)
    affine_key = _save(
        tmp_path / f"{data_id}_affine.npy",
        np.eye(4, dtype=np.float32) if affine is None else affine,
    )
    spacing_key = _save(
        tmp_path / f"{data_id}_spacing.npy",
        np.ones(3, dtype=np.float32) if spacing is None else spacing,
    )
    record: dict[str, object] = {
        "dataset_name": "HipRay",
        "modality": "ct",
        "data_id": data_id,
        "subject_id": data_id,
        "image": image_key,
        "affine": affine_key,
        "spacing": spacing_key,
    }
    if label is not None:
        record["label"] = _save(tmp_path / f"{data_id}_label.npy", label)
    return _manifest(tmp_path, [record])


def _expected_ct_offset(
    spatial_shape: tuple[int, int, int],
    crop_size: tuple[int, int, int],
    *,
    seed: int,
) -> tuple[int, int, int]:
    generator = torch.Generator().manual_seed(seed)
    offsets = []
    for dim, size in zip(spatial_shape, crop_size, strict=True):
        if size <= dim:
            offsets.append(
                int(torch.randint(0, dim - size + 1, (), generator=generator))
            )
        else:
            offsets.append(
                -int(torch.randint(0, size - dim + 1, (), generator=generator))
            )
    return (offsets[0], offsets[1], offsets[2])


def _expected_crop_or_pad(
    tensor: torch.Tensor,
    crop_size: tuple[int, int, int],
    offset: tuple[int, int, int],
    *,
    fill_value: float | int,
) -> torch.Tensor:
    expected = tensor.new_full((*tensor.shape[:-3], *crop_size), fill_value)
    src_slices: list[slice] = [slice(None)] * (tensor.ndim - 3)
    dst_slices: list[slice] = [slice(None)] * (tensor.ndim - 3)
    for dim, size, axis_offset in zip(
        tensor.shape[-3:], crop_size, offset, strict=True
    ):
        src_start = max(axis_offset, 0)
        dst_start = max(-axis_offset, 0)
        length = min(dim - src_start, size - dst_start)
        src_slices.append(slice(src_start, src_start + length))
        dst_slices.append(slice(dst_start, dst_start + length))
    expected[tuple(dst_slices)] = tensor[tuple(src_slices)]
    return expected


def _expected_affine(affine: np.ndarray, offset: tuple[int, int, int]) -> torch.Tensor:
    expected = torch.as_tensor(affine, dtype=torch.float32).clone()
    offset_vector = torch.tensor([*offset, 1.0], dtype=torch.float32)
    expected[:3, 3] = (expected @ offset_vector)[:3]
    return expected


def test_public_dataset_api_exports_planned_names() -> None:
    assert {
        "CTTrainingDataset",
        "DatasetRecord",
        "ManifestStorageBackend",
        "SplitThunderDBStorageBackend",
        "ThunderDBStorageBackend",
        "TrainingLabelRemap",
        "XrayTrainingDataset",
        "compile_training_label_remap_by_name",
    }.issubset(set(public_datasets.__all__))
    assert "NormalizedDatasetConfig" not in public_datasets.__all__
    assert "normalize_dataset_configs" not in public_datasets.__all__
    assert not hasattr(public_datasets, "NormalizedDatasetConfig")
    assert not hasattr(public_datasets, "normalize_dataset_configs")


def test_compile_training_label_remap_by_name_uses_protocol_aliases() -> None:
    protocol = load_protocol_by_name(PROTOCOL_NAME)
    remap = compile_training_label_remap_by_name(PROTOCOL_NAME, "FluXray")

    assert remap.dataset_name == "MOOSE"
    assert remap.label_lut.dtype == torch.long
    assert remap.num_classes == len(protocol.labels)
    assert remap.label_lut[3].item() == protocol.label_to_id["clavicles"]
    assert remap.label_lut[100].item() == 0


def test_compile_training_label_remap_projects_model_subset_to_background() -> None:
    remap = compile_training_label_remap_by_name(
        PROTOCOL_NAME,
        "FluXray",
        model_label_names=("background", "clavicles"),
    )

    assert remap.num_classes == 2
    assert remap.label_lut[3].item() == 1
    assert remap.label_lut[105].item() == 0


def test_manifest_storage_loads_weights_only_tensor_containers(
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "payload.pt"
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    torch.save({"tensor": tensor, "metadata": ["safe", 1, True]}, payload_path)
    manifest = _manifest(
        tmp_path,
        [
            {
                "dataset_name": "SafePayload",
                "modality": "xray",
                "data_id": "sample",
                "image": payload_path.name,
            }
        ],
    )
    backend = ManifestStorageBackend(manifest)

    payload = backend.load(backend.records[0], "image")

    torch.testing.assert_close(payload["tensor"], tensor)
    assert payload["metadata"] == ["safe", 1, True]


def test_manifest_storage_rejects_pickle_execution_in_pth_payload(
    tmp_path: Path,
) -> None:
    _UNSAFE_PAYLOAD_EFFECTS.clear()
    payload_path = tmp_path / "payload.pth"
    torch.save(_UnsafeManifestPayload(), payload_path)
    manifest = _manifest(
        tmp_path,
        [
            {
                "dataset_name": "UnsafePayload",
                "modality": "xray",
                "data_id": "sample",
                "image": payload_path.name,
            }
        ],
    )
    backend = ManifestStorageBackend(manifest)

    with pytest.raises(ValueError, match="weights-only tensor contract"):
        backend.load(backend.records[0], "image")

    assert _UNSAFE_PAYLOAD_EFFECTS == []


def test_xray_training_dataset_native_and_model_modes(tmp_path: Path) -> None:
    image = _save(tmp_path / "img.npy", np.ones((2, 3), dtype=np.float32))
    label = _save(
        tmp_path / "label.npy", np.array([[0, 3, 100], [44, 0, 3]], dtype=np.int64)
    )
    manifest = _manifest(
        tmp_path,
        [
            {
                "dataset_name": "MOOSE",
                "modality": "xray",
                "data_id": "case_001",
                "subject_id": "case_001",
                "split": "train",
                "image": image,
                "label": label,
                "metadata": {"view": "ap"},
            }
        ],
    )
    backend = ManifestStorageBackend(manifest)

    native = XrayTrainingDataset(
        backend=backend,
        dataset_name="FluXray",
        label_mode="native",
        return_metadata=False,
        return_data_id=False,
    )
    native_sample = native[0]
    assert native_sample["image"].shape == (1, 2, 3)
    assert native_sample["label"].dtype == torch.long
    assert "metadata" not in native_sample
    assert "data_id" not in native_sample

    model = XrayTrainingDataset(
        backend=backend,
        protocol_name=PROTOCOL_NAME,
        dataset_name="FluXray",
        label_mode="model",
        return_native_label=True,
    )
    sample = model[0]
    protocol = load_protocol_by_name(PROTOCOL_NAME)
    assert sample["label"].shape == (len(protocol.labels), 2, 3)
    assert sample["label"][protocol.label_to_id["clavicles"], 0, 1].item() == 1.0
    assert sample["label"][0, 0, 2].item() == 1.0
    assert sample["native_label"].shape == (2, 3)
    assert sample["metadata"] == {"view": "ap"}


def test_xray_training_dataset_model_mode_normalizes_channel_aliases(
    tmp_path: Path,
) -> None:
    image = _save(tmp_path / "img.npy", np.ones((2, 2), dtype=np.float32))
    label_array = np.zeros((4, 2, 2), dtype=np.float32)
    label_array[0] = 1.0
    label_array[1, 0, 0] = 0.4
    label_array[2, 0, 0] = 0.9
    label_array[3, 1, 1] = 1.0
    label_array[0, 0, 0] = 0.0
    label_array[0, 1, 1] = 0.0
    label = _save(tmp_path / "label.npy", label_array)
    manifest = _manifest(
        tmp_path,
        [
            {
                "dataset_name": "HandBones",
                "modality": "xray",
                "data_id": "case_001",
                "image": image,
                "label": label,
                "label_names": [
                    "background",
                    "phalange_distal",
                    "phalange_intermediate",
                    "phalange_proximal",
                ],
            }
        ],
    )
    dataset = XrayTrainingDataset(
        backend=ManifestStorageBackend(manifest),
        protocol_name=PROTOCOL_NAME,
        dataset_name="HandBones",
        label_mode="model",
    )

    projected = dataset[0]["label"]
    phalanges_id = load_protocol_by_name(PROTOCOL_NAME).label_to_id["phalanges"]

    assert projected[phalanges_id, 0, 0].item() == pytest.approx(0.9)
    assert projected[phalanges_id, 1, 1].item() == pytest.approx(1.0)


def test_xray_training_dataset_filters_require_seg_skip_and_num_subjects(
    tmp_path: Path,
) -> None:
    image = _save(tmp_path / "img.npy", np.ones((2, 2), dtype=np.float32))
    label = _save(tmp_path / "label.npy", np.array([[0, 1], [2, 0]], dtype=np.int64))
    manifest = _manifest(
        tmp_path,
        [
            {
                "dataset_name": "HipRay",
                "modality": "xray",
                "data_id": "image_001",
                "subject_id": "image_001",
                "split": "train",
                "image": image,
                "label": label,
            },
            {
                "dataset_name": "HipRay",
                "modality": "xray",
                "data_id": "image_002",
                "subject_id": "image_002",
                "split": "train",
                "image": image,
                "label": label,
            },
            {
                "dataset_name": "HipRay",
                "modality": "xray",
                "data_id": "image_003",
                "subject_id": "image_003",
                "split": "train",
                "image": image,
            },
        ],
    )
    backend = ManifestStorageBackend(manifest)

    dataset = XrayTrainingDataset(
        backend=backend,
        protocol_name=PROTOCOL_NAME,
        dataset_name="HipRay",
        split="train",
        require_seg=True,
        num_subjects=1,
    )

    assert len(dataset) == 1
    assert dataset[0]["data_id"] == "image_002"


def test_ct_training_dataset_crop_none_returns_channel_first_and_affine(
    tmp_path: Path,
) -> None:
    image = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    label = np.zeros((3, 4, 5), dtype=np.int64)
    label[1, 2, 3] = 1
    affine = np.array(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, 20.0],
            [0.0, 0.0, 4.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    spacing = np.array([1.0, 1.5, 2.0], dtype=np.float32)
    manifest = _ct_manifest(
        tmp_path,
        image=image,
        label=label,
        affine=affine,
        spacing=spacing,
    )
    backend = ManifestStorageBackend(manifest)

    dataset = CTTrainingDataset(
        backend=backend,
        protocol_name=PROTOCOL_NAME,
        dataset_name="HipRay",
        crop_size=(1, 1, 1),
    )
    sample = dataset[0]

    assert sample["image"].shape == (1, 3, 4, 5)
    assert sample["label"].dtype == torch.long
    assert sample["label"].shape == (1, 3, 4, 5)
    torch.testing.assert_close(sample["image"], torch.as_tensor(image).unsqueeze(0))
    torch.testing.assert_close(
        torch.as_tensor(sample["metadata"]["affine"]), torch.as_tensor(affine)
    )
    torch.testing.assert_close(
        torch.as_tensor(sample["metadata"]["spacing"]), torch.as_tensor(spacing)
    )


def test_ct_training_dataset_random_crop_is_deterministic_and_remaps_dense_label(
    tmp_path: Path,
) -> None:
    image = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    label = (np.arange(4 * 5 * 6).reshape(4, 5, 6) % 3).astype(np.int64)
    affine = np.array(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, 20.0],
            [0.0, 0.0, 4.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    crop_size = (2, 3, 4)
    seed = 13
    offset = _expected_ct_offset((4, 5, 6), crop_size, seed=seed)
    manifest = _ct_manifest(tmp_path, image=image, label=label, affine=affine)

    dataset = CTTrainingDataset.from_manifest(
        manifest,
        protocol_name=PROTOCOL_NAME,
        dataset_name="HipRay",
        crop_mode="random",
        crop_size=crop_size,
        generator=torch.Generator().manual_seed(seed),
        return_native_label=True,
    )
    sample = dataset[0]

    source_image = torch.as_tensor(image).unsqueeze(0)
    source_label = torch.as_tensor(label).unsqueeze(0)
    expected_image = _expected_crop_or_pad(
        source_image, crop_size, offset, fill_value=-1000.0
    )
    expected_native = _expected_crop_or_pad(
        source_label, crop_size, offset, fill_value=0
    )
    protocol = load_protocol_by_name(PROTOCOL_NAME)
    expected_model = expected_native.clone()
    expected_model[expected_native == 1] = protocol.label_to_id["femurs"]
    expected_model[expected_native == 2] = protocol.label_to_id["hips"]

    torch.testing.assert_close(sample["image"], expected_image)
    torch.testing.assert_close(sample["native_label"], expected_native)
    torch.testing.assert_close(sample["label"], expected_model)
    torch.testing.assert_close(
        torch.as_tensor(sample["metadata"]["affine"]), _expected_affine(affine, offset)
    )


def test_ct_training_dataset_pads_with_air_background_and_negative_affine_offset(
    tmp_path: Path,
) -> None:
    image = np.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2)
    label = np.ones((2, 2, 2), dtype=np.int64)
    label[0, 0, 0] = 2
    affine = np.array(
        [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    crop_size = (4, 3, 5)
    seed = 0
    offset = _expected_ct_offset((2, 2, 2), crop_size, seed=seed)
    assert any(axis_offset < 0 for axis_offset in offset)
    manifest = _ct_manifest(tmp_path, image=image, label=label, affine=affine)

    dataset = CTTrainingDataset.from_manifest(
        manifest,
        dataset_name="HipRay",
        label_mode="native",
        crop_mode="random",
        crop_size=crop_size,
        generator=torch.Generator().manual_seed(seed),
    )
    sample = dataset[0]

    source_image = torch.as_tensor(image).unsqueeze(0)
    source_label = torch.as_tensor(label).unsqueeze(0)
    expected_image = _expected_crop_or_pad(
        source_image, crop_size, offset, fill_value=-1000.0
    )
    expected_label = _expected_crop_or_pad(
        source_label, crop_size, offset, fill_value=0
    )
    torch.testing.assert_close(sample["image"], expected_image)
    torch.testing.assert_close(sample["label"], expected_label)
    assert torch.any(sample["image"] == -1000.0)
    assert torch.any(sample["label"] == 0)
    torch.testing.assert_close(
        torch.as_tensor(sample["metadata"]["affine"]), _expected_affine(affine, offset)
    )


def test_ct_training_dataset_random_slices_uses_xyz_crop_size(tmp_path: Path) -> None:
    image = np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)
    label = (np.arange(5 * 6 * 7).reshape(5, 6, 7) % 3).astype(np.int64)
    crop_size = (3, 4, 2)
    seed = 8
    offset = _expected_ct_offset((5, 6, 7), crop_size, seed=seed)
    manifest = _ct_manifest(tmp_path, image=image, label=label)

    dataset = CTTrainingDataset.from_manifest(
        manifest,
        dataset_name="HipRay",
        label_mode="native",
        crop_mode="random_slices",
        crop_size=crop_size,
        generator=torch.Generator().manual_seed(seed),
    )
    sample = dataset[0]

    expected_image = _expected_crop_or_pad(
        torch.as_tensor(image).unsqueeze(0),
        crop_size,
        offset,
        fill_value=-1000.0,
    )
    assert sample["image"].shape == (1, 3, 4, 2)
    assert sample["label"].shape == (1, 3, 4, 2)
    torch.testing.assert_close(sample["image"], expected_image)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"crop_mode": "random"}, "requires crop_size"),
        ({"crop_mode": "middle", "crop_size": (1, 1, 1)}, "crop_mode"),
        ({"crop_mode": "random", "crop_size": 1}, "exactly 3"),
        ({"crop_mode": "random", "crop_size": (1, 1)}, "exactly 3"),
        ({"crop_mode": "random", "crop_size": (1, 0, 1)}, "positive integers"),
    ],
)
def test_ct_training_dataset_validates_crop_config(
    tmp_path: Path,
    kwargs: dict[str, object],
    match: str,
) -> None:
    manifest = _ct_manifest(
        tmp_path,
        image=np.zeros((2, 2, 2), dtype=np.float32),
        label=np.zeros((2, 2, 2), dtype=np.int64),
    )

    with pytest.raises(ValueError, match=match):
        CTTrainingDataset.from_manifest(
            manifest, dataset_name="HipRay", label_mode="native", **kwargs
        )


def test_ct_training_dataset_hu_min_and_air_clamp(tmp_path: Path) -> None:
    image = np.array(
        [[[-1200.0, -950.0], [-800.0, 0.0]], [[-901.0, -900.0], [-899.0, 10.0]]],
        dtype=np.float32,
    )
    label = np.zeros((2, 2, 2), dtype=np.int64)
    manifest = _ct_manifest(tmp_path, image=image, label=label)

    dataset = CTTrainingDataset.from_manifest(
        manifest,
        dataset_name="HipRay",
        label_mode="native",
        hu_min=-500.0,
        air_clamp_hu=-900.0,
    )
    sample = dataset[0]

    expected = torch.clamp_min(torch.as_tensor(image).unsqueeze(0), -500.0)
    expected = expected.masked_fill(
        torch.as_tensor(image).unsqueeze(0) <= -900.0, -1000.0
    )
    torch.testing.assert_close(sample["image"], expected)


def test_ct_training_dataset_foreground_centroids_use_returned_dense_label(
    tmp_path: Path,
) -> None:
    image = np.zeros((3, 4, 5), dtype=np.float32)
    label = np.zeros((3, 4, 5), dtype=np.int64)
    label[0, 0, 0] = 1
    label[2, 0, 0] = 1
    label[1, 2, 3] = 2
    manifest = _ct_manifest(tmp_path, image=image, label=label)

    dataset = CTTrainingDataset.from_manifest(
        manifest,
        protocol_name=PROTOCOL_NAME,
        dataset_name="HipRay",
        compute_fg_centroids=True,
    )
    sample = dataset[0]

    protocol = load_protocol_by_name(PROTOCOL_NAME)
    assert sample["metadata"]["fg_centroids_ijk"] == {
        protocol.label_to_id["femurs"]: (1.0, 0.0, 0.0),
        protocol.label_to_id["hips"]: (1.0, 2.0, 3.0),
    }


def test_ct_dataset_batch_resolves_random_label_drr_with_free_metadata() -> None:
    image = np.linspace(-1000.0, 500.0, 8**3, dtype=np.float32).reshape(8, 8, 8)
    label = np.zeros((8, 8, 8), dtype=np.int64)
    label[1, 2, 3] = 2
    label[6, 5, 4] = 1
    backend = ManifestStorageBackend(
        {
            "payloads": {
                "image": image,
                "label": label,
                "affine": np.eye(4, dtype=np.float32),
                "spacing": np.array([1.0, 1.5, 2.0], dtype=np.float32),
            },
            "records": [
                {
                    "dataset_name": "HipRay",
                    "modality": "ct",
                    "data_id": "ct_runtime",
                    "subject_id": "subject_runtime",
                    "image": "image",
                    "label": "label",
                    "affine": "affine",
                    "spacing": "spacing",
                    "metadata": {
                        "optional_note": None,
                        "preprocessing": {"variable": [1, "two", None]},
                    },
                }
            ],
        }
    )
    dataset = CTTrainingDataset(
        backend=backend,
        dataset_name="HipRay",
        label_mode="native",
        compute_fg_centroids=True,
    )

    batch = next(
        iter(DataLoader(dataset, batch_size=1, collate_fn=_ct_safe_collate))
    )
    inputs = resolve_batch_inputs(batch, modality="ct")

    assert batch["metadata"]["optional_note"] is None
    assert batch["metadata"]["preprocessing"] == {"variable": [1, "two", None]}
    assert batch["metadata"]["spacing"].shape == (1, 3)
    torch.testing.assert_close(
        inputs.fg_centroids_ijk,
        torch.tensor([[6.0, 5.0, 4.0], [1.0, 2.0, 3.0]]),
    )

    config = {
        "protocol": {"name": PROTOCOL_NAME},
        "data": {"CT": {"HipRay": {}}},
        "dataloader": {"batch_size": 2},
        "drr_model": {
            "default": {
                "preset": "frontal",
                "num_views": 2,
                "num_samples": 16,
                "intrinsics_cfg": {
                    "height": 8,
                    "width": 8,
                    "sdd": 1000.0,
                    "delx": 2.0,
                },
                "isocenter_cfg": {
                    "sample_scheme": "random_label",
                    "replacement": False,
                },
                "seg_cfg": {"soft_labels": False, "threshold": 0.0},
            },
            "datasets": {"HipRay": {}},
        },
    }
    pipeline = DrrForwardPipeline.from_config(config, device=torch.device("cpu"))
    assert pipeline is not None

    rendered = pipeline.render(
        volume=inputs.image,
        label=inputs.label,
        affine=inputs.affine,
        dataset_name="HipRay",
        fg_centroids_ijk=inputs.fg_centroids_ijk,
    )

    assert rendered.images.shape == (2, 1, 8, 8)
    assert rendered.labels.shape[0] == 2



def test_ct_collator_copies_read_only_geometry_metadata() -> None:
    affine = np.eye(4, dtype=np.float32)
    spacing = np.array([1.0, 1.5, 2.0], dtype=np.float32)
    centroids = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    arrays = (affine, spacing, centroids)
    for array in arrays:
        array.setflags(write=False)
    sample = {
        "image": torch.zeros((1, 2, 3, 4)),
        "label": torch.zeros((1, 2, 3, 4), dtype=torch.long),
        "metadata": {
            "affine": affine,
            "spacing": spacing,
            "fg_centroids_ijk": centroids,
        },
    }

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        batch = _ct_safe_collate([sample])

    metadata = batch["metadata"]
    assert not any("not writable" in str(item.message) for item in captured)
    torch.testing.assert_close(
        metadata["affine"], torch.from_numpy(affine.copy())[None]
    )
    torch.testing.assert_close(
        metadata["spacing"], torch.from_numpy(spacing.copy())[None]
    )
    torch.testing.assert_close(
        metadata["fg_centroids_ijk"], torch.from_numpy(centroids.copy())[None]
    )
    for tensor, source in zip(
        (
            metadata["affine"],
            metadata["spacing"],
            metadata["fg_centroids_ijk"],
        ),
        arrays,
        strict=True,
    ):
        assert tensor.data_ptr() != source.ctypes.data


def test_ct_collator_sorts_foreground_centroids_by_label_id() -> None:
    sample = {
        "image": torch.zeros(1, 2, 2, 2),
        "label": torch.zeros(1, 2, 2, 2, dtype=torch.long),
        "dataset_name": "CT",
        "modality": "ct",
        "metadata": {
            "affine": torch.eye(4),
            "spacing": torch.ones(3),
            "fg_centroids_ijk": {
                9: (9.0, 8.0, 7.0),
                2: (2.0, 3.0, 4.0),
            },
        },
    }

    batch = _ct_safe_collate([sample])

    torch.testing.assert_close(
        batch["metadata"]["fg_centroids_ijk"],
        torch.tensor([[[2.0, 3.0, 4.0], [9.0, 8.0, 7.0]]]),
    )


def test_ct_training_dataset_foreground_centroids_reject_background_only(
    tmp_path: Path,
) -> None:
    manifest = _ct_manifest(
        tmp_path,
        image=np.zeros((2, 2, 2), dtype=np.float32),
        label=np.zeros((2, 2, 2), dtype=np.int64),
    )
    dataset = CTTrainingDataset.from_manifest(
        manifest,
        dataset_name="HipRay",
        label_mode="native",
        compute_fg_centroids=True,
    )

    with pytest.raises(ValueError, match="background-only"):
        dataset[0]


def test_ct_training_dataset_crops_image_without_label_when_seg_not_required(
    tmp_path: Path,
) -> None:
    image = np.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2)
    affine = np.eye(4, dtype=np.float32)
    crop_size = (3, 3, 3)
    seed = 1
    offset = _expected_ct_offset((2, 2, 2), crop_size, seed=seed)
    manifest = _ct_manifest(tmp_path, image=image, affine=affine)

    dataset = CTTrainingDataset.from_manifest(
        manifest,
        dataset_name="HipRay",
        label_mode="native",
        require_seg=False,
        compute_fg_centroids=True,
        crop_mode="random",
        crop_size=crop_size,
        generator=torch.Generator().manual_seed(seed),
    )
    sample = dataset[0]

    assert "label" not in sample
    assert "native_label" not in sample
    assert "fg_centroids_ijk" not in sample["metadata"]
    torch.testing.assert_close(
        sample["image"],
        _expected_crop_or_pad(
            torch.as_tensor(image).unsqueeze(0),
            crop_size,
            offset,
            fill_value=-1000.0,
        ),
    )
    torch.testing.assert_close(
        torch.as_tensor(sample["metadata"]["affine"]), _expected_affine(affine, offset)
    )


def test_ct_training_dataset_requires_affine_and_spacing(tmp_path: Path) -> None:
    image = _save(tmp_path / "ct.npy", np.ones((2, 2, 2), dtype=np.float32))
    label = _save(tmp_path / "ct_label.npy", np.zeros((2, 2, 2), dtype=np.int64))
    manifest = _manifest(
        tmp_path,
        [
            {
                "dataset_name": "HipRay",
                "modality": "ct",
                "data_id": "ct_001",
                "image": image,
                "label": label,
            }
        ],
    )

    with pytest.raises(ValueError, match="affine and spacing"):
        CTTrainingDataset.from_manifest(
            manifest,
            protocol_name=PROTOCOL_NAME,
            dataset_name="HipRay",
        )


def test_xray_training_dataset_projects_channel_masks(tmp_path: Path) -> None:
    image = _save(tmp_path / "gen_img.npy", np.ones((1, 2, 2), dtype=np.float32))
    label = _save(
        tmp_path / "gen_label.npy",
        np.array(
            [
                [[1, 0], [0, 1]],
                [[0, 1], [0, 0]],
                [[0, 0], [1, 0]],
            ],
            dtype=np.float32,
        ),
    )
    manifest = _manifest(
        tmp_path,
        [
            {
                "dataset_name": "MOOSE",
                "modality": "xray",
                "data_id": "gen_001",
                "image": image,
                "label": label,
                "label_names": ["background", "clavicles", "lungs"],
            }
        ],
    )
    dataset = XrayTrainingDataset.from_manifest(
        manifest,
        protocol_name=PROTOCOL_NAME,
        dataset_name="MOOSE",
        return_native_label=True,
    )
    sample = dataset[0]
    protocol = load_protocol_by_name(PROTOCOL_NAME)

    assert sample["native_label"].shape == (3, 2, 2)
    assert sample["label"].shape == (len(protocol.labels), 2, 2)
    assert sample["label"][protocol.label_to_id["clavicles"], 0, 1].item() == 1.0
    assert sample["label"][protocol.label_to_id["lungs"], 1, 0].item() == 1.0


def test_xray_training_dataset_validates_channel_mask_metadata(tmp_path: Path) -> None:
    image = _save(tmp_path / "gen_img.npy", np.ones((1, 2, 2), dtype=np.float32))
    label = _save(tmp_path / "gen_label.npy", np.ones((2, 2, 2), dtype=np.float32))
    manifest = _manifest(
        tmp_path,
        [
            {
                "dataset_name": "MOOSE",
                "modality": "xray",
                "data_id": "gen_001",
                "image": image,
                "label": label,
                "label_names": ["background"],
            }
        ],
    )
    dataset = XrayTrainingDataset.from_manifest(
        manifest,
        protocol_name=PROTOCOL_NAME,
        dataset_name="MOOSE",
    )

    with pytest.raises(ValueError, match="label channels"):
        dataset[0]


def test_thunderdb_path_constructor_closes_reader_when_record_read_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _CloseCountingDatabase({})
    monkeypatch.setattr(dataset_storage, "_open_thunderdb", lambda path: database)

    with pytest.raises(ValueError, match="requires records"):
        ThunderDBStorageBackend("/fake/database")

    assert database.close_calls == 1


def test_split_thunderdb_path_constructor_closes_reader_when_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _CloseCountingDatabase({})
    monkeypatch.setattr(dataset_storage, "_open_thunderdb", lambda path: database)

    with pytest.raises(KeyError, match="_attrs"):
        SplitThunderDBStorageBackend(
            "/fake/database",
            dataset_name="Broken",
            modality="xray",
            split="train",
        )

    assert database.close_calls == 1


def test_thunderdb_storage_adapter_accepts_dict_like_database() -> None:
    db = {
        "records": [
            {
                "dataset_name": "HipRay",
                "modality": "xray",
                "data_id": "case_001",
                "image": "img",
                "label": "label",
            }
        ],
        "img": np.ones((2, 2), dtype=np.float32),
        "label": np.zeros((2, 2), dtype=np.int64),
    }
    backend = ThunderDBStorageBackend(db)
    record = backend.records[0]

    assert isinstance(record, DatasetRecord)
    assert backend.load(record, "image").shape == (2, 2)


def test_split_thunderdb_backend_materializes_xray_records_from_splits() -> None:
    db = {
        "_attrs": {
            "dataset_name": "HipRay",
            "clean_dataset": True,
            "payload_keys": {"image": "pixels", "label": "mask"},
        },
        "_splits": {"train": ["case_001"]},
        "_metadata": {"case_001": {"view": "ap", "subject_id": "subject_a"}},
        "case_001": {
            "pixels": np.ones((2, 3), dtype=np.float32),
            "mask": np.array([[0, 1, 2], [0, 2, 1]], dtype=np.int64),
        },
    }
    backend = SplitThunderDBStorageBackend(
        db,
        dataset_name="HipRay",
        modality="xray",
        split="train",
        subject_grouping="subject",
    )
    record = backend.records[0]

    assert record.data_id == "case_001"
    assert record.subject_id == "subject_a"
    assert record.metadata == {"view": "ap", "subject_id": "subject_a"}
    assert record.image not in db
    dataset = XrayTrainingDataset(
        backend=backend,
        dataset_name="HipRay",
        label_mode="native",
    )
    sample = dataset[0]
    assert sample["image"].shape == (1, 2, 3)
    torch.testing.assert_close(sample["label"], torch.as_tensor(db["case_001"]["mask"]))


def test_split_thunderdb_backend_materializes_ct_records_from_splits() -> None:
    affine = np.array(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, 20.0],
            [0.0, 0.0, 4.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    spacing = np.array([1.0, 1.5, 2.0], dtype=np.float32)
    db = {
        "_attrs": {
            "dataset_name": "HipRay",
            "clean_dataset": True,
            "payload_fields": {
                "image": "ct",
                "label": "mask",
                "affine": "vox2world",
                "spacing": "spacing_mm",
            }
        },
        "_splits": {"train": ["ct_001"]},
        "_metadata": {"ct_001": {"scanner": "fake"}},
        "ct_001": {
            "ct": np.ones((2, 3, 4), dtype=np.float32),
            "mask": np.zeros((2, 3, 4), dtype=np.int64),
            "vox2world": affine,
            "spacing_mm": spacing,
        },
    }
    backend = SplitThunderDBStorageBackend(
        db,
        dataset_name="HipRay",
        modality="ct",
        split="train",
    )
    record = backend.records[0]

    assert record.affine is not None
    assert record.spacing is not None
    dataset = CTTrainingDataset(
        backend=backend,
        dataset_name="HipRay",
        label_mode="native",
    )
    sample = dataset[0]
    assert sample["image"].shape == (1, 2, 3, 4)
    assert sample["metadata"]["scanner"] == "fake"
    torch.testing.assert_close(
        torch.as_tensor(sample["metadata"]["affine"]), torch.as_tensor(affine)
    )
    torch.testing.assert_close(
        torch.as_tensor(sample["metadata"]["spacing"]), torch.as_tensor(spacing)
    )


def test_ct_dataset_copies_non_writable_numpy_payloads_without_warning() -> None:
    image = np.ones((2, 3, 4), dtype=np.float32)
    label = np.zeros((2, 3, 4), dtype=np.int64)
    image.setflags(write=False)
    label.setflags(write=False)
    db = {
        "_attrs": {"dataset_name": "HipRay", "clean_dataset": True},
        "_splits": {"train": ["ct_001"]},
        "ct_001": {
            "img": image,
            "seg": label,
            "affine": np.eye(4, dtype=np.float32),
            "spacing": np.ones(3, dtype=np.float32),
        },
    }
    backend = SplitThunderDBStorageBackend(
        db, dataset_name="HipRay", modality="ct", split="train"
    )
    dataset = CTTrainingDataset(
        backend=backend, dataset_name="HipRay", label_mode="native"
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        sample = dataset[0]

    assert sample["image"].shape == (1, 2, 3, 4)
    assert not any("not writable" in str(warning.message) for warning in captured)


class _CountingSampleDB(dict):
    """Dict test double that counts sample payload record reads."""

    def __init__(self, *args, sample_key: str, **kwargs) -> None:
        """Initialize the mapping and tracked sample key."""
        super().__init__(*args, **kwargs)
        self.sample_key = sample_key
        self.sample_reads = 0

    def get(self, key, default=None):
        """Return a value while counting tracked sample reads."""
        if key == self.sample_key:
            self.sample_reads += 1
        return super().get(key, default)

    def __getitem__(self, key):
        """Return a value while counting tracked sample reads."""
        if key == self.sample_key:
            self.sample_reads += 1
        return super().__getitem__(key)


def test_split_thunderdb_backend_lazy_payload_refs_defer_ct_sample_reads() -> None:
    db = _CountingSampleDB(
        {
            "_attrs": {"dataset_name": "HipRay", "clean_dataset": True},
            "_splits": {"train": ["ct_001"]},
            "_metadata": {"ct_001": {"scanner": "fake"}},
            "ct_001": {
                "img": np.ones((2, 3, 4), dtype=np.float32),
                "seg": np.zeros((2, 3, 4), dtype=np.int64),
                "affine": np.eye(4, dtype=np.float32),
                "spacing": np.ones(3, dtype=np.float32),
            },
        },
        sample_key="ct_001",
    )

    backend = SplitThunderDBStorageBackend(
        db,
        dataset_name="HipRay",
        modality="ct",
        split="train",
        lazy_payload_refs=True,
    )

    assert db.sample_reads == 0
    record = backend.records[0]
    assert record.image is not None
    assert record.label is not None
    assert record.affine is not None
    assert record.spacing is not None

    _ = backend.load(record, "image")
    assert db.sample_reads == 1


def test_fluxray_thunderdb_backend_projects_native_channel_names() -> None:
    label = np.zeros((len(MOOSE_LABEL_NAMES), 2, 2), dtype=np.float32)
    label[2, 0, 0] = 1.0
    label[3, 0, 1] = 1.0
    label[14, 1, 0] = 1.0
    db = {
        "_attrs": {
            "dataset_name": "FluXray",
            "clean_dataset": True,
            "protocol_name": PROTOCOL_NAME,
            "label_names": list(MOOSE_LABEL_NAMES),
        },
        "_splits": {"train": ["gen_001"]},
        "_metadata": {"gen_001": {"view": "pa"}},
        "gen_001": {
            "img": np.ones((1, 2, 2), dtype=np.float32),
            "seg": label,
        },
    }
    backend = SplitThunderDBStorageBackend(
        db, dataset_name="FluXray", modality="xray", split="train"
    )
    record = backend.records[0]

    assert record.dataset_name == "FluXray"
    assert record.modality == "xray"
    assert record.label_names == MOOSE_LABEL_NAMES
    dataset = XrayTrainingDataset(
        backend=backend,
        protocol_name=PROTOCOL_NAME,
        dataset_name="FluXray",
        return_native_label=True,
    )
    sample = dataset[0]
    protocol = load_protocol_by_name(PROTOCOL_NAME)

    assert sample["native_label"].shape == (len(MOOSE_LABEL_NAMES), 2, 2)
    assert sample["metadata"] == {"view": "pa"}
    assert sample["label"][protocol.label_to_id["clavicles"], 0, 0].item() == 1.0
    assert sample["label"][protocol.label_to_id["clavicles"], 0, 1].item() == 1.0
    assert sample["label"][protocol.label_to_id["lungs"], 1, 0].item() == 1.0


@pytest.mark.parametrize(
    ("attrs", "match"),
    [
        ({"dataset_name": "OtherRay", "clean_dataset": True}, "dataset identity"),
        ({"dataset": "OtherRay"}, "dataset identity"),
        ({"dataset_name": "HipRay", "clean_dataset": False}, "clean_dataset"),
    ],
)
def test_split_thunderdb_backend_validates_dataset_identity_metadata(
    attrs: dict[str, object], match: str
) -> None:
    db = {
        "_attrs": attrs,
        "_splits": {"train": ["case_001"]},
        "case_001": {"img": np.ones((2, 2), dtype=np.float32)},
    }

    with pytest.raises(ValueError, match=match):
        SplitThunderDBStorageBackend(
            db, dataset_name="HipRay", modality="xray", split="train"
        )


def test_split_thunderdb_backend_accepts_legacy_dataset_attr() -> None:
    db = {
        "_attrs": {"dataset": "HipRay", "img_key": "img", "seg_key": "seg"},
        "_splits": {"train": ["case_001"]},
        "case_001": {
            "img": np.ones((2, 2), dtype=np.float32),
            "seg": np.zeros((2, 2), dtype=np.uint8),
        },
    }

    backend = SplitThunderDBStorageBackend(
        db, dataset_name="HipRay", modality="xray", split="train"
    )

    assert backend.records[0].dataset_name == "HipRay"
    assert backend.load(backend.records[0], "image").shape == (2, 2)


def test_split_thunderdb_backend_validates_missing_split_and_image_key() -> None:
    with pytest.raises(KeyError, match="does not define split"):
        SplitThunderDBStorageBackend(
            {
                "_attrs": {"dataset_name": "HipRay", "clean_dataset": True},
                "_splits": {"train": ["case_001"]},
                "case_001": {"img": np.ones((2, 2), dtype=np.float32)},
            },
            dataset_name="HipRay",
            modality="xray",
            split="val",
        )

    with pytest.raises(KeyError, match="image payload field"):
        SplitThunderDBStorageBackend(
            {
                "_attrs": {"dataset_name": "HipRay", "clean_dataset": True},
                "_splits": {"train": ["case_001"]},
                "case_001": {"seg": np.zeros((2, 2), dtype=np.int64)},
            },
            dataset_name="HipRay",
            modality="xray",
            split="train",
        )


def test_split_thunderdb_backend_rejects_unlabeled_require_seg_selection() -> None:
    backend = SplitThunderDBStorageBackend(
        {
            "_attrs": {"dataset_name": "HipRay", "clean_dataset": True},
            "_splits": {"train": ["case_001"]},
            "case_001": {"img": np.ones((2, 2), dtype=np.float32)},
        },
        dataset_name="HipRay",
        modality="xray",
        split="train",
    )

    with pytest.raises(ValueError, match="require_seg=True"):
        XrayTrainingDataset(
            backend=backend,
            dataset_name="HipRay",
            label_mode="native",
            require_seg=True,
        )


@pytest.mark.parametrize(
    ("label_names", "error", "match"),
    [
        ("background", TypeError, "list of strings"),
        (["background", ""], ValueError, "non-empty string"),
    ],
)
def test_fluxray_thunderdb_backend_validates_attrs_label_names(
    label_names: object,
    error: type[Exception],
    match: str,
) -> None:
    db = {
        "_attrs": {
            "dataset_name": "FluXray",
            "clean_dataset": True,
            "label_names": label_names,
        },
        "_splits": {"train": ["gen_001"]},
        "gen_001": {
            "img": np.ones((1, 2, 2), dtype=np.float32),
            "seg": np.ones((1, 2, 2), dtype=np.float32),
        },
    }

    with pytest.raises(error, match=match):
        SplitThunderDBStorageBackend(
            db, dataset_name="FluXray", modality="xray", split="train"
        )


class _TinyDataset(Dataset):
    """Small dataset used to exercise source-aware composition helpers.

    Attributes:
        count: Number of samples exposed by the dataset.
        modality: Modality value emitted in every sample.
    """

    def __init__(self, count: int, modality: str) -> None:
        """Initialize the fixed-size tiny dataset.

        Args:
            count: Number of samples to expose.
            modality: Modality value to include in each sample.

        Returns:
            ``None``.
        """

        self.count = count
        self.modality = modality

    def __len__(self) -> int:
        """Return the configured sample count.

        Returns:
            Number of samples in the tiny dataset.
        """

        return self.count

    def __getitem__(self, index: int) -> dict[str, object]:
        """Return one synthetic sample.

        Args:
            index: Sample index to encode in the image tensor.

        Returns:
            Sample dictionary containing an image tensor and modality value.
        """

        return {"image": torch.tensor([index]), "modality": self.modality}


def test_composite_dataset_and_homogeneous_batch_sampler() -> None:
    composite = CompositeSourceDataset(
        {"xray_source": _TinyDataset(3, "xray"), "ct_source": _TinyDataset(2, "ct")},
        modalities={"xray_source": "xray", "ct_source": "ct"},
    )
    sampler = HomogeneousSourceBatchSampler(composite, batch_size=2)

    assert len(composite) == 5
    assert composite[0]["dataset_name"] == "xray_source"
    assert composite[3]["dataset_name"] == "ct_source"
    assert list(sampler) == [[0, 1], [2], [3, 4]]


def test_mixed_and_sequential_dataloaders_emit_named_batches() -> None:
    xray_loader = DataLoader(_TinyDataset(2, "xray"), batch_size=1)
    ct_loader = DataLoader(_TinyDataset(1, "ct"), batch_size=1)

    mixed = list(
        MixedDataLoader(
            {"xray": xray_loader, "ct": ct_loader},
            modalities={"xray": "xray", "ct": "ct"},
            proportions={"xray": 2, "ct": 1},
        )
    )
    sequential = list(
        SequentialDataLoader(
            {
                "xray": DataLoader(_TinyDataset(1, "xray"), batch_size=1),
                "ct": ct_loader,
            },
            modalities={"xray": "xray", "ct": "ct"},
        )
    )

    assert sorted(batch.source_name for batch in mixed) == ["ct", "xray", "xray"]
    assert all(batch.modality == batch.source_name for batch in mixed)
    assert [batch.source_name for batch in sequential] == ["xray", "ct"]


def test_mixed_dataloader_fixed_epoch_cycles_and_preserves_weights() -> None:
    xray_loader = DataLoader(_TinyDataset(2, "xray"), batch_size=1)
    ct_loader = DataLoader(_TinyDataset(1, "ct"), batch_size=1)
    weights = {"xray": 2, "ct": 1}

    mixed = MixedDataLoader(
        {"xray": xray_loader, "ct": ct_loader},
        modalities={"xray": "xray", "ct": "ct"},
        proportions=weights,
        iters_per_epoch=7,
        seed=5,
    )
    batches = list(mixed)

    source_names = [batch.source_name for batch in batches]
    counts = {name: source_names.count(name) for name in weights}

    assert len(mixed) == len(batches) == 7
    assert sum(counts.values()) == len(mixed)
    for name, weight in weights.items():
        ideal_count = len(mixed) * weight / sum(weights.values())
        assert abs(counts[name] - ideal_count) < 1
    xray_values = [
        int(batch.batch["image"].item())
        for batch in batches
        if batch.source_name == "xray"
    ]
    assert len(xray_values) > len(xray_loader)
    assert set(xray_values) == {0, 1}


def test_mixed_dataloader_apportions_fractional_weights_to_epoch_length() -> None:
    loader = DataLoader(_TinyDataset(1, "xray"), batch_size=1)
    weights = {
        "MOOSE": 0.4,
        "FluXray": 0.4,
        "PedsCT": 0.05,
        "ElbowCT": 0.025,
    }
    loaders = {
        "MOOSE": loader,
        "FluXray": loader,
        "PedsCT": loader,
        "ElbowCT": loader,
    }

    mixed = MixedDataLoader(
        loaders,
        proportions=weights,
        iters_per_epoch=37,
        seed=11,
    )
    source_names = [batch.source_name for batch in mixed]
    total_weight = sum(weights.values())

    assert len(source_names) == len(mixed) == 37
    for name, weight in weights.items():
        ideal_count = len(mixed) * weight / total_weight
        assert abs(source_names.count(name) - ideal_count) < 1


def test_mixed_dataloader_rejects_invalid_options() -> None:
    loader = DataLoader(_TinyDataset(1, "xray"), batch_size=1)
    with pytest.raises(ValueError, match="positive sampling weight"):
        MixedDataLoader({"xray": loader}, proportions={"xray": 0})
    with pytest.raises(ValueError, match=">= 0"):
        MixedDataLoader({"xray": loader}, proportions={"xray": -1})
    with pytest.raises(ValueError, match="positive integer"):
        MixedDataLoader({"xray": loader}, iters_per_epoch=2.5)
    with pytest.raises(ValueError, match="seed must be an integer"):
        MixedDataLoader({"xray": loader}, seed=True)


def test_mixed_dataloader_rejects_empty_sources_clearly() -> None:
    with pytest.raises(ValueError, match="source 'empty' produced no batches"):
        MixedDataLoader({"empty": []})

    lazy_empty = (value for value in ())
    mixed = MixedDataLoader({"empty": lazy_empty}, iters_per_epoch=1)
    with pytest.raises(ValueError, match="source 'empty' produced no batches"):
        list(mixed)


def test_mixed_dataloader_seeded_epochs_are_reproducible_and_interleaved() -> None:
    loaders = {
        "major": ["major"],
        "middle": ["middle"],
        "rare": ["rare"],
    }
    weights = {"major": 4, "middle": 3, "rare": 1}
    epoch_length = 31

    first = MixedDataLoader(
        loaders,
        proportions=weights,
        iters_per_epoch=epoch_length,
        seed=17,
    )
    replay = MixedDataLoader(
        dict(reversed(tuple(loaders.items()))),
        proportions=weights,
        iters_per_epoch=epoch_length,
        seed=17,
    )

    first_epochs = [tuple(batch.source_name for batch in first) for _ in range(2)]
    replay_epochs = [tuple(batch.source_name for batch in replay) for _ in range(2)]

    assert first_epochs == replay_epochs
    assert first_epochs[0] != first_epochs[1]
    for source_names in first_epochs:
        transitions = sum(
            left != right
            for left, right in zip(source_names, source_names[1:])
        )
        assert transitions > len(loaders) - 1

def test_mixed_dataloader_set_epoch_replays_requested_schedule() -> None:
    loaders = {"major": ["major"], "rare": ["rare"]}
    options = {
        "proportions": {"major": 3, "rare": 1},
        "iters_per_epoch": 19,
        "seed": 23,
    }
    uninterrupted = MixedDataLoader(loaders, **options)
    _ = list(uninterrupted)
    expected = tuple(batch.source_name for batch in uninterrupted)

    resumed = MixedDataLoader(loaders, **options)
    resumed.set_epoch(1)
    actual = tuple(batch.source_name for batch in resumed)

    assert actual == expected
    with pytest.raises(ValueError, match="non-negative integer"):
        resumed.set_epoch(-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        resumed.set_epoch(True)


def test_mixed_dataloader_len_without_fixed_epoch_is_one_pass_batches() -> None:
    mixed = MixedDataLoader(
        {
            "xray": DataLoader(_TinyDataset(3, "xray"), batch_size=2),
            "ct": DataLoader(_TinyDataset(2, "ct"), batch_size=1),
        },
        proportions={"xray": 2, "ct": 1},
    )

    assert len(mixed) == 4


def test_datasets_import_does_not_import_heavy_dependencies() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src" / "fxr" / "datasets"
    blocked_roots = {"torchio", "monai"}
    for source_path in source_root.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            imported_roots: list[str] = []
            if isinstance(node, ast.Import):
                imported_roots = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_roots = [node.module.split(".", 1)[0]]
            assert not blocked_roots.intersection(imported_roots)


def test_inverse_label_frequency_weights_use_tempered_inverse_frequency() -> None:
    from fxr.datasets import inverse_label_frequency_weights

    crops = [[1], [1], [1], [1, 2]]
    mean_weights = inverse_label_frequency_weights(crops, [0, 1, 2], tau=1.0)
    max_weights = inverse_label_frequency_weights(
        crops, [0, 1, 2], tau=1.0, class_aggregation="max"
    )
    tempered = inverse_label_frequency_weights(
        crops, [0, 1, 2], tau=0.5, class_aggregation="max"
    )

    assert mean_weights.shape == (4,) and float(mean_weights.mean()) == pytest.approx(1.0)
    assert float(mean_weights[3] / mean_weights[0]) == pytest.approx(2.5)
    assert float(max_weights[3] / max_weights[0]) == pytest.approx(4.0)
    assert float(tempered[3] / tempered[0]) == pytest.approx(2.0)

    lut_weights = inverse_label_frequency_weights([[5], [6], [7]], [0, 0, 0, 0, 0, 1, 1, 0])
    assert float(lut_weights[0]) == pytest.approx(float(lut_weights[1]))
    assert float(lut_weights[2]) == 0.0
    with pytest.raises(AssertionError, match="outside the label LUT"):
        inverse_label_frequency_weights([[9]], [0, 1])


def test_sample_weighting_spec_parse_validates() -> None:
    from fxr.datasets import SampleWeightingSpec

    assert SampleWeightingSpec.parse("uniform", dataset_name="A").scheme == "uniform"
    spec = SampleWeightingSpec.parse(
        {"scheme": "inverse_label_frequency", "tau": 0.5, "class_aggregation": "max"},
        dataset_name="A",
    )
    assert (spec.tau, spec.class_aggregation) == (0.5, "max")
    for bad in (
        "random",
        {"scheme": "uniform", "tau": 0.5},
        {"scheme": "inverse_label_frequency", "tau": 0.0},
        {"scheme": "inverse_label_frequency", "temperature": 1.0},
    ):
        with pytest.raises(AssertionError):
            SampleWeightingSpec.parse(bad, dataset_name="A")


def _crop_backed_ct_db() -> dict:
    image = np.full((4, 4, 4), -500.0, dtype=np.float32)
    label = np.zeros((4, 4, 4), dtype=np.uint8)
    label[1, 1, 1] = 1
    label[2, 2, 2] = 2
    payload = {
        "img": image,
        "seg": label,
        "affine": np.eye(4, dtype=np.float32),
        "spacing": np.ones(3, dtype=np.float32),
    }
    return {
        "_attrs": {"dataset": "HipRay", "storage_layout": "crops"},
        "_splits": {"train": ["s1__crop000", "s1__crop001", "s2__crop000"]},
        "_metadata": {
            "s1__crop000": {
                "subject_id": "s1",
                "crop_foreground_label_ids": [1, 2],
                "fg_centroids_ijk": [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            },
            "s1__crop001": {
                "subject_id": "s1",
                "crop_foreground_label_ids": [1],
                "fg_centroids_ijk": {"1": [1.0, 1.0, 1.0]},
            },
            "s2__crop000": {
                "subject_id": "s2",
                "crop_foreground_label_ids": [],
                "fg_centroids_ijk": [],
            },
        },
        "s1__crop000": payload,
        "s1__crop001": payload,
        "s2__crop000": {**payload, "seg": np.zeros((4, 4, 4), dtype=np.uint8)},
    }


def test_ct_dataset_reads_stored_crop_centroids_and_sample_weights() -> None:
    db = _crop_backed_ct_db()
    backend = SplitThunderDBStorageBackend(
        db, dataset_name="HipRay", modality="ct", split="train", subject_grouping="subject"
    )
    dataset = CTTrainingDataset(
        backend=backend,
        dataset_name="HipRay",
        label_mode="native",
        require_seg=False,
        sample_weighting={"scheme": "inverse_label_frequency", "class_aggregation": "max"},
    )

    assert [record.subject_id for record in dataset.records] == ["s1", "s1", "s2"]
    assert dataset[0]["metadata"]["fg_centroids_ijk"] == {
        1: (1.0, 1.0, 1.0),
        2: (2.0, 2.0, 2.0),
    }
    assert dataset[1]["metadata"]["fg_centroids_ijk"] == {1: (1.0, 1.0, 1.0)}
    assert dataset[2]["metadata"]["fg_centroids_ijk"] == {}

    weights = dataset.sample_weights([0, 1, 2])
    assert float(weights[0]) > float(weights[1]) > 0.0 and float(weights[2]) == 0.0

    uniform = CTTrainingDataset(backend=backend, dataset_name="HipRay", label_mode="native")
    assert uniform.sample_weights([0, 1, 2]) is None

    db["_attrs"].pop("storage_layout")
    subject_backend = SplitThunderDBStorageBackend(
        db, dataset_name="HipRay", modality="ct", split="train"
    )
    weighted = CTTrainingDataset(
        backend=subject_backend,
        dataset_name="HipRay",
        label_mode="native",
        sample_weighting="inverse_label_frequency",
    )
    with pytest.raises(AssertionError, match="crop-backed"):
        weighted.sample_weights([0, 1, 2])


def test_ct_collator_emits_centroid_label_ids_and_allows_empty() -> None:
    base = {
        "image": torch.zeros(1, 2, 2, 2),
        "label": torch.zeros(1, 2, 2, 2, dtype=torch.long),
        "dataset_name": "HipRay",
        "modality": "ct",
    }
    geometry = {"affine": np.eye(4, dtype=np.float32), "spacing": np.ones(3, dtype=np.float32)}

    empty = _ct_safe_collate([{**base, "metadata": {**geometry, "fg_centroids_ijk": {}}}])
    assert tuple(empty["metadata"]["fg_centroids_ijk"].shape) == (1, 0, 3)
    assert tuple(empty["metadata"]["fg_centroid_label_ids"].shape) == (1, 0)

    ordered = _ct_safe_collate(
        [{**base, "metadata": {**geometry, "fg_centroids_ijk": {2: (2, 2, 2), 1: (1, 1, 1)}}}]
    )
    assert ordered["metadata"]["fg_centroid_label_ids"].tolist() == [[1, 2]]
    assert ordered["metadata"]["fg_centroids_ijk"][0, 0].tolist() == [1.0, 1.0, 1.0]
    inputs = resolve_batch_inputs(ordered, modality="ct")
    assert inputs.fg_centroid_label_ids.tolist() == [1, 2]


def test_split_thunderdb_backend_reads_v9_overlapping_channel_attrs() -> None:
    mask = np.zeros((3, 2, 2), dtype=np.uint8)
    mask[1, 0, 0] = 1
    mask[0] = 1 - mask[1:].max(axis=0)
    db = {
        "_attrs": {
            "dataset": "HipRay",
            "label_names": ["background", "femurs", "hips"],
            "label_space": "native",
            "seg_storage": "overlapping_channel_mask",
            "mask_label_ids": [0, 1, 2],
            "mask_label_names": ["background", "femurs", "hips"],
        },
        "_splits": {"train": ["a"]},
        "_metadata": {"a": {}},
        "a": {"img": np.ones((1, 2, 2), dtype=np.float16), "seg": mask},
    }
    backend = SplitThunderDBStorageBackend(
        db, dataset_name="HipRay", modality="xray", split="train"
    )
    assert backend.seg_storage == "overlapping_channel_mask"
    assert backend.label_names == ("background", "femurs", "hips")
    sample = XrayTrainingDataset(backend=backend, dataset_name="HipRay", label_mode="native")[0]
    assert sample["label"].shape == (3, 2, 2)

    db["a"] = {"img": np.ones((1, 2, 2), dtype=np.float16), "seg": mask[1].astype(np.int64)}
    with pytest.raises(ValueError, match="seg_storage"):
        XrayTrainingDataset(backend=backend, dataset_name="HipRay", label_mode="native")[0]

    db["_attrs"]["mask_label_ids"] = [0, 2, 1]
    with pytest.raises(ValueError, match="channel index equals native id"):
        SplitThunderDBStorageBackend(db, dataset_name="HipRay", modality="xray", split="train")
    db["_attrs"]["seg_storage"] = "rle"
    with pytest.raises(ValueError, match="seg_storage"):
        SplitThunderDBStorageBackend(db, dataset_name="HipRay", modality="xray", split="train")


def test_compile_package_label_remap_requires_matching_stored_labels(tmp_path: Path) -> None:
    from fxr.datasets import compile_package_label_remap

    spec = tmp_path / "CustomHips.yml"
    spec.write_text(
        "dataset_name: CustomHips\nskip_subjects: null\n"
        "stored_labels: {0: background, 1: hip_left, 2: hip_right}\n"
        "protocol_label_aliases:\n  all_structures_flexray_v4: {hip_left: hips, hip_right: hips}\n",
        encoding="utf-8",
    )
    stored = {"0": "background", "1": "hip_left", "2": "hip_right"}

    remap = compile_package_label_remap(PROTOCOL_NAME, "CustomHips", stored, dataset_spec=spec)
    hips = load_protocol_by_name(PROTOCOL_NAME).labels.index("hips")
    assert remap.label_lut.tolist() == [0, hips, hips]

    with pytest.raises(ValueError, match="outside protocol"):
        compile_package_label_remap(PROTOCOL_NAME, "CustomHips", stored)
    with pytest.raises(AssertionError, match="stored_labels differ"):
        compile_package_label_remap(
            PROTOCOL_NAME, "CustomHips", {"0": "background", "1": "hip_left"}, dataset_spec=spec
        )
    with pytest.raises(AssertionError, match="declares dataset_name"):
        compile_package_label_remap(PROTOCOL_NAME, "Other", stored, dataset_spec=spec)


def test_builders_pop_dataset_spec_and_require_path(tmp_path: Path) -> None:
    from fxr.datasets.builders import _resolve_dataset_build

    _, config, _ = _resolve_dataset_build(
        "MyXrays", "xray", {"path": str(tmp_path), "dataset_spec": str(tmp_path / "s.yml")}
    )
    assert "dataset_spec" not in config
    with pytest.raises(AssertionError, match="requires an explicit package path"):
        _resolve_dataset_build("HipRay", "xray", {"dataset_spec": str(tmp_path / "s.yml")})
    with pytest.raises(AssertionError, match="absolute"):
        _resolve_dataset_build("MyXrays", "xray", {"path": str(tmp_path), "dataset_spec": "s.yml"})
