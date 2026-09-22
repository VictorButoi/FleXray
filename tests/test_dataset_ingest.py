"""Behavior tests for the data-engine ingestion surface of ``fxr-dataset``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from fxr.datasets import (
    CTTrainingDataset,
    SplitThunderDBStorageBackend,
    pack_dataset,
    validate_dataset_manifest,
    validate_packed_dataset,
)
from fxr.datasets.preprocessing import XrayPreprocessing

thunderpack = pytest.importorskip("thunderpack")


def _write_manifest(root: Path, body: dict) -> Path:
    path = root / "dataset.yml"
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


def _xray_manifest(root: Path, *, preprocessing: dict | None) -> Path:
    image = np.linspace(10, 200, 40 * 64, dtype=np.float64).reshape(40, 64).astype(np.uint8)
    mask = np.zeros((40, 64), dtype=np.uint8)
    mask[5:20, 10:30] = 1
    mask[25:35, 40:60] = 2
    Image.fromarray(image).save(root / "image.png")
    Image.fromarray(mask).save(root / "mask.png")
    body = {
        "schema_version": 1,
        "dataset_name": "Padded",
        "dataset_type": "xray-seg",
        "stored_labels": {0: "background", 1: "lungs", 2: "heart"},
        "samples": [
            {
                "sample_id": "a",
                "subject_id": "a",
                "split": "train",
                "image": "image.png",
                "label": "mask.png",
            }
        ],
    }
    if preprocessing is not None:
        body["preprocessing"] = preprocessing
    return _write_manifest(root, body)


def test_pack_xray_preprocessing_pads_resizes_and_records_geometry(tmp_path: Path) -> None:
    manifest = _xray_manifest(
        tmp_path,
        preprocessing={"intensity": "per_image_minmax", "pad_to_square": True, "output_size": [32, 32]},
    )
    validate_dataset_manifest(manifest)
    report = pack_dataset(manifest, tmp_path / "packed")

    with thunderpack.ThunderDB.open(str(report.path), "r") as db:
        attrs = dict(db["_attrs"])
        geometry = dict(db["_metadata"])["a"]["preprocessing"]
        image = np.asarray(db["a"]["img"])
        mask = np.asarray(db["a"]["seg"])
    assert attrs["preprocessing"] == {
        "intensity": "per_image_minmax",
        "pad_to_square": True,
        "output_size": [32, 32],
    }
    assert geometry["original_shape"] == [40, 64]
    assert geometry["pad_before"] == [12, 0] and geometry["pad_after"] == [12, 0]
    assert geometry["padded_shape"] == [64, 64] and geometry["processed_shape"] == [32, 32]
    assert geometry["resize_scale"] == [0.5, 0.5]
    assert image.shape == (1, 32, 32) and image.dtype == np.float32
    assert float(image.min()) == 0.0 and float(image.max()) <= 1.0
    assert image[0, 0].max() == 0.0  # padded band
    assert set(np.unique(mask).tolist()) == {0, 1, 2}
    assert validate_packed_dataset(report.path).num_samples == 1


def test_pack_xray_without_preprocessing_is_unchanged(tmp_path: Path) -> None:
    report = pack_dataset(_xray_manifest(tmp_path, preprocessing=None), tmp_path / "packed")
    with thunderpack.ThunderDB.open(str(report.path), "r") as db:
        assert "preprocessing" not in dict(db["_attrs"])
        assert "preprocessing" not in dict(db["_metadata"])["a"]
        image = np.asarray(db["a"]["img"])
    assert image.shape == (1, 40, 64)
    assert float(image.min()) == pytest.approx(10 / 255)


def test_preprocessing_minmax_is_invisible_to_percentile_normalization_without_padding() -> None:
    from fxr.augmentation import PercentileMinMaxNormalize

    rng = np.random.default_rng(0)
    image = (rng.uniform(0.2, 0.7, size=(1, 16, 16)).astype(np.float32))
    label = np.zeros((16, 16), dtype=np.uint8)
    scaled, _, _ = XrayPreprocessing(intensity="per_image_minmax").apply(image, label)
    kept, _, _ = XrayPreprocessing(intensity="dtype_range").apply(image, label)

    normalizer = PercentileMinMaxNormalize()
    torch.testing.assert_close(
        normalizer(torch.from_numpy(scaled)[None]), normalizer(torch.from_numpy(kept)[None])
    )
    with pytest.raises(AssertionError, match="intensity"):
        XrayPreprocessing.from_manifest({"intensity": "zscore"})


def _write_nifti_pair(root: Path) -> tuple[np.ndarray, np.ndarray]:
    nibabel = pytest.importorskip("nibabel")
    volume = np.linspace(-1500.0, 3000.0, 8**3, dtype=np.float32).reshape(8, 8, 8)
    label = np.zeros((8, 8, 8), dtype=np.uint8)
    label[2:5, 2:5, 2:5] = 1
    affine = np.diag([1.0, 1.5, 2.0, 1.0]).astype(np.float32)
    affine[:3, 3] = [-4.0, -6.0, -8.0]
    nibabel.save(nibabel.Nifti1Image(volume, affine), str(root / "ct.nii.gz"))
    nibabel.save(nibabel.Nifti1Image(label, affine), str(root / "seg.nii.gz"))
    return volume, affine


def test_pack_ct_nifti_pair_derives_affine_and_spacing_and_clips_hu_window(
    tmp_path: Path,
) -> None:
    _, affine = _write_nifti_pair(tmp_path)
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_name": "NiftiCT",
            "dataset_type": "ct-seg",
            "stored_labels": {0: "background", 1: "femurs"},
            "preprocessing": {"hu_window": [-1000, 2000]},
            "samples": [
                {
                    "sample_id": "v1",
                    "subject_id": "s1",
                    "split": "train",
                    "image": "ct.nii.gz",
                    "label": "seg.nii.gz",
                }
            ],
        },
    )
    report = pack_dataset(manifest, tmp_path / "packed")

    with thunderpack.ThunderDB.open(str(report.path), "r") as db:
        payload = db["v1"]
        image = np.asarray(payload["img"])
        assert image.dtype == np.float16
        assert float(image.min()) == -1000.0 and float(image.max()) == 2000.0
        np.testing.assert_allclose(np.asarray(payload["affine"]), affine, atol=1e-5)
        np.testing.assert_allclose(np.asarray(payload["spacing"]), [1.0, 1.5, 2.0])
        assert dict(db["_attrs"])["preprocessing"] == {"hu_window": [-1000.0, 2000.0]}
        backend = SplitThunderDBStorageBackend(
            db, dataset_name="NiftiCT", modality="ct", split="train"
        )
        sample = CTTrainingDataset(backend=backend, dataset_name="NiftiCT", label_mode="native")[0]
        assert sample["image"].dtype == torch.float32
    assert validate_packed_dataset(report.path).dataset_type == "ct-seg"


def test_ct_manifest_rejects_dicom_and_mismatched_explicit_affine(tmp_path: Path) -> None:
    _write_nifti_pair(tmp_path)
    np.save(tmp_path / "wrong_affine.npy", np.eye(4, dtype=np.float32))
    np.save(tmp_path / "spacing.npy", np.array([1.0, 1.5, 2.0], dtype=np.float32))
    body = {
        "schema_version": 1,
        "dataset_name": "NiftiCT",
        "dataset_type": "ct-seg",
        "stored_labels": {0: "background", 1: "femurs"},
        "samples": [
            {
                "sample_id": "v1",
                "subject_id": "s1",
                "split": "train",
                "image": "ct.nii.gz",
                "label": "seg.nii.gz",
                "affine": "wrong_affine.npy",
                "spacing": "spacing.npy",
            }
        ],
    }
    with pytest.raises(ValueError, match="disagrees with the NIfTI header"):
        validate_dataset_manifest(_write_manifest(tmp_path, body))

    (tmp_path / "scan.dcm").write_bytes(b"not really dicom")
    np.save(tmp_path / "affine.npy", np.eye(4, dtype=np.float32))
    body["samples"][0].update({"image": "scan.dcm", "affine": "affine.npy"})
    with pytest.raises(ValueError, match="dcm2niix"):
        validate_dataset_manifest(_write_manifest(tmp_path, body))


def test_plan_z_crops_default_preserves_minimal_cover_and_padding() -> None:
    from fxr.datasets.ct_crops import plan_z_crops

    assert [c.z_start for c in plan_z_crops(700, 256)] == [0, 222, 444]
    (single,) = plan_z_crops(100, 256)
    assert (single.z_start, single.z_stop, single.z_pad_before, single.z_pad_after) == (
        0, 100, 78, 78
    )
    assert single.z_offset == -78
    assert [c.z_start for c in plan_z_crops(300, 256)] == [0, 44]


@pytest.mark.parametrize(
    ("overlap", "expected_starts"),
    [
        (None, [0, 6, 12]),
        (0.0, [0, 6, 12]),
        (0.5, [0, 4, 8, 12]),
        (0.75, [0, 2, 4, 6, 8, 10, 12]),
    ],
)
def test_pack_ct_crops_uses_manifest_overlap(
    tmp_path: Path, overlap: float | None, expected_starts: list[int]
) -> None:
    """Manifest overlap determines packed samples, payloads, and crop metadata."""

    volume = np.arange(80, dtype=np.float32).reshape(2, 2, 20)
    label = np.ones(volume.shape, dtype=np.uint8)
    affine = np.diag([1.0, 1.0, 2.0, 1.0]).astype(np.float32)
    np.save(tmp_path / "ct.npy", volume)
    np.save(tmp_path / "seg.npy", label)
    np.save(tmp_path / "affine.npy", affine)
    np.save(tmp_path / "spacing.npy", np.array([1.0, 1.0, 2.0], dtype=np.float32))
    crops = {"size": [2, 2, 8]}
    if overlap is not None:
        crops["max_overlap"] = overlap
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_name": "OverlapCT",
            "dataset_type": "ct-seg",
            "stored_labels": {0: "background", 1: "femurs"},
            "crops": crops,
            "samples": [
                {
                    "sample_id": "v1",
                    "subject_id": "s1",
                    "split": "train",
                    "image": "ct.npy",
                    "label": "seg.npy",
                    "affine": "affine.npy",
                    "spacing": "spacing.npy",
                }
            ],
        },
    )
    report = pack_dataset(manifest, tmp_path / "packed")
    assert report.num_samples == len(expected_starts)
    with thunderpack.ThunderDB.open(str(report.path), "r") as db:
        assert db["_attrs"]["crops"]["max_overlap"] == (0.0 if overlap is None else overlap)
        metadata = db["_metadata"]
        sample_ids = [f"v1__crop{index:03d}" for index in range(len(expected_starts))]
        assert list(db["_splits"]["train"]) == sample_ids
        for sample_id, start in zip(sample_ids, expected_starts):
            assert metadata[sample_id]["z_start"] == start
            assert metadata[sample_id]["z_stop"] == start + 8
            payload = db[sample_id]
            np.testing.assert_array_equal(payload["img"], volume[..., start : start + 8])
            np.testing.assert_allclose(np.asarray(payload["affine"])[:3, 3], [0, 0, 2 * start])
    assert validate_packed_dataset(report.path).num_samples == len(expected_starts)


def test_pack_ct_crops_expands_samples_offsets_affine_and_drops_background_only(
    tmp_path: Path,
) -> None:
    volume = np.full((6, 6, 20), -800.0, dtype=np.float32)
    label = np.zeros((6, 6, 20), dtype=np.uint8)
    label[1:4, 1:4, 2:5] = 1
    label[2:5, 2:5, 6:10] = 2
    affine = np.diag([1.0, 1.0, 2.0, 1.0]).astype(np.float32)
    affine[:3, 3] = [10.0, 20.0, 30.0]
    np.save(tmp_path / "ct.npy", volume)
    np.save(tmp_path / "seg.npy", label)
    np.save(tmp_path / "affine.npy", affine)
    np.save(tmp_path / "spacing.npy", np.array([1.0, 1.0, 2.0], dtype=np.float32))
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_name": "CroppedCT",
            "dataset_type": "ct-seg",
            "stored_labels": {0: "background", 1: "femurs", 2: "hips"},
            "crops": {"size": [6, 6, 8], "max_overlap": 0.0},
            "samples": [
                {
                    "sample_id": "v1",
                    "subject_id": "s1",
                    "split": "train",
                    "image": "ct.npy",
                    "label": "seg.npy",
                    "affine": "affine.npy",
                    "spacing": "spacing.npy",
                }
            ],
        },
    )
    report = pack_dataset(manifest, tmp_path / "packed")

    # depth 20 / crop 8 -> starts [0, 6, 12]; the last crop holds no foreground.
    assert report.num_samples == 2 and report.num_subjects == 1
    with thunderpack.ThunderDB.open(str(report.path), "r") as db:
        attrs = dict(db["_attrs"])
        metadata = dict(db["_metadata"])
        assert attrs["storage_layout"] == "crops"
        assert attrs["crops"]["dropped_background_only"] == ["v1__crop002"]
        assert attrs["crops"]["num_source_samples"] == 1
        second = db["v1__crop001"]
        assert np.asarray(second["img"]).shape == (6, 6, 8)
        np.testing.assert_allclose(np.asarray(second["affine"])[:3, 3], [10.0, 20.0, 42.0])
        meta = metadata["v1__crop001"]
        assert (meta["z_start"], meta["z_stop"], meta["subject_id"]) == (6, 14, "s1")
        assert meta["crop_foreground_label_ids"] == [2]
        first = metadata["v1__crop000"]
        assert first["crop_foreground_label_ids"] == [1, 2]
        assert len(first["fg_centroids_ijk"]) == 2
        backend = SplitThunderDBStorageBackend(
            db, dataset_name="CroppedCT", modality="ct", split="train", subject_grouping="subject"
        )
        dataset = CTTrainingDataset(
            backend=backend,
            dataset_name="CroppedCT",
            label_mode="native",
            sample_weighting="inverse_label_frequency",
        )
        assert [r.subject_id for r in dataset.records] == ["s1", "s1"]
        assert set(dataset[0]["metadata"]["fg_centroids_ijk"]) == {1, 2}
        assert dataset.sample_weights([0, 1, 2]).shape == (2,)
    assert validate_packed_dataset(report.path).num_samples == 2

    with pytest.raises(ValueError, match="axial only"):
        body = yaml.safe_load(manifest.read_text())
        body["crops"]["size"] = [4, 6, 8]
        validate_dataset_manifest(_write_manifest(tmp_path, body))


def test_scaffold_xray_dirs_emits_subject_disjoint_seeded_splits(tmp_path: Path) -> None:
    from fxr.datasets import scaffold_manifest, stable_subject_splits
    from fxr.datasets.cli import main as dataset_main

    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir(), masks.mkdir()
    rng = np.random.default_rng(0)
    for patient in range(5):
        for view in ("ap", "lat"):
            stem = f"P{patient:02d}_{view}"
            Image.fromarray(rng.integers(0, 255, (16, 16), dtype=np.uint8)).save(images / f"{stem}.png")
            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[4:8, 4:8] = 1 + patient % 2
            Image.fromarray(mask).save(masks / f"{stem}.png")
    (images / "notes.txt").write_text("ignored", encoding="utf-8")

    report = scaffold_manifest(
        "xray-seg",
        images=images,
        masks=masks,
        dataset_name="Scaffolded",
        output=tmp_path / "dataset.yml",
        seed=3,
        percentages=(60, 20, 20),
        subject_regex=r"^(P\d+)",
    )
    manifest = yaml.safe_load((tmp_path / "dataset.yml").read_text())

    assert report.num_samples == 10 and report.observed_label_ids == (0, 1, 2)
    assert manifest["stored_labels"] == {0: "background", 1: "label_1", 2: "label_2"}
    subjects = {s["subject_id"] for s in manifest["samples"]}
    assert subjects == {f"P{p:02d}" for p in range(5)}
    split_by_subject: dict[str, set[str]] = {}
    for sample in manifest["samples"]:
        split_by_subject.setdefault(sample["subject_id"], set()).add(sample["split"])
    assert all(len(splits) == 1 for splits in split_by_subject.values())
    assert report.split_counts == {"train": 6, "val": 2, "test": 2}
    assert manifest["samples"][0]["image"] == "images/P00_ap.png"
    assert stable_subject_splits(["b", "a", "c"], seed=3, percentages=(60, 20, 20)) == (
        stable_subject_splits(["c", "a", "b"], seed=3, percentages=(60, 20, 20))
    )
    with pytest.raises(AssertionError, match="summing to 100"):
        stable_subject_splits(["a"], seed=0, percentages=(50, 20, 20))

    manifest["stored_labels"] = {0: "background", 1: "lungs", 2: "heart"}
    (tmp_path / "dataset.yml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    assert dataset_main(["validate", str(tmp_path / "dataset.yml")]) == 0


def test_dataset_cli_scaffold_and_inspect_commands(tmp_path: Path, capsys) -> None:
    from fxr.datasets.cli import main as dataset_main

    images, masks = tmp_path / "img", tmp_path / "msk"
    images.mkdir(), masks.mkdir()
    for stem in ("a", "b"):
        Image.fromarray(np.full((8, 8), 100, dtype=np.uint8)).save(images / f"{stem}.png")
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:4, 2:4] = 1
        Image.fromarray(mask).save(masks / f"{stem}.png")
    assert (
        dataset_main(
            [
                "scaffold", "xray-seg", "--images", str(images), "--masks", str(masks),
                "--name", "Two", "--output", str(tmp_path / "two.yml"), "--split", "50", "50", "0",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert out.startswith("Two: 2 samples (train=1, val=1), mask ids [0, 1] ->")
    assert "Rename the stored_labels stub" in out

    manifest = yaml.safe_load((tmp_path / "two.yml").read_text())
    manifest["stored_labels"] = {0: "background", 1: "lungs"}
    (tmp_path / "two.yml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    assert dataset_main(["pack", str(tmp_path / "two.yml"), str(tmp_path / "packed")]) == 0
    capsys.readouterr()
    assert dataset_main(["inspect", str(tmp_path / "packed")]) == 0
    report = capsys.readouterr().out
    assert "package: canonical FleXray" in report
    assert "splits: train=1, val=1" in report and "subjects: 2" in report
    assert "payload img: shape=(1, 8, 8) dtype=float32" in report

    with pytest.raises(SystemExit) as exc_info:
        dataset_main(["scaffold", "xray-seg", "--images", str(images), "--masks", str(tmp_path / "missing"), "--name", "X", "--output", str(tmp_path / "x.yml")])
    assert exc_info.value.code == 2


def test_inspect_reports_non_canonical_thunderdb_that_check_rejects(tmp_path: Path) -> None:
    from fxr.datasets import inspect_thunderdb
    from fxr.datasets.inspect import format_inspection

    with thunderpack.ThunderDB.open(str(tmp_path / "legacy_attrs"), "c") as db:
        db["case"] = {"img": np.zeros((1, 4, 4), dtype=np.float16), "seg": np.zeros((4, 4), dtype=np.uint8)}
        db["_splits"] = {"train": ["case"]}
        db["_metadata"] = {"case": {"subject_id": "p1"}}
        db["_attrs"] = {"dataset": "Legacy", "version": "9.0", "seg_storage": "indexed_label_mask", "label_names": ["background", "hips"]}

    inspection = inspect_thunderdb(tmp_path / "legacy_attrs")
    text = format_inspection(inspection)

    assert inspection.attrs == {"dataset": "Legacy", "seg_storage": "indexed_label_mask", "version": "9.0"}
    assert inspection.label_names == ("background", "hips")
    assert "non-canonical" in text and "payload img: shape=(1, 4, 4) dtype=float16" in text
    with pytest.raises(ValueError, match="schema_name"):
        validate_packed_dataset(tmp_path / "legacy_attrs")
