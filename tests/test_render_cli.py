"""Behavior tests for offline DRR rendering (``fxr-render``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from fxr.datasets import (
    SplitThunderDBStorageBackend,
    XrayTrainingDataset,
    pack_dataset,
    validate_dataset_manifest,
)
from fxr.launch.render import OfflineRenderRequest, render_dataset
from fxr.launch.render_cli import main as render_main
from fxr.protocols import resolve_run_output_label_names

pytest.importorskip("nanodrr")
thunderpack = pytest.importorskip("thunderpack")

_PROTOCOL = "all_structures_flexray_v4"


def _render_config(num_views: int = 2) -> dict:
    return {
        "protocol": {"name": _PROTOCOL},
        "dataloader": {"batch_size": num_views},
        "drr_model": {
            "default": {
                "preset": "frontal",
                "camera_displacement": 800.0,
                "num_views": num_views,
                "num_samples": 16,
                "intrinsics_cfg": {"height": 8, "width": 8, "sdd": 1000.0, "delx": 2.0},
                "isocenter_cfg": {"sample_scheme": "volume_center"},
                "seg_cfg": {"soft_labels": False, "threshold": 0.0},
            },
            "datasets": {
                "MOOSE": {},
                "Other": {
                    "preset": "lateral",
                    "isocenter_cfg": {"sample_scheme": "random_label", "replacement": True},
                },
            },
        },
    }


def _pack_ct(tmp_path: Path, *, name: str = "ToyCT") -> Path:
    rng = np.random.default_rng(0)
    volume = rng.uniform(-200.0, 400.0, size=(16, 16, 16)).astype(np.float32)
    label = np.zeros((16, 16, 16), dtype=np.uint8)
    label[3:8, 3:8, 3:8] = 1
    label[9:13, 9:13, 9:13] = 2
    np.save(tmp_path / "ct.npy", volume)
    np.save(tmp_path / "seg.npy", label)
    np.save(tmp_path / "affine.npy", np.eye(4, dtype=np.float32))
    np.save(tmp_path / "spacing.npy", np.ones(3, dtype=np.float32))
    manifest = tmp_path / "ct.yml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "dataset_name": name,
                "dataset_type": "ct-seg",
                "stored_labels": {0: "background", 1: "femurs", 2: "hips"},
                "samples": [
                    {
                        "sample_id": "vol1",
                        "subject_id": "subj1",
                        "split": "train",
                        "image": "ct.npy",
                        "label": "seg.npy",
                        "affine": "affine.npy",
                        "spacing": "spacing.npy",
                    }
                ],
            },
            sort_keys=False,
        )
    )
    return pack_dataset(manifest, tmp_path / "packed_ct").path


def test_render_writes_packable_manifest_with_output_channels(tmp_path: Path) -> None:
    packed = _pack_ct(tmp_path)
    request = OfflineRenderRequest(
        dataset_path=packed,
        dataset_name="ToyCT",
        config=_render_config(),
        profile="MOOSE",
        output=tmp_path / "rendered",
        renders=2,
        seed=1,
        png=True,
    )
    report = render_dataset(request)
    labels = resolve_run_output_label_names(_render_config())

    assert (report.num_volumes, report.num_samples) == (1, 4)
    assert report.label_names == labels
    manifest = yaml.safe_load(report.manifest.read_text())
    assert manifest["dataset_type"] == "xray-seg" and manifest["label_names"] == list(labels)
    sample = manifest["samples"][0]
    assert sample["subject_id"] == "subj1" and sample["split"] == "train"
    assert sample["metadata"]["source_sample_id"] == "vol1"
    assert len(sample["metadata"]["rot_deg"]) == 3 and "sdd" in sample["metadata"]
    image = np.load(tmp_path / "rendered" / sample["image"])
    mask = np.load(tmp_path / "rendered" / sample["label"])
    assert image.shape == (1, 8, 8) and image.dtype == np.float32
    assert 0.0 <= image.min() and image.max() <= 1.0
    assert len(labels) == 61
    assert mask.shape == (len(labels), 8, 8) and mask.dtype == np.uint8
    assert np.array_equal(mask[0], 1 - mask[1:].max(axis=0))
    nonzero = {labels[c] for c in range(1, len(labels)) if mask[c].any()}
    assert nonzero and nonzero <= {"femurs", "hips"}
    assert (tmp_path / "rendered" / "previews").is_dir()

    validate_dataset_manifest(report.manifest)
    packed_drr = pack_dataset(report.manifest, tmp_path / "packed_drr").path
    with thunderpack.ThunderDB.open(str(packed_drr), "r") as db:
        backend = SplitThunderDBStorageBackend(
            db, dataset_name="ToyCT_drr", modality="xray", split="train"
        )
        assert backend.label_names == labels
        dataset = XrayTrainingDataset(backend=backend, dataset_name="ToyCT_drr", label_mode="native")
        assert dataset[0]["label"].shape == (len(labels), 8, 8)


def test_render_is_deterministic_for_seed(tmp_path: Path) -> None:
    packed = _pack_ct(tmp_path)
    outputs = []
    for run in range(2):
        request = OfflineRenderRequest(
            dataset_path=packed,
            dataset_name="ToyCT",
            config=_render_config(),
            profile="Other",
            output=tmp_path / f"run{run}",
            seed=7,
        )
        render_dataset(request)
        outputs.append(np.load(tmp_path / f"run{run}" / "images" / "vol1__r000_v00.npy"))
    assert np.array_equal(outputs[0], outputs[1])

    with pytest.raises(AssertionError, match="no profile"):
        render_dataset(
            OfflineRenderRequest(
                dataset_path=packed,
                dataset_name="ToyCT",
                config=_render_config(),
                profile="Missing",
                output=tmp_path / "bad",
            )
        )


def test_render_cli_end_to_end_into_training_dry_run(tmp_path: Path, capsys) -> None:
    from fxr.launch.cli import main as train_main

    packed = _pack_ct(tmp_path)
    config_path = tmp_path / "render.yml"
    config_path.write_text(yaml.safe_dump(_render_config(), sort_keys=False))
    assert (
        render_main(
            [
                str(packed), "--dataset-name", "ToyCT", "--base", str(config_path),
                "--profile", "MOOSE", "--output", str(tmp_path / "drr"), "--device", "cpu",
                "--renders", "1", "--output-name", "MyDRRs",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert out.startswith("rendered 2 samples from 1 volumes")
    packed_drr = pack_dataset(tmp_path / "drr" / "manifest.yml", tmp_path / "packed_drr").path

    exit_code = train_main(
        [
            "--base", "base",
            "--set", f"data={{Xray: {{MyDRRs: {{path: {packed_drr}}}}}}}",
            "--set", "dataloader.proportions={MyDRRs: 1}",
            "--set", "loss_func.dataset_losses={MyDRRs: partial_labeled_seg}",
            "--set", "callbacks={}",
            "--set", "drr_model=null",
            "--set", "train.eval_freq=0",
            "--set", f"log.root={tmp_path / 'runs'}",
            "--set", "dataloader.batch_size=2",
            "--set", "dataloader.num_workers=0",
            "--device", "cpu", "--dry-run", "--smoke-data",
        ]
    )
    assert exit_code == 0
