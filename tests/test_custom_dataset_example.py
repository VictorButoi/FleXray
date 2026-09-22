"""End-to-end tests for the ``examples/custom_dataset`` walkthrough."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fxr.config import check_missing
from fxr.datasets.cli import main as dataset_main
from fxr.launch.cli import load_training_base, main as train_main
from fxr.launch.readiness import validate_training_config
from fxr.protocols import compile_training_lut, load_dataset_spec, load_protocol_by_name
from fxr.protocols.cli import main as protocol_main

pytest.importorskip("thunderpack")
pytest.importorskip("kornia")

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "custom_dataset"
PROTOCOL = "all_structures_flexray_v4"


def _example_module():
    spec = importlib.util.spec_from_file_location(
        "make_synthetic_dataset", EXAMPLE / "make_synthetic_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _launch_args(tmp_path: Path, packed: Path, spec: Path) -> list[str]:
    return [
        str(EXAMPLE / "train_custom.yml"),
        "--set", f"data.Xray.CustomHips.path={packed}",
        "--set", f"data.Xray.CustomHips.dataset_spec={spec}",
        "--set", f"log.root={tmp_path / 'runs'}",
        "--device", "cpu",
    ]


def test_example_spec_compiles_to_expected_lut() -> None:
    compiled = compile_training_lut(
        load_protocol_by_name(PROTOCOL), load_dataset_spec(EXAMPLE / "CustomHips.yml")
    )
    assert compiled.label_lut == (0, 54, 54, 11, 0)
    assert compiled.native_id_to_protocol_label == {
        0: "background", 1: "hips", 2: "hips", 3: "femurs", 4: "background"
    }


def test_example_train_config_is_self_contained() -> None:
    config = load_training_base(str(EXAMPLE / "train_custom.yml"))
    assert set(config["data"]) == {"Xray"}
    assert set(config["data"]["Xray"]) == set(config["dataloader"]["proportions"]) == {"CustomHips"}
    assert set(config["loss_func"]["dataset_losses"]) == {"CustomHips"}
    assert config["callbacks"] == {}
    with pytest.raises(ValueError) as exc_info:
        check_missing(config)
    message = str(exc_info.value)
    assert all(key in message for key in ("data.Xray.CustomHips.dataset_spec", "data.Xray.CustomHips.path", "log.root"))


def test_example_runs_end_to_end_on_cpu(tmp_path: Path, capsys) -> None:
    module = _example_module()
    manifest = module.write_dataset(tmp_path / "data")
    masks = sorted((tmp_path / "data" / "masks").glob("*.png"))
    assert len(masks) == 6
    ids = set()
    for path in masks:
        ids |= set(np.unique(np.array(Image.open(path))).tolist())
    assert ids == {0, 1, 2, 3, 4}

    assert dataset_main(["validate", str(manifest)]) == 0
    assert dataset_main(["pack", str(manifest), str(tmp_path / "packed")]) == 0
    assert capsys.readouterr().out.splitlines()[-1].startswith(
        "CustomHips: xray-seg, dense labels, 6 subjects, 6 samples (train=4, val=2) -> "
    )
    assert protocol_main(["compile", "--dataset", str(EXAMPLE / "CustomHips.yml")]) == 0
    assert "label_lut: [0, 54, 54, 11, 0]" in capsys.readouterr().out

    args = _launch_args(tmp_path, tmp_path / "packed", EXAMPLE / "CustomHips.yml")
    assert train_main([*args, "--dry-run", "--smoke-data"]) == 0
    assert not (tmp_path / "runs").exists()


def test_example_readiness_requires_the_spec_for_aliased_labels(tmp_path: Path) -> None:
    module = _example_module()
    manifest = module.write_dataset(tmp_path / "data")
    assert dataset_main(["pack", str(manifest), str(tmp_path / "packed")]) == 0
    config = load_training_base(str(EXAMPLE / "train_custom.yml"))
    config["log"]["root"] = str(tmp_path / "runs")
    config["data"]["Xray"]["CustomHips"] = {"path": str(tmp_path / "packed")}
    with pytest.raises(ValueError, match="hip_left"):
        validate_training_config(config)

    wrong = tmp_path / "Wrong.yml"
    wrong.write_text(
        (EXAMPLE / "CustomHips.yml").read_text().replace("implant", "screw"), encoding="utf-8"
    )
    config["data"]["Xray"]["CustomHips"]["dataset_spec"] = str(wrong)
    with pytest.raises(ValueError, match="stored_labels differ"):
        validate_training_config(config)


def test_example_trains_two_epochs_on_cpu(tmp_path: Path) -> None:
    module = _example_module()
    manifest = module.write_dataset(tmp_path / "data")
    assert dataset_main(["pack", str(manifest), str(tmp_path / "packed")]) == 0
    args = _launch_args(tmp_path, tmp_path / "packed", EXAMPLE / "CustomHips.yml")
    assert train_main([*args, "--set", "log.wandb.mode=disabled"]) == 0
    runs = list((tmp_path / "runs").iterdir())
    assert len(runs) == 1
    assert (runs[0] / "config.yml").is_file()
    assert (runs[0] / "checkpoints" / "last.pt").is_file()
    assert (runs[0] / "augmentations" / "xray.yml").is_file()
