from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import fxr.datasets as public_datasets
from fxr.datasets import (
    CTTrainingDataset,
    DatasetLayout,
    TrainingDatasetBundle,
    XrayTrainingDataset,
    build_multimodal_datasets,
    build_named_datasets,
    build_training_dataset,
    compile_training_label_remap_from_stored_labels,
    load_dataset_layouts,
    pack_dataset,
    register_dataset_layout,
)
from fxr.datasets import builders as dataset_builders
from fxr.launch.readiness import validate_training_config
from fxr.protocols import compile_training_lut_by_name, load_dataset_spec_by_name

PROTOCOL_NAME = "all_structures_flexray_v4"

def _packaged_dataset_config_names() -> tuple[str, ...]:
    dataset_root = files("fxr.configs").joinpath("datasets")
    return tuple(
        sorted(
            path.name.removesuffix(".yml")
            for path in dataset_root.iterdir()
            if path.name.endswith(".yml")
        )
    )


class _ClosableDatabase(dict[str, object]):
    """Dictionary-backed database test double recording close calls.

    Attributes:
        close_calls: Number of times :meth:`close` was invoked.
    """

    def __init__(self, values: dict[str, object]) -> None:
        """Copy database values and initialize the close counter.

        Args:
            values: Database mapping copied into this test double.

        Returns:
            ``None``.
        """

        super().__init__(values)
        self.close_calls = 0

    def close(self) -> None:
        """Record one database close request.

        Returns:
            ``None``.
        """

        self.close_calls += 1


def _ct_db(dataset_name: str = "ElbowCT") -> dict[str, object]:
    return {
        "_attrs": {"dataset_name": dataset_name, "clean_dataset": True},
        "_splits": {"train": ["ct_001"], "val": ["ct_001"]},
        "ct_001": {
            "img": np.arange(8, dtype=np.float32).reshape(2, 2, 2),
            "seg": np.zeros((2, 2, 2), dtype=np.int64),
            "affine": np.eye(4, dtype=np.float32),
            "spacing": np.ones(3, dtype=np.float32),
        },
    }


def _xray_db(
    *, dataset_name: str = "HipRay", with_label: bool = True
) -> dict[str, object]:
    sample = {"img": np.ones((2, 3), dtype=np.float32)}
    if with_label:
        sample["seg"] = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64)
    return {
        "_attrs": {"dataset_name": dataset_name, "clean_dataset": True},
        "_splits": {"train": ["xray_001"], "val": ["xray_001"]},
        "xray_001": sample,
    }


def _fluxray_db() -> dict[str, object]:
    return {
        "_attrs": {
            "dataset_name": "FluXray",
            "clean_dataset": True,
            "protocol_name": PROTOCOL_NAME,
            "label_names": ["background", "clavicles"],
        },
        "_splits": {"train": ["gen_001"], "val": ["gen_001"]},
        "gen_001": {
            "img": np.ones((1, 2, 2), dtype=np.float32),
            "seg": np.ones((2, 2, 2), dtype=np.float32),
        },
    }


def _pack_xray_package(
    root: Path,
    *,
    dataset_name: str = "ClinicXray",
    splits: tuple[str, ...] = ("train", "val"),
) -> Path:
    """Package a small dense X-ray dataset through the public API.

    Args:
        root: Temporary directory that owns source arrays and the package.
        dataset_name: Identity written to canonical package metadata.
        splits: One sample split per generated subject.

    Returns:
        Path to the packaged ThunderDB.
    """

    pytest.importorskip("thunderpack")
    samples: list[dict[str, object]] = []
    for index, split in enumerate(splits):
        image_path = root / f"image_{index}.npy"
        label_path = root / f"label_{index}.npy"
        np.save(
            image_path,
            np.full((64, 64), (index + 1) / (len(splits) + 1), dtype=np.float32),
        )
        np.save(label_path, np.full((64, 64), index % 2, dtype=np.uint8))
        samples.append(
            {
                "sample_id": f"sample_{index}",
                "subject_id": f"subject_{index}",
                "split": split,
                "image": image_path.name,
                "label": label_path.name,
            }
        )
    manifest_path = root / "dataset.yml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "dataset_name": dataset_name,
                "dataset_type": "xray-seg",
                "stored_labels": {0: "background", 1: "clavicles"},
                "samples": samples,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return pack_dataset(manifest_path, root / "packed").path


def test_public_dataset_api_exports_builder_names() -> None:
    assert {
        "DatasetLayout",
        "TrainingDatasetBundle",
        "build_training_dataset",
        "build_named_datasets",
        "build_multimodal_datasets",
        "load_dataset_layouts",
        "register_dataset_layout",
    }.issubset(set(public_datasets.__all__))


def test_package_stored_labels_compile_without_bundled_dataset_spec() -> None:
    remap = compile_training_label_remap_from_stored_labels(
        PROTOCOL_NAME,
        "MyDataset",
        {"0": "background", "1": "clavicles", "2": "heart"},
        model_label_names=("background", "clavicles"),
    )

    assert remap.dataset_name == "MyDataset"
    assert remap.label_names == ("background", "clavicles")
    assert remap.label_lut.tolist() == [0, 1, 0]


def test_package_stored_labels_reject_sparse_ids_and_unknown_names() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        compile_training_label_remap_from_stored_labels(
            PROTOCOL_NAME, "MyDataset", {0: "background", 2: "heart"}
        )
    with pytest.raises(ValueError, match="outside protocol"):
        compile_training_label_remap_from_stored_labels(
            PROTOCOL_NAME, "MyDataset", {0: "background", 1: "not-anatomy"}
        )


def test_packaged_layouts_are_yaml_backed_and_self_consistent() -> None:
    layout_resource = files("fxr.configs").joinpath("dataset_layouts", "training.yml")

    assert layout_resource.is_file()
    layouts = load_dataset_layouts(layout_resource, replace=True)
    names = [layout.dataset_name for layout in layouts]

    assert layouts
    assert len(names) == len(set(names))
    for layout in layouts:
        assert layout.modality in {"ct", "xray"}
        assert layout.root_env_var
        assert not Path(layout.relative_path_template).is_absolute()
        assert set(layout.required_fields) <= (
            dataset_builders._template_fields(layout.relative_path_template)
            | {"dataset_name"}
        )
        spec = load_dataset_spec_by_name(layout.dataset_name)
        assert spec.stored_labels[0] == "background"


@pytest.mark.parametrize(
    ("dataset_name", "modality", "cfg", "env_var"),
    [
        ("MOOSE", "CT", {"version": "2.0"}, "CT_DATAPATH"),
        ("HipRay", "Xray", {}, "XRAY_DATAPATH"),
        ("FluXray", "Xray", {"version": "7.0"}, "GENERATED_DATAPATH"),
    ],
)
def test_layout_resolution_requires_modality_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dataset_name: str,
    modality: str,
    cfg: dict[str, object],
    env_var: str,
) -> None:
    monkeypatch.setenv("CT_DATAPATH", str(tmp_path))
    monkeypatch.setenv("XRAY_DATAPATH", str(tmp_path))
    monkeypatch.setenv("GENERATED_DATAPATH", str(tmp_path))
    monkeypatch.delenv(env_var, raising=False)

    with pytest.raises(KeyError, match=env_var):
        build_training_dataset(dataset_name, "train", modality, cfg)


@pytest.mark.parametrize(
    ("dataset_name", "modality", "cfg", "env_var"),
    [
        ("MOOSE", "CT", {"version": "2.0"}, "CT_DATAPATH"),
        ("HipRay", "Xray", {}, "XRAY_DATAPATH"),
        ("FluXray", "Xray", {"version": "7.0"}, "GENERATED_DATAPATH"),
    ],
)
def test_layout_resolution_rejects_root_search_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dataset_name: str,
    modality: str,
    cfg: dict[str, object],
    env_var: str,
) -> None:
    monkeypatch.setenv("CT_DATAPATH", str(tmp_path))
    monkeypatch.setenv("XRAY_DATAPATH", str(tmp_path))
    monkeypatch.setenv("GENERATED_DATAPATH", str(tmp_path))
    monkeypatch.setenv(env_var, "/data/a:/data/b")

    with pytest.raises(ValueError, match=env_var):
        build_training_dataset(dataset_name, "train", modality, cfg)


def test_layout_resolution_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Path] = {}

    def fake_build(
        layout: DatasetLayout, *, split: str, cfg: dict[str, object], db_path: Path
    ) -> object:
        seen[layout.dataset_name] = db_path
        return object()

    monkeypatch.setenv("CT_DATAPATH", str(tmp_path))
    monkeypatch.setenv("XRAY_DATAPATH", str(tmp_path))
    monkeypatch.setenv("GENERATED_DATAPATH", str(tmp_path))
    monkeypatch.setattr(dataset_builders, "_build_dataset_from_layout", fake_build)

    build_training_dataset("MOOSE", "train", "CT", {"version": "2.0"})
    build_training_dataset("ElbowCT", "train", "CT", {"version": "2.0"})
    build_training_dataset("HipRay", "train", "Xray", {})
    build_training_dataset("MURA_FOREARM", "train", "Xray", {})
    build_training_dataset("MURA_HUMERUS", "train", "Xray", {})
    build_training_dataset("FluXray", "train", "Xray", {"version": "7.0"})

    assert seen["MOOSE"] == tmp_path / "MOOSE/thunder_moose/3D/all_categories/2.0"
    assert seen["ElbowCT"] == tmp_path / "ElbowCT/thunder_dbs/2.0"
    assert seen["HipRay"] == tmp_path / "HipRay"
    assert seen["MURA_FOREARM"] == tmp_path / "MURA_FOREARM"
    assert seen["MURA_HUMERUS"] == tmp_path / "MURA_HUMERUS"
    assert seen["FluXray"] == tmp_path / "FluXray/thunder_dbs/7.0"


@pytest.mark.parametrize("key", ["version", "resolution", "require_seg"])
def test_xray_builder_rejects_stale_builtin_config_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    key: str,
) -> None:
    monkeypatch.setenv("CT_DATAPATH", str(tmp_path))
    monkeypatch.setenv("XRAY_DATAPATH", str(tmp_path))
    monkeypatch.setenv("GENERATED_DATAPATH", str(tmp_path))

    with pytest.raises(ValueError, match=key):
        build_training_dataset("HipRay", "train", "Xray", {key: "unused"})


@pytest.mark.parametrize(
    ("dataset_name", "modality", "cfg", "match"),
    [
        ("MOOSE", "CT", {}, "version"),
        ("FluXray", "Xray", {}, "version"),
    ],
)
def test_required_layout_fields_are_validated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dataset_name: str,
    modality: str,
    cfg: dict[str, object],
    match: str,
) -> None:
    monkeypatch.setenv("CT_DATAPATH", str(tmp_path))
    monkeypatch.setenv("XRAY_DATAPATH", str(tmp_path))
    monkeypatch.setenv("GENERATED_DATAPATH", str(tmp_path))

    with pytest.raises(ValueError, match=match):
        build_training_dataset(dataset_name, "train", modality, cfg)


@pytest.mark.parametrize("key", ["root", "data_root", "_class"])
def test_builder_rejects_private_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    key: str,
) -> None:
    monkeypatch.setenv("CT_DATAPATH", str(tmp_path))
    monkeypatch.setenv("XRAY_DATAPATH", str(tmp_path))
    monkeypatch.setenv("GENERATED_DATAPATH", str(tmp_path))

    with pytest.raises(ValueError, match=key):
        build_training_dataset(
            "HipRay",
            "train",
            "Xray",
            {key: "/tmp/data"},
        )


def test_builder_accepts_absolute_path_for_custom_package(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An explicit package path makes registry setup unnecessary."""

    seen: dict[str, object] = {}

    def fake_build(
        layout: DatasetLayout, *, split: str, cfg: dict[str, object], db_path: Path
    ) -> object:
        seen.update(layout=layout, split=split, cfg=cfg, db_path=db_path)
        return object()

    package_path = tmp_path / "MyDataset"
    monkeypatch.setattr(dataset_builders, "_build_dataset_from_layout", fake_build)

    result = build_training_dataset(
        "MyDataset", "train", "Xray", {"path": str(package_path)}
    )

    assert result is not None
    assert seen["split"] == "train"
    assert seen["cfg"] == {}
    assert seen["db_path"] == package_path
    layout = seen["layout"]
    assert isinstance(layout, DatasetLayout)
    assert layout.dataset_name == "MyDataset"
    assert layout.modality == "xray"


def test_builder_rejects_relative_package_path() -> None:
    """Direct paths are absolute so run configs are location independent."""

    with pytest.raises(ValueError, match="must be absolute"):
        build_training_dataset(
            "MyDataset", "train", "Xray", {"path": "relative/database"}
        )


def test_builder_confines_layout_selectors_to_dataset_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CT_DATAPATH", str(tmp_path))

    with pytest.raises(ValueError, match="outside its dataset root"):
        build_training_dataset(
            "MOOSE",
            "train",
            "CT",
            {"version": "../../../../../escape"},
        )


def test_builder_rejects_model_label_mode_before_storage_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_open(path: Path) -> object:
        raise AssertionError("storage should not be opened")

    monkeypatch.setenv("CT_DATAPATH", str(tmp_path))
    monkeypatch.setenv("XRAY_DATAPATH", str(tmp_path))
    monkeypatch.setenv("GENERATED_DATAPATH", str(tmp_path))
    monkeypatch.setattr(dataset_builders, "_open_database", fail_open)

    with pytest.raises(ValueError, match="label_mode='native'"):
        build_training_dataset(
            "HipRay",
            "train",
            "Xray",
            {"label_mode": "model"},
        )


def test_unknown_dataset_names_include_registration_guidance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CT_DATAPATH", str(tmp_path))
    monkeypatch.setenv("XRAY_DATAPATH", str(tmp_path))
    monkeypatch.setenv("GENERATED_DATAPATH", str(tmp_path))

    with pytest.raises(KeyError, match="register_dataset_layout"):
        build_training_dataset("NoSuchDataset", "train", "Xray", {})


def test_ct_builder_emits_native_dataset_with_crop_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CT_DATAPATH", str(tmp_path))
    monkeypatch.setenv("XRAY_DATAPATH", str(tmp_path))
    monkeypatch.setenv("GENERATED_DATAPATH", str(tmp_path))
    monkeypatch.setattr(dataset_builders, "_open_database", lambda path: _ct_db())

    dataset = build_training_dataset(
        "ElbowCT",
        "train",
        "CT",
        {"version": "2.0", "crop_mode": "random_slices", "crop_size": [2, 2, 2]},
    )
    sample = dataset[0]

    assert isinstance(dataset, CTTrainingDataset)
    assert dataset.label_mode == "native"
    assert dataset.crop_mode == "random_slices"
    assert dataset.crop_size == (2, 2, 2)
    assert sample["label"].dtype == torch.long
    assert sample["image"].shape == (1, 2, 2, 2)
    assert "affine" in sample["metadata"]
    assert "spacing" in sample["metadata"]


def test_xray_builder_emits_native_dataset_with_required_seg_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CT_DATAPATH", str(tmp_path))
    monkeypatch.setenv("XRAY_DATAPATH", str(tmp_path))
    monkeypatch.setenv("GENERATED_DATAPATH", str(tmp_path))
    monkeypatch.setattr(dataset_builders, "_open_database", lambda path: _xray_db())

    dataset = build_training_dataset("HipRay", "train", "Xray", {})
    sample = dataset[0]

    assert isinstance(dataset, XrayTrainingDataset)
    assert dataset.label_mode == "native"
    assert dataset.require_seg is True
    assert sample["label"].dtype == torch.long


@pytest.mark.parametrize(
    ("attrs", "match"),
    [
        ({"dataset_name": "OtherRay", "clean_dataset": True}, "dataset identity"),
        ({"dataset": "OtherRay"}, "dataset identity"),
        ({"dataset_name": "HipRay", "clean_dataset": False}, "clean_dataset"),
    ],
)
def test_builder_validates_dataset_identity_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    attrs: dict[str, object],
    match: str,
) -> None:
    db = _xray_db()
    db["_attrs"] = attrs
    monkeypatch.setenv("CT_DATAPATH", str(tmp_path))
    monkeypatch.setenv("XRAY_DATAPATH", str(tmp_path))
    monkeypatch.setenv("GENERATED_DATAPATH", str(tmp_path))
    monkeypatch.setattr(dataset_builders, "_open_database", lambda path: db)

    with pytest.raises(ValueError, match=match):
        build_training_dataset(
            "HipRay", "train", "Xray", {}
        )


def test_builder_accepts_legacy_dataset_attr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db = _xray_db()
    db["_attrs"] = {"dataset": "HipRay", "img_key": "img", "seg_key": "seg"}
    monkeypatch.setenv("CT_DATAPATH", str(tmp_path))
    monkeypatch.setenv("XRAY_DATAPATH", str(tmp_path))
    monkeypatch.setenv("GENERATED_DATAPATH", str(tmp_path))
    monkeypatch.setattr(dataset_builders, "_open_database", lambda path: db)

    dataset = build_training_dataset("HipRay", "train", "Xray", {})

    assert len(dataset) == 1


def test_fluxray_builder_emits_xray_dataset_with_attrs_label_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CT_DATAPATH", str(tmp_path))
    monkeypatch.setenv("XRAY_DATAPATH", str(tmp_path))
    monkeypatch.setenv("GENERATED_DATAPATH", str(tmp_path))
    monkeypatch.setattr(
        dataset_builders, "_open_database", lambda path: _fluxray_db()
    )

    dataset = build_training_dataset(
        "FluXray",
        "train",
        "Xray",
        {"version": "7.0"},
    )
    sample = dataset[0]

    assert isinstance(dataset, XrayTrainingDataset)
    assert dataset.label_mode == "native"
    assert dataset.records[0].label_names == ("background", "clavicles")
    assert sample["label"].shape == (2, 2, 2)


def test_dense_xray_attrs_label_names_remain_dense(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database = _xray_db()
    attrs = database["_attrs"]
    assert isinstance(attrs, dict)
    attrs["label_names"] = ["background", "femurs"]
    monkeypatch.setenv("XRAY_DATAPATH", str(tmp_path))
    monkeypatch.setattr(dataset_builders, "_open_database", lambda path: database)

    dataset = build_training_dataset("HipRay", "train", "Xray", {})
    sample = dataset[0]

    assert sample["label"].shape == (2, 3)
    assert sample["label"].dtype == torch.long
    assert dataset.records[0].label_names == ("background", "femurs")


def test_build_named_and_multimodal_datasets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_open(path: Path) -> dict[str, object]:
        return _ct_db() if "ElbowCT" in str(path) else _xray_db()

    monkeypatch.setenv("CT_DATAPATH", str(tmp_path))
    monkeypatch.setenv("XRAY_DATAPATH", str(tmp_path))
    monkeypatch.setenv("GENERATED_DATAPATH", str(tmp_path))
    monkeypatch.setattr(dataset_builders, "_open_database", fake_open)
    data_cfg = {
        "data": {
            "CT": {"ElbowCT": {"version": "2.0"}},
            "Xray": {"HipRay": {}},
        }
    }

    xray = build_named_datasets(data_cfg, "train", "Xray")
    bundle = build_multimodal_datasets(data_cfg)

    assert set(xray) == {"HipRay"}
    assert isinstance(bundle, TrainingDatasetBundle)
    assert set(bundle.train) == {"ElbowCT", "HipRay"}
    assert set(bundle.val) == {"ElbowCT", "HipRay"}
    assert bundle.modalities == {"ElbowCT": "ct", "HipRay": "xray"}


def test_multimodal_bundle_closes_each_shared_reader_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    databases = {
        name: _ClosableDatabase(_xray_db(dataset_name=name))
        for name in ("HipRay", "JRST")
    }

    def fake_open(path: Path) -> _ClosableDatabase:
        """Return the reader matching the dataset path.

        Args:
            path: Resolved built-in dataset path.

        Returns:
            Matching closable database.
        """

        name = "JRST" if "JRST" in str(path) else "HipRay"
        return databases[name]

    monkeypatch.setenv("XRAY_DATAPATH", str(tmp_path))
    monkeypatch.setattr(dataset_builders, "_open_database", fake_open)

    with build_multimodal_datasets(
        {"Xray": {"HipRay": {}, "JRST": {}}}
    ) as bundle:
        assert not bundle.closed
        assert bundle.train["HipRay"].backend.db is bundle.val["HipRay"].backend.db
        assert bundle.train["JRST"].backend.db is bundle.val["JRST"].backend.db

    assert bundle.closed
    assert [database.close_calls for database in databases.values()] == [1, 1]
    bundle.close()
    assert [database.close_calls for database in databases.values()] == [1, 1]


def test_multimodal_builder_closes_prior_readers_when_later_source_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = _ClosableDatabase(_xray_db(dataset_name="HipRay"))
    second_values = _xray_db(dataset_name="JRST")
    second_splits = second_values["_splits"]
    assert isinstance(second_splits, dict)
    second_splits.pop("val")
    second = _ClosableDatabase(second_values)

    def fake_open(path: Path) -> _ClosableDatabase:
        """Return a valid first reader and invalid second reader.

        Args:
            path: Resolved built-in dataset path.

        Returns:
            Reader selected from the path.
        """

        return second if "JRST" in str(path) else first

    monkeypatch.setenv("XRAY_DATAPATH", str(tmp_path))
    monkeypatch.setattr(dataset_builders, "_open_database", fake_open)

    with pytest.raises(KeyError, match="missing required split"):
        build_multimodal_datasets({"Xray": {"HipRay": {}, "JRST": {}}})

    assert first.close_calls == 1
    assert second.close_calls == 1


def test_single_dataset_builder_exposes_idempotent_reader_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database = _ClosableDatabase(_xray_db())
    monkeypatch.setenv("XRAY_DATAPATH", str(tmp_path))
    monkeypatch.setattr(dataset_builders, "_open_database", lambda path: database)

    dataset = build_training_dataset("HipRay", "train", "Xray", {})
    dataset.close()
    dataset.close()

    assert database.close_calls == 1


def test_single_dataset_builder_closes_reader_once_when_backend_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database = _ClosableDatabase({})
    monkeypatch.setenv("XRAY_DATAPATH", str(tmp_path))
    monkeypatch.setattr(dataset_builders, "_open_database", lambda path: database)

    with pytest.raises(KeyError, match="_attrs"):
        build_training_dataset("HipRay", "train", "Xray", {})

    assert database.close_calls == 1


def test_single_dataset_builder_closes_reader_on_base_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database = _ClosableDatabase(_xray_db())
    monkeypatch.setenv("XRAY_DATAPATH", str(tmp_path))
    monkeypatch.setattr(dataset_builders, "_open_database", lambda path: database)

    def interrupt_dataset(**kwargs):
        """Simulate interruption after the backend adopts the owned reader."""
        del kwargs
        raise KeyboardInterrupt

    monkeypatch.setattr(dataset_builders, "XrayTrainingDataset", interrupt_dataset)
    with pytest.raises(KeyboardInterrupt):
        build_training_dataset("HipRay", "train", "Xray", {})

    assert database.close_calls == 1


def test_real_package_passes_readiness_and_shares_train_val_reader(
    tmp_path: Path,
) -> None:
    package_path = _pack_xray_package(tmp_path)
    config = {
        "experiment": {"_class": "fxr.experiment.FleXrayTrainExperiment"},
        "train": {"epochs": 1},
        "protocol": {"name": PROTOCOL_NAME},
        "dataloader": {"batch_size": 1, "num_workers": 0},
        "data": {"Xray": {"ClinicXray": {"path": str(package_path)}}},
        "log": {"root": str(tmp_path / "runs")},
    }

    validate_training_config(config, smoke_data=True)
    bundle = build_multimodal_datasets(config)
    database = bundle.train["ClinicXray"].backend.db
    try:
        assert database is bundle.val["ClinicXray"].backend.db
        assert len(bundle.train["ClinicXray"]) == 1
        assert len(bundle.val["ClinicXray"]) == 1
    finally:
        bundle.close()


def test_real_package_rejects_configured_modality_mismatch(tmp_path: Path) -> None:
    package_path = _pack_xray_package(tmp_path)

    with pytest.raises(ValueError, match="declares modality"):
        build_training_dataset(
            "ClinicXray",
            "train",
            "CT",
            {"path": str(package_path)},
        )


def test_multimodal_builder_requires_nonempty_train_and_val(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database = _xray_db()
    splits = database["_splits"]
    assert isinstance(splits, dict)
    splits["val"] = []
    monkeypatch.setenv("XRAY_DATAPATH", str(tmp_path))
    monkeypatch.setattr(dataset_builders, "_open_database", lambda path: database)

    with pytest.raises(ValueError, match="contain at least one sample"):
        build_multimodal_datasets({"Xray": {"HipRay": {}}})


def test_multimodal_builder_allows_train_only_when_validation_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database = _ClosableDatabase(_xray_db())
    splits = database["_splits"]
    assert isinstance(splits, dict)
    splits.pop("val")
    monkeypatch.setenv("XRAY_DATAPATH", str(tmp_path))
    monkeypatch.setattr(dataset_builders, "_open_database", lambda path: database)

    with build_multimodal_datasets(
        {"Xray": {"HipRay": {}}},
        include_validation=False,
    ) as bundle:
        assert set(bundle.train) == {"HipRay"}
        assert bundle.val == {}
        assert bundle.modalities == {"HipRay": "xray"}

    assert database.close_calls == 1


def test_readiness_accepts_train_only_package_when_validation_is_disabled(
    tmp_path: Path,
) -> None:
    package_path = _pack_xray_package(tmp_path, splits=("train",))
    config = {
        "experiment": {"_class": "fxr.experiment.FleXrayTrainExperiment"},
        "train": {"epochs": 1, "eval_freq": 0},
        "protocol": {"name": PROTOCOL_NAME},
        "dataloader": {"batch_size": 1, "num_workers": 0},
        "data": {"Xray": {"ClinicXray": {"path": str(package_path)}}},
        "callbacks": {},
        "log": {"root": str(tmp_path / "runs")},
    }

    validate_training_config(config, smoke_data=True)


def test_callback_local_eval_package_requires_only_its_requested_split(
    tmp_path: Path,
) -> None:
    training_root = tmp_path / "training"
    callback_root = tmp_path / "callback"
    training_root.mkdir()
    callback_root.mkdir()
    training_path = _pack_xray_package(
        training_root, dataset_name="ClinicTrain", splits=("train",)
    )
    callback_path = _pack_xray_package(
        callback_root, dataset_name="ClinicEval", splits=("holdout",)
    )
    config = {
        "experiment": {"_class": "fxr.experiment.FleXrayTrainExperiment"},
        "train": {"epochs": 1, "eval_freq": 0},
        "protocol": {"name": PROTOCOL_NAME},
        "dataloader": {"batch_size": 1, "num_workers": 0},
        "data": {"Xray": {"ClinicTrain": {"path": str(training_path)}}},
        "callbacks": {
            "epoch": {
                "eval": {
                    "_class": "fxr.experiment.EvalSetMetricLogger",
                    "split": "holdout",
                    "data": {
                        "Xray": {"ClinicEval": {"path": str(callback_path)}}
                    },
                }
            }
        },
        "log": {"root": str(tmp_path / "runs")},
    }

    validate_training_config(config, smoke_data=True)


def test_training_source_sample_callback_requires_validation_split(
    tmp_path: Path,
) -> None:
    package_path = _pack_xray_package(tmp_path, splits=("train",))
    config = {
        "experiment": {"_class": "fxr.experiment.FleXrayTrainExperiment"},
        "train": {"epochs": 1, "eval_freq": 0},
        "protocol": {"name": PROTOCOL_NAME},
        "dataloader": {"batch_size": 1, "num_workers": 0},
        "data": {"Xray": {"ClinicXray": {"path": str(package_path)}}},
        "callbacks": {
            "epoch": {
                "samples": {
                    "_class": "fxr.experiment.WandbSamplePredictionLogger",
                    "every": 1,
                }
            }
        },
        "log": {"root": str(tmp_path / "runs")},
    }

    with pytest.raises(ValueError, match="missing required split"):
        validate_training_config(config)


def test_readiness_rejects_real_package_without_validation_split(
    tmp_path: Path,
) -> None:
    package_path = _pack_xray_package(tmp_path, splits=("train",))
    config = {
        "experiment": {"_class": "fxr.experiment.FleXrayTrainExperiment"},
        "train": {"epochs": 1},
        "protocol": {"name": PROTOCOL_NAME},
        "dataloader": {"batch_size": 1, "num_workers": 0},
        "data": {"Xray": {"ClinicXray": {"path": str(package_path)}}},
        "log": {"root": str(tmp_path / "runs")},
    }

    with pytest.raises(ValueError, match="missing required split"):
        validate_training_config(config, smoke_data=True)


def test_duplicate_dataset_names_across_modalities_raise() -> None:
    with pytest.raises(ValueError, match="unique across modalities"):
        build_multimodal_datasets(
            {
                "data": {
                    "CT": {"HipRay": {"version": "1.0"}},
                    "Xray": {"HipRay": {}},
                }
            }
        )


def test_register_dataset_layout_extends_registry() -> None:
    layout = DatasetLayout(
        dataset_name="UnitTestLayout",
        modality="xray",
        relative_path_template="UnitTest/{version}",
    )

    register_dataset_layout(layout, replace=True)
    with pytest.raises(ValueError, match="already registered"):
        register_dataset_layout(layout)
    register_dataset_layout(layout, replace=True)


def test_load_dataset_layouts_validates_yaml_shape() -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        load_dataset_layouts(
            {
                "layouts": [
                    {
                        "dataset_name": "Bad",
                        "modality": "xray",
                        "relative_path_template": "Bad/{version}",
                        "extra": True,
                    }
                ]
            }
        )


def test_packaged_dataset_specs_compile_against_the_training_protocol() -> None:
    config_names = _packaged_dataset_config_names()

    assert config_names
    for public_name in config_names:
        spec = load_dataset_spec_by_name(public_name)
        lut = compile_training_lut_by_name(PROTOCOL_NAME, public_name)

        assert lut.dataset_name == spec.dataset_name
        assert spec.stored_labels[0] == "background"
        assert len(lut.label_lut) == max(spec.stored_labels) + 1
        for native_id, protocol_id in lut.native_id_to_protocol_id.items():
            assert lut.label_lut[native_id] == protocol_id
