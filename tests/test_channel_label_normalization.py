"""Behavior tests for registered named-channel label normalization."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fxr.config import Config
from fxr.experiment import FleXrayTrainExperiment
from fxr.launch import readiness
from fxr.protocols import load_protocol_by_name

_PROTOCOL_NAME = "all_structures_flexray_v4"


def _channel_experiment(
    label_names: tuple[str, ...],
) -> FleXrayTrainExperiment:
    """Build a lightweight experiment around one named-channel HandBones source.

    Args:
        label_names: Ordered names exposed by the package backend.

    Returns:
        Uninitialized experiment populated for channel-name collection.
    """

    experiment = object.__new__(FleXrayTrainExperiment)
    experiment.config = Config({"protocol": {"name": _PROTOCOL_NAME}})
    dataset = SimpleNamespace(
        backend=SimpleNamespace(label_names=label_names),
        records=(),
    )
    experiment.train_datasets = {"HandBones": dataset}
    experiment.val_datasets = {"HandBones": dataset}
    experiment.modalities = {"HandBones": "xray"}
    return experiment


def test_handbones_channel_aliases_collapse_and_merge_by_maximum() -> None:
    source_names = (
        "background",
        "phalange_distal",
        "phalange_intermediate",
        "phalange_proximal",
    )
    experiment = _channel_experiment(source_names)
    protocol = load_protocol_by_name(_PROTOCOL_NAME)

    collected = experiment._collect_channel_source_names()

    assert collected["HandBones"] == (
        "background",
        "phalanges",
        "phalanges",
        "phalanges",
    )
    experiment._channel_source_names = collected
    experiment.model_label_names = protocol.labels
    source = torch.zeros(1, len(source_names), 2, 2)
    source[0, 1, 0, 0] = 0.4
    source[0, 2, 0, 0] = 0.9
    source[0, 3, 1, 1] = 1.0

    projected = experiment._project_channel_label(source, "HandBones")
    phalanges = projected[0, protocol.label_to_id["phalanges"]]

    assert phalanges[0, 0] == pytest.approx(0.9)
    assert phalanges[1, 1] == pytest.approx(1.0)


def test_registered_channel_source_preserves_canonical_protocol_subset() -> None:
    source_names = ("background", "phalanges", "heart")
    experiment = _channel_experiment(source_names)

    collected = experiment._collect_channel_source_names()

    assert collected["HandBones"] == source_names


def test_registered_channel_source_rejects_genuinely_unknown_name() -> None:
    experiment = _channel_experiment(("background", "unknown_hand_structure"))

    with pytest.raises(ValueError, match="unknown_hand_structure"):
        experiment._collect_channel_source_names()


def test_direct_package_readiness_accepts_registered_native_channel_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = SimpleNamespace(
        attrs={"protocol_name": _PROTOCOL_NAME},
        label_names=(
            "background",
            "phalange_distal",
            "phalange_intermediate",
            "phalange_proximal",
        ),
        db={"_splits": {"train": ["train-case"], "val": ["val-case"]}},
    )
    dataset = SimpleNamespace(backend=backend)
    monkeypatch.setattr(
        readiness,
        "build_training_dataset",
        lambda *args, **kwargs: dataset,
    )

    readiness._validate_packaged_protocol_remap(
        _PROTOCOL_NAME,
        "HandBones",
        "xray",
        {"path": "/unused/test-package"},
        model_label_names=None,
        config_root=None,
        required_splits=("train", "val"),
    )
