"""Tests for native-to-model training label projection."""

from __future__ import annotations

import pytest
import torch

from fxr.datasets import compile_training_label_remap_by_name
from fxr.experiment import TrainingLabelProjection


def _hipray_projection() -> TrainingLabelProjection:
    remap = compile_training_label_remap_by_name("all_structures_flexray_v4", "HipRay")
    return TrainingLabelProjection.from_label_remap(remap)


def test_projection_maps_native_ids_to_model_channels():
    projection = _hipray_projection()
    # HipRay native ids {0,1,2} map to background, femurs, and hips.
    label = torch.tensor([[[0, 1], [2, 0]]])  # shape (1, 2, 2)

    projected = projection.project(label, source="test")

    assert projected.shape == (1, projection.num_classes, 2, 2)
    assert projected[0, projection.label_names.index("femurs"), 0, 1] == 1.0
    assert projected[0, projection.label_names.index("hips"), 1, 0] == 1.0
    # Background pixels are channel 0.
    assert projected[0, 0, 0, 0] == 1.0
    assert projected[0, 0, 1, 1] == 1.0


def test_projection_background_is_foreground_complement():
    projection = _hipray_projection()
    label = torch.tensor([[[1, 1], [1, 1]]])  # all foreground

    projected = projection.project(label, source="test")

    # No pixel is background when every pixel is a foreground class.
    assert torch.all(projected[0, 0] == 0.0)
    # Each foreground pixel is one-hot across channels.
    assert torch.all(projected.sum(dim=1) == 1.0)


def test_projection_rejects_unknown_native_id():
    projection = _hipray_projection()
    label = torch.tensor([[[0, 99]]])

    with pytest.raises(ValueError, match="unknown native label id"):
        projection.project(label, source="test")


def test_projection_accepts_channel_singleton_shape():
    projection = _hipray_projection()
    label = torch.zeros(2, 1, 4, 4)  # Bx1xHxW integer map of background

    projected = projection.project(label, source="test")

    assert projected.shape == (2, projection.num_classes, 4, 4)
    assert torch.all(projected[:, 0] == 1.0)


def test_rsnafrac_projection_collapse_map_uses_model_channels():
    remap = compile_training_label_remap_by_name(
        "all_structures_flexray_v4", "RSNAFrac"
    )
    projection = TrainingLabelProjection.from_label_remap(remap)

    collapse_map = projection.foreground_collapse_map()

    assert collapse_map[0] == projection.label_names.index("vertebra_c1")
    assert collapse_map[14] == projection.label_names.index("skull")
    assert projection.native_foreground_label_ids() == tuple(range(1, 16))


@pytest.mark.parametrize("dataset_name", ["MURA_FOREARM", "MURA_HUMERUS"])
def test_mura_projection_maps_stored_ids_to_protocol_channels(dataset_name):
    remap = compile_training_label_remap_by_name(
        "all_structures_flexray_v4", dataset_name
    )
    projection = TrainingLabelProjection.from_label_remap(remap)
    # Stored ids 1/2/3 are humeri/radii/ulnae.
    label = torch.tensor([[[1, 2], [3, 0]]])

    projected = projection.project(label, source="test")

    assert projected.shape == (1, projection.num_classes, 2, 2)
    assert projected[0, projection.label_names.index("humeri"), 0, 0] == 1.0
    assert projected[0, projection.label_names.index("radii"), 0, 1] == 1.0
    assert projected[0, projection.label_names.index("ulnae"), 1, 0] == 1.0
    assert projected[0, 0, 1, 1] == 1.0


def test_ct_validation_rejects_unknown_native_id_before_render():
    remap = compile_training_label_remap_by_name(
        "all_structures_flexray_v4", "RSNAFrac"
    )
    projection = TrainingLabelProjection.from_label_remap(remap)

    with pytest.raises(ValueError, match="unknown native label id"):
        projection.validated_native_ids(
            torch.tensor([[[99]]]), source="CT dataset 'RSNAFrac'", require_mapped=True
        )


def test_undeclared_native_ids_warn_but_declared_drops_remain_silent():
    import warnings

    projection = TrainingLabelProjection((0, -1, 0, 1), ("background", "bone"), 2)
    with pytest.warns(UserWarning, match=r"never declares: \[1\]"):
        projected = projection.project(torch.tensor([[0, 1, 2, 3]]), source="Toy")
    assert projected[0, 0].tolist() == [[1., 1., 1., 0.]]
    assert projected[0, 1].tolist() == [[0., 0., 0., 1.]]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        projection.project(torch.tensor([[0, 2, 3]]), source="Toy")
    assert not caught

    with pytest.raises(ValueError, match="missing from the remap LUT"):
        projection.validated_native_ids(
            torch.tensor([[1]]), source="Toy", require_mapped=True
        )
