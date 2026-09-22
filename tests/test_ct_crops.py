"""Regression tests for configurable axial CT crop spacing."""

from __future__ import annotations

import pytest

from fxr.datasets.ct_crops import CropPlan, plan_z_crops


def test_crop_plan_defaults_preserve_minimal_cover() -> None:
    """Omitted overlap retains the historical crop starts and records zero."""

    plan = CropPlan.from_manifest({"size": [512, 512, 256]})
    assert plan == CropPlan(size=(512, 512, 256), max_overlap=0.0)
    assert CropPlan(size=(512, 512, 256)) == plan
    assert plan.to_attrs() == {"size": [512, 512, 256], "max_overlap": 0.0}
    assert [crop.z_start for crop in plan_z_crops(700, 256)] == [0, 222, 444]


@pytest.mark.parametrize(
    ("depth", "crop_depth", "overlap", "expected_starts"),
    [
        (20, 8, 0.0, [0, 6, 12]),
        (20, 8, 0.5, [0, 4, 8, 12]),
        (20, 8, 0.75, [0, 2, 4, 6, 8, 10, 12]),
        (16, 5, 0.5, [0, 3, 6, 8, 11]),  # 2.5-slice spacing rounds up to 3.
        (8, 3, 0.99, [0, 1, 2, 3, 4, 5]),  # At most one start per slice.
        (9, 8, 0.5, [0, 1]),  # Both ends force more overlap than requested.
        (12, 8, 0.5, [0, 4]),  # Remaining distance equals the spacing limit.
        (300, 256, 0.9, [0, 22, 44]),  # Short excess can need an interior crop.
    ],
)
def test_overlap_controls_crop_starts_and_preserves_coverage(
    depth: int, crop_depth: int, overlap: float, expected_starts: list[int]
) -> None:
    """Configured overlap controls density while crops cover both volume ends."""

    crops = plan_z_crops(depth, crop_depth, max_overlap=overlap)
    assert [crop.z_start for crop in crops] == expected_starts
    assert [crop.crop_index for crop in crops] == list(range(len(expected_starts)))
    assert crops[0].z_start == 0 and crops[-1].z_stop == depth
    for crop in crops:
        assert crop.z_stop - crop.z_start == crop_depth
        assert crop.z_pad_before == crop.z_pad_after == 0
    for left, right in zip(crops, crops[1:]):
        assert left.z_start < right.z_start <= left.z_stop


@pytest.mark.parametrize("overlap", [0.0, 0.5, 0.99])
@pytest.mark.parametrize(("depth", "expected_padding"), [(3, (2, 3)), (8, (0, 0))])
def test_overlap_keeps_one_crop_for_volumes_that_fit(
    overlap: float, depth: int, expected_padding: tuple[int, int]
) -> None:
    """A volume that fits one crop retains symmetric padding at every overlap."""

    (crop,) = plan_z_crops(depth, 8, max_overlap=overlap)
    assert (crop.z_start, crop.z_stop) == (0, depth)
    assert (crop.z_pad_before, crop.z_pad_after) == expected_padding
    assert crop.z_offset == -expected_padding[0]
