"""Crop-backed CT packaging: split volumes into fixed-depth axial crops.

The released FleXray recipe trains on CT volumes stored as overlapping
``512x512x256`` crops so one training sample fits in memory and can be rendered
into DRRs immediately. Crops are taken along the last (axial, ``z``) axis only:
the top and bottom crops are always included, intermediate crops are spread
evenly, and volumes shorter than a crop are padded symmetrically with air.
Every crop stores the label ids and voxel centroids of its foreground so
training can weight rare anatomy and place DRR isocenters without re-reading
the volume.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

AIR_HU = -1000.0


@dataclass(frozen=True)
class CropPlan:
    """Manifest ``crops`` block for ``ct-seg`` packages.

    Attributes:
        size: Required ``(x, y, z)`` crop size; ``x`` and ``y`` must equal the
            source volume extents (crops are axial only).
        max_overlap: Overlap target in [0, 1). Neighbouring crops are spaced
            at most ``ceil(size[2] * (1 - max_overlap))`` slices apart.
            The default ``0.0`` selects the minimal cover; larger values
            request denser crops. Despite its name, this is not an upper
            bound on the actual overlap.
    """

    size: tuple[int, int, int]
    max_overlap: float = 0.0

    @classmethod
    def from_manifest(cls, raw: Mapping[str, Any]) -> "CropPlan":
        """Validate a manifest ``crops`` block.

        Args:
            raw: Mapping with ``size`` and optional ``max_overlap``.

        Returns:
            The validated crop plan.
        """

        assert isinstance(raw, Mapping), "crops must be a mapping."
        unknown = sorted(set(raw) - {"size", "max_overlap"})
        assert not unknown, f"crops has unknown key(s): {unknown}."
        size = tuple(raw.get("size", ()))
        assert len(size) == 3 and all(
            isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in size
        ), f"crops.size must be three positive integers (x, y, z); got {raw.get('size')!r}."
        overlap = float(raw.get("max_overlap", 0.0))
        assert 0.0 <= overlap < 1.0, f"crops.max_overlap must be in [0, 1); got {overlap}."
        return cls(size=(int(size[0]), int(size[1]), int(size[2])), max_overlap=overlap)

    def to_attrs(self) -> dict[str, Any]:
        """Return the JSON-compatible form stored in package ``_attrs``.

        Returns:
            Mapping with ``size`` and ``max_overlap``.
        """

        return {"size": list(self.size), "max_overlap": self.max_overlap}


@dataclass(frozen=True)
class CropSpec:
    """One planned axial crop of a source volume.

    Attributes:
        crop_index: Zero-based crop index within the source volume.
        z_start: First source slice included in the crop.
        z_stop: One past the last source slice included in the crop.
        z_pad_before: Air slices prepended when the volume is shorter than a crop.
        z_pad_after: Air slices appended when the volume is shorter than a crop.
        z_offset: Voxel offset of the crop origin relative to the source volume.
    """

    crop_index: int
    z_start: int
    z_stop: int
    z_pad_before: int
    z_pad_after: int

    @property
    def z_offset(self) -> int:
        """Voxel offset of the crop origin relative to the source volume.

        Returns:
            ``z_start - z_pad_before`` (negative when padding was added).
        """

        return self.z_start - self.z_pad_before


def plan_z_crops(depth: int, crop_depth: int, *, max_overlap: float = 0.0) -> tuple[CropSpec, ...]:
    """Plan axial crop windows covering a volume of ``depth`` slices.

    The first and last crops touch the volume ends. Intermediate starts are
    spread evenly with a maximum step of
    ``ceil(crop_depth * (1 - max_overlap))`` slices. The default ``0.0``
    selects the fewest crops that cover the volume; larger values request
    denser crops. Covering both ends can force extra overlap, and rounding
    the step up to whole slices can give slightly less than the target.
    Volumes shorter than a crop yield one symmetrically air-padded crop.

    Args:
        depth: Source volume depth in slices.
        crop_depth: Crop depth in slices.
        max_overlap: Overlap target in [0, 1); ``0.0`` selects the minimal cover.

    Returns:
        Ordered crop specifications.
    """

    assert depth > 0 and crop_depth > 0, "depth and crop_depth must be positive."
    if depth <= crop_depth:
        pad_before = (crop_depth - depth) // 2
        return (CropSpec(0, 0, depth, pad_before, crop_depth - depth - pad_before),)
    last_start = depth - crop_depth
    max_step = math.ceil(crop_depth * (1.0 - max_overlap))
    if last_start < max_step:
        starts = [0, last_start]
    else:
        intervals = math.ceil(last_start / max_step)
        starts = [round(index * last_start / intervals) for index in range(intervals + 1)]
        starts[0], starts[-1] = 0, last_start
    return tuple(
        CropSpec(index, start, start + crop_depth, 0, 0) for index, start in enumerate(starts)
    )


def crop_z(array: np.ndarray, spec: CropSpec, *, fill_value: float) -> np.ndarray:
    """Extract one axial crop, padding with ``fill_value`` where planned.

    Args:
        array: Source array shaped ``(x, y, z)``.
        spec: Crop window to extract.
        fill_value: Pad value (air HU for images, ``0`` for labels).

    Returns:
        Contiguous crop shaped ``(x, y, z_pad_before + window + z_pad_after)``.
    """

    window = array[..., spec.z_start : spec.z_stop]
    if spec.z_pad_before or spec.z_pad_after:
        pad = [(0, 0)] * (array.ndim - 1) + [(spec.z_pad_before, spec.z_pad_after)]
        window = np.pad(window, pad, constant_values=fill_value)
    return np.ascontiguousarray(window)


def crop_sample_id(sample_id: str, crop_index: int) -> str:
    """Return the packaged sample id of one crop.

    Args:
        sample_id: Source manifest sample id.
        crop_index: Zero-based crop index.

    Returns:
        ``"{sample_id}__crop{crop_index:03d}"``.
    """

    return f"{sample_id}__crop{crop_index:03d}"


def crop_metadata(
    spec: CropSpec,
    label_crop: np.ndarray,
    *,
    subject_id: str,
    source_sample_id: str,
) -> dict[str, Any]:
    """Describe one crop for the training runtime.

    Args:
        spec: Crop window.
        label_crop: Dense label crop shaped ``(x, y, z)``.
        subject_id: Subject owning the source volume.
        source_sample_id: Manifest sample id of the source volume.

    Returns:
        JSON-compatible metadata with crop geometry, ``crop_foreground_label_ids``,
        and aligned ``fg_centroids_ijk`` rows.
    """

    label_ids = [int(v) for v in np.unique(label_crop) if v != 0]
    centroids = [
        [float(c) for c in np.argwhere(label_crop == label_id).mean(axis=0)]
        for label_id in label_ids
    ]
    return {
        "subject_id": subject_id,
        "source_sample_id": source_sample_id,
        "crop_index": spec.crop_index,
        "z_start": spec.z_start,
        "z_stop": spec.z_stop,
        "z_pad_before": spec.z_pad_before,
        "z_pad_after": spec.z_pad_after,
        "affine_offset_xyz": [0, 0, spec.z_offset],
        "crop_foreground_label_ids": label_ids,
        "fg_centroids_ijk": centroids,
    }


def offset_affine(affine: np.ndarray, spec: CropSpec) -> np.ndarray:
    """Shift a voxel-to-world affine to a crop's voxel origin.

    Args:
        affine: Source ``(4, 4)`` affine.
        spec: Crop window.

    Returns:
        ``float32`` affine whose origin is the crop's first voxel.
    """

    shifted = np.array(affine, dtype=np.float64)
    shifted[:3, 3] += shifted[:3, :3] @ np.array([0.0, 0.0, float(spec.z_offset)])
    return shifted.astype(np.float32)
