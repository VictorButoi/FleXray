"""Resolve canonical model inputs from a loader batch.

The X-ray path consumes ``image`` and ``label`` tensors with no affine. The CT
path additionally carries the voxel-to-world ``affine`` (and, for the
``random_label`` isocenter scheme, per-class ``fg_centroids_ijk``) under the
sample ``metadata``, which the DRR runtime needs to render the CT volume into
DRRs on the fly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class BatchInputs:
    """Canonical per-step model inputs resolved from a loader batch.

    Attributes:
        modality: Resolved modality, ``"xray"`` or ``"ct"``.
        image: X-ray image batch ``(B, 1, H, W)`` or CT volume
            ``(1, 1, D, H, W)``.
        label: Native X-ray integer labels or channel-first masks
            ``(B, C, H, W)``, or dense native CT labels.
        affine: Voxel-to-world affine ``(4, 4)`` for CT, ``None`` otherwise.
        fg_centroids_ijk: Per-class voxel centroids ``(N, 3)`` for CT when present,
            otherwise ``None``.
        fg_centroid_label_ids: Native label id ``(N,)`` of each centroid row when
            the CT batch carries them, otherwise ``None``.
    """

    modality: str
    image: Tensor
    label: Tensor
    affine: Tensor | None
    fg_centroids_ijk: Tensor | None
    fg_centroid_label_ids: Tensor | None = None


def resolve_batch_inputs(
    batch: Mapping[str, Any],
    *,
    modality: str | None,
) -> BatchInputs:
    """Resolve modality and model inputs from a loader batch.

    Args:
        batch: Collated batch mapping. X-ray batches carry ``image`` and
            ``label``; CT batches additionally carry ``metadata.affine`` and
            optionally ``metadata.fg_centroids_ijk``.
        modality: Modality label from the loader, or ``None`` to infer it from
            the batch.

    Returns:
        Resolved :class:`BatchInputs` for the X-ray or CT path.

    Raises:
        NotImplementedError: If the modality is not X-ray or CT.
        KeyError: If a CT batch lacks ``metadata.affine``.
    """

    name = str(modality if modality is not None else _infer_modality(batch)).lower()
    if name == "xray":
        return BatchInputs(
            "xray", batch["image"].float(), batch["label"].float(), None, None
        )
    if name == "ct":
        return _resolve_ct_inputs(batch)
    raise NotImplementedError(f"Unsupported batch modality {name!r}.")


def _resolve_ct_inputs(batch: Mapping[str, Any]) -> BatchInputs:
    """Resolve CT volume, dense label, affine, and centroids from a batch."""
    metadata = batch.get("metadata")
    if not isinstance(metadata, Mapping) or "affine" not in metadata:
        raise KeyError("CT batches require metadata.affine for DRR rendering.")
    affine = _strip_batch_dim(torch.as_tensor(metadata["affine"]).float(), ndim=2)
    fg_centroids = metadata.get("fg_centroids_ijk")
    if fg_centroids is not None:
        fg_centroids = _strip_batch_dim(torch.as_tensor(fg_centroids), ndim=2)
    label_ids = metadata.get("fg_centroid_label_ids")
    if label_ids is not None:
        label_ids = _strip_batch_dim(torch.as_tensor(label_ids), ndim=1)
    return BatchInputs(
        "ct", batch["image"].float(), batch["label"], affine, fg_centroids, label_ids
    )


def _strip_batch_dim(tensor: Tensor, *, ndim: int) -> Tensor:
    """Drop a leading singleton batch dim; CT rendering uses one subject per step."""
    if tensor.ndim == ndim + 1:
        if int(tensor.shape[0]) != 1:
            raise ValueError("CT DRR rendering requires a CT loader batch size of 1.")
        return tensor[0]
    return tensor


def _infer_modality(batch: Mapping[str, Any]) -> str:
    """Infer a single modality string from a batch ``modality`` field."""
    modality = batch.get("modality")
    if isinstance(modality, str):
        return modality
    if isinstance(modality, (list, tuple)) and modality:
        return str(modality[0])
    if "image" in batch:
        return "xray"
    raise KeyError("Could not infer modality from batch; expected an 'image' key.")
