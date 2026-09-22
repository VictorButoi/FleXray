"""Internal collation helpers for segmentation training samples.

CT rendering consumes one volume at a time and requires structured affine,
spacing, and optional foreground-centroid metadata. Other metadata is deliberately
free-form and must not be passed through PyTorch's recursive default collator.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torch.utils.data._utils.collate import default_collate


def _ct_safe_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate one CT subject while preserving arbitrary metadata safely.

    Args:
        samples: Single-sample list emitted by a CT training dataset.

    Returns:
        Batch mapping whose image and label have a leading singleton batch
        dimension. ``metadata.affine`` has shape ``(1, 4, 4)``,
        ``metadata.spacing`` has shape ``(1, 3)``, and an optional
        ``metadata.fg_centroids_ijk`` has shape ``(1, N, 3)``. Centroid mappings
        additionally yield ``metadata.fg_centroid_label_ids`` shaped ``(1, N)``
        (``N`` may be zero for crops without foreground). All other metadata
        values are retained unchanged from the sole sample.

    Raises:
        ValueError: If the batch does not contain exactly one subject or a
            required geometric tensor has the wrong shape or non-finite values.
        TypeError: If CT metadata or foreground centroid ids are malformed.
        KeyError: If CT metadata lacks ``affine`` or ``spacing``.
    """

    if len(samples) != 1:
        raise ValueError(
            "CT DRR collation requires exactly one subject per loader batch."
        )
    sample = samples[0]
    metadata = sample.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError("CT samples require metadata to be a mapping.")

    stripped = {key: value for key, value in sample.items() if key != "metadata"}
    batch = default_collate([stripped])
    batch["image"], batch["label"] = _canonical_ct_image_and_label(batch)
    collated_metadata = dict(metadata)
    collated_metadata["affine"] = _batch_geometry_tensor(
        metadata,
        field="affine",
        shape=(4, 4),
    )
    collated_metadata["spacing"] = _batch_geometry_tensor(
        metadata,
        field="spacing",
        shape=(3,),
    )
    centroids = metadata.get("fg_centroids_ijk")
    if isinstance(centroids, Mapping):
        label_ids = sorted(_positive_label_id(key) for key in centroids)
        collated_metadata["fg_centroid_label_ids"] = torch.tensor(
            label_ids, dtype=torch.long
        ).unsqueeze(0)
        collated_metadata["fg_centroids_ijk"] = (
            _canonical_foreground_centroids(centroids).unsqueeze(0)
            if centroids
            else torch.zeros((1, 0, 3), dtype=torch.float32)
        )
    elif centroids is not None:
        collated_metadata["fg_centroids_ijk"] = (
            _canonical_foreground_centroids(centroids).unsqueeze(0)
        )
    batch["metadata"] = collated_metadata
    return batch


def _metadata_safe_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate 2D samples without recursively collating free-form metadata.

    Args:
        samples: X-ray samples whose tensor fields should be collated normally.

    Returns:
        Batch mapping with metadata retained as a list of per-sample values.
    """

    metadata = [sample.get("metadata") for sample in samples]
    stripped = [
        {key: value for key, value in sample.items() if key != "metadata"}
        for sample in samples
    ]
    batch = default_collate(stripped)
    if any(item is not None for item in metadata):
        batch["metadata"] = metadata
    return batch


def _owned_float_tensor(value: Any) -> Tensor:
    """Copy an array-like metadata value into an owned float tensor.

    Args:
        value: Tensor or array-like geometry metadata.

    Returns:
        Float tensor that does not alias the input storage.
    """

    if isinstance(value, Tensor):
        return value.detach().to(dtype=torch.float32).clone()
    return torch.tensor(value, dtype=torch.float32)


def _batch_geometry_tensor(
    metadata: Mapping[str, Any],
    *,
    field: str,
    shape: tuple[int, ...],
) -> Tensor:
    """Return one finite geometry value with a singleton batch dimension.

    Args:
        metadata: CT metadata mapping.
        field: Geometry field to read.
        shape: Required unbatched tensor shape.

    Returns:
        Float tensor shaped ``(1, *shape)``.

    Raises:
        KeyError: If ``field`` is absent.
        ValueError: If its value has the wrong shape or non-finite values.
    """

    if field not in metadata:
        raise KeyError(f"CT samples require metadata.{field}.")
    tensor = _owned_float_tensor(metadata[field])
    if tensor.ndim == len(shape) + 1 and int(tensor.shape[0]) == 1:
        tensor = tensor[0]
    if tuple(tensor.shape) != shape:
        raise ValueError(
            f"CT metadata.{field} must have shape {shape}; "
            f"got {tuple(tensor.shape)}."
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"CT metadata.{field} must contain only finite values.")
    return tensor.unsqueeze(0)


def _canonical_ct_image_and_label(batch: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
    """Validate and normalize the dense CT tensors in one collated batch.

    Args:
        batch: Default-collated single-subject batch.

    Returns:
        Contiguous float image and integer label tensors, each shaped
        ``(1, 1, D, H, W)``.

    Raises:
        KeyError: If ``image`` or ``label`` is missing.
        ValueError: If either tensor has an invalid shape, their spatial shapes
            differ, or the label contains non-integer floating-point values.
    """

    if "image" not in batch or "label" not in batch:
        raise KeyError("CT segmentation samples require image and label tensors.")
    image = torch.as_tensor(batch["image"]).float()
    label = torch.as_tensor(batch["label"])
    expected_prefix = (1, 1)
    if image.ndim != 5 or tuple(image.shape[:2]) != expected_prefix:
        raise ValueError(
            "Collated CT image must have shape (1, 1, D, H, W); "
            f"got {tuple(image.shape)}."
        )
    if label.ndim != 5 or tuple(label.shape[:2]) != expected_prefix:
        raise ValueError(
            "Collated dense CT label must have shape (1, 1, D, H, W); "
            f"got {tuple(label.shape)}."
        )
    if tuple(image.shape[-3:]) != tuple(label.shape[-3:]):
        raise ValueError("Collated CT image and label spatial shapes must match.")
    if label.is_floating_point():
        rounded = label.round()
        if not bool(torch.equal(label, rounded)):
            raise ValueError("Collated dense CT label must contain integer ids.")
        label = rounded
    return image.contiguous(), label.to(dtype=torch.long).contiguous()


def _canonical_foreground_centroids(value: Any) -> Tensor:
    """Convert centroid mappings or arrays to a deterministic ``(N, 3)`` tensor.

    Mapping entries are sorted by their positive integer foreground label id.
    The ids are used only to establish deterministic order; the DRR isocenter
    sampler consumes the resulting centroid coordinates.

    Args:
        value: Mapping from label ids to coordinates, or an array-like centroid
            matrix.

    Returns:
        Finite float tensor shaped ``(N, 3)``.

    Raises:
        TypeError: If mapping keys are not positive integer label ids.
        ValueError: If ids collide after normalization, the matrix is empty, or
            coordinates have an invalid shape or non-finite values.
    """

    if isinstance(value, Mapping):
        normalized: dict[int, Any] = {}
        for raw_label_id, coordinates in value.items():
            label_id = _positive_label_id(raw_label_id)
            if label_id in normalized:
                raise ValueError(
                    "fg_centroids_ijk contains duplicate label ids after "
                    f"normalization: {label_id}."
                )
            normalized[label_id] = coordinates
        value = [normalized[label_id] for label_id in sorted(normalized)]

    tensor = _owned_float_tensor(value)
    if tensor.ndim == 3 and int(tensor.shape[0]) == 1:
        tensor = tensor[0]
    if tensor.ndim != 2 or int(tensor.shape[1]) != 3:
        raise ValueError(
            "CT metadata.fg_centroids_ijk must have shape (N, 3); "
            f"got {tuple(tensor.shape)}."
        )
    if int(tensor.shape[0]) == 0:
        raise ValueError(
            "CT metadata.fg_centroids_ijk must contain at least one centroid."
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(
            "CT metadata.fg_centroids_ijk must contain only finite values."
        )
    return tensor


def _positive_label_id(value: Any) -> int:
    """Normalize one centroid mapping key to a positive integer label id.

    Args:
        value: Raw mapping key.

    Returns:
        Positive integer label id.

    Raises:
        TypeError: If ``value`` is not an integer or canonical integer string.
    """

    if isinstance(value, bool):
        raise TypeError("Foreground centroid label ids must be positive integers.")
    try:
        label_id = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Foreground centroid label ids must be positive integers."
        ) from exc
    if label_id < 1 or str(value).strip() not in {str(label_id), f"+{label_id}"}:
        raise TypeError("Foreground centroid label ids must be positive integers.")
    return label_id
