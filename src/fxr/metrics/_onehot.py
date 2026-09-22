"""Private one-hot conversion and score-reduction helpers for ``fxr.metrics``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F

InputMode = Literal["binary", "multiclass", "onehot", "auto"]
Reduction = Literal["mean", "sum", "none"] | None
MetricIgnoreIndex = int | list[int] | tuple[int, ...] | None
MetricWeights = Tensor | list[float] | tuple[float, ...] | None


def resolve_mode(y_pred: Tensor, y_true: Tensor, mode: InputMode) -> str:
    """Resolve an explicit or automatic segmentation input mode."""

    if y_pred.ndim <= 2:
        raise ValueError("y_pred must have at least 3 dimensions.")
    if mode != "auto":
        return mode

    num_classes = y_pred.shape[1]
    if y_pred.shape == y_true.shape:
        return "binary" if num_classes == 1 else "onehot"
    return "multiclass"


def hard_onehot(x: Tensor, *, threshold: float = 0.5) -> Tensor:
    """Return a hard one-hot or binary-thresholded tensor.

    ``threshold`` applies to single-channel inputs only; multi-channel inputs
    are discretized with ``argmax`` whatever the channel count.
    """

    if x.shape[1] > 1:
        labels = x.argmax(dim=1)
        order = (0, x.ndim - 1, *range(1, x.ndim - 1))
        return F.one_hot(labels, num_classes=x.shape[1]).permute(order)

    return (x > float(threshold)).long()


def inputs_as_onehot(
    y_pred: Tensor,
    y_true: Tensor,
    *,
    mode: InputMode,
    from_logits: bool,
    discretize: bool = False,
    threshold: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """Convert segmentation predictions and targets to flattened one-hot tensors."""

    mode = resolve_mode(y_pred, y_true, mode)
    batch_size, num_classes = y_pred.shape[:2]

    if from_logits:
        if mode == "binary":
            y_pred = torch.sigmoid(y_pred)
        else:
            y_pred = torch.softmax(y_pred, dim=1)

    if mode in {"binary", "onehot"}:
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"{mode!r} segmentation expects targets with the same shape as predictions; "
                f"got {tuple(y_true.shape)} != {tuple(y_pred.shape)}."
            )
        y_true = y_true.to(device=y_pred.device)
        if discretize:
            if mode == "binary":
                y_pred = (y_pred > float(threshold)).to(dtype=y_pred.dtype)
                y_true = (y_true > 0.5).to(dtype=y_pred.dtype)
            else:
                y_pred = hard_onehot(y_pred, threshold=threshold).to(dtype=y_pred.dtype)
                y_true = hard_onehot(y_true, threshold=0.5).to(dtype=y_pred.dtype)
        else:
            y_true = y_true.to(dtype=y_pred.dtype)
        return (
            y_pred.reshape(batch_size, num_classes, -1).float(),
            y_true.reshape(batch_size, num_classes, -1).float(),
        )

    if y_true.ndim == y_pred.ndim:
        if y_true.shape[1] != 1:
            raise ValueError(
                "Multiclass segmentation expects dense labels with no class "
                "axis or a singleton class axis."
            )
        y_true = y_true.squeeze(1)

    y_true = y_true.to(device=y_pred.device).long()
    if discretize:
        y_pred = hard_onehot(y_pred, threshold=threshold)

    y_pred = y_pred.reshape(batch_size, num_classes, -1)
    y_true = y_true.reshape(batch_size, -1)
    y_true = F.one_hot(y_true, num_classes=num_classes).permute(0, 2, 1)
    return y_pred.float(), y_true.float()


def inputs_as_spatial_onehot(
    y_pred: Tensor,
    y_true: Tensor,
    *,
    mode: InputMode,
    threshold: float,
    from_logits: bool,
) -> tuple[Tensor, Tensor]:
    """Convert inputs to hard one-hot tensors with spatial dimensions preserved."""

    spatial_shape = y_pred.shape[2:]
    y_pred_flat, y_true_flat = inputs_as_onehot(
        y_pred,
        y_true,
        mode=mode,
        from_logits=from_logits,
        discretize=True,
        threshold=threshold,
    )
    batch_size, num_classes = y_pred_flat.shape[:2]
    return (
        y_pred_flat.reshape(batch_size, num_classes, *spatial_shape),
        y_true_flat.reshape(batch_size, num_classes, *spatial_shape),
    )


def coerce_metric_weights(weights: MetricWeights, reference: Tensor) -> Tensor | None:
    """Broadcast class or batch/class metric weights to a score tensor."""

    if weights is None:
        return None

    if isinstance(weights, Tensor):
        weights = weights.to(device=reference.device, dtype=reference.dtype)
    elif isinstance(weights, Sequence) and not isinstance(weights, (str, bytes)):
        weights = torch.tensor(
            list(weights),
            device=reference.device,
            dtype=reference.dtype,
        )
    else:
        raise TypeError("weights must be a tensor or a numeric sequence.")

    if weights.ndim == 1:
        if weights.shape[0] != reference.shape[1]:
            raise ValueError(
                "weights must match the number of channels; "
                f"got {weights.shape[0]} != {reference.shape[1]}."
            )
        weights = weights.unsqueeze(0).expand(reference.shape[0], -1)
    elif weights.ndim == 2:
        if weights.shape != reference.shape:
            raise ValueError(
                f"weights must match score shape; got {weights.shape} != {reference.shape}."
            )
    else:
        raise ValueError(
            "weights must be a 1D class vector or a 2D batch/class matrix."
        )

    return weights


def normalize_metric_ignore_index(ignore_index: MetricIgnoreIndex) -> tuple[int, ...]:
    """Return ignored class indices as a tuple of integers."""

    if ignore_index is None:
        return ()
    if isinstance(ignore_index, int):
        return (int(ignore_index),)
    if isinstance(ignore_index, Sequence) and not isinstance(ignore_index, (str, bytes)):
        return tuple(int(index) for index in ignore_index)
    raise TypeError("ignore_index must be an int, a sequence of ints, or None.")


def metric_reduction(
    scores: Tensor,
    *,
    reduction: Reduction,
    batch_reduction: Reduction,
    weights: MetricWeights = None,
    ignore_index: MetricIgnoreIndex = None,
) -> Tensor:
    """Reduce batch/class score tensors with optional weights and class ignores."""

    if scores.ndim != 2:
        raise ValueError(
            "Reducible score tensors must have batch and channel dimensions; "
            f"got shape {tuple(scores.shape)}."
        )

    batch_size, num_classes = scores.shape
    weights = coerce_metric_weights(weights, scores)

    ignored_indices = normalize_metric_ignore_index(ignore_index)
    if ignored_indices:
        invalid = [
            index for index in ignored_indices if not 0 <= int(index) < num_classes
        ]
        if invalid:
            raise ValueError("ignore_index must be in [0, channels).")
        ignore_weights = torch.ones_like(scores)
        ignore_weights[:, list(ignored_indices)] = 0.0
        weights = ignore_weights if weights is None else weights * ignore_weights

    if weights is None:
        weights = torch.ones(
            (batch_size, num_classes), device=scores.device, dtype=scores.dtype
        )

    weighted_scores = scores * weights.to(dtype=scores.dtype)
    included: Tensor | None = None
    if reduction == "mean":
        denominator = weights.sum(dim=1)
        numerator = weighted_scores.sum(dim=1)
        included = denominator > 0
        safe_denominator = torch.where(
            included, denominator, torch.ones_like(denominator)
        )
        scores = torch.where(included, numerator / safe_denominator, numerator * 0.0)
    elif reduction == "sum":
        scores = weighted_scores.sum(dim=1)
    elif reduction in {"none", None}:
        scores = weighted_scores
    else:
        raise ValueError(f"Unsupported reduction {reduction!r}.")

    if batch_reduction == "mean":
        if included is None:
            return scores.mean(dim=0)
        return scores.sum(dim=0) / included.sum().clamp_min(1)
    if batch_reduction == "sum":
        return scores.sum(dim=0)
    if batch_reduction in {"none", None}:
        return scores
    raise ValueError(f"Unsupported batch_reduction {batch_reduction!r}.")


def mask_background_when_foreground_present(
    scores: Tensor,
    weights: Tensor | None,
) -> Tensor:
    """Zero background weight for samples that retain any foreground class."""

    if weights is None:
        weights = torch.ones_like(scores)
    if scores.shape[1] <= 1:
        return weights

    non_bg_sum = weights.sum(dim=1) - weights[:, 0]
    has_foreground = non_bg_sum > 0
    weights = weights.clone()
    weights[:, 0] = weights[:, 0] * (~has_foreground).float()
    return weights


def flatten_metric_pixel_weights(
    pixel_weights: Tensor | None,
    *,
    reference: Tensor,
    spatial_ndim: int,
) -> Tensor | None:
    """Flatten optional pixel weights to match flattened one-hot metric inputs."""

    if pixel_weights is None:
        return None

    batch_size, num_classes, num_pixels = reference.shape
    pixel_weights = pixel_weights.to(device=reference.device, dtype=reference.dtype)
    if pixel_weights.shape[0] != batch_size:
        raise ValueError("pixel_weights batch dimension must match y_pred.")

    if pixel_weights.ndim == 2:
        flat_weights = pixel_weights.reshape(batch_size, 1, -1)
    elif pixel_weights.ndim == spatial_ndim + 1:
        flat_weights = pixel_weights.reshape(batch_size, 1, -1)
    elif pixel_weights.ndim == spatial_ndim + 2:
        flat_weights = pixel_weights.reshape(batch_size, pixel_weights.shape[1], -1)
    else:
        raise ValueError(
            "pixel_weights must include batch and spatial dimensions, with an "
            "optional channel dimension."
        )

    if flat_weights.shape[2] != num_pixels:
        raise ValueError(
            "pixel_weights spatial size must match the flattened prediction size; "
            f"got {flat_weights.shape[2]} != {num_pixels}."
        )
    if flat_weights.shape[1] == 1 and num_classes != 1:
        flat_weights = flat_weights.expand(-1, num_classes, -1)
    elif flat_weights.shape[1] != num_classes:
        raise ValueError("pixel_weights channel dimension must be 1 or match y_pred.")

    return flat_weights
