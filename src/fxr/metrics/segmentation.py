from __future__ import annotations

import numpy as np
import torch
from pydantic import validate_call
from scipy import ndimage
from torch import Tensor

from fxr.metrics._onehot import (
    InputMode,
    MetricIgnoreIndex,
    MetricWeights,
    Reduction,
    coerce_metric_weights,
    flatten_metric_pixel_weights,
    inputs_as_onehot,
    inputs_as_spatial_onehot,
    mask_background_when_foreground_present,
    metric_reduction,
    normalize_metric_ignore_index,
)


def _reject_single_channel_ignore_index(
    y_pred: Tensor,
    ignore_index: MetricIgnoreIndex,
) -> None:
    """Reject class-index ignores for single-channel binary masks."""

    if y_pred.ndim > 1 and y_pred.shape[1] == 1:
        if normalize_metric_ignore_index(ignore_index):
            raise ValueError("ignore_index is not supported for binary segmentation.")


def _reject_non_finite_predictions(y_pred: Tensor) -> None:
    """Reject predictions that discretize into a plausible-looking score."""

    if not bool(torch.isfinite(y_pred).all()):
        raise ValueError(
            "y_pred contains non-finite values; discretizing them would score a "
            "diverged prediction as background."
        )


def _apply_empty_and_background_weights(
    scores: Tensor,
    true_amounts: Tensor,
    *,
    weights: MetricWeights,
    ignore_empty_labels: bool,
    ignore_background: bool,
) -> Tensor | None:
    """Build class weights after empty-label and background filtering."""

    class_weights = coerce_metric_weights(weights, scores)

    if ignore_empty_labels:
        existing_label = (true_amounts > 0).float()
        class_weights = (
            existing_label if class_weights is None else class_weights * existing_label
        )

    if ignore_background:
        class_weights = mask_background_when_foreground_present(scores, class_weights)

    return class_weights


@validate_call(config={"arbitrary_types_allowed": True})
def dice_score(
    y_pred: Tensor,
    y_true: Tensor,
    eps: float = 1e-7,
    smooth: float = 1e-7,
    threshold: float = 0.5,
    mode: InputMode = "auto",
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    from_logits: bool = False,
    ignore_empty_labels: bool = True,
    ignore_index: MetricIgnoreIndex = None,
    ignore_background: bool = False,
    weights: MetricWeights = None,
) -> Tensor:
    """Return hard Dice scores after discretizing segmentation predictions.

    Args:
        y_pred: Prediction tensor with shape ``(B, C, ...)``.
        y_true: Target tensor in the format selected by ``mode``.
        eps: Minimum denominator used for numerical stability.
        smooth: Additive smoothing term for numerator and denominator.
        threshold: Probability threshold for each independent channel in
            binary mode. Categorical multi-channel predictions use argmax.
        mode: Input interpretation: ``"binary"``, ``"onehot"``,
            ``"multiclass"``, or ``"auto"``.
        reduction: Channel reduction: ``"mean"``, ``"sum"``, ``"none"``, or
            ``None``.
        batch_reduction: Batch reduction with the same accepted values.
        from_logits: Whether ``y_pred`` contains logits instead of probabilities.
        ignore_empty_labels: Whether to zero target-empty class weights.
        ignore_index: Optional class index or indices to exclude.
        ignore_background: Whether to drop channel zero when foreground remains.
        weights: Optional class vector or batch/class weight matrix.

    Returns:
        Tensor containing reduced hard Dice scores.

    Raises:
        ValueError: If ``y_pred`` contains non-finite values.
    """

    _reject_single_channel_ignore_index(y_pred, ignore_index)
    _reject_non_finite_predictions(y_pred)
    y_pred, y_true = inputs_as_onehot(
        y_pred,
        y_true,
        mode=mode,
        from_logits=from_logits,
        discretize=True,
        threshold=threshold,
    )

    intersection = torch.logical_and(y_pred == 1.0, y_true == 1.0).sum(dim=-1)
    pred_amounts = (y_pred == 1.0).sum(dim=-1)
    true_amounts = (y_true == 1.0).sum(dim=-1)
    cardinalities = pred_amounts + true_amounts
    scores = (2 * intersection + smooth) / (cardinalities + smooth).clamp_min(eps)
    class_weights = _apply_empty_and_background_weights(
        scores,
        true_amounts,
        weights=weights,
        ignore_empty_labels=ignore_empty_labels,
        ignore_background=ignore_background,
    )

    return metric_reduction(
        scores,
        reduction=reduction,
        batch_reduction=batch_reduction,
        weights=class_weights,
        ignore_index=ignore_index,
    )


def _mask_surface(mask: np.ndarray) -> np.ndarray:
    """Return the binary surface voxels/pixels of a mask."""

    if not mask.any():
        return mask
    eroded = ndimage.binary_erosion(mask, border_value=0)
    return np.logical_and(mask, np.logical_not(eroded))


def _spatial_diagonal(spatial_shape: tuple[int, ...]) -> float:
    """Return the maximum spatial distance within an image grid."""

    return float(np.linalg.norm([max(int(size) - 1, 0) for size in spatial_shape]))


def _hd95_binary_masks(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    *,
    empty_penalty: float,
) -> float:
    """Return the symmetric 95th percentile Hausdorff distance for two masks.

    The two directional surface-distance sets are reduced separately and the
    larger 95th percentile wins, matching the FleXray paper's definition for
    nonempty masks. Pooling them into one percentile gives a different metric.
    """

    pred_present = bool(pred_mask.any())
    true_present = bool(true_mask.any())
    if not pred_present and not true_present:
        return 0.0
    if not pred_present or not true_present:
        return empty_penalty

    pred_surface = _mask_surface(pred_mask)
    true_surface = _mask_surface(true_mask)
    pred_to_true = ndimage.distance_transform_edt(~true_surface)[pred_surface]
    true_to_pred = ndimage.distance_transform_edt(~pred_surface)[true_surface]
    if pred_to_true.size == 0 or true_to_pred.size == 0:
        return 0.0
    return float(
        max(np.percentile(pred_to_true, 95), np.percentile(true_to_pred, 95))
    )


@validate_call(config={"arbitrary_types_allowed": True})
def hd95(
    y_pred: Tensor,
    y_true: Tensor,
    threshold: float = 0.5,
    mode: InputMode = "auto",
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    from_logits: bool = False,
    ignore_empty_labels: bool = True,
    ignore_index: MetricIgnoreIndex = None,
    ignore_background: bool = False,
    weights: MetricWeights = None,
) -> Tensor:
    """Return symmetric 95th percentile Hausdorff distance for segmentation masks.

    Args:
        y_pred: Prediction tensor with shape ``(B, C, ...)``.
        y_true: Target tensor in the format selected by ``mode``.
        threshold: Probability threshold for each independent channel in
            binary mode. Categorical multi-channel predictions use argmax.
        mode: Input interpretation: ``"binary"``, ``"onehot"``,
            ``"multiclass"``, or ``"auto"``.
        reduction: Channel reduction: ``"mean"``, ``"sum"``, ``"none"``, or
            ``None``.
        batch_reduction: Batch reduction with the same accepted values.
        from_logits: Whether ``y_pred`` contains logits instead of probabilities.
        ignore_empty_labels: Whether to zero target-empty class weights.
        ignore_index: Optional class index or indices to exclude.
        ignore_background: Whether to drop channel zero when foreground remains.
        weights: Optional class vector or batch/class weight matrix.

    Returns:
        Tensor containing reduced HD95 distances on the prediction device.
    """

    _reject_single_channel_ignore_index(y_pred, ignore_index)
    y_pred, y_true = inputs_as_spatial_onehot(
        y_pred,
        y_true,
        mode=mode,
        threshold=threshold,
        from_logits=from_logits,
    )

    spatial_shape = tuple(int(size) for size in y_pred.shape[2:])
    empty_penalty = _spatial_diagonal(spatial_shape)
    pred_np = y_pred.detach().cpu().numpy().astype(bool, copy=False)
    true_np = y_true.detach().cpu().numpy().astype(bool, copy=False)

    batch_size, num_classes = pred_np.shape[:2]
    hd95_scores = np.empty((batch_size, num_classes), dtype=np.float32)
    for batch_idx in range(batch_size):
        for class_idx in range(num_classes):
            hd95_scores[batch_idx, class_idx] = _hd95_binary_masks(
                pred_np[batch_idx, class_idx],
                true_np[batch_idx, class_idx],
                empty_penalty=empty_penalty,
            )

    scores = torch.as_tensor(hd95_scores, device=y_pred.device, dtype=torch.float32)
    true_amounts = y_true.reshape(batch_size, num_classes, -1).sum(dim=-1)
    class_weights = _apply_empty_and_background_weights(
        scores,
        true_amounts,
        weights=weights,
        ignore_empty_labels=ignore_empty_labels,
        ignore_background=ignore_background,
    )

    return metric_reduction(
        scores,
        reduction=reduction,
        batch_reduction=batch_reduction,
        weights=class_weights,
        ignore_index=ignore_index,
    )


@validate_call(config={"arbitrary_types_allowed": True})
def soft_dice_score(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    smooth: float = 1e-7,
    eps: float = 1e-7,
    square_denom: bool = True,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    ignore_empty_labels: bool = False,
    weights: MetricWeights = None,
    ignore_index: MetricIgnoreIndex = None,
    ignore_background: bool = False,
    from_logits: bool = False,
    pixel_weights: Tensor | None = None,
) -> Tensor:
    """Return soft Dice scores for probabilistic segmentation predictions.

    Args:
        y_pred: Prediction tensor with shape ``(B, C, ...)``.
        y_true: Target tensor in the format selected by ``mode``.
        mode: Input interpretation: ``"binary"``, ``"onehot"``,
            ``"multiclass"``, or ``"auto"``.
        smooth: Additive smoothing term for numerator and denominator.
        eps: Minimum denominator used for numerical stability.
        square_denom: Whether to square prediction and target denominator terms.
        reduction: Channel reduction: ``"mean"``, ``"sum"``, ``"none"``, or
            ``None``.
        batch_reduction: Batch reduction with the same accepted values.
        ignore_empty_labels: Whether to zero target-empty class weights.
        weights: Optional class vector or batch/class weight matrix.
        ignore_index: Optional class index or indices to exclude.
        ignore_background: Whether to drop channel zero when foreground remains.
        from_logits: Whether ``y_pred`` contains logits instead of probabilities.
        pixel_weights: Optional spatial or channel/spatial pixel weights.

    Returns:
        Tensor containing reduced soft Dice scores.
    """

    _reject_single_channel_ignore_index(y_pred, ignore_index)
    spatial_ndim = y_pred.ndim - 2
    y_pred, y_true = inputs_as_onehot(
        y_pred,
        y_true,
        mode=mode,
        from_logits=from_logits,
    )
    pixel_weights = flatten_metric_pixel_weights(
        pixel_weights,
        reference=y_pred,
        spatial_ndim=spatial_ndim,
    )
    pixel_weight_factor: Tensor | float = (
        1.0 if pixel_weights is None else pixel_weights
    )

    intersection = torch.sum(y_pred * y_true * pixel_weight_factor, dim=-1)
    if square_denom:
        pred_amounts = (y_pred.square() * pixel_weight_factor).sum(dim=-1)
        true_amounts = (y_true.square() * pixel_weight_factor).sum(dim=-1)
    else:
        pred_amounts = (y_pred * pixel_weight_factor).sum(dim=-1)
        true_amounts = (y_true * pixel_weight_factor).sum(dim=-1)

    cardinalities = pred_amounts + true_amounts
    scores = (2 * intersection + smooth) / (cardinalities + smooth).clamp_min(eps)
    class_weights = _apply_empty_and_background_weights(
        scores,
        true_amounts,
        weights=weights,
        ignore_empty_labels=ignore_empty_labels,
        ignore_background=ignore_background,
    )

    return metric_reduction(
        scores,
        reduction=reduction,
        batch_reduction=batch_reduction,
        weights=class_weights,
        ignore_index=ignore_index,
    )

