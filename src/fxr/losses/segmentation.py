from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from pydantic import validate_call
from torch import Tensor
from torch.nn import functional as F

from ._config import loss_module_from_func

InputMode = Literal["binary", "multiclass", "onehot", "auto"]
Reduction = Literal["mean", "sum", "none"] | None


def _resolve_mode(y_pred: Tensor, y_true: Tensor, mode: InputMode) -> str:
    if y_pred.ndim <= 2:
        raise ValueError("y_pred must have at least 3 dimensions.")
    if mode != "auto":
        return mode

    num_classes = y_pred.shape[1]
    if y_pred.shape == y_true.shape:
        return "binary" if num_classes == 1 else "onehot"
    return "multiclass"


def _inputs_as_onehot(
    y_pred: Tensor,
    y_true: Tensor,
    *,
    mode: InputMode,
    from_logits: bool,
) -> tuple[Tensor, Tensor]:
    mode = _resolve_mode(y_pred, y_true, mode)
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
        y_true = y_true.to(device=y_pred.device, dtype=y_pred.dtype)
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

    y_pred = y_pred.reshape(batch_size, num_classes, -1)
    y_true = y_true.reshape(batch_size, -1).long()
    y_true = F.one_hot(y_true, num_classes=num_classes).permute(0, 2, 1)
    return y_pred.float(), y_true.float()


def _coerce_metric_weights(
    weights: Tensor | list[float] | None, reference: Tensor
) -> Tensor | None:
    if weights is None:
        return None

    if isinstance(weights, list):
        weights = torch.tensor(weights, device=reference.device, dtype=reference.dtype)
    else:
        weights = weights.to(device=reference.device, dtype=reference.dtype)

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


def _metric_reduction(
    scores: Tensor,
    *,
    reduction: Reduction,
    batch_reduction: Reduction,
    weights: Tensor | list[float] | None = None,
    ignore_index: int | None = None,
) -> Tensor:
    if scores.ndim != 2:
        raise ValueError(
            "Reducible score tensors must have batch and channel dimensions; "
            f"got shape {tuple(scores.shape)}."
        )

    batch_size, num_classes = scores.shape
    weights = _coerce_metric_weights(weights, scores)

    if ignore_index is not None:
        if not 0 <= int(ignore_index) < num_classes:
            raise ValueError("ignore_index must be in [0, channels).")
        ignore_weights = torch.ones_like(scores)
        ignore_weights[:, int(ignore_index)] = 0.0
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


def _mask_background_when_foreground_present(
    scores: Tensor,
    weights: Tensor | None,
) -> Tensor:
    if weights is None:
        weights = torch.ones_like(scores)
    if scores.shape[1] <= 1:
        return weights

    non_bg_sum = weights.sum(dim=1) - weights[:, 0]
    has_foreground = non_bg_sum > 0
    weights = weights.clone()
    weights[:, 0] = weights[:, 0] * (~has_foreground).float()
    return weights


def _flatten_metric_pixel_weights(
    pixel_weights: Tensor | None,
    *,
    reference: Tensor,
    spatial_ndim: int,
) -> Tensor | None:
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


@validate_call(config={"arbitrary_types_allowed": True})
def soft_dice_loss(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    weights: Tensor | list[float] | None = None,
    ignore_index: int | None = None,
    ignore_background: bool = False,
    ignore_empty_labels: bool = False,
    from_logits: bool = False,
    smooth: float = 1e-7,
    eps: float = 1e-7,
    square_denom: bool = True,
    log_loss: bool = False,
    pixel_weights: Tensor | None = None,
) -> Tensor:
    if y_pred.shape[1] == 1 and ignore_index is not None:
        raise ValueError("ignore_index is not supported for binary segmentation.")

    spatial_ndim = y_pred.ndim - 2
    y_pred, y_true = _inputs_as_onehot(
        y_pred,
        y_true,
        mode=mode,
        from_logits=from_logits,
    )
    pixel_weights = _flatten_metric_pixel_weights(
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
    class_weights = _coerce_metric_weights(weights, scores)

    if ignore_empty_labels:
        existing_label = (true_amounts > 0).float()
        class_weights = (
            existing_label if class_weights is None else class_weights * existing_label
        )

    if ignore_background:
        class_weights = _mask_background_when_foreground_present(scores, class_weights)

    scores = _metric_reduction(
        scores,
        reduction=reduction,
        batch_reduction=batch_reduction,
        weights=class_weights,
        ignore_index=ignore_index,
    )
    if log_loss:
        return -torch.log(scores.clamp_min(eps))
    return 1.0 - scores


def _coerce_class_weights(
    weights: Tensor | list[float] | None,
    *,
    reference: Tensor,
    num_classes: int,
) -> Tensor | None:
    if weights is None:
        return None
    if isinstance(weights, list):
        weights = torch.tensor(weights, device=reference.device, dtype=reference.dtype)
    else:
        weights = weights.to(device=reference.device, dtype=reference.dtype)
    if weights.ndim != 1 or weights.shape[0] != num_classes:
        raise ValueError(
            "weights must be a 1D class vector matching the number of classes; "
            f"got shape {tuple(weights.shape)} for {num_classes} classes."
        )
    return weights


def _apply_loss_reduction(
    loss: Tensor,
    *,
    reduction: Reduction,
    batch_reduction: Reduction,
    batch_weights: Tensor | None = None,
    pixel_weights: Tensor | None = None,
) -> Tensor:
    spatial_dims = tuple(range(1, loss.ndim))

    if pixel_weights is not None:
        pixel_weights = pixel_weights.to(device=loss.device, dtype=loss.dtype)
        if pixel_weights.shape != loss.shape:
            raise ValueError(
                "pixel_weights must match the unreduced loss shape; "
                f"got {tuple(pixel_weights.shape)} != {tuple(loss.shape)}."
            )
        weighted_loss = loss * pixel_weights
        if reduction == "mean":
            denominator = (
                pixel_weights.sum(dim=spatial_dims) if spatial_dims else pixel_weights
            )
            numerator = (
                weighted_loss.sum(dim=spatial_dims) if spatial_dims else weighted_loss
            )
            loss = numerator / denominator.clamp_min(1.0)
            loss = torch.where(denominator > 0, loss, numerator * 0.0)
        elif reduction == "sum":
            loss = (
                weighted_loss.sum(dim=spatial_dims) if spatial_dims else weighted_loss
            )
        elif reduction in {"none", None}:
            loss = weighted_loss
        else:
            raise ValueError(f"Unsupported reduction {reduction!r}.")
    else:
        if reduction == "mean" and spatial_dims:
            loss = loss.mean(dim=spatial_dims)
        elif reduction == "sum" and spatial_dims:
            loss = loss.sum(dim=spatial_dims)
        elif reduction in {"mean", "sum", "none", None}:
            loss = loss
        else:
            raise ValueError(f"Unsupported reduction {reduction!r}.")

    if batch_weights is not None:
        batch_weights = batch_weights.to(device=loss.device, dtype=loss.dtype)
        if batch_weights.ndim != 1 or batch_weights.shape[0] != loss.shape[0]:
            raise ValueError(
                "batch_weights must be a 1D tensor matching the batch dimension of the loss."
            )
        batch_weights = batch_weights.view(-1, *([1] * (loss.ndim - 1)))
        loss = loss * batch_weights

    if batch_reduction == "mean":
        if batch_weights is None:
            loss = loss.mean(dim=0)
        else:
            denominator = batch_weights.sum(dim=0)
            numerator = loss.sum(dim=0)
            loss = numerator / denominator.clamp_min(1.0)
            loss = torch.where(denominator > 0, loss, numerator * 0.0)
    elif batch_reduction == "sum":
        loss = loss.sum(dim=0)
    elif batch_reduction in {"none", None}:
        loss = loss
    else:
        raise ValueError(f"Unsupported batch_reduction {batch_reduction!r}.")

    return loss


def _prepare_dense_loss_pixel_weights(
    pixel_weights: Tensor | None, loss: Tensor
) -> Tensor | None:
    if pixel_weights is None:
        return None
    pixel_weights = pixel_weights.to(device=loss.device, dtype=loss.dtype)
    if pixel_weights.shape[0] != loss.shape[0]:
        raise ValueError("pixel_weights batch dimension must match y_pred.")
    if pixel_weights.ndim == loss.ndim + 1 and pixel_weights.shape[1] == 1:
        pixel_weights = pixel_weights.squeeze(1)
    if pixel_weights.ndim != loss.ndim:
        raise ValueError(
            "pixel_weights must either match the loss shape or include a singleton channel axis."
        )
    if pixel_weights.shape != loss.shape:
        raise ValueError(
            "pixel_weights must match the unreduced loss shape; "
            f"got {tuple(pixel_weights.shape)} != {tuple(loss.shape)}."
        )
    return pixel_weights


@validate_call(config={"arbitrary_types_allowed": True})
def _validate_supervise_empty_label_ids(
    label_ids: Sequence[int],
    *,
    mode: str,
    ignore_empty_labels: bool,
    num_classes: int,
) -> None:
    """Validate channel ids that stay supervised despite empty targets."""
    if not ignore_empty_labels or mode != "binary":
        raise ValueError(
            "supervise_empty_label_ids requires ignore_empty_labels=True and binary mode."
        )
    ids = [int(label_id) for label_id in label_ids]
    if len(set(ids)) != len(ids) or any(not 0 <= i < num_classes for i in ids):
        raise ValueError(
            f"supervise_empty_label_ids must be unique ids in [0, {num_classes}); got {ids}."
        )


def pixel_crossentropy_loss(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    weights: Tensor | list[float] | None = None,
    ignore_index: int | None = None,
    ignore_empty_labels: bool = False,
    ignore_background: bool = False,
    from_logits: bool = False,
    pixel_weights: Tensor | None = None,
    supervise_empty_label_ids: Sequence[int] | None = None,
) -> Tensor:
    if y_pred.ndim <= 2:
        raise ValueError("y_pred must have at least 3 dimensions.")
    if y_pred.shape[1] == 1 and ignore_index is not None:
        raise ValueError("ignore_index is not supported for binary segmentation.")

    mode = _resolve_mode(y_pred, y_true, mode)
    batch_size, num_classes = y_pred.shape[:2]
    if supervise_empty_label_ids is not None:
        _validate_supervise_empty_label_ids(
            supervise_empty_label_ids,
            mode=mode,
            ignore_empty_labels=ignore_empty_labels,
            num_classes=num_classes,
        )
    class_weights = _coerce_class_weights(
        weights,
        reference=y_pred,
        num_classes=num_classes,
    )

    batch_flattened_shape: tuple[int, int] | None = None
    batch_weights: Tensor | None = None
    bg_keep_flat: Tensor | None = None

    if mode == "binary":
        if y_pred.shape != y_true.shape:
            raise ValueError(
                "Binary segmentation expects targets with the same shape as predictions."
            )
        if class_weights is not None:
            raise ValueError("weights are not supported for binary segmentation.")

        y_true = y_true.to(device=y_pred.device, dtype=y_pred.dtype)
        if ignore_background and y_pred.shape[1] > 1:
            spatial_dims_full = tuple(range(2, y_true.ndim))
            non_bg_has_fg = (y_true[:, 1:].sum(dim=spatial_dims_full) > 0).any(dim=1)
            bg_keep = torch.ones(
                (y_true.shape[0], y_pred.shape[1]),
                device=y_pred.device,
                dtype=y_pred.dtype,
            )
            bg_keep[:, 0] = (~non_bg_has_fg).to(bg_keep.dtype)
            bg_keep_flat = bg_keep.reshape(-1)

        if pixel_weights is not None:
            pixel_weights = pixel_weights.to(device=y_pred.device, dtype=y_pred.dtype)
            if pixel_weights.shape[0] != y_pred.shape[0]:
                raise ValueError("pixel_weights batch dimension must match y_pred.")
            if pixel_weights.ndim == y_pred.ndim - 1:
                pixel_weights = pixel_weights.unsqueeze(1)
            if pixel_weights.ndim != y_pred.ndim:
                raise ValueError(
                    "pixel_weights must either match the prediction rank or omit the channel dimension."
                )
            if pixel_weights.shape[1] == 1 and y_pred.shape[1] != 1:
                pixel_weights = pixel_weights.expand(
                    -1, y_pred.shape[1], *pixel_weights.shape[2:]
                )
            if pixel_weights.shape != y_pred.shape:
                raise ValueError(
                    "pixel_weights must broadcast to the binary prediction shape; "
                    f"got {tuple(pixel_weights.shape)} != {tuple(y_pred.shape)}."
                )

        if y_pred.shape[1] != 1:
            batch_flattened_shape = (batch_size, num_classes)
            flat_shape = (batch_size * num_classes, 1, *y_pred.shape[2:])
            y_pred = y_pred.reshape(flat_shape)
            y_true = y_true.reshape(flat_shape)
            if pixel_weights is not None:
                pixel_weights = pixel_weights.reshape(flat_shape)

        if from_logits:
            loss = F.binary_cross_entropy_with_logits(
                input=y_pred,
                target=y_true,
                reduction="none",
            )
        else:
            loss = F.binary_cross_entropy(input=y_pred, target=y_true, reduction="none")
        loss = loss.squeeze(dim=1)
        if pixel_weights is not None:
            pixel_weights = pixel_weights.squeeze(dim=1)

        if ignore_empty_labels:
            spatial_dims = tuple(range(2, y_true.ndim))
            batch_weights = y_true.sum(dim=spatial_dims).squeeze(1) > 0
            if supervise_empty_label_ids:
                # Known-absent channels stay supervised as negatives.
                supervise_empty = torch.zeros(
                    (batch_size, num_classes), dtype=torch.bool, device=y_pred.device
                )
                supervise_empty[:, list(supervise_empty_label_ids)] = True
                batch_weights = batch_weights | supervise_empty.reshape(-1)
        if bg_keep_flat is not None:
            bg_keep_flat = bg_keep_flat.to(device=y_pred.device, dtype=y_pred.dtype)
            batch_weights = (
                bg_keep_flat
                if batch_weights is None
                else batch_weights.to(bg_keep_flat.dtype) * bg_keep_flat
            )

    elif mode == "onehot":
        if ignore_empty_labels:
            raise ValueError(
                "ignore_empty_labels is only supported for binary segmentation."
            )
        if ignore_background:
            raise ValueError(
                "ignore_background is only supported for binary segmentation."
            )
        if y_pred.shape != y_true.shape:
            raise ValueError(
                "One-hot segmentation expects targets with the same shape as predictions."
            )

        log_probs = F.log_softmax(y_pred, dim=1) if from_logits else y_pred
        y_true = y_true.to(device=y_pred.device, dtype=log_probs.dtype)
        if ignore_index is not None:
            if not 0 <= int(ignore_index) < num_classes:
                raise ValueError("ignore_index must be in [0, channels).")
            valid_mask = y_true[:, int(ignore_index), ...] <= 0
            y_true = y_true.clone()
            y_true[:, int(ignore_index), ...] = 0
        else:
            valid_mask = torch.ones_like(y_true[:, 0, ...], dtype=torch.bool)

        channel_loss = -(y_true * log_probs)
        if class_weights is not None:
            view_shape = (1, num_classes, *([1] * (y_true.ndim - 2)))
            channel_loss = channel_loss * class_weights.view(view_shape)
        loss = channel_loss.sum(dim=1)
        loss = loss * valid_mask.to(dtype=loss.dtype)
        pixel_weights = _prepare_dense_loss_pixel_weights(pixel_weights, loss)

    else:
        if ignore_empty_labels:
            raise ValueError(
                "ignore_empty_labels is only supported for binary segmentation."
            )
        if ignore_background:
            raise ValueError(
                "ignore_background is only supported for binary segmentation."
            )

        if y_true.ndim == y_pred.ndim:
            if y_true.shape[1] != 1:
                raise ValueError(
                    "Multiclass segmentation expects dense labels with no class "
                    "axis or a singleton class axis."
                )
            y_true = y_true.squeeze(1)
        y_true = y_true.long()

        loss_kwargs: dict[str, object] = {
            "reduction": "none",
            "weight": class_weights,
        }
        if ignore_index is not None:
            loss_kwargs["ignore_index"] = int(ignore_index)

        if from_logits:
            loss = F.cross_entropy(y_pred, y_true, **loss_kwargs)
        else:
            loss = F.nll_loss(y_pred, y_true, **loss_kwargs)
        pixel_weights = _prepare_dense_loss_pixel_weights(pixel_weights, loss)

    reduced = _apply_loss_reduction(
        loss,
        reduction=reduction,
        batch_reduction=batch_reduction,
        batch_weights=batch_weights if mode == "binary" else None,
        pixel_weights=pixel_weights,
    )

    if batch_flattened_shape is not None and batch_reduction in {"none", None}:
        reduced = reduced.reshape(*batch_flattened_shape, *reduced.shape[1:])

    return reduced


PixelCELoss = loss_module_from_func("PixelCELoss", pixel_crossentropy_loss)
SoftDiceLoss = loss_module_from_func("SoftDiceLoss", soft_dice_loss)
