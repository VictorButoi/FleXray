from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import torch
from torch import Tensor

ProbabilityMode = Literal[
    "binary",
    "multilabel",
    "multiclass",
    "onehot",
    "sigmoid",
    "softmax",
]

_SIGMOID_MODES = {"binary", "multilabel", "sigmoid"}
_SOFTMAX_MODES = {"multiclass", "onehot", "softmax"}
_KNOWN_MODES = _SIGMOID_MODES | _SOFTMAX_MODES


def probabilities_from_logits(logits: Tensor, *, mode: ProbabilityMode | str) -> Tensor:
    """Convert model logits into probabilities.

    Args:
        logits: Tensor with batch and channel dimensions, conventionally
            ``BxCx...``.
        mode: Probability mode. ``"binary"``, ``"sigmoid"``, and
            ``"multilabel"`` apply a sigmoid. ``"onehot"``, ``"multiclass"``,
            and ``"softmax"`` apply softmax over channel dimension, except that
            any single-channel output uses sigmoid.

    Returns:
        Tensor of probabilities with the same shape, dtype, and device as
        ``logits``.

    Raises:
        ValueError: If ``logits`` has no channel dimension or ``mode`` is
            unknown.
    """

    if logits.ndim < 2:
        raise ValueError(
            "logits must include batch and channel dimensions; "
            f"got shape {tuple(logits.shape)}."
        )

    resolved_mode = str(mode).strip().lower()
    if resolved_mode not in _KNOWN_MODES:
        raise ValueError(
            "Unsupported probability mode "
            f"{mode!r}; expected one of {sorted(_KNOWN_MODES)}."
        )

    if int(logits.shape[1]) == 1 or resolved_mode in _SIGMOID_MODES:
        return torch.sigmoid(logits)
    return torch.softmax(logits, dim=1)


def quantized_probabilities(probs: Tensor | np.ndarray) -> np.ndarray:
    """Quantize probabilities into a CPU ``uint8`` sidecar array.

    Args:
        probs: Probability tensor or array. Values are scaled by 255, rounded,
            clipped to ``[0, 255]``, and stored as ``uint8``.

    Returns:
        NumPy array on CPU with the same shape as ``probs`` and dtype
        ``np.uint8``.
    """

    if isinstance(probs, torch.Tensor):
        return (
            probs.mul(255.0)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )

    array = np.asarray(probs)
    return np.clip(np.rint(array.astype(np.float32, copy=False) * 255.0), 0, 255).astype(
        np.uint8
    )


def dequantize_saved_probabilities(probs: np.ndarray) -> np.ndarray:
    """Normalize saved probability sidecars into ``float32`` probabilities.

    Args:
        probs: Channel-first probability sidecar with shape ``CxHxW``. ``uint8``
            values are interpreted as quantized probabilities in ``[0, 255]``.
            Bool, non-``uint8`` integer, and floating-point sidecars are accepted
            with warnings when their values can be interpreted unambiguously.

    Returns:
        ``float32`` probabilities with shape ``CxHxW`` and values in ``[0, 1]``.

    Raises:
        ValueError: If shape is not ``CxHxW`` or numeric values are outside the
            supported range.
        TypeError: If the sidecar dtype is unsupported.
    """

    probs = np.asarray(probs)
    if probs.ndim != 3:
        raise ValueError(
            f"Expected saved probability tensor with shape CxHxW, got {probs.shape}."
        )

    if probs.size == 0:
        return np.zeros_like(probs, dtype=np.float32)

    if probs.dtype == np.uint8:
        return probs.astype(np.float32) / 255.0

    if np.issubdtype(probs.dtype, np.bool_):
        warnings.warn(
            "Expected quantized uint8 probability sidecars; got bool sidecars instead. "
            "Treating them as already-thresholded probabilities in [0, 1].",
            stacklevel=2,
        )
        return probs.astype(np.float32, copy=False)

    if np.issubdtype(probs.dtype, np.integer):
        probs_f = probs.astype(np.float32, copy=False)
        min_value = float(np.min(probs_f))
        max_value = float(np.max(probs_f))
        if 0.0 <= min_value and max_value <= 1.0:
            warnings.warn(
                "Expected quantized uint8 probability sidecars; got integer sidecars in "
                "[0, 1] instead. Treating them as already-thresholded probabilities.",
                stacklevel=2,
            )
            return probs_f
        if min_value < 0.0 or max_value > 255.0:
            raise ValueError(
                "Integer probability sidecars must be stored as quantized probabilities "
                f"in [0, 255]; got min={min_value:.3f}, max={max_value:.3f}."
            )
        warnings.warn(
            "Expected quantized uint8 probability sidecars; got a non-uint8 integer dtype. "
            "Interpreting values as quantized probabilities in [0, 255].",
            stacklevel=2,
        )
        return probs_f / 255.0

    if np.issubdtype(probs.dtype, np.floating):
        probs_f = probs.astype(np.float32, copy=False)
        min_value = float(np.min(probs_f))
        max_value = float(np.max(probs_f))
        if min_value < 0.0 or max_value > 1.0:
            raise ValueError(
                "Floating-point probability sidecars must already be in [0, 1]; "
                f"got min={min_value:.3f}, max={max_value:.3f}."
            )
        warnings.warn(
            "Expected quantized uint8 probability sidecars; got floating-point sidecars "
            "instead. Treating them as probabilities in [0, 1].",
            stacklevel=2,
        )
        return probs_f

    raise TypeError(
        "Unsupported probability sidecar dtype "
        f"{probs.dtype!r}; expected quantized uint8 probabilities."
    )


__all__ = [
    "ProbabilityMode",
    "dequantize_saved_probabilities",
    "probabilities_from_logits",
    "quantized_probabilities",
]
