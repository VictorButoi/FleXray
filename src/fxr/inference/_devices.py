"""Shared device selection for inference entry points."""

from __future__ import annotations

import torch


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    """Resolve a user-facing device choice into a concrete torch device.

    Args:
        device: Device name such as ``"cuda"``, ``"cuda:1"``, or ``"cpu"``,
            a ``torch.device``, or ``"auto"``/``None`` to pick CUDA when it is
            available and CPU otherwise.

    Returns:
        Concrete ``torch.device``.

    Raises:
        ValueError: If a CUDA device is requested but CUDA is unavailable, if
            a CUDA ordinal is out of range, or if the device string is not
            recognized by PyTorch.
    """

    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        resolved = torch.device(device)
    except RuntimeError as exc:
        raise ValueError(f"Unrecognized device {device!r}: {exc}") from exc
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            f"Device {device!r} was requested but CUDA is not available; "
            "use --device cpu or run on a machine with a CUDA-capable GPU."
        )
    if resolved.type == "cuda" and resolved.index is not None:
        count = torch.cuda.device_count()
        if resolved.index >= count:
            raise ValueError(
                f"Device {device!r} is out of range; this machine exposes "
                f"{count} CUDA device(s) (valid indices: 0-{count - 1})."
            )
    return resolved
