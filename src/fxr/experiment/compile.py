"""Validation and construction helpers for optional single-process compilation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import torch

_COMPILE_KWARGS = frozenset(
    {"backend", "mode", "options", "fullgraph", "dynamic"}
)


def resolve_torch_compile_config(config: Any) -> tuple[bool, dict[str, Any]]:
    """Validate ``model.compile_cfg`` and return enabled state plus Torch kwargs.

    Args:
        config: ``None`` or a mapping with ``enabled`` and optional arguments
            accepted by :func:`torch.compile`.

    Returns:
        Pair ``(enabled, compile_kwargs)``. A missing config disables compilation.

    Raises:
        TypeError: If the config or one of its values has an invalid type.
        ValueError: If the config contains unsupported keys.
    """

    if config is None:
        return False, {}
    cfg = _plain_mapping(config, name="model.compile_cfg")
    unknown = sorted(set(cfg) - ({"enabled"} | _COMPILE_KWARGS))
    if unknown:
        raise ValueError(f"Unexpected model.compile_cfg keys: {unknown}.")

    enabled = cfg.pop("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("model.compile_cfg.enabled must be a bool.")

    for key in ("fullgraph", "dynamic"):
        value = cfg.get(key)
        if value is not None and not isinstance(value, bool):
            raise TypeError(f"model.compile_cfg.{key} must be a bool or null.")
    for key in ("backend", "mode"):
        value = cfg.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise TypeError(f"model.compile_cfg.{key} must be a non-empty string or null.")
    options = cfg.get("options")
    if options is not None:
        cfg["options"] = _plain_mapping(
            options, name="model.compile_cfg.options"
        )
    return enabled, cfg


def maybe_compile_model(
    model: torch.nn.Module,
    config: Any,
) -> torch.nn.Module:
    """Return ``model`` or a compiled forward wrapper sharing its parameters.

    The raw module remains the checkpoint and optimizer owner. Compilation is
    intentionally limited to a single process; distributed training is outside
    the current FleXray contract.

    Args:
        model: Raw model already moved to its training device.
        config: ``model.compile_cfg`` mapping or ``None``.

    Returns:
        Raw model when disabled, otherwise the object returned by
        :func:`torch.compile`.

    Raises:
        RuntimeError: If compilation is requested in a multi-process runtime.
    """

    enabled, kwargs = resolve_torch_compile_config(config)
    if not enabled:
        return model
    world_size = _runtime_world_size()
    if world_size != 1:
        raise RuntimeError(
            "model.compile_cfg.enabled=true currently requires single-process "
            f"training; detected world size {world_size}."
        )
    _configure_compiled_backward_autocast()
    return torch.compile(model, **kwargs)


def _configure_compiled_backward_autocast() -> None:
    """Disable compiled-backward autocast for an out-of-context backward call.

    FleXray executes the compiled forward inside ``torch.autocast`` and calls
    backward after leaving that context. Current PyTorch guidance requires the
    compiled backward to mirror that outer state rather than the forward state.

    Returns:
        ``None``. Older PyTorch versions without the setting are left unchanged.
    """

    try:
        from torch._functorch import config as functorch_config
    except (AttributeError, ImportError):
        return
    if hasattr(functorch_config, "backward_pass_autocast"):
        functorch_config.backward_pass_autocast = "off"


def _runtime_world_size() -> int:
    """Return initialized distributed size or the launcher environment hint.

    Args:
        None.

    Returns:
        Positive process count, defaulting to ``1``.

    Raises:
        ValueError: If ``WORLD_SIZE`` is present but not a positive integer.
    """

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_world_size())
    raw_world_size = os.environ.get("WORLD_SIZE", "1")
    try:
        world_size = int(raw_world_size)
    except ValueError as exc:
        raise ValueError("WORLD_SIZE must be a positive integer.") from exc
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be a positive integer.")
    return world_size


def _plain_mapping(value: Any, *, name: str) -> dict[str, Any]:
    """Return a Config-like value as a plain dictionary.

    Args:
        value: Mapping or object exposing ``to_dict``.
        name: Config path used in validation errors.

    Returns:
        Plain dictionary copy.

    Raises:
        TypeError: If ``value`` is not mapping-like.
    """

    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}.")
    return dict(value)
