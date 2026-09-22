from __future__ import annotations

from collections.abc import Iterator

import torch.nn as nn


def iter_leaf_loss_modules(loss_module: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    """Yield leaf loss modules inside combo or routed loss containers."""
    if hasattr(loss_module, "fn_module_dict"):
        for child in loss_module.fn_module_dict.values():
            yield from iter_leaf_loss_modules(child)
        return

    yield type(loss_module).__name__, loss_module


def set_batch_reduction(loss_module: nn.Module, batch_reduction: str) -> None:
    """Set ``batch_reduction`` on every configurable leaf loss module."""
    if batch_reduction not in {"mean", "sum", "none", None}:
        raise ValueError(
            "batch_reduction must be one of 'mean', 'sum', 'none', or None; "
            f"got {batch_reduction!r}."
        )

    for loss_name, module in iter_leaf_loss_modules(loss_module):
        func_kwargs = getattr(module, "_func_kwargs", None)
        if func_kwargs is None or "batch_reduction" not in func_kwargs:
            raise ValueError(
                f"Loss module {loss_name} does not expose a configurable batch_reduction."
            )
        func_kwargs["batch_reduction"] = batch_reduction
        module.batch_reduction = batch_reduction
