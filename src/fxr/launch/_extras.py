"""Shared optional-dependency guards for training launch commands."""

from __future__ import annotations

import importlib.util

TRAINING_EXTRA_MESSAGE = (
    "Install FleXray with .[train] or .[full] to use this command."
)


def require_training_extra(import_name: str) -> None:
    """Ensure one training-extra module is importable.

    Args:
        import_name: Top-level module name used as a representative dependency.

    Returns:
        None.

    Raises:
        SystemExit: If the requested module cannot be found.
    """

    if importlib.util.find_spec(import_name) is None:
        raise SystemExit(TRAINING_EXTRA_MESSAGE)


__all__ = ["TRAINING_EXTRA_MESSAGE", "require_training_extra"]
