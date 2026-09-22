"""Optional WandB import shared by the trainer and its logging callbacks.

WandB ships with the ``train`` extra. Training and logging callbacks explain
how to install it when missing. ``fxr.experiment`` still needs the other training
dependencies; the inference surface does not import this module.
"""

from __future__ import annotations

from typing import Any

try:
    import wandb
except ModuleNotFoundError as exc:
    if exc.name != "wandb":
        raise
    wandb = None

WANDB_EXTRA_MESSAGE = (
    "Training logs to WandB, which ships with the train extra: "
    'python -m pip install "flexray[train]".'
)


def require_wandb() -> Any:
    """Return the imported ``wandb`` module or explain how to install it.

    Returns:
        The ``wandb`` module.

    Raises:
        ImportError: If WandB is not installed.
    """

    if wandb is None:
        raise ImportError(WANDB_EXTRA_MESSAGE)
    return wandb


__all__ = ["WANDB_EXTRA_MESSAGE", "require_wandb", "wandb"]
