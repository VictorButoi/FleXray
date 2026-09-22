"""Exercise a base-only installation without pytest or downloaded model weights."""

from __future__ import annotations

import importlib.util
import sys

TRAINING_MODULES = (
    "kornia",
    "nanodrr",
    "pandas",
    "pydantic",
    "submitit",
    "tabulate",
    "thunderpack",
    "wandb",
    "zstd",
)


def main() -> None:
    """Check absent training dependencies and run a small CPU prediction.

    Returns:
        None. Failed checks raise ``AssertionError``.
    """
    assert all(importlib.util.find_spec(name) is None for name in TRAINING_MODULES)

    import torch
    from fxr.inference import InferenceRunner
    from fxr.models import UNet

    torch.set_num_threads(1)
    model = UNet(in_channels=1, out_channels=2, filters=[2], convs_per_block=1)
    prediction = InferenceRunner(model).predict(torch.zeros(1, 1, 16, 16))
    assert prediction.probabilities.shape == (1, 2, 16, 16)
    assert torch.isfinite(prediction.probabilities).all()
    assert all(sys.modules.get(name) is None for name in TRAINING_MODULES)


if __name__ == "__main__":
    main()
