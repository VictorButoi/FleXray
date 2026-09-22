from __future__ import annotations

import torch
import torch.nn as nn


class ConstantLoss(nn.Module):
    """Deterministic scalar loss module used by loss tests.

    Attributes:
        value: Constant scalar returned by ``forward``.
    """

    def __init__(self, value: float) -> None:
        """Store the constant loss value.

        Args:
            value: Scalar value returned by ``forward``.

        Returns:
            ``None``.
        """

        super().__init__()
        self.value = float(value)

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        **loss_kwargs: object,
    ) -> torch.Tensor:
        """Return the configured constant as a tensor on the output device.

        Args:
            outputs: Output tensor used for device and dtype placement.
            targets: Ignored target tensor.
            **loss_kwargs: Ignored per-call loss keyword arguments.

        Returns:
            Scalar tensor containing ``value``.
        """

        del targets, loss_kwargs
        return outputs.new_tensor(self.value)


class DifferenceLoss(nn.Module):
    """Mean absolute difference loss module used by loss tests.

    Attributes:
        No additional public attributes beyond ``nn.Module`` state.
    """

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        **loss_kwargs: object,
    ) -> torch.Tensor:
        """Return mean absolute difference between outputs and targets.

        Args:
            outputs: Predicted tensor.
            targets: Target tensor.
            **loss_kwargs: Ignored per-call loss keyword arguments.

        Returns:
            Scalar mean absolute error tensor.
        """

        del loss_kwargs
        return (outputs - targets).abs().mean()
