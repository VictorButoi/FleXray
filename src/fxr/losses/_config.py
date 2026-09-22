from __future__ import annotations

from copy import deepcopy
from importlib import import_module
from inspect import Parameter, signature
from typing import Any, Callable, Mapping

import torch.nn as nn


def loss_module_from_func(name: str, loss_func: Callable[..., Any]) -> type[nn.Module]:
    """Wrap a tensor loss function in a small configurable ``nn.Module``."""

    class _LossWrapper(nn.Module):
        """Configurable ``nn.Module`` adapter for a tensor loss function.

        Attributes:
            _func_kwargs: Stored keyword arguments applied to every wrapped loss
                function call.
        """

        def __init__(self, **kwargs: Any) -> None:
            """Store configured keyword arguments for the wrapped function.

            Args:
                **kwargs: Loss-function keyword overrides or additions.

            Returns:
                ``None``.
            """

            super().__init__()
            defaults = {
                param_name: param.default
                for param_name, param in signature(loss_func).parameters.items()
                if param_name not in {"y_pred", "y_true"}
                and param.default is not Parameter.empty
            }
            defaults.update(kwargs)
            self._func_kwargs = defaults
            self.__dict__.update(defaults)

        def forward(self, y_pred: Any, y_true: Any, **loss_kwargs: Any) -> Any:
            """Evaluate the wrapped tensor loss function.

            Args:
                y_pred: Predicted tensor or structure accepted by the wrapped
                    loss function.
                y_true: Target tensor or structure accepted by the wrapped loss
                    function.
                **loss_kwargs: Per-call keyword overrides passed after stored
                    configuration.

            Returns:
                Loss value returned by the wrapped function.
            """

            return loss_func(y_pred, y_true, **{**self._func_kwargs, **loss_kwargs})

    _LossWrapper.__name__ = name
    _LossWrapper.__qualname__ = name
    return type(name, (_LossWrapper,), {"__module__": "fxr.losses"})


def build_loss_from_config(config: Mapping[str, Any]) -> nn.Module:
    """Instantiate the focused FleXray loss config dictionary format."""
    if not isinstance(config, Mapping):
        raise TypeError(f"Loss config must be a mapping, got {type(config).__name__}.")

    config_copy = deepcopy(dict(config))
    if "_combo_class" in config_copy:
        from .combo import build_combined_loss

        return build_combined_loss(config_copy)

    class_path = config_copy.pop("_class", None)
    if class_path is None:
        raise ValueError(
            "Loss config must define `_class` or `_combo_class`; "
            f"got keys: {sorted(config_copy.keys())}."
        )

    loss_class = _resolve_loss_class(class_path)
    return loss_class(**config_copy)


def _resolve_loss_class(value: Any) -> type[nn.Module]:
    if isinstance(value, type):
        return value
    if not isinstance(value, str):
        raise TypeError(
            "`_class` must be a class or import string; " f"got {type(value).__name__}."
        )

    if "." not in value:
        from .segmentation import PixelCELoss, SoftDiceLoss
        from .combo import CombinedLoss
        from .routed import DatasetRoutedLoss

        local_classes: dict[str, type[nn.Module]] = {
            "CombinedLoss": CombinedLoss,
            "DatasetRoutedLoss": DatasetRoutedLoss,
            "PixelCELoss": PixelCELoss,
            "SoftDiceLoss": SoftDiceLoss,
        }
        try:
            return local_classes[value]
        except KeyError as exc:
            raise ValueError(f"Unknown FleXray loss class {value!r}.") from exc

    module_name, attr_name = value.rsplit(".", 1)
    module = import_module(module_name)
    loss_class = getattr(module, attr_name)
    if not isinstance(loss_class, type):
        raise TypeError(f"Resolved `_class` {value!r} is not a class.")
    return loss_class
