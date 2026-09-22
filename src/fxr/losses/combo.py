from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import torch
import torch.nn as nn

from ._config import build_loss_from_config


class CombinedLoss(nn.Module):
    """Weighted sum of named loss modules.

    ``fn_label_types`` optionally routes each component to a key in mapping-like
    ``outputs`` and ``targets`` before evaluating that component.

    Attributes:
        fn_module_dict: Named component loss modules.
        fn_weights: Scalar weight for each component loss name.
        fn_label_types: Optional output/target mapping key for each component
            loss name.
        _last_loss_breakdown: Detached raw component losses from the most recent
            forward call.
    """

    def __init__(
        self,
        fn_dict: Mapping[str, nn.Module],
        fn_weights: Mapping[str, float],
        fn_label_types: Mapping[str, str | None],
    ) -> None:
        """Initialize weighted component losses.

        Args:
            fn_dict: Mapping from component name to loss module.
            fn_weights: Mapping from component name to scalar loss weight.
            fn_label_types: Mapping from component name to optional output and
                target key used before evaluating that component.

        Returns:
            ``None``.

        Raises:
            ValueError: If components are empty or weights/label types are
                missing for any component.
        """

        super().__init__()
        if not fn_dict:
            raise ValueError("CombinedLoss requires at least one component loss.")

        missing_weights = sorted(set(fn_dict) - set(fn_weights))
        missing_label_types = sorted(set(fn_dict) - set(fn_label_types))
        if missing_weights:
            raise ValueError(f"Missing component weights for {missing_weights}.")
        if missing_label_types:
            raise ValueError(
                f"Missing component label types for {missing_label_types}."
            )

        self.fn_module_dict = nn.ModuleDict(dict(fn_dict))
        self.fn_weights = {
            name: float(fn_weights[name]) for name in self.fn_module_dict
        }
        self.fn_label_types = {
            name: fn_label_types[name] for name in self.fn_module_dict
        }
        self._last_loss_breakdown: dict[str, torch.Tensor] = {}

    def _compute_raw_loss(
        self,
        outputs: Any,
        targets: Any,
        loss_name: str,
        loss_func: nn.Module,
        **loss_kwargs: Any,
    ) -> torch.Tensor:
        """Compute one unweighted component loss.

        Args:
            outputs: Model outputs or mapping of outputs by label type.
            targets: Training targets or mapping of targets by label type.
            loss_name: Component name being evaluated.
            loss_func: Component loss module.
            **loss_kwargs: Additional keyword arguments passed to the component.

        Returns:
            Raw unweighted component loss tensor.
        """

        label_type = self.fn_label_types[loss_name]
        if label_type is not None:
            outputs = outputs[label_type]
            targets = targets[label_type]
        return loss_func(outputs, targets, **loss_kwargs)

    def forward(
        self,
        outputs: Any,
        targets: Any,
        *,
        component_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
        **loss_kwargs: Any,
    ) -> torch.Tensor:
        """Evaluate and sum all weighted component losses.

        Args:
            outputs: Model outputs passed to components directly or by label
                type routing.
            targets: Training targets passed to components directly or by label
                type routing.
            component_kwargs: Optional per-component keyword arguments keyed by
                component name, merged over ``loss_kwargs`` for that component.
            **loss_kwargs: Additional keyword arguments forwarded to every
                component loss.

        Returns:
            Weighted total loss tensor.

        Raises:
            KeyError: If ``component_kwargs`` names an unknown component.
            RuntimeError: If the module has no component losses at call time.
        """

        per_component = dict(component_kwargs or {})
        unknown = sorted(set(per_component) - set(self.fn_module_dict))
        if unknown:
            raise KeyError(f"component_kwargs references unknown component(s): {unknown}.")
        total_loss: torch.Tensor | None = None
        loss_breakdown: dict[str, torch.Tensor] = {}
        for loss_name, loss_func in self.fn_module_dict.items():
            kwargs = {**loss_kwargs, **per_component.get(loss_name, {})}
            raw_loss = self._compute_raw_loss(
                outputs, targets, loss_name, loss_func, **kwargs
            )
            loss_breakdown[loss_name] = raw_loss.detach()
            weighted_loss = self.fn_weights[loss_name] * raw_loss
            total_loss = (
                weighted_loss if total_loss is None else total_loss + weighted_loss
            )

        self._last_loss_breakdown = loss_breakdown
        if total_loss is None:
            raise RuntimeError("CombinedLoss has no component losses.")
        return total_loss

    def get_last_loss_breakdown(self) -> dict[str, torch.Tensor]:
        """Return detached raw component losses from the previous call.

        Returns:
            Dictionary keyed by component name with cloned tensor values.
        """

        return {
            loss_name: value.clone() if isinstance(value, torch.Tensor) else value
            for loss_name, value in self._last_loss_breakdown.items()
        }


def build_combined_loss(loss_cfg_dict: Mapping[str, Any]) -> CombinedLoss:
    """Build ``CombinedLoss`` from a ``{"_combo_class": ...}`` config body."""
    config = deepcopy(dict(loss_cfg_dict))
    combo_classes = config.pop("_combo_class", None)
    if config:
        raise ValueError(
            "Combined loss config only supports `_combo_class` at the profile root; "
            f"got extra keys: {sorted(config.keys())}."
        )
    if not isinstance(combo_classes, Mapping) or not combo_classes:
        raise ValueError("`_combo_class` must be a non-empty mapping.")

    loss_fn_dict: dict[str, nn.Module] = {}
    loss_fn_weights: dict[str, float] = {}
    loss_fn_label_types: dict[str, str | None] = {}
    for loss_name, loss_fn_cfg in combo_classes.items():
        if not isinstance(loss_fn_cfg, Mapping):
            raise TypeError(
                f"Combined loss component {loss_name!r} must be a mapping, "
                f"got {type(loss_fn_cfg).__name__}."
            )
        component_cfg = deepcopy(dict(loss_fn_cfg))
        loss_fn_label_types[loss_name] = component_cfg.pop("label_type", None)
        loss_fn_weights[loss_name] = float(component_cfg.pop("weight", 1.0))
        loss_fn_dict[loss_name] = build_loss_from_config(component_cfg)

    return CombinedLoss(
        fn_dict=loss_fn_dict,
        fn_weights=loss_fn_weights,
        fn_label_types=loss_fn_label_types,
    )
