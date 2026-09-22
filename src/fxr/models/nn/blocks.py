"""Compact neural-network blocks used by FleXray model definitions."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as torch_nn

__all__ = ["ConvBlock", "get_activation"]


def get_activation(name: str | type[torch_nn.Module] | torch_nn.Module) -> torch_nn.Module:
    """Resolve an activation into a ``torch.nn.Module`` instance.

    Existing module instances are returned unchanged. Module classes are
    constructed without arguments. String names are resolved against
    ``torch.nn`` first by exact attribute name, then by case-insensitive module
    class name.

    Args:
        name: Activation module instance, activation module class, or non-empty
            ``torch.nn`` module name.

    Returns:
        Activation module instance ready to insert into a network.

    Raises:
        ValueError: If ``name`` is empty, is not an accepted input type, does
            not identify a supported ``torch.nn.Module`` subclass, or names a
            module class that cannot be constructed without arguments.
    """

    if isinstance(name, torch_nn.Module):
        return name
    if isinstance(name, type) and issubclass(name, torch_nn.Module):
        return name()
    if not isinstance(name, str) or not name:
        raise ValueError("activation must be a non-empty torch.nn module name")

    activation_cls = getattr(torch_nn, name, None)
    if activation_cls is None:
        activation_cls = _case_insensitive_torch_nn_class(name)
    if activation_cls is None or not issubclass(activation_cls, torch_nn.Module):
        raise ValueError(f"Unsupported activation: {name!r}")

    try:
        return activation_cls()
    except TypeError as exc:
        raise ValueError(f"Activation {name!r} cannot be constructed without arguments") from exc


class ConvBlock(torch_nn.Module):
    """2D convolutional block with optional normalization and residual output.

    Each convolution stage uses a padded ``3x3`` convolution followed by the
    requested normalization layer, activation, and optional dropout. The
    normalization and activation order is configurable for imported checkpoints.
    When ``residual`` is enabled, the block adds either an identity shortcut or
    a ``1x1`` projection shortcut to the stacked-layer output.

    Attributes:
        layers: Sequential stack containing the convolution, normalization,
            activation, and dropout modules.
        norm_after_activation: Whether activation precedes normalization in
            each convolution stage.
        residual_projection: Residual projection shortcut policy.
        residual_shortcut_norm: Whether projected shortcuts use normalization.
        shortcut: Optional residual path. This is ``None`` when residual output
            is disabled, ``Identity`` for equal input and output channels, and a
            ``1x1`` convolution when channels differ.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_convs: int = 1,
        dropout: float = 0.0,
        activation: str | type[torch_nn.Module] | torch_nn.Module = "LeakyReLU",
        norm: str | None = None,
        residual: bool = False,
        norm_after_activation: bool = False,
        residual_projection: Literal["auto", "always"] = "auto",
        residual_shortcut_norm: bool = False,
    ) -> None:
        """Initialize the convolutional block.

        Args:
            in_channels: Number of channels expected in the input tensor.
            out_channels: Number of channels produced by each convolution and
                by the block output.
            num_convs: Number of convolution stages to stack.
            dropout: Dropout probability for ``Dropout2d`` layers. A value of
                ``0`` omits dropout layers.
            activation: Activation passed to ``get_activation`` for each
                convolution stage.
            norm: Optional normalization mode. Supported values are ``None``,
                ``"none"``, ``"batch"``, ``"instance"``, ``"group"``, and
                ``"layer"``.
            residual: Whether to add a residual shortcut to the stacked-layer
                output.
            norm_after_activation: Whether to apply activation before
                normalization within each convolution stage. ``False`` keeps the
                default ``conv -> norm -> activation`` order.
            residual_projection: Projection shortcut policy when ``residual``
                is enabled. ``"auto"`` keeps an identity shortcut when channel
                counts match and projects only on mismatch. ``"always"`` always
                uses a ``1x1`` convolution.
            residual_shortcut_norm: Whether projection shortcuts should include
                the same normalization mode as the main convolution stack.

        Returns:
            ``None``.

        Raises:
            ValueError: If channel counts or ``num_convs`` are not positive, if
                ``dropout`` is negative, or if activation or normalization
                configuration cannot be resolved.
        """

        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        if num_convs <= 0:
            raise ValueError("num_convs must be positive")
        if dropout < 0:
            raise ValueError("dropout must be non-negative")
        if residual_projection not in {"auto", "always"}:
            raise ValueError("residual_projection must be 'auto' or 'always'")

        layers: list[torch_nn.Module] = []
        current_channels = in_channels
        for _ in range(num_convs):
            layers.append(
                torch_nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                )
            )
            norm_layer = _make_norm(norm, out_channels)
            activation_layer = get_activation(activation)
            if norm_after_activation:
                layers.append(activation_layer)
                if norm_layer is not None:
                    layers.append(norm_layer)
            else:
                if norm_layer is not None:
                    layers.append(norm_layer)
                layers.append(activation_layer)
            if dropout > 0:
                layers.append(torch_nn.Dropout2d(p=dropout))
            current_channels = out_channels

        self.layers = torch_nn.Sequential(*layers)
        self.norm_after_activation = bool(norm_after_activation)
        self.residual_projection = residual_projection
        self.residual_shortcut_norm = bool(residual_shortcut_norm)
        self.shortcut: torch_nn.Module | None
        if not residual:
            self.shortcut = None
        elif residual_projection == "auto" and in_channels == out_channels:
            self.shortcut = torch_nn.Identity()
        elif residual_projection == "auto" and not residual_shortcut_norm:
            self.shortcut = torch_nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        else:
            shortcut = torch_nn.Sequential()
            shortcut.add_module(
                "conv",
                torch_nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
            )
            shortcut_norm = _make_norm(norm, out_channels) if residual_shortcut_norm else None
            if shortcut_norm is not None:
                shortcut.add_module(_norm_module_name(norm), shortcut_norm)
            self.shortcut = shortcut

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset trainable parameters owned by this block.

        Convolution weights use Kaiming normal initialization and convolution
        biases are zeroed. Affine normalization weights are set to one and
        affine normalization biases are zeroed.

        Returns:
            ``None``. Modules are updated in place.
        """

        for module in self.modules():
            if isinstance(module, torch_nn.Conv2d):
                _init_conv(module)
            elif isinstance(
                module,
                (
                    torch_nn.BatchNorm2d,
                    torch_nn.GroupNorm,
                    torch_nn.InstanceNorm2d,
                ),
            ):
                if module.weight is not None:
                    torch_nn.init.ones_(module.weight)
                if module.bias is not None:
                    torch_nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolutional stack and optional residual shortcut.

        Args:
            x: Input tensor with shape ``(B, in_channels, H, W)``.

        Returns:
            Tensor with shape ``(B, out_channels, H, W)``.
        """

        out = self.layers(x)
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        return out


def _case_insensitive_torch_nn_class(name: str) -> type[torch_nn.Module] | None:
    """Find a ``torch.nn`` module class by case-insensitive name.

    Args:
        name: Module class name to search for.

    Returns:
        Matching ``torch.nn.Module`` subclass, or ``None`` when no class name
        matches.
    """

    lowered = name.lower()
    for attr_name in dir(torch_nn):
        if attr_name.lower() != lowered:
            continue
        attr = getattr(torch_nn, attr_name)
        if isinstance(attr, type) and issubclass(attr, torch_nn.Module):
            return attr
    return None


def _make_norm(norm: str | None, channels: int) -> torch_nn.Module | None:
    """Create the requested 2D normalization module.

    Args:
        norm: Normalization mode. Supported values are ``None``, ``"none"``,
            ``"batch"``, ``"instance"``, ``"group"``, and ``"layer"``.
        channels: Number of feature channels to normalize.

    Returns:
        Normalization module for ``channels``, or ``None`` when normalization is
        disabled.

    Raises:
        ValueError: If ``norm`` is not a string-or-``None`` mode or names an
            unsupported mode.
    """

    if norm is None:
        return None
    if not isinstance(norm, str):
        raise ValueError(
            "norm must be one of None, 'none', 'batch', 'instance', 'group', or 'layer'"
        )

    mode = norm.lower()
    if mode == "none":
        return None
    if mode == "batch":
        return torch_nn.BatchNorm2d(channels)
    if mode == "instance":
        return torch_nn.InstanceNorm2d(channels, affine=True)
    if mode == "group":
        return torch_nn.GroupNorm(_group_count(channels), channels)
    if mode == "layer":
        return torch_nn.GroupNorm(1, channels)
    raise ValueError(f"Unsupported norm: {norm!r}")


def _norm_module_name(norm: str | None) -> str:
    """Return the stable shortcut module name for a normalization mode.

    Args:
        norm: Normalization mode accepted by ``_make_norm``.

    Returns:
        Module name used inside residual shortcut projections.
    """

    if norm is None:
        return "norm"
    mode = norm.lower()
    if mode == "batch":
        return "batchnorm"
    if mode == "instance":
        return "instancenorm"
    if mode == "group":
        return "groupnorm"
    if mode == "layer":
        return "layernorm"
    return f"{mode}norm"


def _group_count(channels: int) -> int:
    """Choose a valid group count for ``GroupNorm``.

    Args:
        channels: Number of channels that must be divisible by the result.

    Returns:
        The largest valid group count at or below ``32``, preferring ``32``
        when possible.
    """

    if channels % 32 == 0:
        return 32
    for groups in range(min(32, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _init_conv(module: torch_nn.Conv2d) -> None:
    """Initialize a convolution used inside a ``ConvBlock``.

    Args:
        module: Convolution whose weights and optional bias should be reset.

    Returns:
        ``None``. The convolution is modified in place.
    """

    torch_nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")
    if module.bias is not None:
        torch_nn.init.zeros_(module.bias)
