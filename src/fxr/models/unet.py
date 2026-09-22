"""Compact 2D segmentation UNet."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as torch_nn
import torch.nn.functional as F

from fxr.models.nn import ConvBlock


class UNet(torch_nn.Module):
    """Compact 2D UNet for segmentation logits.

    The network applies a configurable encoder, an optional skip-connected
    decoder, and a final ``1x1`` convolution. Outputs are raw logits with no
    sigmoid, softmax, or other activation applied, so callers can choose the
    loss or post-processing appropriate for their label space.

    Attributes:
        skip_connections: Whether decoder blocks concatenate matching encoder
            feature maps before convolution.
        upsample_align_corners: ``align_corners`` setting used by decoder
            interpolation.
        pool: Max-pooling layer used between encoder stages.
        down_blocks: Encoder and bottleneck ``ConvBlock`` stages.
        up_blocks: Decoder ``ConvBlock`` stages.
        output_conv: Final ``1x1`` convolution that maps decoder features to
            ``out_channels`` logits.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filters: Sequence[int],
        up_filters: Sequence[int] | None = None,
        convs_per_block: int = 1,
        bottleneck_convs: int | None = None,
        skip_connections: bool = True,
        dropout: float = 0.0,
        activation: str | type[torch_nn.Module] | torch_nn.Module = "LeakyReLU",
        norm: str | None = None,
        residual: bool = False,
        norm_after_activation: bool = False,
        upsample_align_corners: bool = False,
        residual_projection: Literal["auto", "always"] = "auto",
        residual_shortcut_norm: bool = False,
    ) -> None:
        """Initialize the UNet architecture.

        Args:
            in_channels: Number of input image channels.
            out_channels: Number of output logit channels.
            filters: Encoder channel counts, including the bottleneck stage.
                The sequence must contain at least one positive value.
            up_filters: Decoder channel counts ordered from bottleneck toward
                the output resolution. When omitted, the decoder mirrors
                ``filters[:-1]`` in reverse order.
            convs_per_block: Number of ``3x3`` convolutions in each non-
                bottleneck ``ConvBlock``.
            bottleneck_convs: Optional number of convolutions in the final
                encoder block. When omitted, ``convs_per_block`` is used.
            skip_connections: Whether to concatenate encoder feature maps into
                decoder blocks.
            dropout: Dropout probability passed to each ``ConvBlock``. A value
                of ``0`` disables dropout layers.
            activation: Activation passed to ``ConvBlock``. May be a
                ``torch.nn`` module name, module class, or module instance.
            norm: Normalization mode passed to ``ConvBlock``. Supported values
                are ``None``, ``"none"``, ``"batch"``, ``"instance"``,
                ``"group"``, and ``"layer"``.
            residual: Whether each ``ConvBlock`` adds a residual shortcut around
                its stacked convolutions (identity when channel counts match, a
                ``1x1`` projection otherwise).
            norm_after_activation: Whether ``ConvBlock`` stages apply activation
                before normalization. ``False`` keeps the default order.
            upsample_align_corners: ``align_corners`` value used by bilinear
                decoder interpolation.
            residual_projection: Projection shortcut policy passed to each
                ``ConvBlock`` when residual blocks are enabled.
            residual_shortcut_norm: Whether residual projection shortcuts should
                include the same normalization mode as the main block.

        Returns:
            ``None``.

        Raises:
            ValueError: If channel counts, convolution counts, or decoder stage
                counts are invalid. ``ConvBlock`` may also raise ``ValueError``
                for unsupported activation or normalization settings.
        """

        super().__init__()
        filters = tuple(filters)
        if not filters:
            raise ValueError("filters must contain at least one encoder stage")
        if any(channel_count <= 0 for channel_count in filters):
            raise ValueError("filters must contain positive channel counts")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        if convs_per_block <= 0:
            raise ValueError("convs_per_block must be positive")
        if bottleneck_convs is not None and bottleneck_convs <= 0:
            raise ValueError("bottleneck_convs must be positive when provided")

        decoder_stage_count = len(filters) - 1
        if up_filters is None:
            decoder_filters = tuple(reversed(filters[:-1]))
        else:
            decoder_filters = tuple(up_filters)
            if len(decoder_filters) != decoder_stage_count:
                raise ValueError(
                    "up_filters length must match the number of decoder stages "
                    f"({decoder_stage_count})"
                )
            if any(channel_count <= 0 for channel_count in decoder_filters):
                raise ValueError("up_filters must contain positive channel counts")

        self.skip_connections = bool(skip_connections)
        self.upsample_align_corners = bool(upsample_align_corners)
        self.pool = torch_nn.MaxPool2d(kernel_size=2, stride=2)

        down_blocks: list[ConvBlock] = []
        current_channels = in_channels
        for index, channel_count in enumerate(filters):
            num_convs = bottleneck_convs if index == len(filters) - 1 else convs_per_block
            down_blocks.append(
                ConvBlock(
                    current_channels,
                    channel_count,
                    num_convs=num_convs or convs_per_block,
                    dropout=dropout,
                    activation=activation,
                    norm=norm,
                    residual=residual,
                    norm_after_activation=norm_after_activation,
                    residual_projection=residual_projection,
                    residual_shortcut_norm=residual_shortcut_norm,
                )
            )
            current_channels = channel_count
        self.down_blocks = torch_nn.ModuleList(down_blocks)

        up_blocks: list[ConvBlock] = []
        skip_channels = tuple(reversed(filters[:-1]))
        for skip_channel_count, channel_count in zip(skip_channels, decoder_filters, strict=True):
            block_in_channels = current_channels
            if self.skip_connections:
                block_in_channels += skip_channel_count
            up_blocks.append(
                ConvBlock(
                    block_in_channels,
                    channel_count,
                    num_convs=convs_per_block,
                    dropout=dropout,
                    activation=activation,
                    norm=norm,
                    residual=residual,
                    norm_after_activation=norm_after_activation,
                    residual_projection=residual_projection,
                    residual_shortcut_norm=residual_shortcut_norm,
                )
            )
            current_channels = channel_count
        self.up_blocks = torch_nn.ModuleList(up_blocks)

        self.output_conv = torch_nn.Conv2d(current_channels, out_channels, kernel_size=1, bias=True)
        _init_output_conv(self.output_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the UNet to a batch of 2D images.

        Args:
            x: Input tensor with shape ``(B, C, H, W)``.

        Returns:
            Raw segmentation logits with shape ``(B, out_channels, H, W)``.

        Raises:
            ValueError: If ``x`` is not a 4D tensor representing 2D images.
        """

        if x.ndim != 4:
            raise ValueError("UNet expects 2D image tensors with shape (B, C, H, W)")

        out = x
        skips: list[torch.Tensor] = []
        for index, block in enumerate(self.down_blocks):
            out = block(out)
            if index < len(self.down_blocks) - 1:
                skips.append(out)
                out = self.pool(out)

        for block, skip in zip(self.up_blocks, reversed(skips), strict=True):
            out = F.interpolate(
                out,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=self.upsample_align_corners,
            )
            if self.skip_connections:
                out = torch.cat((out, skip), dim=1)
            out = block(out)

        return self.output_conv(out)


def _init_output_conv(module: torch_nn.Conv2d) -> None:
    """Initialize the final logit projection convolution.

    Args:
        module: ``1x1`` convolution whose weights and optional bias should be
            reset.

    Returns:
        ``None``. The convolution is modified in place.
    """

    torch_nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")
    if module.bias is not None:
        torch_nn.init.zeros_(module.bias)
