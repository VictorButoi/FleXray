"""Projected-label post-processing for CT->DRR rendering.

``fxr.drr.render_drr`` already projects rendered foreground channels into an
explicit-background label mask. These helpers add the optional Gaussian label
smoothing applied by the segmentation DRR runtime on top of that projection.
"""

from __future__ import annotations

import torch
from kornia.filters import gaussian_blur2d
from torch import Tensor

__all__ = ["prepend_background_channel", "smooth_projected_fg_masks"]


def smooth_projected_fg_masks(
    fg_masks: Tensor,
    *,
    sigma: float,
    kernel_size: tuple[int, int] | None,
) -> Tensor:
    """Gaussian-smooth projected foreground masks.

    Args:
        fg_masks: Foreground masks shaped ``(V, C_fg, H, W)``.
        sigma: Gaussian standard deviation; non-positive values disable smoothing.
        kernel_size: Odd ``(kh, kw)`` kernel size, or ``None`` to disable.

    Returns:
        Smoothed masks clamped to ``[0, 1]``, or ``fg_masks`` unchanged when
        smoothing is disabled.
    """

    if sigma <= 0 or kernel_size is None:
        return fg_masks
    smoothed = gaussian_blur2d(fg_masks, kernel_size=kernel_size, sigma=(sigma, sigma))
    return smoothed.clamp(0.0, 1.0)


def prepend_background_channel(fg_masks: Tensor, *, soft_background: bool) -> Tensor:
    """Prepend a background channel computed from the foreground union.

    Args:
        fg_masks: Foreground masks shaped ``(V, C_fg, H, W)``.
        soft_background: When ``True`` the background is ``1 - max(fg)`` clamped
            to ``[0, 1]``; when ``False`` it is the hard complement of any
            foreground activation.

    Returns:
        Label masks shaped ``(V, 1 + C_fg, H, W)`` with background at channel 0.
    """

    fg_union = fg_masks.amax(dim=1, keepdim=True)
    if soft_background:
        background = (1.0 - fg_union).clamp(0.0, 1.0)
    else:
        background = (fg_union <= 0).to(fg_masks.dtype)
    return torch.cat([background, fg_masks], dim=1)
