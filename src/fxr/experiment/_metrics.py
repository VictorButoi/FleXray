"""Sample counts for the experiment's independent-channel Dice metric."""

from torch import Tensor


def binary_dice_sample_count(y_true: Tensor) -> int:
    """Count images retained by binary Dice with empty labels ignored.

    Background-only images remain eligible because the metric keeps background
    when no foreground channel survives. An image with no positive target pixel
    in any channel contributes neither a score nor a sample to the mean.

    Args:
        y_true: Binary-mode targets shaped ``(B, C, ...)``.

    Returns:
        Number of images with at least one target pixel above ``0.5``.
    """

    return int((y_true > 0.5).flatten(1).any(dim=1).sum().item())
