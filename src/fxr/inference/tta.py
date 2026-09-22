"""Test-time augmentation for FleXray inference.

Implements the released ``tta_v3`` prediction contract: one un-augmented
forward pass plus ``tta_samples - 1`` randomly augmented passes, merged as a
mean in probability space and converted back to logits. The only geometric
augmentation is a horizontal flip, which is exactly inverted on the logits
before merging. The intensity chain is invert, mutually exclusive CLAHE or
gamma correction, contrast, sharpness, and Gaussian noise. Inputs are expected
as ``BxCxHxW`` tensors with 1 or 3 channels and values in ``[0, 1]``.

Ensembles pass several forward callables: every view (plain and augmented) is
drawn once and run through each member, and the mean is taken over all
``members x views`` probability maps.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
from torch import Tensor
from torch.nn import functional as nnf
from torchvision.transforms import functional as tvf

from .probabilities import _SIGMOID_MODES, ProbabilityMode, probabilities_from_logits

# Released ``tta_v3`` preset. These values intentionally stay local to
# the dependency-light inference surface rather than importing train-only
# Kornia augmentation modules.
_FLIP_PROB = 0.5
_INVERT_PROB = 0.5
_CLAHE_PROB = 0.1
_CLAHE_CLIP_LIMIT_RANGE = (1.0, 2.0)
_CLAHE_GRID_SIZE = (8, 8)
_GAMMA_PROB = 0.25
_GAMMA_RANGE = (0.9, 1.1)
_GAIN_RANGE = (0.9, 1.1)
_CONTRAST_PROB = 0.25
_CONTRAST_RANGE = (0.7, 1.3)
_SHARPNESS_PROB = 0.5
_SHARPNESS_RANGE = (0.7, 1.3)
_GAUSSIAN_NOISE_PROB = 0.25
_GAUSSIAN_NOISE_STD = 0.01


def predict_with_tta(
    forward_fn: Callable[[Tensor], Tensor] | Sequence[Callable[[Tensor], Tensor]],
    images: Tensor,
    *,
    mode: ProbabilityMode | str,
    tta_samples: int,
    seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Run test-time-augmented prediction and merge in probability space.

    Runs every forward callable once on the un-augmented batch and
    ``tta_samples - 1`` times on randomly augmented copies (each augmented view
    is drawn once and shared by all callables), averages the per-pass
    probabilities, and converts the mean back to logits so downstream consumers
    keep a logits/probabilities pair.

    Args:
        forward_fn: Callable, or sequence of ensemble-member callables, mapping
            a ``BxCxHxW`` image tensor to ``BxCxHxW`` logits. The caller owns
            eval mode and gradient/inference guards.
        images: Prepared input tensor with shape ``BxCxHxW`` and values in
            ``[0, 1]``.
        mode: Probability conversion mode forwarded to
            ``probabilities_from_logits``.
        tta_samples: Total number of forward passes per member. Values ``<= 1``
            disable TTA; a single callable then returns its plain single-pass
            logits/probabilities unchanged.
        seed: Optional seed for the augmentation draws, applied to a forked
            RNG so the global one is untouched. Draws are made on the CPU, so
            one seed reproduces the same views on any device.

    Returns:
        Tuple of ``(logits, probabilities)`` tensors with shape ``BxCxHxW``.
        With TTA or several members, ``probabilities`` is the mean over
        ``members x passes`` and ``logits`` is the re-logit-ized mean.

    Raises:
        ValueError: If ``tta_samples`` is negative or no callable is given.
    """

    return _predict_with_tta(
        forward_fn,
        images,
        mode=mode,
        tta_samples=tta_samples,
        normalizer=None,
        seed=seed,
    )


def _predict_with_tta(
    forward_fn: Callable[[Tensor], Tensor]
    | Sequence[Callable[[Tensor], Tensor]],
    images: Tensor,
    *,
    mode: ProbabilityMode | str,
    tta_samples: int,
    normalizer: Callable[[Tensor], Tensor] | None,
    seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Run TTA with an optional bundle-specific augmented-view normalizer.

    Args:
        forward_fn: One model callable or a sequence of ensemble callables.
        images: Prepared input batch shaped ``BxCxHxW``.
        mode: Logits-to-probability conversion mode.
        tta_samples: Total pass count per ensemble member.
        normalizer: Optional callable applied to every augmented batch after
            the intensity chain and before model execution. The plain pass is
            already normalized and is left untouched.
        seed: Optional seed for the augmentation draws.

    Returns:
        Tuple of merged logits and probabilities shaped ``BxCxHxW``.
    """

    forward_fns = _forward_fns(forward_fn)
    resolved_samples = max(_validate_tta_samples(tta_samples), 1)
    logits = forward_fns[0](images)
    probabilities = probabilities_from_logits(logits, mode=mode)
    if resolved_samples == 1 and len(forward_fns) == 1:
        return logits, probabilities

    probability_sum = probabilities
    for member_fn in forward_fns[1:]:
        member_logits = member_fn(images)
        probability_sum = probability_sum + probabilities_from_logits(
            member_logits, mode=mode
        )

    with torch.random.fork_rng(devices=[], enabled=seed is not None):
        if seed is not None:
            # Augmentation draws only from CPU; do not reseed accelerator RNGs.
            torch.random.default_generator.manual_seed(seed)
        for _ in range(resolved_samples - 1):
            augmented, flip_mask = _augment_batch(images)
            if normalizer is not None:
                augmented = normalizer(augmented)
            for member_fn in forward_fns:
                view_logits = _unflip_logits(member_fn(augmented), flip_mask)
                probability_sum = probability_sum + probabilities_from_logits(
                    view_logits, mode=mode
                )
    divisor = float(resolved_samples * len(forward_fns))
    mean_probabilities = probability_sum / divisor
    return (
        _logits_from_probabilities(mean_probabilities, mode=mode),
        mean_probabilities,
    )


def _forward_fns(
    forward_fn: Callable[[Tensor], Tensor] | Sequence[Callable[[Tensor], Tensor]],
) -> tuple[Callable[[Tensor], Tensor], ...]:
    """Normalize one callable or a sequence of callables to a non-empty tuple.

    Args:
        forward_fn: Single forward callable or sequence of member callables.

    Returns:
        Tuple of forward callables.

    Raises:
        ValueError: If the sequence is empty.
    """

    forward_fns = (forward_fn,) if callable(forward_fn) else tuple(forward_fn)
    if not forward_fns:
        raise ValueError("predict_with_tta requires at least one forward callable.")
    return forward_fns


def _validate_tta_samples(tta_samples: int) -> int:
    """Normalize and validate the requested number of TTA passes.

    Args:
        tta_samples: Requested total number of forward passes.

    Returns:
        ``tta_samples`` as an ``int``.

    Raises:
        ValueError: If ``tta_samples`` is negative or not an integer value.
    """

    resolved = int(tta_samples)
    if resolved != tta_samples or resolved < 0:
        raise ValueError(
            f"tta_samples must be a non-negative integer, got {tta_samples!r}."
        )
    return resolved


def _augment_batch(images: Tensor) -> tuple[Tensor, Tensor]:
    """Randomly augment every batch item and record which were flipped.

    Args:
        images: Input tensor with shape ``BxCxHxW``.

    Returns:
        Tuple of the augmented ``BxCxHxW`` tensor and a boolean flip mask with
        shape ``B``.
    """

    views = []
    flip_flags = []
    for image in images:
        view, flipped = _augment_sample(image)
        views.append(view)
        flip_flags.append(flipped)
    flip_mask = torch.tensor(flip_flags, dtype=torch.bool, device=images.device)
    return torch.stack(views, dim=0), flip_mask


def _augment_sample(image: Tensor) -> tuple[Tensor, bool]:
    """Apply the released ``tta_v3`` chain to one ``CxHxW`` image.

    Args:
        image: One image tensor with shape ``CxHxW`` and values in ``[0, 1]``.

    Returns:
        Tuple of the augmented image and whether it was horizontally flipped.
    """

    flipped = _chance(_FLIP_PROB)
    if flipped:
        image = torch.flip(image, dims=(-1,))
    if _chance(_INVERT_PROB):
        image = 1.0 - image
    image = _apply_clahe_or_gamma(image)
    if _chance(_CONTRAST_PROB):
        # Kornia RandomContrast multiplies raw intensities; torchvision uses a
        # different mean-centred definition.
        image = (image * _uniform(*_CONTRAST_RANGE)).clamp(0.0, 1.0)
    if _chance(_SHARPNESS_PROB):
        image = tvf.adjust_sharpness(image, _uniform(*_SHARPNESS_RANGE))
    if _chance(_GAUSSIAN_NOISE_PROB):
        # Drawn on the CPU like every other sample, so one seed reproduces
        # the same views on CPU and CUDA.
        noise = torch.randn(
            image.shape, dtype=image.dtype, device="cpu"
        ).to(image.device)
        image = image + noise * _GAUSSIAN_NOISE_STD
    return image, flipped


def _apply_clahe_or_gamma(image: Tensor) -> Tensor:
    """Apply the mutually exclusive ``tta_v3`` tone-transform branch.

    Args:
        image: One image shaped ``CxHxW`` with values in ``[0, 1]``.

    Returns:
        CLAHE-equalized, gamma-corrected, or unchanged image.
    """

    choice = _uniform(0.0, 1.0)
    if choice < _CLAHE_PROB:
        return _clahe(
            image,
            clip_limit=_uniform(*_CLAHE_CLIP_LIMIT_RANGE),
            grid_size=_CLAHE_GRID_SIZE,
        )
    if choice < _CLAHE_PROB + _GAMMA_PROB:
        return tvf.adjust_gamma(
            image,
            _uniform(*_GAMMA_RANGE),
            gain=_uniform(*_GAIN_RANGE),
        )
    return image


def _clahe(
    image: Tensor,
    *,
    clip_limit: float,
    grid_size: tuple[int, int],
) -> Tensor:
    """Apply dependency-light CLAHE using clipped tile histograms.

    The 256-bin lookup tables and tile-centre interpolation follow the same
    OpenCV contract as Kornia RandomClahe without making Kornia a required
    dependency of the inference-only installation.

    Args:
        image: One image shaped ``CxHxW`` with values in ``[0, 1]``.
        clip_limit: Positive histogram clip-limit factor.
        grid_size: Requested ``(rows, columns)`` tile grid.

    Returns:
        Equalized image with the original shape, dtype, and device.

    Raises:
        ValueError: If the image shape or CLAHE settings are invalid.
    """

    if image.ndim != 3:
        raise ValueError(
            f"CLAHE expects a CxHxW image, got shape {tuple(image.shape)}."
        )
    if not math.isfinite(float(clip_limit)) or float(clip_limit) <= 0.0:
        raise ValueError(
            f"CLAHE clip_limit must be positive and finite, got {clip_limit!r}."
        )
    if len(grid_size) != 2 or any(int(size) <= 0 for size in grid_size):
        raise ValueError(
            f"CLAHE grid_size must contain two positive integers, got {grid_size!r}."
        )

    channels, height, width = (int(value) for value in image.shape)
    grid_rows = min(int(grid_size[0]), height)
    grid_cols = min(int(grid_size[1]), width)
    tile_height = math.ceil(height / grid_rows)
    tile_width = math.ceil(width / grid_cols)
    tile_height += tile_height % 2
    tile_width += tile_width % 2
    padded_height = tile_height * grid_rows
    padded_width = tile_width * grid_cols
    pad_height = padded_height - height
    pad_width = padded_width - width

    work = image.float().clamp(0.0, 1.0)
    if pad_height or pad_width:
        mode = (
            "reflect"
            if pad_height < height and pad_width < width
            else "replicate"
        )
        work = nnf.pad(work, (0, pad_width, 0, pad_height), mode=mode)

    tiles = (
        work.unfold(1, tile_height, tile_height)
        .unfold(2, tile_width, tile_width)
        .permute(1, 2, 0, 3, 4)
        .contiguous()
    )
    tile_pixels = tile_height * tile_width
    flattened = tiles.reshape(-1, tile_pixels)
    bins = (flattened * 255.0).long().clamp_(0, 255)
    offsets = torch.arange(
        flattened.shape[0], device=image.device, dtype=torch.long
    ).unsqueeze(1)
    histograms = torch.bincount(
        (bins + offsets * 256).reshape(-1),
        minlength=int(flattened.shape[0]) * 256,
    ).reshape(-1, 256)

    max_count = max(math.floor(float(clip_limit) * tile_pixels / 256), 1)
    histograms = histograms.clamp(max=max_count)
    clipped = tile_pixels - histograms.sum(dim=1)
    residual = torch.remainder(clipped, 256)
    histograms = histograms + ((clipped - residual) // 256).unsqueeze(1)
    histograms = histograms + (
        torch.arange(256, device=image.device).unsqueeze(0)
        < residual.unsqueeze(1)
    )
    lookup_tables = (
        torch.cumsum(histograms, dim=1).float() * (255.0 / tile_pixels)
    ).floor().clamp_(0.0, 255.0)
    lookup_tables = lookup_tables.reshape(
        grid_rows, grid_cols, channels, 256
    ).permute(2, 0, 1, 3)

    pixel_bins = (work * 255.0).long().clamp_(0, 255)
    y0, y1, y_weight = _clahe_interpolation_axis(
        padded_height, tile_height, grid_rows, device=image.device
    )
    x0, x1, x_weight = _clahe_interpolation_axis(
        padded_width, tile_width, grid_cols, device=image.device
    )
    channel_index = torch.arange(channels, device=image.device)[:, None, None]
    values_00 = lookup_tables[
        channel_index, y0[None, :, None], x0[None, None, :], pixel_bins
    ]
    values_01 = lookup_tables[
        channel_index, y0[None, :, None], x1[None, None, :], pixel_bins
    ]
    values_10 = lookup_tables[
        channel_index, y1[None, :, None], x0[None, None, :], pixel_bins
    ]
    values_11 = lookup_tables[
        channel_index, y1[None, :, None], x1[None, None, :], pixel_bins
    ]
    wy = y_weight[None, :, None]
    wx = x_weight[None, None, :]
    top = values_00 * (1.0 - wx) + values_01 * wx
    bottom = values_10 * (1.0 - wx) + values_11 * wx
    equalized = (top * (1.0 - wy) + bottom * wy) / 255.0
    return equalized[:, :height, :width].to(dtype=image.dtype)


def _clahe_interpolation_axis(
    length: int,
    tile_size: int,
    tile_count: int,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return neighboring tile indices and weights for one CLAHE axis.

    Args:
        length: Padded image-axis length.
        tile_size: Tile extent along the axis.
        tile_count: Number of tiles along the axis.
        device: Device for the returned tensors.

    Returns:
        Lower indices, upper indices, and upper-tile blend weights.
    """

    position = torch.arange(length, device=device, dtype=torch.float32)
    position = position / float(tile_size) - 0.5
    lower_unclamped = torch.floor(position).long()
    upper_unclamped = lower_unclamped + 1
    weight = position - lower_unclamped.float()
    lower = lower_unclamped.clamp(0, tile_count - 1)
    upper = upper_unclamped.clamp(0, tile_count - 1)
    return lower, upper, weight


def _unflip_logits(logits: Tensor, flip_mask: Tensor) -> Tensor:
    """Invert the horizontal flip on logits for the flipped batch items.

    Args:
        logits: Model output with shape ``BxCxHxW`` for one augmented view.
        flip_mask: Boolean tensor with shape ``B`` marking flipped items.

    Returns:
        Logits with flipped items un-flipped along the width axis.
    """

    if not bool(flip_mask.any()):
        return logits
    if bool(flip_mask.all()):
        return torch.flip(logits, dims=(-1,))
    unflipped = logits.clone()
    unflipped[flip_mask] = torch.flip(logits[flip_mask], dims=(-1,))
    return unflipped


def _logits_from_probabilities(
    probabilities: Tensor,
    *,
    mode: ProbabilityMode | str,
) -> Tensor:
    """Convert merged probabilities back into logits.

    Args:
        probabilities: Mean probability tensor with shape ``BxCxHxW``.
        mode: Probability mode used for the forward conversion. Sigmoid modes
            (and any single-channel output) use the logit function; softmax
            modes use ``log``.

    Returns:
        Logit tensor with the same shape, dtype, and device.
    """

    eps = 1e-4 if probabilities.dtype == torch.float16 else 1e-6
    clamped = probabilities.clamp(eps, 1.0 - eps)
    resolved_mode = str(mode).strip().lower()
    if int(probabilities.shape[1]) == 1 or resolved_mode in _SIGMOID_MODES:
        return torch.log(clamped) - torch.log1p(-clamped)
    return torch.log(clamped)


def _chance(prob: float) -> bool:
    """Sample one Bernoulli decision from the torch RNG.

    Args:
        prob: Probability of returning ``True``.

    Returns:
        Whether the event fired.
    """

    return bool(torch.rand((), device="cpu") < prob)


def _uniform(low: float, high: float) -> float:
    """Sample one uniform float from the torch RNG.

    Args:
        low: Inclusive lower bound.
        high: Exclusive upper bound.

    Returns:
        Sampled value.
    """

    return float(torch.empty((), device="cpu").uniform_(low, high))


__all__ = [
    "predict_with_tta",
]
