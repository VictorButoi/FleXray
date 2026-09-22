from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import kornia.augmentation as KA
import torch
import torch.nn as nn
from torch.nn import functional as F


class RandomIntensityScale(nn.Module):
    """Randomly scale image intensity per batch item.

    Attributes:
        factor_min: Lower additive scale factor sampled for each item.
        factor_max: Upper additive scale factor sampled for each item.
        prob: Per-item probability of applying the sampled scale.
    """

    def __init__(
        self,
        factors: float | Sequence[float] = 0.1,
        prob: float = 0.7,
    ) -> None:
        """Initialize per-item random intensity scaling.

        Args:
            factors: Scalar symmetric factor range or ``(min, max)`` additive
                scale-factor range. A sampled factor ``f`` scales images by
                ``1 + f``.
            prob: Per-item probability of applying the sampled factor.

        Returns:
            ``None``.

        Raises:
            ValueError: If the factor range or probability is invalid.
        """

        super().__init__()
        factor_min, factor_max = _normalize_float_range(
            factors,
            name="factors",
            symmetric_scalar=True,
        )
        prob = _normalize_probability(prob, name="prob")
        self.register_buffer(
            "factor_min", torch.tensor(factor_min, dtype=torch.float32)
        )
        self.register_buffer(
            "factor_max", torch.tensor(factor_max, dtype=torch.float32)
        )
        self.prob = prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale a 4D image batch when the module is in training mode.

        Args:
            x: Image tensor shaped ``(N, C, H, W)``.

        Returns:
            Intensity-scaled tensor, or ``x`` unchanged in evaluation mode.

        Raises:
            ValueError: If ``x`` is not 4D.
        """

        _require_4d_tensor(x, name="x")
        if not self.training or self.prob == 0.0:
            return x
        batch_size = x.shape[0]
        sample_shape = (batch_size, 1, 1, 1)
        apply_mask = torch.rand(sample_shape, device=x.device) < self.prob
        factors = (
            torch.rand(sample_shape, device=x.device)
            * (self.factor_max.to(x.device) - self.factor_min.to(x.device))
            + self.factor_min.to(x.device)
        ).to(dtype=x.dtype)
        scale = 1.0 + factors * apply_mask.to(dtype=x.dtype)
        return x * scale


class RandomLetterDrop(nn.Module):
    """Stamp a random bright ``L`` or ``R`` marker onto an image batch.

    Attributes:
        p: Per-item probability of stamping a marker.
        min_size: Minimum marker height as a fraction of image height.
        max_size: Maximum marker height as a fraction of image height.
        intensity_range: Inclusive range used to sample marker brightness.
        _l_template: Binary ``L`` template stored as a module buffer.
        _r_template: Binary ``R`` template stored as a module buffer.
    """

    _L_TEMPLATE = (
        (1, 0, 0, 0, 0),
        (1, 0, 0, 0, 0),
        (1, 0, 0, 0, 0),
        (1, 0, 0, 0, 0),
        (1, 0, 0, 0, 0),
        (1, 0, 0, 0, 0),
        (1, 1, 1, 1, 1),
    )
    _R_TEMPLATE = (
        (1, 1, 1, 1, 0),
        (1, 0, 0, 0, 1),
        (1, 0, 0, 0, 1),
        (1, 1, 1, 1, 0),
        (1, 0, 1, 0, 0),
        (1, 0, 0, 1, 0),
        (1, 0, 0, 0, 1),
    )

    def __init__(
        self,
        p: float = 0.25,
        min_size: float = 0.03,
        max_size: float = 0.10,
        intensity_range: Sequence[float] = (0.8, 1.0),
    ) -> None:
        """Initialize random letter marker stamping.

        Args:
            p: Per-item probability of stamping a marker.
            min_size: Minimum marker height as a fraction of image height.
            max_size: Maximum marker height as a fraction of image height.
            intensity_range: Two-value marker brightness range.

        Returns:
            ``None``.

        Raises:
            ValueError: If probabilities, size fractions, or intensity range are
                invalid.
        """

        super().__init__()
        self.p = _normalize_probability(p, name="p")
        self.min_size, self.max_size = _normalize_size_fraction_range(
            min_size,
            max_size,
        )
        self.intensity_range = _normalize_float_range(
            intensity_range,
            name="intensity_range",
        )
        self.register_buffer(
            "_l_template",
            torch.tensor(self._L_TEMPLATE, dtype=torch.float32).view(1, 1, 7, 5),
        )
        self.register_buffer(
            "_r_template",
            torch.tensor(self._R_TEMPLATE, dtype=torch.float32).view(1, 1, 7, 5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Stamp random laterality markers when the module is in training mode.

        Args:
            x: Image tensor shaped ``(N, C, H, W)``.

        Returns:
            Tensor with bright markers stamped onto selected batch items, or
            ``x`` unchanged in evaluation mode.

        Raises:
            ValueError: If ``x`` is not 4D.
        """

        _require_4d_tensor(x, name="x")
        if not self.training or self.p == 0.0:
            return x

        batch_size, channels, height, width = x.shape
        apply_mask = torch.rand(batch_size, device=x.device) < self.p
        if not bool(apply_mask.any()):
            return x

        result = x.clone()
        for batch_idx in range(batch_size):
            if not bool(apply_mask[batch_idx]):
                continue
            template = (
                self._l_template
                if float(torch.rand((), device=x.device)) < 0.5
                else self._r_template
            )
            letter_h = self._sample_marker_height(height, device=x.device)
            letter_w = max(1, min(width, round(letter_h * 5 / 7)))
            stamp = F.interpolate(
                template.to(device=x.device, dtype=x.dtype),
                size=(letter_h, letter_w),
                mode="nearest",
            )
            top = int(
                torch.randint(
                    0,
                    max(1, height - letter_h + 1),
                    (1,),
                    device=x.device,
                ).item()
            )
            left = int(
                torch.randint(
                    0,
                    max(1, width - letter_w + 1),
                    (1,),
                    device=x.device,
                ).item()
            )
            low, high = self.intensity_range
            intensity = float(torch.empty((), device=x.device).uniform_(low, high))
            region = result[
                batch_idx : batch_idx + 1,
                :,
                top : top + letter_h,
                left : left + letter_w,
            ]
            result[
                batch_idx : batch_idx + 1,
                :,
                top : top + letter_h,
                left : left + letter_w,
            ] = torch.maximum(region, stamp.expand(1, channels, -1, -1) * intensity)
        return result

    def _sample_marker_height(self, image_height: int, *, device: torch.device) -> int:
        """Sample a marker height in pixels.

        Args:
            image_height: Height of the image receiving the marker.
            device: Device used for random sampling.

        Returns:
            Integer marker height clamped to fit the image.
        """

        min_height = max(1, int(self.min_size * image_height))
        max_height = max(min_height, int(self.max_size * image_height))
        max_height = min(max_height, image_height)
        min_height = min(min_height, max_height)
        return int(
            torch.randint(min_height, max_height + 1, (1,), device=device).item()
        )


class RandomLabelZoom(nn.Module):
    """Zoom into a randomly selected foreground label centroid.

    Attributes:
        label_aware: Marker used by the pipeline builder to run this transform
            before the Kornia sequence.
        zoom_min: Minimum sampled zoom factor.
        zoom_max: Maximum sampled zoom factor.
        p: Per-item probability of applying label zoom.
        label_ids: Optional sorted foreground channel ids eligible for zooming.
        image_interpolation: Interpolation mode used for image crops.
        label_interpolation: Interpolation mode used for label crops.
    """

    label_aware = True

    def __init__(
        self,
        zoom_range: Sequence[float] = (1.5, 4.0),
        p: float = 0.5,
        label_ids: Sequence[int] | None = None,
        image_interpolation: str = "bilinear",
        label_interpolation: str = "nearest",
    ) -> None:
        """Initialize foreground-centered random zoom.

        Args:
            zoom_range: Two-value inclusive range of zoom factors. The minimum
                must be at least ``1.0``.
            p: Per-item probability of applying the transform.
            label_ids: Optional foreground label-channel ids to sample from.
            image_interpolation: ``torch.nn.functional.interpolate`` mode for
                image crops.
            label_interpolation: ``torch.nn.functional.interpolate`` mode for
                label crops.

        Returns:
            ``None``.

        Raises:
            ValueError: If the zoom range, probability, label ids, or
                interpolation modes are invalid.
        """

        super().__init__()
        zoom_min, zoom_max = _normalize_float_range(zoom_range, name="zoom_range")
        if zoom_min < 1.0:
            raise ValueError(f"zoom_range minimum must be >= 1.0, got {zoom_min}.")
        self.zoom_min = zoom_min
        self.zoom_max = zoom_max
        self.p = _normalize_probability(p, name="p")
        self.label_ids = _normalize_label_ids(label_ids)
        self.image_interpolation = _normalize_interpolation_mode(
            image_interpolation,
            name="image_interpolation",
            allowed={"nearest", "nearest-exact", "bilinear", "bicubic", "area"},
        )
        self.label_interpolation = _normalize_interpolation_mode(
            label_interpolation,
            name="label_interpolation",
            allowed={"nearest", "nearest-exact"},
        )

    def forward(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply foreground-centered random zoom to image and label batches.

        Args:
            image: Image tensor shaped ``(N, C, H, W)``.
            label: Multi-channel label tensor shaped ``(N, C_label, H, W)`` with
                foreground channels at indices greater than zero.

        Returns:
            Zoomed image and label tensors. Samples without eligible foreground
            pixels are returned unchanged.

        Raises:
            ValueError: If tensors are not 4D, devices differ, or batch/spatial
                dimensions do not align.
        """

        _validate_image_label(image, label)
        if not self.training or self.p == 0.0 or label.shape[1] < 2:
            return image, label

        foreground_channels = self._foreground_channels(
            num_label_channels=label.shape[1],
            device=label.device,
        )
        if foreground_channels.numel() == 0:
            return image, label

        output_image = image.clone()
        output_label = label.clone()
        for batch_idx in range(image.shape[0]):
            if float(torch.rand((), device=image.device)) >= self.p:
                continue
            present_channels = self._present_channels(
                label[batch_idx],
                foreground_channels,
            )
            if present_channels.numel() == 0:
                continue
            selected_channel = int(
                present_channels[
                    torch.randint(
                        present_channels.numel(),
                        (1,),
                        device=image.device,
                    )
                ].item()
            )
            crop_slices = self._sample_crop_slices(
                label[batch_idx, selected_channel],
                image_height=image.shape[-2],
                image_width=image.shape[-1],
                device=image.device,
            )
            output_image[batch_idx] = self._resize_image_crop(
                image[batch_idx : batch_idx + 1, :, crop_slices[0], crop_slices[1]],
                output_size=image.shape[-2:],
            )[0]
            output_label[batch_idx] = self._resize_label_crop(
                label[batch_idx : batch_idx + 1, :, crop_slices[0], crop_slices[1]],
                output_size=label.shape[-2:],
            )[0]
        return output_image, output_label

    def _foreground_channels(
        self,
        *,
        num_label_channels: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return eligible foreground channel ids.

        Args:
            num_label_channels: Total number of label channels.
            device: Device for the returned tensor.

        Returns:
            Long tensor of foreground channel ids.
        """

        if self.label_ids is None:
            channels = range(1, num_label_channels)
        else:
            channels = (idx for idx in self.label_ids if 1 <= idx < num_label_channels)
        return torch.tensor(tuple(channels), dtype=torch.long, device=device)

    def _present_channels(
        self,
        sample_label: torch.Tensor,
        foreground_channels: torch.Tensor,
    ) -> torch.Tensor:
        """Filter candidate channels to those with positive pixels.

        Args:
            sample_label: One sample label tensor shaped ``(C_label, H, W)``.
            foreground_channels: Candidate foreground channel ids.

        Returns:
            Long tensor of channels present in ``sample_label``.
        """

        present_mask = (sample_label[foreground_channels] > 0).flatten(1).any(dim=1)
        return foreground_channels[present_mask]

    def _sample_crop_slices(
        self,
        foreground_mask: torch.Tensor,
        *,
        image_height: int,
        image_width: int,
        device: torch.device,
    ) -> tuple[slice, slice]:
        """Sample a zoom crop centered on one foreground mask centroid.

        Args:
            foreground_mask: Two-dimensional selected foreground mask.
            image_height: Full image height.
            image_width: Full image width.
            device: Device used for random sampling.

        Returns:
            Pair of height and width slices into the original image.
        """

        coords = torch.nonzero(foreground_mask > 0, as_tuple=False).to(
            dtype=torch.float32
        )
        center_y, center_x = coords.mean(dim=0).tolist()
        zoom = float(
            torch.empty((), device=device).uniform_(self.zoom_min, self.zoom_max)
        )
        crop_h = max(1, min(image_height, int(round(image_height / zoom))))
        crop_w = max(1, min(image_width, int(round(image_width / zoom))))
        top = int(round(center_y - crop_h / 2))
        left = int(round(center_x - crop_w / 2))
        top = max(0, min(image_height - crop_h, top))
        left = max(0, min(image_width - crop_w, left))
        return slice(top, top + crop_h), slice(left, left + crop_w)

    def _resize_image_crop(
        self,
        crop: torch.Tensor,
        *,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        """Resize an image crop to the original spatial size.

        Args:
            crop: Cropped image tensor shaped ``(1, C, h, w)``.
            output_size: Target ``(H, W)`` size.

        Returns:
            Resized image crop.
        """

        kwargs: dict[str, Any] = {"mode": self.image_interpolation}
        if self.image_interpolation in {"bilinear", "bicubic"}:
            kwargs["align_corners"] = False
        return F.interpolate(crop, size=output_size, **kwargs)

    def _resize_label_crop(
        self,
        crop: torch.Tensor,
        *,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        """Resize a label crop to the original spatial size.

        Args:
            crop: Cropped label tensor shaped ``(1, C_label, h, w)``.
            output_size: Target ``(H, W)`` size.

        Returns:
            Resized label crop.
        """

        return F.interpolate(crop, size=output_size, mode=self.label_interpolation)


def _normalize_float_range(
    value: float | Sequence[float],
    *,
    name: str,
    symmetric_scalar: bool = False,
) -> tuple[float, float]:
    """Normalize a scalar or two-value floating-point range.

    Args:
        value: Scalar or sequence to normalize.
        name: Field name for validation errors.
        symmetric_scalar: Whether scalar inputs create ``(-abs(x), abs(x))``.

    Returns:
        ``(minimum, maximum)`` float tuple.

    Raises:
        ValueError: If the value cannot define a non-decreasing range.
    """

    if isinstance(value, (int, float)):
        scalar = float(value)
        if symmetric_scalar:
            minimum, maximum = -abs(scalar), abs(scalar)
        else:
            minimum, maximum = scalar, scalar
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError(f"{name} must contain exactly two values.")
        minimum, maximum = float(value[0]), float(value[1])
    else:
        raise ValueError(f"{name} must be a number or two-value range.")
    if minimum > maximum:
        raise ValueError(f"{name} minimum must be <= maximum, got {value!r}.")
    return minimum, maximum


def _normalize_probability(value: float, *, name: str) -> float:
    """Validate a probability value.

    Args:
        value: Candidate probability.
        name: Field name for validation errors.

    Returns:
        Float probability.

    Raises:
        ValueError: If the value is outside ``[0, 1]``.
    """

    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}.")
    return probability


def _normalize_size_fraction_range(
    min_size: float,
    max_size: float,
) -> tuple[float, float]:
    """Validate marker size fractions.

    Args:
        min_size: Minimum marker height fraction.
        max_size: Maximum marker height fraction.

    Returns:
        ``(min_size, max_size)`` as floats.

    Raises:
        ValueError: If either fraction is invalid.
    """

    min_value = float(min_size)
    max_value = float(max_size)
    if min_value <= 0.0 or max_value <= 0.0:
        raise ValueError("min_size and max_size must be positive.")
    if min_value > max_value:
        raise ValueError("min_size must be <= max_size.")
    return min_value, max_value


def _normalize_label_ids(label_ids: Sequence[int] | None) -> tuple[int, ...] | None:
    """Validate optional foreground label ids.

    Args:
        label_ids: Optional sequence of label-channel ids.

    Returns:
        Sorted unique label ids, or ``None``.

    Raises:
        ValueError: If any id is negative.
    """

    if label_ids is None:
        return None
    normalized = tuple(sorted({int(label_id) for label_id in label_ids}))
    if any(label_id < 0 for label_id in normalized):
        raise ValueError("label_ids must be non-negative channel ids.")
    return normalized


def _normalize_interpolation_mode(
    mode: str,
    *,
    name: str,
    allowed: set[str],
) -> str:
    """Validate an interpolation mode.

    Args:
        mode: Candidate interpolation mode.
        name: Field name for validation errors.
        allowed: Accepted mode names.

    Returns:
        Normalized interpolation mode.

    Raises:
        ValueError: If the mode is not supported for this transform.
    """

    normalized = str(mode).strip()
    if normalized not in allowed:
        allowed_modes = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of {allowed_modes}, got {mode!r}.")
    return normalized


def _require_4d_tensor(tensor: torch.Tensor, *, name: str) -> None:
    """Validate a tensor has 4D image-batch shape.

    Args:
        tensor: Tensor to validate.
        name: Tensor name for validation errors.

    Returns:
        ``None``.

    Raises:
        ValueError: If ``tensor`` is not 4D.
    """

    if tensor.ndim != 4:
        raise ValueError(
            f"{name} must be a 4D (N, C, H, W) tensor, got shape {tuple(tensor.shape)}."
        )


def _validate_image_label(image: torch.Tensor, label: torch.Tensor) -> None:
    """Validate paired image and label tensors.

    Args:
        image: Image tensor candidate.
        label: Label tensor candidate.

    Returns:
        ``None``.

    Raises:
        ValueError: If shapes or devices are incompatible.
    """

    _require_4d_tensor(image, name="image")
    _require_4d_tensor(label, name="label")
    if image.device != label.device:
        raise ValueError(
            f"image and label must be on the same device, got {image.device} and {label.device}."
        )
    if image.shape[0] != label.shape[0] or image.shape[-2:] != label.shape[-2:]:
        raise ValueError(
            "image and label must have aligned batch and spatial dimensions, "
            f"got image={tuple(image.shape)} and label={tuple(label.shape)}."
        )


class RandomClaheOrGamma(nn.Module):
    """Apply CLAHE, gamma correction, or neither, exclusively per sample.

    ``clahe_prob`` and ``gamma_prob`` are marginal probabilities whose sum must
    not exceed one; the remainder leaves the image unchanged. Keeping the two
    branches exclusive avoids stacking two strong tone transforms on one image.

    Attributes:
        clahe_prob: Probability of applying CLAHE to a sample.
        gamma_prob: Probability of applying gamma correction to a sample.
        same_on_batch: Whether one draw decides the branch for the whole batch.
        clahe: Kornia CLAHE transform applied to the selected samples.
        gamma: Kornia gamma transform applied to the selected samples.
    """

    def __init__(
        self,
        clip_limit: Sequence[float] = (1.0, 1.5),
        gamma: Sequence[float] = (0.92, 1.08),
        gain: Sequence[float] = (1.0, 1.0),
        clahe_prob: float = 0.1,
        gamma_prob: float = 0.25,
        grid_size: Sequence[int] = (8, 8),
        slow_and_differentiable: bool = False,
        same_on_batch: bool = False,
    ) -> None:
        """Configure the exclusive CLAHE/gamma branches.

        Args:
            clip_limit: CLAHE clip-limit sampling range.
            gamma: Gamma exponent sampling range.
            gain: Gamma gain sampling range.
            clahe_prob: Marginal probability of the CLAHE branch.
            gamma_prob: Marginal probability of the gamma branch.
            grid_size: CLAHE tile grid ``(rows, cols)``.
            slow_and_differentiable: Kornia CLAHE differentiable mode flag.
            same_on_batch: Whether the branch choice is shared across the batch.

        Returns:
            ``None``.
        """

        super().__init__()
        self.clahe_prob = _normalize_probability(clahe_prob, name="clahe_prob")
        self.gamma_prob = _normalize_probability(gamma_prob, name="gamma_prob")
        assert self.clahe_prob + self.gamma_prob <= 1.0, (
            f"clahe_prob + gamma_prob must be <= 1; got {self.clahe_prob + self.gamma_prob}."
        )
        self.same_on_batch = bool(same_on_batch)
        self.clahe = KA.RandomClahe(
            clip_limit=tuple(clip_limit),
            grid_size=tuple(grid_size),
            slow_and_differentiable=slow_and_differentiable,
            same_on_batch=same_on_batch,
            p=1.0,
            keepdim=True,
        )
        self.gamma = KA.RandomGamma(
            gamma=tuple(gamma),
            gain=tuple(gain),
            same_on_batch=same_on_batch,
            p=1.0,
            keepdim=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one exclusive branch per sample while training.

        Args:
            x: Image batch shaped ``(N, C, H, W)`` with intensities in ``[0, 1]``.

        Returns:
            Batch with CLAHE, gamma, or no change applied per sample.
        """

        if not self.training:
            return x
        _require_4d_tensor(x, name="x")
        batch_size = int(x.shape[0])
        draws = torch.rand(1 if self.same_on_batch else batch_size, device=x.device)
        choices = draws.expand(batch_size) if self.same_on_batch else draws
        clahe_mask = choices < self.clahe_prob
        gamma_mask = (choices >= self.clahe_prob) & (
            choices < self.clahe_prob + self.gamma_prob
        )
        output = x.clone()
        if clahe_mask.any():
            output[clahe_mask] = self.clahe(x[clahe_mask])
        if gamma_mask.any():
            output[gamma_mask] = self.gamma(x[gamma_mask])
        return output


class RandomAspectCrop(KA.AugmentationBase2D):
    """Center-crop one spatial axis and zero-pad back to the input shape.

    Each applied sample independently chooses a landscape crop (reduced height,
    full width) or a portrait crop (full height, reduced width). Target sizes are
    sampled with the same parity as the input axis so both margins are equal.
    The whole configured range must fit inside the input axis;
    route smaller inputs to a preset without this transform. Masks are cropped
    identically so labels outside the crop disappear.

    Attributes:
        height_range: Inclusive target-height range for landscape crops.
        width_range: Inclusive target-width range for portrait crops.
    """

    def __init__(
        self,
        height_range: Sequence[int],
        width_range: Sequence[int],
        p: float = 0.5,
    ) -> None:
        """Validate the crop ranges.

        Args:
            height_range: Inclusive ``(min, max)`` target heights.
            width_range: Inclusive ``(min, max)`` target widths.
            p: Per-sample probability of applying the crop.

        Returns:
            ``None``.
        """

        super().__init__(p=p)
        self.height_range = _normalize_int_range(height_range, name="height_range")
        self.width_range = _normalize_int_range(width_range, name="width_range")

    @staticmethod
    def _parity_bounds(
        configured: tuple[int, int], axis_size: int, *, name: str
    ) -> tuple[int, int]:
        """Return the first parity-compatible size and the count of candidates.

        Args:
            configured: Inclusive configured size range.
            axis_size: Input size along the cropped axis.
            name: Range name used in error messages.

        Returns:
            ``(first, count)`` describing sizes ``first + 2 * k`` for ``k < count``.

        Raises:
            ValueError: If the range exceeds the axis or has no value with the
                axis parity.
        """

        minimum, maximum = configured
        if maximum > axis_size:
            raise ValueError(
                f"RandomAspectCrop {name} {configured} exceeds input axis size "
                f"{axis_size}; the complete configured range must fit within its input axis."
            )
        first = minimum + ((axis_size - minimum) % 2)
        if first > maximum:
            raise ValueError(
                f"RandomAspectCrop {name} {configured} has no value with the same "
                f"parity as input axis size {axis_size}."
            )
        return first, ((maximum - first) // 2) + 1

    def generate_parameters(self, input_shape: torch.Size) -> dict[str, torch.Tensor]:
        """Sample the per-sample crop orientation and target size.

        Args:
            input_shape: Batch shape ``(N, C, H, W)``.

        Returns:
            Parameters ``is_landscape``, ``target_height``, ``target_width``.
        """

        batch_size, _, height, width = (int(value) for value in input_shape)
        first_height, height_count = self._parity_bounds(
            self.height_range, height, name="height_range"
        )
        first_width, width_count = self._parity_bounds(
            self.width_range, width, name="width_range"
        )
        is_landscape = torch.rand(batch_size) < 0.5
        target_heights = torch.full((batch_size,), height, dtype=torch.long)
        target_widths = torch.full((batch_size,), width, dtype=torch.long)
        landscape_count = int(is_landscape.sum().item())
        target_heights[is_landscape] = first_height + 2 * torch.randint(
            0, height_count, (landscape_count,)
        )
        target_widths[~is_landscape] = first_width + 2 * torch.randint(
            0, width_count, (batch_size - landscape_count,)
        )
        return {
            "is_landscape": is_landscape,
            "target_height": target_heights,
            "target_width": target_widths,
        }

    def apply_transform(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],
        transform: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Zero everything outside the centred crop window.

        Args:
            input: Batch shaped ``(N, C, H, W)``.
            params: Parameters from :meth:`generate_parameters`.
            flags: Kornia flags (unused).
            transform: Kornia transform matrix (unused).

        Returns:
            Cropped-and-padded batch with the input shape.
        """

        _, _, height, width = input.shape
        target_heights = params["target_height"].to(device=input.device)
        target_widths = params["target_width"].to(device=input.device)
        tops = (height - target_heights) // 2
        lefts = (width - target_widths) // 2
        rows = torch.arange(height, device=input.device).unsqueeze(0)
        columns = torch.arange(width, device=input.device).unsqueeze(0)
        keep_rows = (rows >= tops.unsqueeze(1)) & (rows < (tops + target_heights).unsqueeze(1))
        keep_columns = (columns >= lefts.unsqueeze(1)) & (
            columns < (lefts + target_widths).unsqueeze(1)
        )
        keep = keep_rows.unsqueeze(2) & keep_columns.unsqueeze(1)
        return torch.where(keep.unsqueeze(1), input, torch.zeros_like(input))

    def apply_non_transform(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],
        transform: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return unselected image samples unchanged.

        Args:
            input: Batch shaped ``(N, C, H, W)``.
            params: Parameters from :meth:`generate_parameters` (unused).
            flags: Kornia flags (unused).
            transform: Kornia transform matrix (unused).

        Returns:
            ``input`` unchanged.
        """

        return input

    def apply_non_transform_mask(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],
        transform: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return unselected mask samples unchanged.

        Args:
            input: Mask batch shaped ``(N, C, H, W)``.
            params: Parameters from :meth:`generate_parameters` (unused).
            flags: Kornia flags (unused).
            transform: Kornia transform matrix (unused).

        Returns:
            ``input`` unchanged.
        """

        return input

    def apply_transform_mask(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],
        transform: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Crop masks with the same window as the image.

        Args:
            input: Mask batch shaped ``(N, C, H, W)``.
            params: Parameters from :meth:`generate_parameters`.
            flags: Kornia flags (unused).
            transform: Kornia transform matrix (unused).

        Returns:
            Cropped mask batch.
        """

        return self.apply_transform(input, params, flags, transform)


def _normalize_int_range(value: Sequence[int], *, name: str) -> tuple[int, int]:
    """Validate an inclusive ``(min, max)`` integer range with positive bounds."""
    values = tuple(value)
    assert len(values) == 2 and all(
        isinstance(v, int) and not isinstance(v, bool) for v in values
    ), f"{name} must contain exactly two integers, got {value!r}."
    minimum, maximum = values
    assert 0 < minimum <= maximum, f"{name} must satisfy 0 < min <= max, got {value!r}."
    return minimum, maximum
