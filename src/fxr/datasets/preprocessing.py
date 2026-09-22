"""Optional packaging-time preprocessing for X-ray and CT samples.

The released FleXray weights were trained on X-rays that were min-max scaled
per image, zero-padded to a square, and area-resized to ``256x256`` (masks with
nearest-neighbour resampling), and on CT volumes clipped to an HU window. These
helpers reproduce that pipeline inside ``fxr-dataset pack`` and record the
resulting geometry so predictions can be mapped back to the original frame.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

_INTENSITY_MODES = ("per_image_minmax", "dtype_range")


@dataclass(frozen=True)
class XrayPreprocessing:
    """Packaging-time X-ray preprocessing declared by a manifest.

    Attributes:
        intensity: ``"per_image_minmax"`` rescales each image to span ``[0, 1]``
            before padding; ``"dtype_range"`` keeps the packager's fixed
            dtype-range scaling.
        pad_to_square: Whether to centre-pad image and mask to a square with
            zeros (background) before resizing.
        output_size: Optional ``(height, width)`` to resize to; images use area
            (box) averaging and masks nearest-neighbour sampling.
    """

    intensity: str = "per_image_minmax"
    pad_to_square: bool = False
    output_size: tuple[int, int] | None = None

    @classmethod
    def from_manifest(cls, raw: Mapping[str, Any]) -> "XrayPreprocessing":
        """Validate a manifest ``preprocessing`` block for ``xray-seg``.

        Args:
            raw: Mapping with optional ``intensity``, ``pad_to_square``, and
                ``output_size`` keys.

        Returns:
            The validated preprocessing spec.
        """

        assert isinstance(raw, Mapping), "xray-seg preprocessing must be a mapping."
        unknown = sorted(set(raw) - {"intensity", "pad_to_square", "output_size"})
        assert not unknown, f"xray-seg preprocessing has unknown key(s): {unknown}."
        intensity = raw.get("intensity", "per_image_minmax")
        assert intensity in _INTENSITY_MODES, (
            f"preprocessing.intensity must be one of {_INTENSITY_MODES}; got {intensity!r}."
        )
        pad_to_square = raw.get("pad_to_square", False)
        assert isinstance(pad_to_square, bool), "preprocessing.pad_to_square must be a bool."
        output_size = raw.get("output_size")
        if output_size is not None:
            sizes = tuple(output_size)
            assert len(sizes) == 2 and all(
                isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in sizes
            ), f"preprocessing.output_size must be two positive integers; got {output_size!r}."
            output_size = (int(sizes[0]), int(sizes[1]))
        return cls(intensity=str(intensity), pad_to_square=pad_to_square, output_size=output_size)

    def to_attrs(self) -> dict[str, Any]:
        """Return the JSON-compatible form stored in package ``_attrs``.

        Returns:
            Mapping with ``intensity``, ``pad_to_square``, and ``output_size``.
        """

        size = None if self.output_size is None else list(self.output_size)
        return {
            "intensity": self.intensity,
            "pad_to_square": self.pad_to_square,
            "output_size": size,
        }

    def apply(
        self, image: np.ndarray, label: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Preprocess one canonical image/label pair.

        Args:
            image: Canonical ``float32`` image shaped ``(1, H, W)`` in ``[0, 1]``.
            label: Dense ``(H, W)`` map or channel-first ``(C, H, W)`` mask.

        Returns:
            ``(image, label, geometry)`` where ``geometry`` records
            ``original_shape``, ``pad_before``, ``pad_after``, ``padded_shape``,
            ``processed_shape``, and ``resize_scale``.
        """

        original_shape = tuple(int(v) for v in image.shape[-2:])
        if self.intensity == "per_image_minmax":
            image = per_image_minmax(image)
        pad_before, pad_after = (0, 0), (0, 0)
        if self.pad_to_square:
            pad_before, pad_after = _square_padding(original_shape)
            image = _pad_spatial(image, pad_before, pad_after)
            label = _pad_spatial(label, pad_before, pad_after)
        padded_shape = tuple(int(v) for v in image.shape[-2:])
        processed_shape = padded_shape
        if self.output_size is not None:
            processed_shape = self.output_size
            image = _resize_image_area(image, self.output_size)
            label = _resize_mask_nearest(label, self.output_size)
        geometry = {
            "original_shape": list(original_shape),
            "pad_before": list(pad_before),
            "pad_after": list(pad_after),
            "padded_shape": list(padded_shape),
            "processed_shape": list(processed_shape),
            "resize_scale": [
                processed_shape[0] / padded_shape[0],
                processed_shape[1] / padded_shape[1],
            ],
        }
        return np.ascontiguousarray(image), np.ascontiguousarray(label), geometry


@dataclass(frozen=True)
class CTPreprocessing:
    """Packaging-time CT preprocessing declared by a manifest.

    Attributes:
        hu_window: Inclusive ``(low, high)`` HU clipping window; clipped volumes
            are stored as ``float16``.
    """

    hu_window: tuple[float, float]

    @classmethod
    def from_manifest(cls, raw: Mapping[str, Any]) -> "CTPreprocessing":
        """Validate a manifest ``preprocessing`` block for ``ct-seg``.

        Args:
            raw: Mapping with a required ``hu_window`` pair.

        Returns:
            The validated preprocessing spec.
        """

        assert isinstance(raw, Mapping), "ct-seg preprocessing must be a mapping."
        unknown = sorted(set(raw) - {"hu_window"})
        assert not unknown, f"ct-seg preprocessing has unknown key(s): {unknown}."
        window = tuple(float(v) for v in raw.get("hu_window", ()))
        assert len(window) == 2 and window[0] < window[1], (
            f"preprocessing.hu_window must be an increasing (low, high) pair; got {raw!r}."
        )
        return cls(hu_window=(window[0], window[1]))

    def to_attrs(self) -> dict[str, Any]:
        """Return the JSON-compatible form stored in package ``_attrs``.

        Returns:
            Mapping with ``hu_window``.
        """

        return {"hu_window": list(self.hu_window)}

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Clip a CT volume to the HU window and store it as ``float16``.

        Args:
            image: CT volume in Hounsfield units.

        Returns:
            Clipped ``float16`` volume with the input shape.
        """

        low, high = self.hu_window
        return np.ascontiguousarray(np.clip(image, low, high).astype(np.float16))


def per_image_minmax(image: np.ndarray) -> np.ndarray:
    """Rescale one image so its minimum maps to 0 and maximum to 1.

    Args:
        image: Floating-point image array.

    Returns:
        ``float32`` array in ``[0, 1]``; constant images map to zeros.
    """

    values = np.asarray(image, dtype=np.float32)
    minimum, maximum = float(values.min()), float(values.max())
    if maximum <= minimum:
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def _square_padding(shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return centred ``(pad_before, pad_after)`` per axis to make ``shape`` square."""
    size = max(shape)
    before = tuple((size - extent) // 2 for extent in shape)
    after = tuple(size - extent - b for extent, b in zip(shape, before))
    return (before[0], before[1]), (after[0], after[1])


def _pad_spatial(array: np.ndarray, before: tuple[int, int], after: tuple[int, int]) -> np.ndarray:
    """Zero-pad the trailing two (spatial) axes of an array."""
    leading = [(0, 0)] * (array.ndim - 2)
    return np.pad(array, leading + [(before[0], after[0]), (before[1], after[1])])


def _resize_image_area(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a ``(1, H, W)`` float image with area (box) averaging."""
    height, width = size
    resized = Image.fromarray(np.asarray(image[0], dtype=np.float32), mode="F").resize(
        (width, height), Image.BOX
    )
    return np.clip(np.asarray(resized, dtype=np.float32), 0.0, 1.0)[None]


def _resize_mask_nearest(label: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a dense or channel-first mask with nearest-neighbour sampling.

    Uses the OpenCV ``INTER_NEAREST`` index rule ``src = floor(dst * in / out)``
    so label ids stay exact.
    """

    in_height, in_width = label.shape[-2:]
    rows = np.floor(np.arange(size[0]) * in_height / size[0]).astype(np.intp)
    columns = np.floor(np.arange(size[1]) * in_width / size[1]).astype(np.intp)
    return label[..., rows[:, None], columns[None, :]]
