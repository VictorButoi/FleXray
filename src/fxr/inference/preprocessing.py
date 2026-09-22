"""Image preprocessing used by FleXray inference entry points."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.nn import functional as nnf

from .dicom import is_dicom_path, load_dicom_grayscale, scale_to_unit_range

DEFAULT_IMAGE_SIZE = (256, 256)
DEFAULT_NORMALIZATION_EPS = 1.0e-8
DEFAULT_NORMALIZATION_PERCENTILES = (0.5, 99.5)
_NORMALIZATION_SCHEMES = frozenset(
    {"none", "minmax", "percentile_minmax", "standardize"}
)


@dataclass(frozen=True)
class ImagePreprocessing:
    """Preprocessing contract for public FleXray inference.

    Attributes:
        image_size: Output ``(height, width)`` for every input type.
        color_mode: Color conversion mode. The public v1 contract supports only
            ``"grayscale"``.
        pad_to_square: Whether inputs are zero-padded to square
            before resizing.
        scale: Pixel scaling mode. ``"zero_one"`` maps integer images by
            their dtype range up to 16 bits and by their per-image range for
            wider integers; ``"none"`` leaves values unchanged after
            grayscale conversion and ``float32`` casting.
        mean: Optional scalar subtracted after scaling.
        std: Optional positive scalar divisor applied after ``mean``.
        normalization_scheme: Per-sample normalization applied after scaling.
            ``"none"`` preserves the legacy scalar ``mean``/``std`` behavior;
            the other schemes mirror FleXray training normalization.
        percentiles: Lower and upper percentiles used by
            ``"percentile_minmax"``.
        eps: Positive denominator floor used by per-sample normalization.
    """

    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
    color_mode: str = "grayscale"
    pad_to_square: bool = True
    scale: str = "zero_one"
    mean: float | None = None
    std: float | None = None
    normalization_scheme: str = "none"
    percentiles: tuple[float, float] | None = None
    eps: float = DEFAULT_NORMALIZATION_EPS

    def __post_init__(self) -> None:
        """Validate directly constructed preprocessing contracts.

        Returns:
            ``None``.
        """

        _validate_image_size(self.image_size)
        if self.color_mode != "grayscale":
            raise ValueError(
                "FleXray inference currently supports only grayscale preprocessing; "
                f"got color_mode={self.color_mode!r}."
            )
        if self.scale not in {"zero_one", "none"}:
            raise ValueError(
                "preprocessing scale must be 'zero_one' or 'none', "
                f"got {self.scale!r}."
            )
        _validate_normalization(
            scheme=self.normalization_scheme,
            percentiles=self.percentiles,
            eps=self.eps,
            mean=self.mean,
            std=self.std,
        )

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any] | None) -> "ImagePreprocessing":
        """Build preprocessing from ``preprocessing.json`` metadata.

        Args:
            metadata: Parsed metadata mapping, or ``None`` for the default v1
                preprocessing contract.

        Returns:
            ``ImagePreprocessing`` with validated values.

        Raises:
            ValueError: If size, color mode, scale, or normalization values are
                unsupported.
        """

        data = dict(metadata or {})
        size_value = data.get("image_size", data.get("size", DEFAULT_IMAGE_SIZE))
        image_size = _validate_image_size(size_value)
        color_mode = str(data.get("color_mode", "grayscale")).lower()
        if color_mode != "grayscale":
            raise ValueError(
                "FleXray inference currently supports only grayscale preprocessing; "
                f"got color_mode={color_mode!r}."
            )
        scale = str(data.get("scale", "zero_one")).lower()
        if scale not in {"zero_one", "none"}:
            raise ValueError(
                "preprocessing scale must be 'zero_one' or 'none', "
                f"got {scale!r}."
            )
        normalization = _normalization_values(data)
        return cls(
            image_size=image_size,
            color_mode=color_mode,
            pad_to_square=bool(data.get("pad_to_square", True)),
            scale=scale,
            mean=normalization["mean"],
            std=normalization["std"],
            normalization_scheme=normalization["scheme"],
            percentiles=normalization["percentiles"],
            eps=normalization["eps"],
        )

    def to_metadata(self) -> dict[str, Any]:
        """Return this preprocessing contract as JSON-serializable metadata.

        Args:
            None.

        Returns:
            Dictionary suitable for ``preprocessing.json``.
        """

        metadata: dict[str, Any] = {
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "color_mode": self.color_mode,
            "pad_to_square": bool(self.pad_to_square),
            "scale": self.scale,
            "mean": self.mean,
            "std": self.std,
        }
        if self.normalization_scheme != "none":
            normalization: dict[str, Any] = {
                "scheme": self.normalization_scheme,
                "eps": float(self.eps),
            }
            if self.percentiles is not None:
                normalization["percentiles"] = [
                    float(self.percentiles[0]),
                    float(self.percentiles[1]),
                ]
            metadata["normalization"] = normalization
        return metadata

    def content_box(self, height: int, width: int) -> tuple[int, int, int, int]:
        """Locate an input image on the prepared canvas.

        Args:
            height: Input image height in pixels.
            width: Input image width in pixels.

        Returns:
            ``(top, left, bottom, right)`` bounds, in canvas pixels, of the
            rounded input footprint after padding and resizing. A footprint
            smaller than one pixel is expanded to one canvas pixel. Bilinear
            interpolation can mix input and padding at the boundary.

        Raises:
            ValueError: If either input dimension is not positive.
        """

        if height <= 0 or width <= 0:
            raise ValueError("Input height and width must be positive.")
        side = max(height, width)
        canvas_height = side if self.pad_to_square else height
        canvas_width = side if self.pad_to_square else width
        top = (canvas_height - height) // 2
        left = (canvas_width - width) // 2
        scale_y = self.image_size[0] / canvas_height
        scale_x = self.image_size[1] / canvas_width
        out_top = min(round(top * scale_y), self.image_size[0] - 1)
        out_left = min(round(left * scale_x), self.image_size[1] - 1)
        bottom = max(out_top + 1, round((top + height) * scale_y))
        right = max(out_left + 1, round((left + width) * scale_x))
        return out_top, out_left, bottom, right

    def prepare(self, image: str | Path | Image.Image | np.ndarray | Tensor) -> Tensor:
        """Convert one image-like input to a batched tensor.

        Args:
            image: Image path, Pillow image, NumPy array, or tensor. Path and
                Pillow inputs receive the full resize/pad/scale contract. DICOM
                paths (``.dcm``/``.dicom`` or extension-less files with the
                ``DICM`` magic) are decoded with ``pydicom`` and min-max scaled
                to ``[0, 1]`` before the same padding and resize. Tensor
                inputs are coerced to ``BxCxHxW`` ``float32`` and receive the
                same padding and bilinear resize, so every input type reaches
                the model at ``image_size``. Accepted tensor layouts are
                ``HxW``, ``1xHxW``, ``HxWxC``, and ``BxCxHxW``; a ``3xHxW`` or
                ``4xHxW`` tensor is rejected because it could be one color
                image or a batch of grayscale images. Use ``1xCxHxW`` for
                channel-last images whose height is also 1, 3, or 4.

        Returns:
            Tensor shaped ``Bx1xHxW`` at the configured ``image_size``; ``B``
            is 1 except for batched tensor inputs.

        Raises:
            TypeError: If the image type is unsupported.
            ValueError: If tensor or array shape cannot be interpreted.
        """

        if isinstance(image, (str, Path)):
            image = _load_image_file(image)
        if isinstance(image, Image.Image):
            return self._prepare_pillow(image)
        if isinstance(image, np.ndarray):
            return self._prepare_array(image)
        if isinstance(image, torch.Tensor):
            return self._prepare_tensor(image)
        raise TypeError(
            "FleXraySegmenter.predict expects an image path, Pillow image, "
            f"NumPy array, or torch.Tensor; got {type(image).__name__}."
        )

    def _prepare_pillow(
        self, image: Image.Image) -> Tensor:
        """Apply file-style preprocessing without discarding grayscale depth.

        Args:
            image: Pillow image to convert, scale, pad, and resize.

        Returns:
            Normalized tensor shaped ``1x1xHxW``.
        """

        image = ImageOps.exif_transpose(image)
        if (
            image.mode in {"RGBA", "LA", "PA"}
            or "transparency" in image.info
            or (image.palette is not None and image.palette.mode == "RGBA")
        ):
            # Composite over black so transparent pixels match the zero padding.
            canvas = Image.new("RGBA", image.size, (0, 0, 0, 255))
            image = Image.alpha_composite(canvas, image.convert("RGBA"))
        grayscale = (
            image
            if image.mode in {"1", "L", "I", "I;16", "F"}
            else image.convert("L")
        )
        array = np.array(grayscale, copy=True)
        return self._prepare_grayscale_array(
            array, source_dtype=array.dtype
        )

    def _prepare_array(
        self, array: np.ndarray) -> Tensor:
        """Convert a NumPy image array without quantizing integer precision.

        Args:
            array: Grayscale or channel-last image array.

        Returns:
            Normalized tensor shaped ``1x1xHxW``.
        """

        np_array = np.asarray(array)
        if np_array.size == 0:
            raise ValueError("Image inputs must be non-empty.")
        if np_array.ndim not in {2, 3}:
            raise ValueError(
                "NumPy image inputs must have shape HxW or HxWxC; "
                f"got {np_array.shape}."
            )
        source_dtype = np_array.dtype
        if np_array.ndim == 3:
            channels = int(np_array.shape[-1])
            if channels == 1:
                np_array = np_array[..., 0]
            elif channels in {3, 4}:
                rgb = np_array[..., :3].astype(np.float64, copy=False)
                if channels == 4:
                    alpha = _alpha_unit_range(np_array[..., 3], source_dtype=source_dtype)
                    rgb = rgb * alpha[..., None]
                np_array = (
                    rgb[..., 0] * 0.299
                    + rgb[..., 1] * 0.587
                    + rgb[..., 2] * 0.114
                )
            else:
                raise ValueError(
                    "NumPy color images must have 1, 3, or 4 channels; "
                    f"got shape {np_array.shape}."
                )
        return self._prepare_grayscale_array(
            np_array, source_dtype=source_dtype
        )

    def _prepare_grayscale_array(
        self,
        array: np.ndarray,
        *,
        source_dtype: np.dtype[Any],
    ) -> Tensor:
        """Scale, pad, and resize one two-dimensional grayscale array.

        Args:
            array: Two-dimensional grayscale values.
            source_dtype: Original pixel dtype used for integer range scaling.

        Returns:
            Normalized tensor shaped ``1x1xHxW``.
        """

        if array.size == 0:
            raise ValueError("Image inputs must be non-empty.")
        if self.scale == "zero_one":
            values = _scale_numpy_integer_range(array, source_dtype=source_dtype)
        else:
            values = np.asarray(array, dtype=np.float32)
        tensor = torch.from_numpy(np.array(values, copy=True)).unsqueeze(0).unsqueeze(0)
        values = _check_pixel_values(tensor)[0, 0].numpy()
        prepared = Image.fromarray(values)
        if self.pad_to_square:
            prepared = _pad_to_square(prepared)
        height, width = self.image_size
        resized = prepared.resize(
            (width, height), resample=Image.Resampling.BILINEAR
        )
        resized_array = np.array(resized, dtype=np.float32, copy=True)
        tensor = torch.from_numpy(resized_array).unsqueeze(0).unsqueeze(0)
        return self._normalize_tensor(tensor)

    def _prepare_tensor(self, image: Tensor) -> Tensor:
        """Scale, pad, resize, and normalize an already-loaded tensor or batch.

        Args:
            image: Image or batch tensor in one of the accepted layouts.

        Returns:
            Float tensor shaped ``Bx1xHxW`` after configured normalization.

        Raises:
            ValueError: If the tensor layout cannot be interpreted, including a
                ``3xHxW``/``4xHxW`` tensor for a grayscale bundle.
        """

        source_dtype = image.dtype
        if image.numel() == 0:
            raise ValueError("Image inputs must be non-empty.")
        tensor = image.detach().clone()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3:
            leading = int(tensor.shape[0])
            if leading == 1:
                tensor = tensor.unsqueeze(0)
            elif leading in {3, 4}:
                raise ValueError(
                    f"A {leading}xHxW tensor is ambiguous for a grayscale bundle: it "
                    f"could be one {'RGB' if leading == 3 else 'RGBA'} image or a batch "
                    f"of {leading} grayscale images. Pass 1x{leading}xHxW for one color "
                    f"image or {leading}x1xHxW for a batch; got shape {tuple(tensor.shape)}."
                )
            elif int(tensor.shape[-1]) in {1, 3, 4}:
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError(
                    "3D tensor image inputs must be channel-first CxHxW or "
                    f"channel-last HxWxC; got shape {tuple(tensor.shape)}."
                )
        elif tensor.ndim != 4:
            raise ValueError(
                "Tensor image inputs must have shape HxW, CxHxW, HxWxC, or "
                f"BxCxHxW; got shape {tuple(tensor.shape)}."
            )

        if int(tensor.shape[1]) != 1:
            tensor = _rgb_like_to_grayscale(tensor, source_dtype=source_dtype)
        if not torch.is_floating_point(image) and self.scale == "zero_one":
            tensor = _scale_tensor_integer_range(tensor, source_dtype=source_dtype)
        tensor = _check_pixel_values(tensor.to(dtype=torch.float32))
        if self.pad_to_square:
            tensor = _pad_tensor_to_square(tensor)
        if tuple(int(dim) for dim in tensor.shape[-2:]) != tuple(self.image_size):
            tensor = nnf.interpolate(
                tensor,
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return self._normalize_tensor(tensor)

    def _normalize_tensor(self, tensor: Tensor) -> Tensor:
        """Apply configured per-sample or legacy scalar normalization.

        Args:
            tensor: Prepared floating-point image batch.

        Returns:
            Contiguous normalized tensor with the same shape.
        """

        normalized = tensor
        dims = tuple(range(1, tensor.ndim))
        if self.normalization_scheme == "standardize":
            mean = normalized.mean(dim=dims, keepdim=True)
            std = normalized.std(dim=dims, keepdim=True)
            normalized = (normalized - mean) / (std + float(self.eps))
        elif self.normalization_scheme == "minmax":
            batch_size = normalized.shape[0]
            flat = normalized.reshape(batch_size, -1)
            lower, upper = flat.aminmax(dim=1)
            view_shape = [batch_size] + [1] * (normalized.ndim - 1)
            lower = lower.view(*view_shape)
            upper = upper.view(*view_shape)
            normalized = (normalized - lower) / (upper - lower).clamp(
                min=float(self.eps)
            )
        elif self.normalization_scheme == "percentile_minmax":
            normalized = _percentile_minmax(
                normalized,
                percentiles=self.percentiles
                or DEFAULT_NORMALIZATION_PERCENTILES,
                eps=self.eps,
            )
        if self.mean is not None:
            normalized = normalized - float(self.mean)
        if self.std is not None:
            normalized = normalized / float(self.std)
        return normalized.contiguous()


def _load_image_file(path: str | Path) -> Image.Image | np.ndarray:
    """Decode an image file without applying model preprocessing.

    Args:
        path: Pillow-readable image or DICOM path.

    Returns:
        Detached Pillow image or DICOM grayscale array. No file handle remains
        open, and decoder errors propagate to the caller.
    """

    if is_dicom_path(path):
        return load_dicom_grayscale(path)
    with Image.open(path) as opened:
        opened.load()
        return opened.copy()


def _validate_image_size(size: Sequence[int]) -> tuple[int, int]:
    """Validate and return an image ``(height, width)`` pair.

    Args:
        size: Candidate image size.

    Returns:
        Two positive integers.

    Raises:
        ValueError: If the size is malformed or non-positive.
    """

    if len(size) != 2:
        raise ValueError(f"Image size must be an H W pair, got {tuple(size)}.")
    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError(
            f"Image size dimensions must be positive, got {(height, width)}."
        )
    return height, width


def _scale_numpy_integer_range(
    array: np.ndarray,
    *,
    source_dtype: np.dtype[Any],
) -> np.ndarray:
    """Map integer pixels into ``[0, 1]`` and return ``float32`` values.

    Eight- and sixteen-bit integers use their dtype bounds. Wider integers
    rarely fill their container, and subtracting their dtype minimum collapses
    the image to a constant in ``float32``, so they are min-max scaled per
    image exactly like DICOM pixels.

    Args:
        array: Source pixels before any floating-point cast.
        source_dtype: Original NumPy dtype before grayscale conversion.

    Returns:
        Scaled ``float32`` pixels for integers, otherwise a ``float32`` cast.
    """

    dtype = np.dtype(source_dtype)
    if np.issubdtype(dtype, np.bool_) or not np.issubdtype(dtype, np.integer):
        return np.asarray(array, dtype=np.float32)
    if dtype.itemsize > 2:
        return scale_to_unit_range(array)
    values = np.asarray(array, dtype=np.float32)
    limits = np.iinfo(dtype)
    if limits.min >= 0:
        return values / float(limits.max)
    return (values - float(limits.min)) / float(limits.max - limits.min)


def _scale_tensor_integer_range(tensor: Tensor, *, source_dtype: torch.dtype) -> Tensor:
    """Map an integer-sourced ``BxCxHxW`` tensor into ``[0, 1]``.

    Mirrors :func:`_scale_numpy_integer_range`: dtype bounds for integers up
    to sixteen bits, per-sample min-max scaling for wider ones.

    Args:
        tensor: Integer pixels, or ``float64`` luminance after color conversion.
        source_dtype: Original torch dtype of the input.

    Returns:
        Scaled ``float32`` tensor; ``bool`` inputs retain their 0/1 values.
    """

    if source_dtype == torch.bool:
        return tensor.float()
    dtype_info = torch.iinfo(source_dtype)
    if dtype_info.bits > 16:
        view_shape = [tensor.shape[0]] + [1] * (tensor.ndim - 1)
        if not torch.is_floating_point(tensor):
            tensor = tensor.to(torch.int64)
            if dtype_info.min == 0 and dtype_info.bits == 64:
                # Remap uint64 to signed order without losing low pixel bits.
                tensor = tensor.bitwise_xor(torch.iinfo(torch.int64).min)
            lower, upper = tensor.reshape(tensor.shape[0], -1).aminmax(dim=1)
            # Subtract the endpoint nearest zero when all pixels share a sign.
            # This preserves small differences even beyond float64 precision,
            # while subtraction stays in range for full-width integer inputs.
            origin = lower.clamp(min=0) + upper.clamp(max=0)
            tensor = tensor - origin.view(*view_shape)
        tensor = tensor.double()
        flat = tensor.reshape(tensor.shape[0], -1)
        lower, upper = flat.aminmax(dim=1)
        span = (upper - lower).view(*view_shape)
        span = torch.where(span > 0, span, torch.ones_like(span))
        scaled = (tensor - lower.view(*view_shape)) / span
        return scaled.to(dtype=torch.float32)
    tensor = tensor.float()
    if dtype_info.min >= 0:
        return tensor / float(dtype_info.max)
    return (tensor - float(dtype_info.min)) / float(dtype_info.max - dtype_info.min)


def _normalization_values(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return validated scalar or per-sample normalization metadata.

    Args:
        data: Parsed preprocessing metadata.

    Returns:
        Canonical normalization fields accepted by ``ImagePreprocessing``.
    """

    normalization = data.get("normalization")
    if isinstance(normalization, Mapping):
        if "scheme" in normalization:
            unknown = sorted(
                set(normalization).difference({"scheme", "percentiles", "eps"})
            )
            if unknown:
                raise ValueError(
                    f"Unsupported preprocessing normalization keys: {unknown}."
                )
            scheme = normalization.get("scheme")
            percentiles = normalization.get("percentiles")
            eps = normalization.get("eps", DEFAULT_NORMALIZATION_EPS)
            mean = data.get("mean")
            std = data.get("std")
        else:
            scheme = "none"
            percentiles = None
            eps = DEFAULT_NORMALIZATION_EPS
            mean = normalization.get("mean", data.get("mean"))
            std = normalization.get("std", data.get("std"))
    elif normalization is None:
        scheme = "none"
        percentiles = None
        eps = DEFAULT_NORMALIZATION_EPS
        mean = data.get("mean")
        std = data.get("std")
    else:
        raise TypeError("preprocessing normalization must be a mapping.")
    canonical_mean = None if mean is None else float(mean)
    canonical_std = None if std is None else float(std)
    canonical_percentiles = (
        None if percentiles is None else _validate_percentiles(percentiles)
    )
    canonical_eps = _validate_eps(eps)
    _validate_normalization(
        scheme=scheme,
        percentiles=canonical_percentiles,
        eps=canonical_eps,
        mean=canonical_mean,
        std=canonical_std,
    )
    return {
        "scheme": scheme,
        "percentiles": canonical_percentiles,
        "eps": canonical_eps,
        "mean": canonical_mean,
        "std": canonical_std,
    }


def _validate_normalization(
    *,
    scheme: object,
    percentiles: Sequence[float] | None,
    eps: object,
    mean: float | None,
    std: float | None,
) -> None:
    """Validate one inference normalization contract.

    Args:
        scheme: Requested per-sample normalization scheme.
        percentiles: Optional lower and upper percentiles.
        eps: Candidate positive denominator floor.
        mean: Optional legacy scalar mean.
        std: Optional legacy scalar standard deviation.

    Returns:
        ``None``.
    """

    if not isinstance(scheme, str) or scheme not in _NORMALIZATION_SCHEMES:
        raise ValueError(
            "preprocessing normalization scheme must be one of "
            f"{sorted(_NORMALIZATION_SCHEMES)}; got {scheme!r}."
        )
    _validate_eps(eps)
    if scheme == "percentile_minmax":
        if percentiles is None:
            raise ValueError(
                "percentile_minmax preprocessing requires two percentiles."
            )
        _validate_percentiles(percentiles)
    elif percentiles is not None:
        raise ValueError(
            "preprocessing percentiles are valid only for percentile_minmax."
        )
    if scheme != "none" and (mean is not None or std is not None):
        raise ValueError(
            "Per-sample preprocessing normalization cannot be combined with "
            "legacy scalar mean/std."
        )
    if mean is not None and not math.isfinite(mean):
        raise ValueError("preprocessing mean must be finite.")
    if std is not None and (not math.isfinite(std) or std <= 0.0):
        raise ValueError("preprocessing std must be a positive finite number.")


def _validate_percentiles(values: Sequence[float]) -> tuple[float, float]:
    """Validate and return an ordered percentile pair.

    Args:
        values: Candidate lower and upper percentiles.

    Returns:
        Finite pair satisfying ``0 <= lower < upper <= 100``.
    """

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("preprocessing percentiles must be a two-value sequence.")
    if len(values) != 2:
        raise ValueError("preprocessing percentiles must contain two values.")
    if any(isinstance(value, bool) for value in values):
        raise TypeError("preprocessing percentiles must contain two numbers.")
    try:
        lower, upper = float(values[0]), float(values[1])
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "preprocessing percentiles must contain two numbers."
        ) from exc
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("preprocessing percentiles must be finite.")
    if not 0.0 <= lower < upper <= 100.0:
        raise ValueError(
            "preprocessing percentiles must satisfy "
            "0 <= lower < upper <= 100."
        )
    return lower, upper


def _validate_eps(value: object) -> float:
    """Validate and return one positive finite denominator floor.

    Args:
        value: Candidate epsilon.

    Returns:
        Positive finite floating-point value.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("preprocessing normalization eps must be a number.")
    eps = float(value)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(
            "preprocessing normalization eps must be a positive finite number."
        )
    return eps


def _percentile_minmax(
    tensor: Tensor,
    *,
    percentiles: tuple[float, float],
    eps: float,
) -> Tensor:
    """Apply training-equivalent percentile clipping per batch sample.

    Args:
        tensor: Batched image tensor.
        percentiles: Validated lower and upper percentages.
        eps: Positive denominator floor.

    Returns:
        Tensor clipped and scaled independently per sample into ``[0, 1]``.
    """

    batch_size = tensor.shape[0]
    flat = tensor.reshape(batch_size, -1)
    quantile_input = (
        flat if flat.dtype in {torch.float32, torch.float64} else flat.float()
    )
    quantiles = quantile_input.new_tensor(
        [percentiles[0] / 100.0, percentiles[1] / 100.0]
    )
    bounds = torch.quantile(
        quantile_input,
        quantiles,
        dim=1,
        interpolation="linear",
    ).to(dtype=tensor.dtype)
    view_shape = [batch_size] + [1] * (tensor.ndim - 1)
    lower = bounds[0].view(*view_shape)
    upper = bounds[1].view(*view_shape)
    clipped = tensor.clamp(min=lower, max=upper)
    return (clipped - lower) / (upper - lower).clamp(min=float(eps))


def _check_pixel_values(tensor: Tensor) -> Tensor:
    """Warn about non-finite or contrast-free pixels before normalization.

    Args:
        tensor: Grayscale ``BxCxHxW`` batch before padding and resizing.

    Returns:
        ``tensor`` with non-finite values replaced by ``0``.
    """

    finite = torch.isfinite(tensor)
    if not bool(finite.all()):
        warnings.warn(
            f"Image contains {int((~finite).sum().item())} non-finite pixel values; "
            "they were replaced with 0.",
            RuntimeWarning,
            stacklevel=2,
        )
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    flat = tensor.reshape(tensor.shape[0], -1)
    if bool((flat.amax(dim=1) == flat.amin(dim=1)).any()):
        warnings.warn(
            "Image has no contrast (every pixel is identical).",
            RuntimeWarning,
            stacklevel=2,
        )
    return tensor


def _pad_to_square(image: Image.Image) -> Image.Image:
    """Return ``image`` zero-padded to a square canvas.

    Args:
        image: Pillow image to pad.

    Returns:
        Original image when already square, otherwise a zero-padded copy.
    """

    width, height = image.size
    side = max(width, height)
    if width == height:
        return image
    pad_left = (side - width) // 2
    pad_right = side - width - pad_left
    pad_top = (side - height) // 2
    pad_bottom = side - height - pad_top
    return ImageOps.expand(
        image,
        border=(pad_left, pad_top, pad_right, pad_bottom),
        fill=0,
    )


def _pad_tensor_to_square(tensor: Tensor) -> Tensor:
    """Zero-pad a ``BxCxHxW`` tensor to square with the :func:`_pad_to_square` margins.

    Args:
        tensor: Batched image tensor.

    Returns:
        Original tensor when already square, otherwise a zero-padded copy.
    """

    height, width = (int(dim) for dim in tensor.shape[-2:])
    side = max(height, width)
    if height == width:
        return tensor
    pad_left = (side - width) // 2
    pad_top = (side - height) // 2
    return nnf.pad(
        tensor, (pad_left, side - width - pad_left, pad_top, side - height - pad_top)
    )


def _alpha_unit_range(alpha: np.ndarray, *, source_dtype: np.dtype[Any]) -> np.ndarray:
    """Scale an alpha channel into ``[0, 1]`` with the pixels' own scaling rule.

    Args:
        alpha: Alpha channel values.
        source_dtype: Original NumPy dtype of the RGBA array.

    Returns:
        ``float64`` alpha in ``[0, 1]``: 8/16-bit integers divide by the dtype
        maximum, wider integers by the channel maximum, floats use their original values.
        The result is clamped to ``[0, 1]``.
    """

    dtype = np.dtype(source_dtype)
    values = alpha.astype(np.float64, copy=False)
    if not np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
        return np.where(np.isfinite(values), np.clip(values, 0.0, 1.0), values)
    if dtype.itemsize > 2:
        values = values / max(float(values.max()), 1.0)
    else:
        values = values / float(np.iinfo(dtype).max)
    return np.clip(values, 0.0, 1.0)


def _rgb_like_to_grayscale(tensor: Tensor, *, source_dtype: torch.dtype) -> Tensor:
    """Convert raw batched RGB/RGBA pixels to luminance over black.

    Args:
        tensor: Batched color pixels before integer scaling.
        source_dtype: Pixel dtype used to interpret the alpha channel.

    Returns:
        ``float64`` luminance with one channel, before pixel scaling.

    Raises:
        ValueError: If the tensor has neither three nor four channels.
    """

    channels = int(tensor.shape[1])
    if channels == 3 or channels == 4:
        tensor = tensor.double()
        rgb = tensor[:, :3]
        if channels == 4:
            alpha = tensor[:, 3:4]
            if source_dtype != torch.bool and not source_dtype.is_floating_point:
                limits = torch.iinfo(source_dtype)
                if limits.bits > 16:
                    alpha = alpha / alpha.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1)
                else:
                    alpha = alpha / float(limits.max)
            alpha = torch.where(torch.isfinite(alpha), alpha.clamp(0.0, 1.0), alpha)
            rgb = rgb * alpha
        weights = tensor.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (rgb * weights).sum(dim=1, keepdim=True)
    raise ValueError(
        "Grayscale preprocessing expects 1, 3, or 4 input channels; "
        f"got {channels}."
    )


__all__ = [
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_NORMALIZATION_EPS",
    "DEFAULT_NORMALIZATION_PERCENTILES",
    "ImagePreprocessing",
]
