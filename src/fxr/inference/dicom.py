"""DICOM image loading for FleXray inference inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

DICOM_SUFFIXES = frozenset({".dcm", ".dicom"})
_DICOM_PREAMBLE_LENGTH = 128
_DICOM_MAGIC = b"DICM"


def is_dicom_path(path: str | Path) -> bool:
    """Return whether a path should be read as a DICOM file.

    A path qualifies by suffix (``.dcm``/``.dicom``, case-insensitive) or, for
    existing extension-less files, by the ``DICM`` magic at byte offset 128.

    Args:
        path: Candidate file path.

    Returns:
        ``True`` if the path is a DICOM file.
    """

    candidate = Path(path)
    if candidate.suffix.lower() in DICOM_SUFFIXES:
        return True
    if candidate.suffix or not candidate.is_file():
        return False
    try:
        with candidate.open("rb") as handle:
            handle.seek(_DICOM_PREAMBLE_LENGTH)
            return handle.read(len(_DICOM_MAGIC)) == _DICOM_MAGIC
    except OSError:
        return False


def load_dicom_grayscale(path: str | Path) -> np.ndarray:
    """Read one DICOM file into a two-dimensional ``float32`` image in ``[0, 1]``.

    The stored pixels are decoded with ``pydicom``, ``RescaleSlope`` and
    ``RescaleIntercept`` are applied when present, ``MONOCHROME1`` images are
    inverted so higher values are brighter, color images are converted to
    luminance, and the result is min-max scaled to ``[0, 1]`` over its own
    range. Per-image scaling is used because DICOM bit depths
    (``BitsStored``) vary and rarely fill their container dtype.

    Args:
        path: DICOM file path.

    Returns:
        Two-dimensional ``float32`` array with values in ``[0, 1]``.

    Raises:
        ImportError: If ``pydicom`` is not installed.
        ValueError: If the file holds no pixel data or has an unsupported
            shape (only single-frame 2D images are accepted).
    """

    try:
        import pydicom
    except ImportError as error:  # pragma: no cover - exercised without extra
        raise ImportError(
            "Reading DICOM inputs requires pydicom. Install it with "
            "`pip install pydicom` or `pip install flexray[dicom]`."
        ) from error

    dataset = pydicom.dcmread(str(path), force=True)
    if "PixelData" not in dataset:
        raise ValueError(f"DICOM file has no pixel data: {path}.")
    pixels = np.asarray(dataset.pixel_array)
    if pixels.ndim == 3 and pixels.shape[-1] in {3, 4}:
        rgb = pixels[..., :3].astype(np.float64, copy=False)
        pixels = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    if pixels.ndim != 2:
        raise ValueError(
            "Only single-frame 2D DICOM images are supported; got pixel shape "
            f"{pixels.shape} for {path}."
        )
    values = pixels.astype(np.float64, copy=False)
    slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
    values = values * slope + intercept
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        values = values.max() - values
    return scale_to_unit_range(values)


def scale_to_unit_range(values: np.ndarray) -> np.ndarray:
    """Min-max scale one image into ``[0, 1]`` over its own range.

    Args:
        values: Pixel array of any numeric dtype.

    Returns:
        ``float32`` array in ``[0, 1]``; a constant image becomes all zeros.
    """

    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.integer):
        # Remove large offsets before casting so int64 low bits survive.
        # The endpoint nearest zero keeps subtraction inside the dtype range.
        low, high = int(values.min()), int(values.max())
        origin = max(low, 0) + min(high, 0)
        values = values - np.asarray(origin, dtype=values.dtype)
    values = np.asarray(values, dtype=np.float64)
    low = float(values.min())
    high = float(values.max())
    if high > low:
        values = (values - low) / (high - low)
    else:
        values = np.zeros_like(values)
    return values.astype(np.float32)
