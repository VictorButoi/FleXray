"""NIfTI volume loading for CT dataset packaging.

``nibabel`` is an optional dependency (installed with the ``train`` extra), so it
is imported lazily and only when a manifest references ``.nii``/``.nii.gz``
payloads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_NIFTI_SUFFIXES = (".nii", ".nii.gz")


def is_nifti_path(path: Path) -> bool:
    """Return whether a payload path names a NIfTI file.

    Args:
        path: Payload path.

    Returns:
        ``True`` for ``.nii`` and ``.nii.gz`` paths (case-insensitive).
    """

    return str(path).lower().endswith(_NIFTI_SUFFIXES)


def load_nifti(path: Path, *, context: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a NIfTI volume with its voxel-to-world affine and spacing.

    Args:
        path: ``.nii`` or ``.nii.gz`` path.
        context: Manifest location used in errors.

    Returns:
        ``(array, affine, spacing)``: the voxel array (trailing singleton axes
        squeezed to three dimensions, native dtype), the ``float32`` ``(4, 4)``
        affine, and the ``float32`` ``(3,)`` voxel spacing in millimetres.

    Raises:
        ImportError: If ``nibabel`` is not installed.
        ValueError: If the file cannot be read or is not a 3D volume.
    """

    nibabel = _import_nibabel()
    try:
        volume = nibabel.load(str(path))
    except Exception as exc:  # nibabel raises several unrelated error types
        raise ValueError(f"{context} could not be read as NIfTI: {path} ({exc}).") from exc
    array = np.asarray(volume.dataobj)
    while array.ndim > 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 3:
        raise ValueError(f"{context} NIfTI must be a 3D volume, got shape {array.shape}.")
    affine = np.asarray(volume.affine, dtype=np.float32)
    spacing = np.asarray(volume.header.get_zooms()[:3], dtype=np.float32)
    return np.ascontiguousarray(array), affine, spacing


def _import_nibabel():
    """Import ``nibabel`` lazily with an install hint.

    Returns:
        The ``nibabel`` module.

    Raises:
        ImportError: If the package is missing.
    """

    try:
        import nibabel
    except ImportError as exc:
        raise ImportError(
            "NIfTI payloads require nibabel; install it with `pip install 'flexray[train]'`."
        ) from exc
    return nibabel
