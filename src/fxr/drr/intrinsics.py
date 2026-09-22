"""DRR camera intrinsics helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

__all__ = ["DRRIntrinsics", "build_drr_camera_info", "build_render_intrinsics"]


ScalarOrSequence = float | int | Sequence[float | int] | Tensor


@dataclass(frozen=True)
class DRRIntrinsics:
    """Resolved per-view cone-beam camera intrinsics for nanoDRR.

    Attributes:
        k_inv: Per-view inverse camera intrinsic matrices with shape
            ``(V, 3, 3)``.
        sdd: Per-view source-to-detector distances.
        delx: Per-view detector pixel spacing along x.
        dely: Per-view detector pixel spacing along y.
        x0: Per-view detector principal-point x offsets in world units.
        y0: Per-view detector principal-point y offsets in world units.
        height: Detector image height in pixels.
        width: Detector image width in pixels.
        orthographic: Whether rendering should use orthographic projection.
        num_views: Number of per-view camera entries represented by the object.
    """

    k_inv: Tensor
    sdd: Tensor
    delx: Tensor
    dely: Tensor
    x0: Tensor
    y0: Tensor
    height: int
    width: int
    orthographic: bool = False

    @property
    def num_views(self) -> int:
        """Return the number of camera views in this intrinsics bundle.

        Returns:
            Number of scalar ``sdd`` entries, one per view.
        """

        return int(self.sdd.numel())

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "DRRIntrinsics":
        """Return a copy with tensor fields moved to a device or dtype.

        Args:
            device: Optional target torch device.
            dtype: Optional target tensor dtype.

        Returns:
            New ``DRRIntrinsics`` with tensor fields converted and scalar fields
            preserved.
        """

        return DRRIntrinsics(
            k_inv=self.k_inv.to(device=device, dtype=dtype),
            sdd=self.sdd.to(device=device, dtype=dtype),
            delx=self.delx.to(device=device, dtype=dtype),
            dely=self.dely.to(device=device, dtype=dtype),
            x0=self.x0.to(device=device, dtype=dtype),
            y0=self.y0.to(device=device, dtype=dtype),
            height=self.height,
            width=self.width,
            orthographic=self.orthographic,
        )


def build_render_intrinsics(
    *,
    sdd: ScalarOrSequence,
    delx: ScalarOrSequence,
    dely: ScalarOrSequence,
    height: int,
    width: int,
    x0: ScalarOrSequence = 0.0,
    y0: ScalarOrSequence = 0.0,
    num_views: int | None = None,
    orthographic: bool = False,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> DRRIntrinsics:
    """Normalize scalar or per-view camera values into nanoDRR intrinsics.

    ``sdd`` determines the number of views unless ``num_views`` is provided.
    Other scalar fields are expanded to match that view count; sequences must
    have length one or the resolved number of views.
    """

    height = int(height)
    width = int(width)
    if height < 1 or width < 1:
        raise ValueError("height and width must be positive integers.")

    dtype = dtype or torch.float32
    sdd_t = _as_1d_tensor(sdd, key="sdd", device=device, dtype=dtype)
    if num_views is None:
        num_views = int(sdd_t.numel())
    num_views = int(num_views)
    if num_views < 1:
        raise ValueError("num_views must be >= 1.")

    sdd_t = _expand_view_tensor(sdd_t, key="sdd", num_views=num_views)
    delx_t = _expand_view_tensor(
        _as_1d_tensor(delx, key="delx", device=device, dtype=dtype),
        key="delx",
        num_views=num_views,
    )
    dely_t = _expand_view_tensor(
        _as_1d_tensor(dely, key="dely", device=device, dtype=dtype),
        key="dely",
        num_views=num_views,
    )
    x0_t = _expand_view_tensor(
        _as_1d_tensor(x0, key="x0", device=device, dtype=dtype),
        key="x0",
        num_views=num_views,
    )
    y0_t = _expand_view_tensor(
        _as_1d_tensor(y0, key="y0", device=device, dtype=dtype),
        key="y0",
        num_views=num_views,
    )

    if torch.any(sdd_t <= 0):
        raise ValueError("sdd values must be positive.")
    if torch.any(delx_t <= 0) or torch.any(dely_t <= 0):
        raise ValueError("delx and dely values must be positive.")

    fx = sdd_t / delx_t
    fy = sdd_t / dely_t
    cx = x0_t / delx_t + width / 2.0
    cy = y0_t / dely_t + height / 2.0

    k_inv = torch.zeros((num_views, 3, 3), device=sdd_t.device, dtype=dtype)
    k_inv[:, 0, 0] = 1.0 / fx
    k_inv[:, 0, 2] = -cx / fx
    k_inv[:, 1, 1] = 1.0 / fy
    k_inv[:, 1, 2] = -cy / fy
    k_inv[:, 2, 2] = 1.0

    return DRRIntrinsics(
        k_inv=k_inv,
        sdd=sdd_t,
        delx=delx_t,
        dely=dely_t,
        x0=x0_t,
        y0=y0_t,
        height=height,
        width=width,
        orthographic=bool(orthographic),
    )


def build_drr_camera_info(
    intrinsics: DRRIntrinsics | Mapping[str, Any],
) -> dict[str, Any]:
    """Return a plain per-view camera metadata dictionary."""

    resolved = _coerce_intrinsics(intrinsics)
    return {
        "sdd": _tensor_to_float_list(resolved.sdd),
        "delx": _tensor_to_float_list(resolved.delx),
        "dely": _tensor_to_float_list(resolved.dely),
        "x0": _tensor_to_float_list(resolved.x0),
        "y0": _tensor_to_float_list(resolved.y0),
        "height": int(resolved.height),
        "width": int(resolved.width),
        "orthographic": bool(resolved.orthographic),
    }


def _coerce_intrinsics(intrinsics: DRRIntrinsics | Mapping[str, Any]) -> DRRIntrinsics:
    if isinstance(intrinsics, DRRIntrinsics):
        return intrinsics
    required = {"sdd", "delx", "dely", "height", "width"}
    missing = sorted(required - set(intrinsics))
    if missing:
        raise KeyError(f"intrinsics is missing required key(s): {missing}.")
    return build_render_intrinsics(
        sdd=intrinsics["sdd"],
        delx=intrinsics["delx"],
        dely=intrinsics["dely"],
        x0=intrinsics.get("x0", 0.0),
        y0=intrinsics.get("y0", 0.0),
        height=int(intrinsics["height"]),
        width=int(intrinsics["width"]),
        orthographic=bool(intrinsics.get("orthographic", False)),
    )


def _as_1d_tensor(
    value: ScalarOrSequence,
    *,
    key: str,
    device: torch.device | str | None,
    dtype: torch.dtype,
) -> Tensor:
    if isinstance(value, Tensor):
        tensor = value.to(device=device, dtype=dtype).reshape(-1)
    elif isinstance(value, (int, float)):
        tensor = torch.tensor([float(value)], device=device, dtype=dtype)
    elif isinstance(value, Sequence):
        if len(value) == 0:
            raise ValueError(f"intrinsics[{key!r}] cannot be empty.")
        tensor = torch.tensor([float(v) for v in value], device=device, dtype=dtype)
    else:
        raise TypeError(
            f"intrinsics[{key!r}] must be a scalar, tensor, or sequence; "
            f"got {type(value).__name__}."
        )
    return tensor


def _expand_view_tensor(tensor: Tensor, *, key: str, num_views: int) -> Tensor:
    if tensor.numel() == num_views:
        return tensor
    if tensor.numel() == 1:
        return tensor.repeat(num_views)
    raise ValueError(
        f"intrinsics[{key!r}] has {tensor.numel()} values but expected 1 "
        f"or {num_views}."
    )


def _tensor_to_float_list(value: Tensor) -> list[float]:
    return [float(v) for v in value.detach().cpu().reshape(-1).tolist()]
