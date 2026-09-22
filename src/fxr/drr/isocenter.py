"""World-space isocenter helpers for DRR rendering."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .subjects import (
    _normalize_affine_tensor,
    _normalize_label_tensor,
    _normalize_volume_tensor,
)

__all__ = [
    "IsocenterConfig",
    "compute_isocenter",
    "sample_isocenters_from_centroids",
]


@dataclass(frozen=True)
class IsocenterConfig:
    """Configuration for selecting a DRR camera target.

    Attributes:
        sample_scheme: Isocenter selection mode, such as ``"volume_center"``,
            ``"label_centroid"``, or ``"random_label"`` with precomputed
            centroids.
        replacement: Optional sampling replacement policy for centroid-based
            selection.
    """

    sample_scheme: str
    replacement: bool | None = None


def compute_isocenter(
    volume: Tensor,
    label: Tensor | None,
    affine: Tensor,
    config: IsocenterConfig | str = "volume_center",
    *,
    num_views: int = 1,
) -> Tensor:
    """Compute a world-space isocenter from a CT volume and optional labelmap."""

    if num_views < 1:
        raise ValueError(f"num_views must be >= 1. Got {num_views}.")

    volume_5d = _normalize_volume_tensor(volume)
    label_5d = _normalize_label_tensor(label, reference=volume_5d)
    affine_t = _normalize_affine_tensor(affine, reference=volume_5d)
    scheme = config.sample_scheme if isinstance(config, IsocenterConfig) else str(config)

    if scheme == "volume_center":
        return _volume_center(volume_5d, affine_t)

    if scheme == "label_centroid":
        label_3d = label_5d[0, 0]
        foreground = label_3d > 0
        if not bool(foreground.any()):
            return _volume_center(volume_5d, affine_t)
        centroid_ijk = torch.nonzero(foreground, as_tuple=False).to(
            dtype=affine_t.dtype,
            device=affine_t.device,
        ).mean(dim=0)
        return _ijk_to_world(centroid_ijk.reshape(1, 3), affine_t)[0]

    if scheme == "random_label":
        raise ValueError(
            "sample_scheme='random_label' requires precomputed centroids; "
            "use sample_isocenters_from_centroids(...)."
        )

    if scheme == "label_random_class":
        raise ValueError(
            "sample_scheme='label_random_class' is not supported; use "
            "'random_label' with precomputed centroids."
        )

    raise ValueError(
        "Unknown isocenter sample scheme "
        f"{scheme!r}; expected 'volume_center', 'label_centroid', or 'random_label'."
    )


def sample_isocenters_from_centroids(
    centroids_ijk: Tensor,
    affine: Tensor,
    num_views: int,
    *,
    replacement: bool = True,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample per-view isocenters from precomputed voxel-space centroids."""

    num_views = int(num_views)
    if num_views < 1:
        raise ValueError(f"num_views must be >= 1. Got {num_views}.")
    centroids = torch.as_tensor(centroids_ijk)
    if centroids.ndim != 2 or int(centroids.shape[1]) != 3:
        raise ValueError(
            f"centroids_ijk must have shape (N, 3); got {tuple(centroids.shape)}."
        )
    if int(centroids.shape[0]) == 0:
        raise ValueError("centroids_ijk must contain at least one centroid.")

    affine_t = torch.as_tensor(affine, device=centroids.device)
    if affine_t.shape != (4, 4):
        raise ValueError(f"affine must have shape (4, 4); got {tuple(affine_t.shape)}.")
    if not affine_t.is_floating_point():
        affine_t = affine_t.to(torch.float32)
    if not centroids.is_floating_point():
        centroids = centroids.to(affine_t.dtype)
    else:
        centroids = centroids.to(dtype=affine_t.dtype)

    n_centroids = int(centroids.shape[0])
    if replacement:
        chosen = torch.randint(
            n_centroids,
            (num_views,),
            device=centroids.device,
            generator=generator,
        )
    elif num_views <= n_centroids:
        chosen = torch.randperm(
            n_centroids,
            device=centroids.device,
            generator=generator,
        )[:num_views]
    else:
        first_pass = torch.randperm(
            n_centroids,
            device=centroids.device,
            generator=generator,
        )
        remaining = torch.randint(
            n_centroids,
            (num_views - n_centroids,),
            device=centroids.device,
            generator=generator,
        )
        chosen = torch.cat([first_pass, remaining])

    return _ijk_to_world(centroids[chosen], affine_t)


def _volume_center(volume_5d: Tensor, affine: Tensor) -> Tensor:
    d_size, h_size, w_size = volume_5d.shape[-3:]
    center_ijk = torch.tensor(
        [(d_size - 1) / 2.0, (h_size - 1) / 2.0, (w_size - 1) / 2.0],
        dtype=affine.dtype,
        device=affine.device,
    )
    return _ijk_to_world(center_ijk.reshape(1, 3), affine)[0]


def _ijk_to_world(ijk: Tensor, affine: Tensor) -> Tensor:
    ones = torch.ones(ijk.shape[0], 1, dtype=affine.dtype, device=affine.device)
    homog = torch.cat([ijk.to(device=affine.device, dtype=affine.dtype), ones], dim=1)
    return (affine @ homog.T).T[:, :3]
