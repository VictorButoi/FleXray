"""CT subject and attenuation helpers for nanoDRR."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from nanodrr.data import Subject
from nanodrr.data.preprocess import hu_to_mu
from torch import Tensor

__all__ = ["sample_attenuation", "subject_from_tensors"]

_SQRT_2 = math.sqrt(2.0)


def sample_attenuation(
    bounds: tuple[float, float] | list[float],
    n: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    dist: Mapping[str, Any] | None = None,
) -> Tensor:
    """Sample ``n`` attenuation multipliers within ``bounds``.

    Supported distributions are uniform, beta, and truncated lognormal. The
    lognormal distribution may be named ``"lognormal"`` or
    ``"truncated_lognormal"``.
    """

    n = int(n)
    if n < 0:
        raise ValueError("n must be >= 0.")
    lo, hi = _parse_bounds(bounds, "attenuation bounds")
    dist_cfg = dict(dist or {})
    dist_type = str(dist_cfg.get("type", "uniform"))

    if dist_type == "uniform":
        _reject_unknown_keys(dist_cfg, {"type"}, "uniform attenuation distribution")
        if n == 0:
            return torch.empty((0,), device=device, dtype=dtype)
        if lo == hi:
            return torch.full((n,), lo, device=device, dtype=dtype)
        x = torch.rand(n, device=device, dtype=dtype)
        return lo + (hi - lo) * x

    if dist_type == "beta":
        _reject_unknown_keys(
            dist_cfg,
            {"type", "alpha", "beta"},
            "beta attenuation distribution",
        )
        if "alpha" not in dist_cfg or "beta" not in dist_cfg:
            raise ValueError("beta attenuation distribution requires alpha and beta.")
        alpha = float(dist_cfg["alpha"])
        beta = float(dist_cfg["beta"])
        if not math.isfinite(alpha) or not math.isfinite(beta):
            raise ValueError("beta attenuation alpha and beta must be finite.")
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError("beta attenuation alpha and beta must be > 0.")
        if n == 0:
            return torch.empty((0,), device=device, dtype=dtype)
        if lo == hi:
            return torch.full((n,), lo, device=device, dtype=dtype)
        alpha_t = torch.as_tensor(alpha, device=device, dtype=dtype)
        beta_t = torch.as_tensor(beta, device=device, dtype=dtype)
        x = torch.distributions.Beta(alpha_t, beta_t).sample((n,)).to(dtype)
        return lo + (hi - lo) * x

    if dist_type in {"lognormal", "truncated_lognormal"}:
        _reject_unknown_keys(
            dist_cfg,
            {"type", "mode", "sigma"},
            "lognormal attenuation distribution",
        )
        return _sample_truncated_lognormal((lo, hi), n, device, dtype, dist_cfg)

    raise ValueError(f"Unknown attenuation distribution type: {dist_type!r}.")


def subject_from_tensors(
    volume_hu: Tensor,
    label: Tensor | None,
    affine: Tensor,
    *,
    attenuation: float | tuple[float, float] | list[float] | None = None,
    attenuated_label_ids: Sequence[int] | Tensor | None = None,
    do_per_label_attenuation: bool = False,
    max_label: int | None = None,
    attenuation_dist: Mapping[str, Any] | None = None,
) -> Subject:
    """Build a nanoDRR ``Subject`` from one FleXray CT image and dense labels.

    ``volume_hu`` accepts ``(1, D, H, W)`` or legacy ``(1, 1, D, H, W)`` input.
    ``label`` accepts ``(D, H, W)``, ``(1, D, H, W)``, or legacy
    ``(1, 1, D, H, W)`` dense integer labels. The affine is voxel-to-world.
    """

    volume_5d = _normalize_volume_tensor(volume_hu)
    label_5d = _normalize_label_tensor(label, reference=volume_5d)
    affine_t = _normalize_affine_tensor(affine, reference=volume_5d)

    observed_max_label = int(label_5d.max().item())
    if max_label is None:
        subject_max_label = observed_max_label
    else:
        subject_max_label = int(max_label)
        if subject_max_label < observed_max_label:
            raise ValueError(
                "max_label must be >= the largest dense label id in label."
            )

    i_size, j_size, k_size = volume_5d.shape[-3:]
    volume_perm = volume_5d.permute(0, 1, 4, 3, 2)
    label_perm = label_5d.permute(0, 1, 4, 3, 2).contiguous()

    bounds = _attenuation_bounds(attenuation)
    if do_per_label_attenuation:
        if attenuated_label_ids is None:
            raise ValueError(
                "attenuated_label_ids must be set when do_per_label_attenuation=True."
            )
        if not isinstance(attenuation, (tuple, list)):
            raise ValueError(
                "attenuation must be a (lo, hi) tuple/list when "
                "do_per_label_attenuation=True."
            )
        ids = torch.as_tensor(
            attenuated_label_ids,
            device=volume_5d.device,
            dtype=torch.long,
        ).reshape(-1)
        ids = ids[ids > 0].unique(sorted=True)
        if ids.numel() == 0:
            raise ValueError(
                "do_per_label_attenuation=True requires at least one foreground label."
            )

        lookup_size = max(subject_max_label + 1, int(ids.max().item()) + 1)
        multipliers = torch.ones(
            lookup_size,
            device=volume_5d.device,
            dtype=volume_5d.dtype,
        )
        multipliers[ids] = sample_attenuation(
            bounds,
            int(ids.numel()),
            volume_5d.device,
            volume_5d.dtype,
            attenuation_dist,
        )
        in_lookup_range = label_perm < lookup_size
        lookup_idx = torch.where(
            in_lookup_range,
            label_perm,
            torch.zeros_like(label_perm),
        ).long()
        per_voxel_multiplier = torch.where(
            in_lookup_range,
            multipliers[lookup_idx],
            torch.ones_like(label_perm, dtype=volume_5d.dtype),
        )
    else:
        per_voxel_multiplier = sample_attenuation(
            bounds,
            1,
            volume_5d.device,
            volume_5d.dtype,
            attenuation_dist,
        )[0]

    image_mu = hu_to_mu(volume_perm) * per_voxel_multiplier
    world_to_voxel = torch.linalg.inv(affine_t.double()).to(dtype=volume_5d.dtype)
    voxel_to_grid = Subject._make_voxel_to_grid(image_mu.shape).to(
        device=volume_5d.device,
        dtype=volume_5d.dtype,
    )

    center_voxel = torch.tensor(
        [(i_size - 1) / 2.0, (j_size - 1) / 2.0, (k_size - 1) / 2.0, 1.0],
        dtype=affine_t.dtype,
        device=affine_t.device,
    )
    isocenter = (affine_t @ center_voxel)[:3]

    return Subject(
        image_mu,
        label_perm,
        affine_t,
        world_to_voxel,
        voxel_to_grid,
        isocenter,
        max_label=subject_max_label,
        convert_to_mu=False,
    )


def _normalize_volume_tensor(volume_hu: Tensor) -> Tensor:
    volume = torch.as_tensor(volume_hu)
    if volume.ndim == 4 and int(volume.shape[0]) == 1:
        volume = volume.unsqueeze(0)
    elif volume.ndim == 5 and tuple(volume.shape[:2]) == (1, 1):
        pass
    else:
        raise ValueError(
            "volume_hu must have shape (1, D, H, W) or (1, 1, D, H, W)."
        )
    if not volume.is_floating_point():
        volume = volume.to(torch.float32)
    return volume.contiguous()


def _normalize_label_tensor(label: Tensor | None, *, reference: Tensor) -> Tensor:
    if label is None:
        return torch.zeros_like(reference)

    label_t = torch.as_tensor(label, device=reference.device)
    if label_t.ndim == 3:
        label_t = label_t.unsqueeze(0).unsqueeze(0)
    elif label_t.ndim == 4 and int(label_t.shape[0]) == 1:
        label_t = label_t.unsqueeze(0)
    elif label_t.ndim == 5 and tuple(label_t.shape[:2]) == (1, 1):
        pass
    else:
        raise ValueError(
            "label must have shape (D, H, W), (1, D, H, W), "
            "or (1, 1, D, H, W)."
        )
    if tuple(label_t.shape[-3:]) != tuple(reference.shape[-3:]):
        raise ValueError("label spatial shape must match volume_hu.")
    if label_t.is_floating_point():
        rounded = label_t.round()
        if not torch.allclose(label_t, rounded):
            raise ValueError("label must contain dense integer ids.")
        label_t = rounded
    if torch.any(label_t < 0):
        raise ValueError("label ids must be non-negative.")
    return label_t.to(device=reference.device, dtype=reference.dtype).contiguous()


def _normalize_affine_tensor(affine: Tensor, *, reference: Tensor) -> Tensor:
    affine_t = torch.as_tensor(affine, device=reference.device)
    if affine_t.shape != (4, 4):
        raise ValueError(f"affine must have shape (4, 4); got {tuple(affine_t.shape)}.")
    if not affine_t.is_floating_point():
        affine_t = affine_t.to(reference.dtype)
    else:
        affine_t = affine_t.to(dtype=reference.dtype)
    return affine_t


def _attenuation_bounds(
    attenuation: float | tuple[float, float] | list[float] | None,
) -> tuple[float, float]:
    if attenuation is None:
        return 1.0, 1.0
    if isinstance(attenuation, (tuple, list)):
        return _parse_bounds(attenuation, "attenuation")
    value = float(attenuation)
    if not math.isfinite(value):
        raise ValueError("attenuation must be finite.")
    return value, value


def _parse_bounds(value: tuple[float, float] | list[float], name: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must have exactly two bounds.")
    lo = float(value[0])
    hi = float(value[1])
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError(f"{name} bounds must be finite.")
    if lo > hi:
        raise ValueError(f"{name} lower bound must be <= upper bound.")
    return lo, hi


def _reject_unknown_keys(
    mapping: Mapping[str, Any],
    allowed: set[str],
    context: str,
) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"{context} has unknown key(s): {unknown}.")


def _normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / _SQRT_2)


def _normal_cdf_diff(lo: float, hi: float) -> float:
    if lo >= hi:
        return 0.0
    if hi <= 0.0:
        diff = 0.5 * (math.erfc(-hi / _SQRT_2) - math.erfc(-lo / _SQRT_2))
    elif lo >= 0.0:
        diff = 0.5 * (math.erfc(lo / _SQRT_2) - math.erfc(hi / _SQRT_2))
    else:
        diff = _normal_cdf(hi) - _normal_cdf(lo)
    return max(diff, 0.0)


def _sample_truncated_lognormal(
    bounds: tuple[float, float],
    n: int,
    device: torch.device | str | None,
    dtype: torch.dtype,
    dist: Mapping[str, Any],
) -> Tensor:
    lo, hi = bounds
    mode, sigma = _parse_lognormal_config(bounds, dist)
    if n == 0:
        return torch.empty((0,), device=device, dtype=dtype)
    if lo == hi:
        return torch.full((n,), lo, device=device, dtype=dtype)

    mu = math.log(mode) + sigma * sigma
    sample_dtype = dtype if dtype not in (torch.float16, torch.bfloat16) else torch.float32
    distribution = torch.distributions.LogNormal(
        torch.as_tensor(mu, device=device, dtype=sample_dtype),
        torch.as_tensor(sigma, device=device, dtype=sample_dtype),
    )
    prob = _normal_cdf_diff((math.log(lo) - mu) / sigma, (math.log(hi) - mu) / sigma)

    samples = torch.empty((n,), device=device, dtype=dtype)
    filled = 0
    empty_draws = 0
    while filled < n:
        remaining = n - filled
        draw_count = max(remaining, int(math.ceil(1.1 * remaining / max(prob, 1e-3))))
        draws = distribution.sample((draw_count,))
        valid = draws[(draws >= lo) & (draws <= hi)]
        if valid.numel() == 0:
            empty_draws += 1
            if empty_draws > 100:
                raise RuntimeError(
                    "Unable to sample valid lognormal attenuation values within "
                    f"range ({lo}, {hi})."
                )
            continue
        empty_draws = 0
        take = min(remaining, int(valid.numel()))
        samples[filled : filled + take] = valid[:take].to(dtype=dtype)
        filled += take
    return samples


def _parse_lognormal_config(
    bounds: tuple[float, float],
    dist: Mapping[str, Any],
) -> tuple[float, float]:
    lo, hi = bounds
    if lo <= 0.0 or hi <= 0.0:
        raise ValueError("lognormal attenuation bounds must be positive.")
    missing = sorted({"mode", "sigma"} - set(dist))
    if missing:
        raise ValueError(
            f"lognormal attenuation distribution is missing key(s): {missing}."
        )
    mode = float(dist["mode"])
    sigma = float(dist["sigma"])
    if not math.isfinite(mode):
        raise ValueError("lognormal attenuation mode must be finite.")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("lognormal attenuation sigma must be finite and > 0.")
    if not lo <= mode <= hi:
        raise ValueError("lognormal attenuation mode must be within bounds.")
    return mode, sigma
