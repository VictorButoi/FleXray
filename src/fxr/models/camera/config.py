"""Static configuration for the CT->DRR segmentation rendering runtime.

A training config declares CT rendering under a per-dataset ``drr_model`` block::

    drr_model:
      default: {<shared DRR profile>}
      datasets:
        <ct_dataset_name>: {<overrides>}   # one entry per configured CT dataset

``resolve_ct_profiles`` merges ``default`` with each dataset override into an
*effective* flat profile, and ``DRRRenderConfig.from_profile`` parses one
effective profile into the immutable render-time configuration consumed by
:class:`fxr.models.camera.runtime.SegmentationDRRRuntime`.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from fxr.config.training import resolve_num_views
from fxr.drr import IsocenterConfig

__all__ = [
    "DRRRenderConfig",
    "ScalarRangeSampler",
    "collect_configured_ct_dataset_names",
    "resolve_ct_profiles",
    "resolve_num_views",
]

_VALID_ISOCENTER_SCHEMES = frozenset({"volume_center", "label_centroid", "random_label"})
_VALID_ATTENUATION_SCOPES = frozenset({"per_label", "global"})
_ATTENUATION_CFG_KEYS = frozenset({"enabled", "prob", "range", "scope", "distribution"})


class ScalarRangeSampler:
    """Parse scalar or ``[low, high]`` config values and sample within bounds."""

    @staticmethod
    def parse(value: Any, name: str) -> tuple[float, float]:
        """Parse a scalar or two-value range into ``(low, high)`` bounds.

        Args:
            value: Scalar, one-element, or two-element sequence config value.
            name: Config path used in error messages.

        Returns:
            Inclusive ``(low, high)`` bounds; a scalar yields equal endpoints.

        Raises:
            ValueError: If a sequence has the wrong length or an inverted range.
            TypeError: If ``value`` is neither scalar nor sequence.
        """

        if isinstance(value, (int, float)):
            return float(value), float(value)
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                return float(value[0]), float(value[0])
            if len(value) != 2:
                raise ValueError(f"{name} must be a scalar or 2-value range; got {value}.")
            low, high = float(value[0]), float(value[1])
            if high < low:
                raise ValueError(f"{name} has inverted range [{low}, {high}].")
            return low, high
        raise TypeError(f"{name} must be a scalar or sequence, got {type(value).__name__}.")

    @staticmethod
    def sample(bounds: tuple[float, float]) -> float:
        """Sample uniformly within ``bounds`` (deterministic for equal endpoints).

        Args:
            bounds: Inclusive ``(low, high)`` sampling bounds.

        Returns:
            Sampled scalar, or the shared endpoint for a degenerate range.
        """
        low, high = bounds
        return low if high == low else random.uniform(low, high)


@dataclass(frozen=True)
class DRRRenderConfig:
    """Immutable render-time configuration for one CT DRR profile.

    Attributes:
        height: Detector height in pixels.
        width: Detector width in pixels.
        sdd_bounds: Source-to-detector distance sampling bounds.
        delx_bounds: Detector column spacing sampling bounds.
        dely_bounds: Detector row spacing bounds, or ``None`` to follow ``delx``.
        x0_bounds: Principal-point x-offset sampling bounds.
        y0_bounds: Principal-point y-offset sampling bounds.
        orientation: nanoDRR camera orientation string.
        isocenter_cfg: Isocenter sampling configuration.
        ortho_prob: Probability of rendering a view orthographically.
        num_views: Number of DRR views rendered per CT subject.
        n_samples: Number of samples along each projection ray.
        preset: Pose-sampler preset name or multi-preset schedule.
        camera_displacement: Default fixed-pose source displacement.
        sample_params: Random pose axis ranges used by ``preset="random"``.
        attenuation_range: Per-render attenuation multiplier bounds, or ``None``.
        do_per_label_attenuation: Whether each foreground label is attenuated
            independently.
        attenuated_label_ids: Foreground model label ids eligible for attenuation.
        attenuation_dist: Optional attenuation sampling distribution config.
        attenuation_prob: Per-render probability of applying the sampled
            attenuation multipliers; ``0.0`` never attenuates.
        input_num_label_channels: Model label channels (incl. background) the CT
            mask is rendered into.
        output_num_label_channels: Model label channels (incl. background) of the
            produced training labels.
        render_soft_labels: Whether projected labels retain soft foreground.
        seg_threshold: Threshold used for hard projected label masks.
        label_smoothing_sigma: Gaussian sigma for optional label smoothing.
        label_smoothing_kernel_size: Odd ``(kh, kw)`` smoothing kernel, or ``None``.
        follows_delx: Whether detector row spacing follows column spacing.
    """

    height: int
    width: int
    sdd_bounds: tuple[float, float]
    delx_bounds: tuple[float, float]
    dely_bounds: tuple[float, float] | None
    x0_bounds: tuple[float, float]
    y0_bounds: tuple[float, float]
    orientation: str
    isocenter_cfg: IsocenterConfig
    ortho_prob: float
    num_views: int
    n_samples: int
    preset: Any
    camera_displacement: float
    sample_params: dict[str, Any]
    attenuation_range: tuple[float, float] | None
    do_per_label_attenuation: bool
    attenuated_label_ids: list[int]
    attenuation_dist: dict[str, Any] | None
    attenuation_prob: float
    input_num_label_channels: int
    output_num_label_channels: int
    render_soft_labels: bool
    seg_threshold: float
    label_smoothing_sigma: float
    label_smoothing_kernel_size: tuple[int, int] | None

    @classmethod
    def from_profile(
        cls,
        profile: Mapping[str, Any],
        *,
        num_views: int,
        input_num_label_channels: int,
        output_num_label_channels: int,
        attenuated_label_ids: list[int],
    ) -> "DRRRenderConfig":
        """Parse one effective (merged) DRR profile into a render config.

        Args:
            profile: Flat effective DRR profile (``default`` merged with a dataset
                override).
            num_views: Number of DRR views to render per subject.
            input_num_label_channels: Model label channels the CT mask renders into.
            output_num_label_channels: Model label channels of produced labels.
            attenuated_label_ids: Foreground model ids eligible for attenuation.

        Returns:
            The parsed immutable render configuration.
        """

        cam_cfg = dict(profile.get("intrinsics_cfg", {}))
        seg_cfg = dict(profile.get("seg_cfg", {}))
        has_dely = "dely" in cam_cfg
        attenuation = _parse_attenuation(
            profile.get("attenuation_cfg"),
            attenuated_label_ids=attenuated_label_ids,
        )
        sigma, kernel_size = _parse_label_smoothing(seg_cfg)
        return cls(
            height=int(cam_cfg["height"]),
            width=int(cam_cfg.get("width", cam_cfg.get("height"))),
            sdd_bounds=ScalarRangeSampler.parse(cam_cfg["sdd"], "intrinsics_cfg.sdd"),
            delx_bounds=ScalarRangeSampler.parse(cam_cfg["delx"], "intrinsics_cfg.delx"),
            dely_bounds=(
                ScalarRangeSampler.parse(cam_cfg["dely"], "intrinsics_cfg.dely")
                if has_dely
                else None
            ),
            x0_bounds=ScalarRangeSampler.parse(cam_cfg.get("x0", 0.0), "intrinsics_cfg.x0"),
            y0_bounds=ScalarRangeSampler.parse(cam_cfg.get("y0", 0.0), "intrinsics_cfg.y0"),
            orientation=str(profile.get("extrinsics_cfg", {}).get("orientation", "AP")),
            isocenter_cfg=_parse_isocenter(profile.get("isocenter_cfg")),
            ortho_prob=_parse_projection_mode(cam_cfg.get("projection_mode", "cone")),
            num_views=int(num_views),
            n_samples=int(profile.get("num_samples", 500)),
            preset=profile["preset"],
            camera_displacement=float(profile.get("camera_displacement", 1200.0)),
            sample_params=dict(profile.get("sample_params_cfg", {})),
            attenuation_range=attenuation["range"],
            do_per_label_attenuation=attenuation["do_per_label"],
            attenuated_label_ids=attenuation["ids"],
            attenuation_dist=attenuation["dist"],
            attenuation_prob=attenuation["prob"],
            input_num_label_channels=int(input_num_label_channels),
            output_num_label_channels=int(output_num_label_channels),
            render_soft_labels=bool(seg_cfg.get("soft_labels", False)),
            seg_threshold=float(seg_cfg.get("threshold", 0.0)),
            label_smoothing_sigma=sigma,
            label_smoothing_kernel_size=kernel_size,
        )

    @property
    def follows_delx(self) -> bool:
        """Return whether the detector row spacing mirrors the column spacing.

        Returns:
            ``True`` when no independent row-spacing range is configured.
        """
        return self.dely_bounds is None


def _parse_isocenter(cfg: Any) -> IsocenterConfig:
    """Parse a mandatory ``isocenter_cfg`` block into an ``IsocenterConfig``."""
    if not isinstance(cfg, Mapping):
        valid = ", ".join(sorted(_VALID_ISOCENTER_SCHEMES))
        raise ValueError(f"drr_model.isocenter_cfg is required with sample_scheme in {valid}.")
    scheme = cfg.get("sample_scheme")
    if scheme not in _VALID_ISOCENTER_SCHEMES:
        valid = ", ".join(sorted(_VALID_ISOCENTER_SCHEMES))
        raise ValueError(f"isocenter_cfg.sample_scheme must be one of {valid}; got {scheme!r}.")
    if scheme != "random_label":
        if "replacement" in cfg:
            raise ValueError("isocenter_cfg.replacement is only valid for 'random_label'.")
        return IsocenterConfig(sample_scheme=scheme, replacement=None)
    replacement = cfg.get("replacement", True)
    if not isinstance(replacement, bool):
        raise TypeError("isocenter_cfg.replacement must be a boolean.")
    return IsocenterConfig(sample_scheme=scheme, replacement=replacement)


def _parse_projection_mode(cfg: Any) -> float:
    """Parse ``projection_mode`` into the probability of orthographic rendering."""
    valid = {"cone", "orthographic"}
    if isinstance(cfg, str):
        if cfg not in valid:
            raise ValueError(f"projection_mode must be one of {sorted(valid)}; got {cfg!r}.")
        return 1.0 if cfg == "orthographic" else 0.0
    if isinstance(cfg, Mapping):
        if not set(cfg).issubset(valid):
            raise ValueError(f"projection_mode keys must be a subset of {sorted(valid)}.")
        p_ortho = float(cfg.get("orthographic", 0.0))
        p_cone = float(cfg.get("cone", 0.0))
        if p_ortho < 0 or p_cone < 0:
            raise ValueError("projection_mode probabilities must be non-negative.")
        if not math.isclose(p_ortho + p_cone, 1.0, abs_tol=1e-6):
            raise ValueError("projection_mode probabilities must sum to 1.0.")
        return p_ortho
    raise TypeError(f"projection_mode must be a string or mapping, got {type(cfg).__name__}.")


def _parse_attenuation(
    cfg: Any,
    *,
    attenuated_label_ids: list[int],
) -> dict[str, Any]:
    """Parse an optional ``attenuation_cfg`` block into render attenuation kwargs.

    ``prob`` is the per-render probability of applying the sampled multipliers;
    an absent or disabled block never attenuates.
    """
    disabled = {
        "range": None,
        "do_per_label": False,
        "ids": list(attenuated_label_ids),
        "dist": None,
        "prob": 0.0,
    }
    if cfg is None:
        return disabled
    if not isinstance(cfg, Mapping):
        raise TypeError("attenuation_cfg must be a mapping with 'prob', 'range', and 'scope'.")
    if not bool(cfg.get("enabled", True)):
        return disabled

    unknown = sorted(set(cfg) - _ATTENUATION_CFG_KEYS)
    if unknown:
        raise ValueError(f"attenuation_cfg has unknown key(s): {unknown}.")
    missing = sorted({"prob", "range", "scope"} - set(cfg))
    if missing:
        raise ValueError(f"attenuation_cfg is missing required key(s): {missing}.")
    scope = cfg["scope"]
    if scope not in _VALID_ATTENUATION_SCOPES:
        valid = ", ".join(sorted(_VALID_ATTENUATION_SCOPES))
        raise ValueError(f"attenuation_cfg.scope must be one of {valid}; got {scope!r}.")
    do_per_label = scope == "per_label"
    if do_per_label and not attenuated_label_ids:
        raise ValueError("attenuation_cfg.scope='per_label' requires a foreground label.")

    dist = cfg.get("distribution")
    if dist is not None and not isinstance(dist, Mapping):
        raise TypeError("attenuation_cfg.distribution must be a mapping when set.")
    prob = cfg["prob"]
    if isinstance(prob, bool) or not isinstance(prob, (int, float)) or not 0.0 <= prob <= 1.0:
        raise ValueError(f"attenuation_cfg.prob must be a number in [0, 1]; got {prob!r}.")
    return {
        "range": ScalarRangeSampler.parse(cfg["range"], "attenuation_cfg.range"),
        "do_per_label": do_per_label,
        "ids": list(attenuated_label_ids),
        "dist": dict(dist) if dist is not None else None,
        "prob": float(prob),
    }


def _parse_label_smoothing(seg_cfg: Mapping[str, Any]) -> tuple[float, tuple[int, int] | None]:
    """Parse optional ``seg_cfg.label_smoothing`` into sigma and kernel size."""
    smoothing = seg_cfg.get("label_smoothing")
    if isinstance(smoothing, Mapping):
        sigma = float(smoothing.get("sigma", 0.0) or 0.0)
        kernel = smoothing.get("kernel_size")
    elif smoothing is not None:
        sigma, kernel = float(smoothing or 0.0), None
    else:
        return 0.0, None
    return sigma, _resolve_smoothing_kernel(kernel, sigma)


def _resolve_smoothing_kernel(kernel_size: Any, sigma: float) -> tuple[int, int] | None:
    """Resolve a Gaussian kernel size, defaulting to a 3-sigma odd support."""
    if sigma <= 0:
        return None
    if kernel_size is None:
        auto = max(3, int(2 * math.ceil(3 * sigma) + 1))
        return auto, auto
    if isinstance(kernel_size, int):
        size = (kernel_size, kernel_size)
    elif isinstance(kernel_size, (list, tuple)) and len(kernel_size) == 2:
        size = (int(kernel_size[0]), int(kernel_size[1]))
    else:
        raise TypeError("label_smoothing kernel_size must be null, an int, or a length-2 sequence.")
    if any(k <= 0 or k % 2 == 0 for k in size):
        raise ValueError("label_smoothing kernel_size values must be positive odd integers.")
    return size


def _plain(value: Any) -> Any:
    """Recursively convert config objects with ``to_dict`` into plain structures."""
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, Mapping):
        return {key: _plain(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_plain(child) for child in value]
    return value


def _merge_profile(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge a dataset ``override`` profile onto the shared ``base`` profile."""
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_profile(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def collect_configured_ct_dataset_names(config: Mapping[str, Any]) -> set[str]:
    """Return CT dataset names declared under ``data.CT``."""
    data_cfg = _plain(config).get("data")
    if not isinstance(data_cfg, Mapping):
        return set()
    ct_cfg = data_cfg.get("CT")
    if ct_cfg is None:
        return set()
    if not isinstance(ct_cfg, Mapping):
        raise TypeError("data.CT must be a mapping keyed by dataset name.")
    return {str(name) for name in ct_cfg}


def resolve_ct_profiles(config: Mapping[str, Any]) -> dict[str, dict[str, Any]] | None:
    """Resolve effective DRR profiles keyed by configured CT dataset name.

    Args:
        config: Full experiment config mapping.

    Returns:
        Mapping from CT dataset name to its effective (merged) DRR profile, or
        ``None`` when no ``drr_model`` block is configured.

    Raises:
        ValueError: If ``drr_model.default``/``drr_model.datasets`` is missing,
            the dataset keys do not match the configured CT datasets, or the
            profiles resolve to differing detector geometry or view counts.
        TypeError: If config sections have the wrong type.
    """

    config = _plain(config)
    drr_cfg = config.get("drr_model")
    if drr_cfg is None:
        return None
    if not isinstance(drr_cfg, Mapping):
        raise TypeError("drr_model must be a mapping with 'default' and 'datasets'.")

    default_cfg = drr_cfg.get("default")
    datasets_cfg = drr_cfg.get("datasets")
    if not isinstance(default_cfg, Mapping):
        raise ValueError("drr_model.default is required and must be a mapping.")
    if not isinstance(datasets_cfg, Mapping):
        raise ValueError(
            "drr_model.datasets is required and must list every CT dataset "
            "(use an empty mapping for datasets that use the default)."
        )

    ct_names = collect_configured_ct_dataset_names(config)
    profile_names = {str(name) for name in datasets_cfg}
    missing = sorted(ct_names - profile_names)
    stray = sorted(profile_names - ct_names)
    if missing or stray:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if stray:
            details.append(f"stray {stray}")
        raise ValueError(
            "drr_model.datasets keys must exactly match configured CT datasets; "
            + ", ".join(details)
            + "."
        )

    profiles: dict[str, dict[str, Any]] = {}
    for name in sorted(ct_names):
        override = datasets_cfg[name] or {}
        if not isinstance(override, Mapping):
            raise TypeError(f"drr_model.datasets.{name} must be a mapping or null.")
        profiles[name] = _merge_profile(default_cfg, override)

    _validate_uniform_geometry(profiles, config)
    return profiles


def _validate_uniform_geometry(
    profiles: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    """Require every CT profile to share detector geometry and view count."""
    sizes = {name: _profile_render_size(profile) for name, profile in profiles.items()}
    views = {name: resolve_num_views(profile, config) for name, profile in profiles.items()}
    if len(set(sizes.values())) > 1:
        raise ValueError(f"All CT DRR profiles must share intrinsics height/width; got {sizes}.")
    if len(set(views.values())) > 1:
        raise ValueError(f"All CT DRR profiles must share num_views; got {views}.")


def _profile_render_size(profile: Mapping[str, Any]) -> tuple[int, int]:
    """Return the ``(height, width)`` detector size declared by a profile."""
    cam_cfg = profile.get("intrinsics_cfg")
    if not isinstance(cam_cfg, Mapping) or "height" not in cam_cfg:
        raise ValueError("Each CT DRR profile must define intrinsics_cfg.height and width.")
    height = int(cam_cfg["height"])
    return height, int(cam_cfg.get("width", height))
