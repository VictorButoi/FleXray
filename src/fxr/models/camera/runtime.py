"""CT volume to DRR rendering runtime for segmentation training.

``SegmentationDRRRuntime`` owns the parsed DRR render configuration and the pose
sampler for one CT dataset profile. Each call to :meth:`render` samples camera
intrinsics, poses, and an isocenter, then delegates the actual projection to
:func:`fxr.drr.render_drr`, adding the optional segmentation label smoothing on
top. The runtime is a *pure renderer*: image normalization and augmentation are
applied by the experiment's forward pass, exactly as on the X-ray path.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from fxr.drr import (
    DRRRenderRequest,
    IsocenterConfig,
    PoseSampler,
    compute_isocenter,
    render_drr,
    sample_isocenters_from_centroids,
)
from fxr.models.camera.config import (
    DRRRenderConfig,
    ScalarRangeSampler,
    resolve_ct_profiles,
    resolve_num_views,
)
from fxr.models.camera.labels import (
    prepend_background_channel,
    smooth_projected_fg_masks,
)
from fxr.protocols import (
    resolve_run_attenuated_label_ids,
    resolve_run_output_label_names,
    resolve_run_render_label_names,
)

__all__ = ["CTDRRRenderResult", "SegmentationDRRRuntime"]


@dataclass(frozen=True)
class CTDRRRenderResult:
    """Rendered CT->DRR training inputs.

    Attributes:
        images: Raw DRR images shaped ``(V, 1, H, W)`` before normalization.
        labels: Projected model-channel labels shaped ``(V, C_out, H, W)`` with
            background at channel 0.
        request: Sampled camera request (poses, intrinsics, isocenter) used for
            the render, or ``None`` when not recorded.
    """

    images: Tensor
    labels: Tensor
    request: DRRRenderRequest | None = None


def scale_rendered_images(images: Tensor, eps: float = 1e-6) -> Tensor:
    """Map rendered DRR intensities to a finite per-view ``[0, 1]`` range.

    This is the scaling training applies before augmentation, so offline
    renders match what the model sees.

    Args:
        images: Rendered DRR image tensor shaped ``(V, 1, H, W)``.
        eps: Minimum range used for constant images.

    Returns:
        Per-view min-max scaled image tensor with the same shape as ``images``.
    """

    finite = torch.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0)
    dims = tuple(range(1, finite.ndim))
    min_value = finite.amin(dim=dims, keepdim=True)
    max_value = finite.amax(dim=dims, keepdim=True)
    scale = (max_value - min_value).clamp_min(float(eps))
    return (finite - min_value) / scale


class SegmentationDRRRuntime:
    """Sample DRR cameras and render a CT subject into training images and labels.

    Attributes:
        render_config: Parsed immutable DRR render configuration.
        device: Device used for pose-sampler state.
        pose_sampler: Camera pose sampler built from the configured preset.
        num_views: Number of DRR views rendered per CT subject.
    """

    def __init__(self, render_config: DRRRenderConfig, *, device: torch.device) -> None:
        """Store the render config and build the configured pose sampler.

        Args:
            render_config: Parsed DRR render configuration for one CT profile.
            device: Device used for pose-sampler buffers and random state.

        Returns:
            ``None``.
        """

        self.render_config = render_config
        self.device = device
        self.pose_sampler = PoseSampler(
            preset=render_config.preset,
            camera_displacement=render_config.camera_displacement,
            sample_params=render_config.sample_params,
        ).to(device)

    # --------------------------------------------------------------- builders
    @classmethod
    def from_effective_config(
        cls,
        config: Mapping[str, Any],
        *,
        device: torch.device,
    ) -> "SegmentationDRRRuntime":
        """Build a runtime from a config whose ``drr_model`` is one flat profile.

        Args:
            config: Experiment config with ``drr_model`` set to a single effective
                (merged) CT DRR profile and a top-level ``protocol`` block.
            device: Device used for pose-sampler state.

        Returns:
            A configured runtime instance.

        Raises:
            ValueError: If the config has no protocol-derived model label space.
        """

        profile = config["drr_model"]
        output_labels = resolve_run_output_label_names(config)
        render_labels = resolve_run_render_label_names(config)
        attenuated_ids = resolve_run_attenuated_label_ids(config)
        if output_labels is None or render_labels is None or attenuated_ids is None:
            raise ValueError(
                "CT->DRR rendering requires a top-level protocol.name to derive the "
                "model label space."
            )
        render_config = DRRRenderConfig.from_profile(
            profile,
            num_views=resolve_num_views(profile, config),
            input_num_label_channels=len(render_labels),
            output_num_label_channels=len(output_labels),
            attenuated_label_ids=list(attenuated_ids),
        )
        return cls(render_config, device=device)

    @classmethod
    def build_runtimes_from_config(
        cls,
        config: Mapping[str, Any],
        *,
        device: torch.device,
    ) -> dict[str, "SegmentationDRRRuntime"] | None:
        """Build one runtime per configured CT dataset profile.

        Args:
            config: Full experiment config mapping.
            device: Device used for pose-sampler state.

        Returns:
            Mapping from CT dataset name to its runtime, or ``None`` when no
            ``drr_model`` block is configured.
        """

        profiles = resolve_ct_profiles(config)
        if not profiles:
            return None
        return {
            name: cls.from_effective_config(
                _effective_config(config, profile), device=device
            )
            for name, profile in profiles.items()
        }

    @staticmethod
    def get_configured_render_size(config: Mapping[str, Any]) -> tuple[int, int] | None:
        """Return the shared detector ``(height, width)`` without building state.

        Args:
            config: Full experiment config mapping.

        Returns:
            Shared detector size, or ``None`` without a ``drr_model`` block.
        """
        profiles = resolve_ct_profiles(config)
        if not profiles:
            return None
        cam_cfg = next(iter(profiles.values()))["intrinsics_cfg"]
        height = int(cam_cfg["height"])
        return height, int(cam_cfg.get("width", height))

    @property
    def num_views(self) -> int:
        """Return the number of DRR views rendered per CT subject.

        Returns:
            Configured number of views.
        """
        return self.render_config.num_views

    def get_render_size(self) -> tuple[int, int]:
        """Return the configured detector ``(height, width)``.

        Returns:
            Detector height and width in pixels.
        """
        return self.render_config.height, self.render_config.width

    # ---------------------------------------------------------------- render
    def render(
        self,
        *,
        volume: Tensor,
        label: Tensor,
        affine: Tensor,
        isocenter: Tensor | None = None,
        fg_centroids_ijk: Tensor | None = None,
        render_kwargs: Mapping[str, Any] | None = None,
        foreground_collapse_map: tuple[int, ...] | Tensor | None = None,
        attenuated_label_ids: tuple[int, ...] | Tensor | None = None,
    ) -> CTDRRRenderResult:
        """Render one CT subject into DRR images and projected model labels.

        Args:
            volume: CT volume for a single subject, shaped ``(1, D, H, W)`` or
                ``(1, 1, D, H, W)``.
            label: Dense integer label aligned with ``volume``.
            affine: Voxel-to-world affine, shaped ``(4, 4)`` or ``(1, 4, 4)``.
            isocenter: Optional precomputed world-space isocenter overriding the
                configured sampling scheme.
            fg_centroids_ijk: Per-class voxel centroids required by the
                ``random_label`` isocenter scheme.
            render_kwargs: Optional per-call nanoDRR render overrides.
            foreground_collapse_map: Optional native-foreground to model-channel
                collapse map used after DRR label projection.
            attenuated_label_ids: Optional foreground ids eligible for per-label
                attenuation, overriding the config-derived model ids.

        Returns:
            The rendered DRR images and projected labels.
        """

        cfg = self.render_config
        affine_2d = affine[0] if affine.ndim == 3 else affine
        resolved_isocenter = self._resolve_isocenter(
            volume, label, affine_2d, isocenter, fg_centroids_ijk
        )
        request = self.sample_request(
            isocenter=resolved_isocenter, render_kwargs=render_kwargs
        )
        # Bernoulli gate on attenuation randomization (attenuation_cfg.prob).
        attenuate = random.random() < cfg.attenuation_prob
        rendered = render_drr(
            volume=volume,
            label=label,
            affine=affine_2d,
            request=request,
            attenuation=cfg.attenuation_range if attenuate else None,
            attenuated_label_ids=(
                attenuated_label_ids
                if attenuated_label_ids is not None
                else cfg.attenuated_label_ids
            ),
            do_per_label_attenuation=cfg.do_per_label_attenuation and attenuate,
            max_label=_render_max_label(cfg, foreground_collapse_map),
            attenuation_dist=cfg.attenuation_dist if attenuate else None,
            foreground_collapse_map=foreground_collapse_map,
            output_num_label_channels=(
                cfg.output_num_label_channels
                if foreground_collapse_map is not None
                else None
            ),
        )

        labels = rendered.labels
        if cfg.label_smoothing_sigma > 0:
            smoothed = smooth_projected_fg_masks(
                labels[:, 1:],
                sigma=cfg.label_smoothing_sigma,
                kernel_size=cfg.label_smoothing_kernel_size,
            )
            labels = prepend_background_channel(smoothed, soft_background=True)
        return CTDRRRenderResult(images=rendered.images, labels=labels, request=request)

    def sample_request(
        self,
        *,
        isocenter: Tensor | None = None,
        render_kwargs: Mapping[str, Any] | None = None,
    ) -> DRRRenderRequest:
        """Sample camera intrinsics and poses for one render request.

        Args:
            isocenter: Optional world-space isocenter passed through to rendering.
            render_kwargs: Optional per-call nanoDRR render overrides.

        Returns:
            A populated :class:`fxr.drr.DRRRenderRequest`.
        """

        cfg = self.render_config
        intrinsics = self._sample_intrinsics()
        sampled_sdd = torch.tensor(intrinsics["sdd"], dtype=torch.float32)
        rot, xyz = self.pose_sampler.sample(
            num_poses=cfg.num_views, sampled_sdd=sampled_sdd
        )
        intrinsics["sdd"] = [float(value) for value in sampled_sdd.reshape(-1).tolist()]
        return DRRRenderRequest(
            rot=rot,
            xyz=xyz,
            intrinsics=intrinsics,
            isocenter=isocenter,
            orientation=cfg.orientation,
            n_samples=cfg.n_samples,
            orthographic=random.random() < cfg.ortho_prob,
            render_soft_labels=cfg.render_soft_labels,
            seg_threshold=cfg.seg_threshold,
            render_kwargs=dict(render_kwargs or {}),
        )

    # --------------------------------------------------------------- helpers
    def _resolve_isocenter(
        self,
        volume: Tensor,
        label: Tensor,
        affine_2d: Tensor,
        isocenter: Tensor | None,
        fg_centroids_ijk: Tensor | None,
    ) -> Tensor:
        """Resolve the world-space isocenter for the configured sampling scheme.

        Args:
            volume: CT volume for the subject.
            label: Dense label aligned with ``volume``.
            affine_2d: Voxel-to-world affine matrix.
            isocenter: Optional precomputed world-space isocenter.
            fg_centroids_ijk: Optional per-class voxel centroids. An empty
                ``(0, 3)`` tensor (no supervised foreground in the crop) falls
                back to the volume center.

        Returns:
            World-space isocenter tensor for each requested view.

        Raises:
            ValueError: If random-label sampling lacks foreground centroids.
        """
        if isocenter is not None:
            return isocenter
        cfg = self.render_config
        if cfg.isocenter_cfg.sample_scheme == "random_label":
            if fg_centroids_ijk is None:
                raise ValueError(
                    "isocenter_cfg.sample_scheme='random_label' requires the CT batch "
                    "to provide fg_centroids_ijk."
                )
            if torch.as_tensor(fg_centroids_ijk).numel() == 0:
                fallback = IsocenterConfig(sample_scheme="volume_center")
                return compute_isocenter(
                    volume, label, affine_2d, fallback, num_views=cfg.num_views
                )
            return sample_isocenters_from_centroids(
                fg_centroids_ijk,
                affine_2d,
                cfg.num_views,
                replacement=bool(cfg.isocenter_cfg.replacement),
            )
        return compute_isocenter(
            volume, label, affine_2d, cfg.isocenter_cfg, num_views=cfg.num_views
        )

    def _sample_intrinsics(self) -> dict[str, Any]:
        """Sample per-view detector intrinsics as length-``num_views`` lists.

        Returns:
            Mapping of sampled detector geometry fields.
        """
        cfg = self.render_config
        num_views = cfg.num_views
        delx = [ScalarRangeSampler.sample(cfg.delx_bounds) for _ in range(num_views)]
        dely = (
            list(delx)
            if cfg.follows_delx
            else [ScalarRangeSampler.sample(cfg.dely_bounds) for _ in range(num_views)]
        )
        return {
            "sdd": [
                ScalarRangeSampler.sample(cfg.sdd_bounds) for _ in range(num_views)
            ],
            "delx": delx,
            "dely": dely,
            "x0": [ScalarRangeSampler.sample(cfg.x0_bounds) for _ in range(num_views)],
            "y0": [ScalarRangeSampler.sample(cfg.y0_bounds) for _ in range(num_views)],
            "height": cfg.height,
            "width": cfg.width,
        }


def _render_max_label(
    cfg: DRRRenderConfig,
    foreground_collapse_map: tuple[int, ...] | Tensor | None,
) -> int:
    """Return the dense maximum label id exposed to nanoDRR.

    Args:
        cfg: Runtime render configuration.
        foreground_collapse_map: Optional native foreground collapse map.

    Returns:
        Maximum dense label id expected in the render input label.
    """

    if foreground_collapse_map is None:
        return cfg.input_num_label_channels - 1
    return int(torch.as_tensor(foreground_collapse_map).numel())


def _effective_config(
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a config copy whose ``drr_model`` is one effective flat profile."""
    config_copy = deepcopy(dict(config))
    config_copy["drr_model"] = deepcopy(dict(profile))
    return config_copy
