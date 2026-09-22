"""Raw nanoDRR rendering helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from nanodrr.camera import make_rt_inv
import nanodrr.drr as nanodrr_drr
from torch import Tensor

from .intrinsics import DRRIntrinsics, build_drr_camera_info, build_render_intrinsics
from .subjects import subject_from_tensors

__all__ = ["DRRRenderRequest", "DRRRenderResult", "render_drr"]


@dataclass(frozen=True)
class DRRRenderRequest:
    """Per-render pose, camera, and label-output settings.

    Attributes:
        rot: Per-view ZXY Euler rotations in degrees with shape ``(V, 3)`` or
            one rotation with shape ``(3,)``.
        xyz: Per-view camera translations with shape matching ``rot``.
        intrinsics: Camera intrinsics object or mapping accepted by
            ``build_render_intrinsics``.
        isocenter: Optional world-space isocenter with shape ``(3,)`` or
            per-view shape ``(V, 3)``.
        orientation: Optional nanoDRR orientation string.
        n_samples: Default number of samples along each projection ray.
        orthographic: Optional projection-mode override for ``intrinsics``.
        render_soft_labels: Whether projected labels retain soft foreground
            values instead of thresholded masks.
        seg_threshold: Threshold used for hard projected label masks.
        render_kwargs: Additional keyword arguments forwarded to nanoDRR render.
    """

    rot: Tensor
    xyz: Tensor
    intrinsics: DRRIntrinsics | Mapping[str, Any]
    isocenter: Tensor | None = None
    orientation: str | None = "AP"
    n_samples: int = 500
    orthographic: bool | None = None
    render_soft_labels: bool = False
    seg_threshold: float = 0.5
    render_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DRRRenderResult:
    """Rendered raw DRR image, projected labels, and camera metadata.

    Attributes:
        images: Rendered single-channel DRR images with explicit channel axis.
        labels: Projected label masks with explicit background channel.
        request: Original render request used to produce the result.
        intrinsics: Resolved camera intrinsics used for rendering.
        rt_inv: Per-view inverse camera extrinsic matrices passed to nanoDRR.
        rendered: Raw multi-channel nanoDRR render output before image and label
            projection post-processing.
        camera_info: Plain dictionary metadata derived from ``intrinsics``.
    """

    images: Tensor
    labels: Tensor
    request: DRRRenderRequest
    intrinsics: DRRIntrinsics
    rt_inv: Tensor
    rendered: Tensor

    @property
    def camera_info(self) -> dict[str, Any]:
        """Return plain per-view camera metadata for serialization.

        Returns:
            Dictionary containing camera spacing, offsets, image size, SDD, and
            projection mode.
        """

        return build_drr_camera_info(self.intrinsics)


def render_drr(
    *,
    volume: Tensor,
    label: Tensor | None,
    affine: Tensor,
    request: DRRRenderRequest,
    attenuation: float | tuple[float, float] | list[float] | None = None,
    attenuated_label_ids: list[int] | tuple[int, ...] | Tensor | None = None,
    do_per_label_attenuation: bool = False,
    max_label: int | None = None,
    attenuation_dist: Mapping[str, Any] | None = None,
    foreground_collapse_map: list[int] | tuple[int, ...] | Tensor | None = None,
    output_num_label_channels: int | None = None,
) -> DRRRenderResult:
    """Render raw DRR image tensors and explicit-background label masks.

    Args:
        volume: CT image tensor accepted by ``subject_from_tensors``.
        label: Dense integer label tensor aligned with ``volume``, or ``None``.
        affine: Voxel-to-world affine matrix.
        request: Camera pose, intrinsics, and projection settings.
        attenuation: Optional global or per-label attenuation bounds.
        attenuated_label_ids: Label ids eligible for per-label attenuation.
        do_per_label_attenuation: Whether attenuation is sampled per label id.
        max_label: Maximum dense label id to expose to nanoDRR.
        attenuation_dist: Optional attenuation sampling distribution config.
        foreground_collapse_map: Optional mapping from rendered foreground
            channels to output label channel ids, including background id ``0``.
        output_num_label_channels: Output channel count when collapsing labels.

    Returns:
        Render result containing raw DRR images and projected label masks.
    """

    volume_t = torch.as_tensor(volume)
    if not volume_t.is_floating_point():
        volume_t = volume_t.to(torch.float32)
    rot = _pose_tensor(request.rot, "rot", reference=volume_t)
    xyz = _pose_tensor(request.xyz, "xyz", reference=volume_t)
    if rot.shape != xyz.shape:
        raise ValueError("request.rot and request.xyz must have the same shape.")
    num_views = int(rot.shape[0])

    intrinsics = _resolve_intrinsics(
        request.intrinsics,
        num_views=num_views,
        reference=volume_t,
        orthographic_override=request.orthographic,
    )

    subject = subject_from_tensors(
        volume_t,
        label,
        affine,
        attenuation=attenuation,
        attenuated_label_ids=attenuated_label_ids,
        do_per_label_attenuation=do_per_label_attenuation,
        max_label=max_label,
        attenuation_dist=attenuation_dist,
    )

    isocenter = (
        subject.isocenter
        if request.isocenter is None
        else torch.as_tensor(
            request.isocenter,
            device=volume_t.device,
            dtype=volume_t.dtype,
        )
    )
    rt_inv = _make_rt_inv_per_view(
        rot,
        xyz,
        orientation=request.orientation,
        isocenter=isocenter,
    )

    render_kwargs = dict(request.render_kwargs)
    n_samples = int(render_kwargs.pop("n_samples", request.n_samples))
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2.")
    orthographic = bool(render_kwargs.pop("orthographic", intrinsics.orthographic))

    rendered = nanodrr_drr.render(
        subject=subject,
        k_inv=intrinsics.k_inv,
        rt_inv=rt_inv,
        sdd=intrinsics.sdd,
        height=intrinsics.height,
        width=intrinsics.width,
        n_samples=n_samples,
        orthographic=orthographic,
        **render_kwargs,
    )

    images = rendered.sum(dim=1, keepdim=True)
    labels = _project_labels(
        rendered,
        render_soft_labels=bool(request.render_soft_labels),
        seg_threshold=float(request.seg_threshold),
        foreground_collapse_map=foreground_collapse_map,
        output_num_label_channels=output_num_label_channels,
    )
    result_intrinsics = (
        intrinsics
        if orthographic == intrinsics.orthographic
        else DRRIntrinsics(
            k_inv=intrinsics.k_inv,
            sdd=intrinsics.sdd,
            delx=intrinsics.delx,
            dely=intrinsics.dely,
            x0=intrinsics.x0,
            y0=intrinsics.y0,
            height=intrinsics.height,
            width=intrinsics.width,
            orthographic=orthographic,
        )
    )

    return DRRRenderResult(
        images=images,
        labels=labels,
        request=request,
        intrinsics=result_intrinsics,
        rt_inv=rt_inv,
        rendered=rendered,
    )


def _pose_tensor(value: Tensor, name: str, *, reference: Tensor) -> Tensor:
    tensor = torch.as_tensor(value, device=reference.device)
    if not tensor.is_floating_point():
        tensor = tensor.to(reference.dtype)
    else:
        tensor = tensor.to(dtype=reference.dtype)
    if tensor.ndim == 1 and int(tensor.numel()) == 3:
        tensor = tensor.reshape(1, 3)
    if tensor.ndim != 2 or int(tensor.shape[1]) != 3:
        raise ValueError(f"request.{name} must have shape (3,) or (V, 3).")
    return tensor.contiguous()


def _resolve_intrinsics(
    intrinsics: DRRIntrinsics | Mapping[str, Any],
    *,
    num_views: int,
    reference: Tensor,
    orthographic_override: bool | None,
) -> DRRIntrinsics:
    if isinstance(intrinsics, DRRIntrinsics):
        orthographic = (
            intrinsics.orthographic
            if orthographic_override is None
            else bool(orthographic_override)
        )
        if intrinsics.num_views not in (1, num_views):
            raise ValueError(
                "DRRIntrinsics must contain one view or match the pose view count."
            )
        if intrinsics.num_views == num_views:
            resolved = intrinsics.to(device=reference.device, dtype=reference.dtype)
            if resolved.orthographic == orthographic:
                return resolved
        return build_render_intrinsics(
            sdd=intrinsics.sdd,
            delx=intrinsics.delx,
            dely=intrinsics.dely,
            x0=intrinsics.x0,
            y0=intrinsics.y0,
            height=intrinsics.height,
            width=intrinsics.width,
            num_views=num_views,
            orthographic=orthographic,
            device=reference.device,
            dtype=reference.dtype,
        )

    orthographic = bool(
        intrinsics.get("orthographic", False)
        if orthographic_override is None
        else orthographic_override
    )
    return build_render_intrinsics(
        sdd=intrinsics["sdd"],
        delx=intrinsics["delx"],
        dely=intrinsics["dely"],
        x0=intrinsics.get("x0", 0.0),
        y0=intrinsics.get("y0", 0.0),
        height=int(intrinsics["height"]),
        width=int(intrinsics["width"]),
        num_views=num_views,
        orthographic=orthographic,
        device=reference.device,
        dtype=reference.dtype,
    )


def _make_rt_inv_per_view(
    rot: Tensor,
    xyz: Tensor,
    *,
    orientation: str | None,
    isocenter: Tensor,
) -> Tensor:
    if isocenter.ndim == 1:
        if int(isocenter.numel()) != 3:
            raise ValueError("isocenter must have shape (3,) or (V, 3).")
        return make_rt_inv(rot, xyz, orientation=orientation, isocenter=isocenter)
    if isocenter.ndim == 2:
        if isocenter.shape != rot.shape:
            raise ValueError(
                "Per-view isocenter must have shape (V, 3) matching request.rot."
            )
        return torch.cat(
            [
                make_rt_inv(
                    rot[idx : idx + 1],
                    xyz[idx : idx + 1],
                    orientation=orientation,
                    isocenter=isocenter[idx],
                )
                for idx in range(int(rot.shape[0]))
            ],
            dim=0,
        )
    raise ValueError("isocenter must have shape (3,) or (V, 3).")


def _project_labels(
    rendered: Tensor,
    *,
    render_soft_labels: bool,
    seg_threshold: float,
    foreground_collapse_map: list[int] | tuple[int, ...] | Tensor | None = None,
    output_num_label_channels: int | None = None,
) -> Tensor:
    if rendered.ndim != 4:
        raise ValueError("nanoDRR render output must have shape (V, C, H, W).")
    foreground = rendered[:, 1:, :, :]
    if render_soft_labels:
        foreground_out = foreground.clamp(0.0, 1.0)
        soft_background = True
    else:
        foreground_out = (foreground > seg_threshold).to(dtype=rendered.dtype)
        soft_background = False

    if foreground_collapse_map is not None:
        foreground_out = _collapse_foreground_channels(
            foreground_out,
            foreground_collapse_map,
            output_num_label_channels=output_num_label_channels,
        )
    elif foreground_out.shape[1] == 0:
        output_channels = (
            1 if output_num_label_channels is None else int(output_num_label_channels)
        )
        if output_channels <= 0:
            raise ValueError("output_num_label_channels must be positive.")
        foreground_out = rendered.new_zeros(
            rendered.shape[0], output_channels - 1, rendered.shape[2], rendered.shape[3]
        )

    if foreground_out.shape[1] == 0:
        background = torch.ones(
            rendered.shape[0],
            1,
            rendered.shape[2],
            rendered.shape[3],
            device=rendered.device,
            dtype=rendered.dtype,
        )
    else:
        foreground_union = foreground_out.amax(dim=1, keepdim=True)
        if soft_background:
            background = (1.0 - foreground_union).clamp(0.0, 1.0)
        else:
            background = (foreground_union <= 0).to(dtype=rendered.dtype)
    return torch.cat([background, foreground_out], dim=1)


def _collapse_foreground_channels(
    foreground: Tensor,
    collapse_map: list[int] | tuple[int, ...] | Tensor,
    *,
    output_num_label_channels: int | None,
) -> Tensor:
    """Collapse rendered foreground masks into output model channels.

    Args:
        foreground: Rendered foreground masks shaped ``(V, C_fg, H, W)``.
        collapse_map: Target output channel id for each foreground channel.
        output_num_label_channels: Total output channels including background.

    Returns:
        Foreground masks shaped ``(V, output_num_label_channels - 1, H, W)``.

    Raises:
        ValueError: If the collapse map length or target ids are invalid.
    """

    targets = torch.as_tensor(collapse_map, device=foreground.device, dtype=torch.long)
    targets = targets.reshape(-1)
    if int(targets.numel()) != int(foreground.shape[1]):
        raise ValueError(
            "foreground_collapse_map length must match rendered foreground "
            f"channels ({foreground.shape[1]}), got {targets.numel()}."
        )
    if torch.any(targets < 0):
        raise ValueError("foreground_collapse_map target ids must be non-negative.")

    if output_num_label_channels is None:
        output_channels = int(targets.max().item()) + 1 if targets.numel() else 1
    else:
        output_channels = int(output_num_label_channels)
    if output_channels <= 0:
        raise ValueError("output_num_label_channels must be positive.")
    if targets.numel() and int(targets.max().item()) >= output_channels:
        raise ValueError(
            "foreground_collapse_map target ids must be less than "
            "output_num_label_channels."
        )

    collapsed = foreground.new_zeros(
        foreground.shape[0],
        output_channels - 1,
        foreground.shape[2],
        foreground.shape[3],
    )
    for source_idx, target_id_t in enumerate(targets.tolist()):
        target_id = int(target_id_t)
        if target_id <= 0:
            continue
        target = collapsed[:, target_id - 1]
        collapsed[:, target_id - 1] = torch.maximum(target, foreground[:, source_idx])
    return collapsed
