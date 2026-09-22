"""Low-level nanoDRR rendering primitives for FleXray."""

from .intrinsics import DRRIntrinsics, build_drr_camera_info, build_render_intrinsics
from .isocenter import (
    IsocenterConfig,
    compute_isocenter,
    sample_isocenters_from_centroids,
)
from .poses import PoseSampler, RandomPoseGenerator
from .rendering import DRRRenderRequest, DRRRenderResult, render_drr
from .subjects import sample_attenuation, subject_from_tensors

__all__ = [
    "DRRIntrinsics",
    "DRRRenderRequest",
    "DRRRenderResult",
    "IsocenterConfig",
    "PoseSampler",
    "RandomPoseGenerator",
    "build_drr_camera_info",
    "build_render_intrinsics",
    "compute_isocenter",
    "render_drr",
    "sample_attenuation",
    "sample_isocenters_from_centroids",
    "subject_from_tensors",
]
