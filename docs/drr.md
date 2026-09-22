# FleXray DRR Primitives

`fxr.drr` is a low-level CT-to-DRR rendering surface built on nanoDRR. It
contains reusable rendering primitives only; it does not include training
runtime configuration, augmentation callbacks, inference runners, or Flux
rendering.

## Tensor Contracts

`subject_from_tensors(...)` accepts one CT image and optional dense labels:

- CT image: `(1, D, H, W)` or legacy `(1, 1, D, H, W)` in Hounsfield units.
- Dense label: `(D, H, W)`, `(1, D, H, W)`, or legacy `(1, 1, D, H, W)`.
- Affine: `(4, 4)` voxel-to-world transform using the same voxel axis order as
  the CT tensor.

Dense labels use non-negative integer ids with background at `0`; callers may
pass dataset-native or model-channel ids. The helper permutes CT and label
axes into nanoDRR's internal `(1, 1, W, H, D)` layout and converts HU values to linear attenuation coefficients before creating
the nanoDRR `Subject`.

`render_drr(...)` returns `DRRRenderResult.images` as raw projected DRR tensors
with shape `(V, 1, H, W)`. `DRRRenderResult.labels` has shape
`(V, C, H, W)` and always includes an explicit background channel at `0`.

## Pose Sampling

`PoseSampler` returns `(rot, xyz)` tensors with shape `(V, 3)`. Rotations are ZXY
Euler angles in degrees, matching nanoDRR extrinsics.

Fixed presets are:

- `frontal`
- `lateral`
- `lateral_left`
- `offangle`
- `above`

Use `preset="fixed"` with `fixed_params={"rot": (...), "xyz": (...)}` for an
explicit pose. A list of presets cycles across calls, and nested lists randomly
choose one fixed pose for that slot. Pass `seed=...` for deterministic nested
choices and random pose sampling.

`preset="random"` requires explicit axis ranges:

```python
sample_params = {
    "rot_range": {"alpha": [-10, 10], "beta": [0, 5], "gamma": [-3, 3]},
    "xyz_range": {"x": [-20, 20], "y": [900, 1100], "z": [-20, 20]},
}
```

When a sampled SDD tensor is passed to `sample(...)`, the sampled y translation
and SDD are sorted per view so the source-detector geometry remains valid.

## Intrinsics

`build_render_intrinsics(...)` normalizes scalar or per-view values for `sdd`,
`delx`, `dely`, `x0`, and `y0` into a `DRRIntrinsics` dataclass. The returned
`k_inv` has shape `(V, 3, 3)` and matches nanoDRR `make_k_inv` for each view.

```python
intrinsics = build_render_intrinsics(
    sdd=[1050.0, 1200.0],
    delx=0.7,
    dely=0.7,
    x0=0.0,
    y0=0.0,
    height=256,
    width=256,
)
```

`build_drr_camera_info(...)` converts resolved intrinsics into plain Python
metadata with per-view lists and the detector size.

## Isocenters

`compute_isocenter(...)` supports:

- `volume_center`: world-space center of the CT tensor.
- `label_centroid`: world-space centroid of all non-background voxels, falling
  back to volume center when no foreground exists.

For random label targets, precompute foreground centroids in voxel coordinates
and call `sample_isocenters_from_centroids(...)`. It supports replacement and
non-replacement sampling, returning one world-space isocenter per requested view.

## Attenuation

`sample_attenuation(...)` supports:

- uniform sampling over `(lo, hi)`
- beta sampling with `{"type": "beta", "alpha": a, "beta": b}`
- truncated lognormal sampling with `{"type": "lognormal", "mode": m, "sigma": s}`

`subject_from_tensors(...)` accepts global attenuation or per-label attenuation.
Per-label attenuation uses dense lookup ids, so non-contiguous model labels are
supported:

```python
subject = subject_from_tensors(
    volume_hu,
    label,
    affine,
    attenuation=(0.85, 1.15),
    attenuated_label_ids=[2, 7, 19],
    do_per_label_attenuation=True,
    max_label=63,
)
```

## Direct Rendering

```python
from fxr.drr import DRRRenderRequest, PoseSampler, build_render_intrinsics, render_drr

pose_sampler = PoseSampler("frontal", camera_displacement=1050.0)
rot, xyz = pose_sampler.sample(num_poses=1)
intrinsics = build_render_intrinsics(
    sdd=1200.0,
    delx=0.7,
    dely=0.7,
    height=256,
    width=256,
)

request = DRRRenderRequest(rot=rot, xyz=xyz, intrinsics=intrinsics)
result = render_drr(volume=volume_hu, label=label, affine=affine, request=request)
```

Hard labels threshold foreground projections with `seg_threshold` and rebuild
background from the thresholded foreground union. Set `render_soft_labels=True`
on `DRRRenderRequest` to return clamped soft foreground projections and a soft
background channel.

## Package Boundary

Training-time YAML profiles, source routing, label collapse, label smoothing,
augmentation, and normalization are owned by `fxr.models.camera` and
`fxr.experiment`. `fxr.drr` does not provide learnable camera parameters,
inference runtimes, a `DRRModel` compatibility alias, or Flux rendering.
