# FleXray CT→DRR Rendering Runtime

`fxr.models.camera` is the training-runtime rendering layer that turns CT volumes
into segmentation DRRs on the fly. It is built on the low-level `fxr.drr`
primitives (`PoseSampler`, `render_drr`, `compute_isocenter`,
`build_render_intrinsics`, `subject_from_tensors`) and adds the config parsing,
per-view sampling, and label post-processing needed for training.

The runtime is a **pure renderer**: it returns raw DRR image intensities and
projected labels. `FleXrayTrainExperiment` scales rendered CT images per view
into a finite `[0, 1]` range before applying the shared augmentation preset,
then applies the same configured per-sample normalization used on X-ray batches.

## Config schema

CT rendering is declared per dataset under a top-level `drr_model` block:

```yaml
data:
  CT:
    TotalSegCT: {}
    VerseCT: {}

drr_model:
  default:                       # shared profile for every CT dataset
    preset: random               # fxr.drr.PoseSampler preset
    num_views: 4                 # DRR views rendered per CT subject
    camera_displacement: 800.0
    intrinsics_cfg:
      height: 256
      width: 256                 # defaults to height when omitted
      sdd: [900, 1100]           # scalar or [low, high] sampling range
      delx: 1.4                  # dely follows delx unless given
      projection_mode: cone      # "cone" | "orthographic" | {cone: p, orthographic: q}
    extrinsics_cfg: {orientation: AP}
    sample_params_cfg:
      rot_range:
        alpha: [-10, 10]
        beta: [-5, 5]
        gamma: [-3, 3]
      xyz_range:
        x: [-20, 20]
        y: [700, 900]
        z: [-20, 20]
    isocenter_cfg:
      sample_scheme: label_centroid   # volume_center | label_centroid | random_label
    attenuation_cfg:
      prob: 0.5                  # per-render probability of attenuating
      range: [0.8, 1.2]
      scope: per_label           # per_label | global
      distribution: {type: lognormal, mode: 1.0, sigma: 1.0}
    seg_cfg:
      soft_labels: false
      threshold: 0.0
      label_smoothing: {sigma: 0.0}
  datasets:                      # one entry per configured data.CT dataset
    TotalSegCT: {}               # empty mapping uses the default profile
    VerseCT: {attenuation_cfg: {prob: 1.0, range: [0.9, 1.1], scope: per_label}}
```

`drr_model.datasets` keys must exactly match the datasets under `data.CT`. All
resolved profiles must share the same detector `height`/`width` and `num_views`.
`resolve_ct_profiles(config)` performs the merge and validation.

## Runtime

`SegmentationDRRRuntime.build_runtimes_from_config(config, device=...)` returns
one runtime per CT dataset. Each `render(...)` call:

1. resolves a world-space isocenter (config scheme, or precomputed
   `fg_centroids_ijk` for `random_label`; an empty centroid set falls back to
   the volume center);
2. samples per-view intrinsics and camera poses, building a
   `fxr.drr.DRRRenderRequest`;
3. renders via `fxr.drr.render_drr` (subject build, attenuation applied with
   probability `attenuation_cfg.prob`, projection); and
4. applies optional Gaussian label smoothing.

It returns `CTDRRRenderResult` with `images` shaped `(num_views, 1, H, W)` and
`labels` shaped `(num_views, C_model, H, W)` with background at channel 0.

FleXray training passes dense **dataset-native** CT labels into the renderer.
`FleXrayTrainExperiment` validates those native ids with `TrainingLabelProjection`,
renders native foreground channels, then supplies a foreground-collapse map so
projected masks land in the configured model-output channels. Per-label
attenuation also receives native foreground ids, so attenuation samples the source
label ids rather than model-channel ids. Calling the lower-level renderer without
a collapse map returns one foreground channel per rendered dense id.

## Offline rendering with `fxr-render`

`fxr-render` runs the same projection, runtime, and per-view `[0, 1]` scaling
that training uses on a packed `ct-seg` dataset and writes image/mask arrays
plus an `xray-seg` manifest, so rendered DRRs can be inspected or trained on as
an X-ray source. It is the raw-DRR path of the data engine (no diffusion
refinement).

```bash
fxr-render /data/MyCT --dataset-name MyCT --base base --profile MOOSE \
    --output ./rendered --split train --renders 4 --seed 0 --device auto [--png]
fxr-dataset pack ./rendered/manifest.yml /data/MyDRRs
```

`--base` names a packaged training config (or a YAML path) whose
`drr_model.datasets.<profile>` entry is merged onto `drr_model.default`;
`--set KEY=VALUE` overrides apply first. Each of `--renders` camera draws per
volume yields the profile's `num_views` images. Offline rendering defaults to one
view when the effective profile omits it, regardless of `dataloader.batch_size`.
Use `--set drr_model.default.num_views=8` to request eight views explicitly.
The output holds
`images/<sample>__r<k>_v<j>.npy` (`float32 (1, H, W)` in `[0, 1]`),
`masks/...npy` (`uint8 (C, H, W)` in model-output channel order with background
at channel 0; `float16` when the profile renders soft labels), optional
`previews/*.png`, and `manifest.yml` (`schema_version: 1`, `dataset_type:
xray-seg`, `label_names` = the resolved model-output labels, source subject/split,
and per-view camera metadata: `rot_deg`, `xyz_mm`, `orthographic`, `sdd`, `delx`).
Train on the packed result under `data.Xray.<name>.path` with the
`partial_labeled_seg` or `standard_seg` route. Renders are deterministic for a
given `--seed`. The Python entry point is `fxr.launch.render.render_dataset`.

## Experiment wiring

`DrrForwardPipeline` (in `fxr.experiment`) owns the per-dataset runtimes and routes
a CT batch to the correct one by source name. `FleXrayTrainExperiment` renders the
CT batch, scales each rendered view into `[0, 1]` for augmentation, then runs the
same augment → normalize → model → loss tail as the X-ray path. CT loaders
use a batch size of 1 because each volume becomes `num_views` DRRs that form the
effective training batch.

## Training view counts

`drr_model.default.num_views` sets views per CT volume independently of the
global image batch size. Dataset profiles may override it, provided every active
CT profile resolves to the same count. For example:

```yaml
dataloader:
  batch_size: 8
  CT: {batch_size: 1}
drr_model:
  default:
    num_views: 8
```

This fragment keeps an eight-image CT training batch while allowing the global
batch size to change later. Merge it into a complete training config.

A profile that omits `num_views` falls back to the global
`dataloader.batch_size`, and to one when neither is set. `fxr-train` and
`fxr-submit` resolve that count on each dataset profile after all overrides and
record it in the saved run config; resume reads the persisted config as-is.
A profile's explicit count always wins. `resolve_num_views(profile, config)`
applies the same rule programmatically, while `resolve_num_views(profile)`
defaults to one for standalone profiles. Offline `fxr-render` uses the
standalone rule and records an explicit count in its render config, so training
batch sizes do not silently multiply exported images.

## Runtime Boundary

The runtime implements sampled CT-to-segmentation-DRR rendering for the
FleXray training path. It does not provide a learnable-pose `DRRModel`, the
latent-segmentation variant, or Flux rendering.
