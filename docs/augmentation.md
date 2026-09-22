# Train Augmentation

`fxr.augmentation` provides the train-only 2D segmentation augmentation surface.
It loads packaged presets from the `fxr.configs/augmentations` resource path,
builds one centralized image/label pipeline per preset, and applies it to 4D
`(N, C, H, W)` tensors. `FleXrayTrainExperiment` augments every training batch
with the preset of its source modality and leaves validation batches unchanged.

```python
from fxr.augmentation import (
    apply_segmentation_augmentation,
    build_segmentation_augmentation_pipeline,
    load_named_augmentation_preset,
)

preset = load_named_augmentation_preset("Xray_base")
pipeline = build_segmentation_augmentation_pipeline(preset).train()
image, label = apply_segmentation_augmentation(pipeline, image, label)
```

## Packaged presets and per-modality routing

Two presets reproduce the training recipe behind the released FleXray
weights:

- `CT_base` — applied to rendered DRR batches.
- `Xray_base` — applied to real and generated (FluXray) X-ray batches; it is the
  CT chain plus `RandomLabelZoom` (`zoom_range: [1.1, 4.0]`, `p: 0.5`).

`base_light` is a reduced chain for smoke tests. Both base chains run, in order:
horizontal flip → (label zoom) → affine → elastic deformation → letter drop →
invert → CLAHE-or-gamma → contrast → plasma brightness → sharpness → Gaussian
noise → aspect crop.

Routing is configured under a top-level `augmentation` block; omitted entries
use the defaults above:

```yaml
augmentation:
  presets:
    CT: CT_base
    Xray: Xray_base
```

`resolve_augmentation_presets(config)` returns the resolved `{"ct": ..., "xray": ...}`
names. At run start the resolved transform maps are written once to
`<run>/augmentations/{ct,xray}.yml` (`snapshot_augmentation_presets`), so the
exact chain of a run stays inspectable after packaged presets change.

## Input normalization

Training normalization is configured independently under
`train.normalization`. The packaged base recipe clips each image to its own 0.5th and
99.5th percentiles, then rescales the clipped range to `[0, 1]`:

```yaml
train:
  normalization:
    scheme: percentile_minmax
    percentiles: [0.5, 99.5]
    eps: 1.0e-8
```

`scheme: minmax` selects ordinary per-sample min-max scaling, while
`scheme: standardize` selects zero-mean/unit-variance scaling. Configs without a
`train.normalization` block use zero-mean/unit-variance standardization. All
schemes compute bounds independently for each batch item, and constant images map to
zeros rather than producing non-finite values. The public
`build_input_normalizer(config)` helper validates the same contract used by
`FleXrayTrainExperiment`.

## Spatial and intensity transforms

Presets keep Kornia transforms under `kornia.augmentation` and use local FleXray
transforms for `RandomLabelZoom`, `RandomLetterDrop`, `RandomClaheOrGamma`,
`RandomAspectCrop`, and `RandomIntensityScale`. `RandomLabelZoom` is label-aware
and runs before the Kornia sequence so it can sample foreground label centroids.
`RandomClaheOrGamma` applies CLAHE, gamma, or neither per sample (marginal
probabilities `clahe_prob` and `gamma_prob`). `RandomAspectCrop` keeps a
centred strip of one axis and zeroes the rest of image and mask; its configured
range must fit inside the input axis, so the base presets require inputs of at
least 224 pixels per axis — route smaller images to `base_light` or a custom
preset via `augmentation.presets`.

After the Kornia sequence, `restore_hard_background` re-thresholds foreground
channels at `0.5` and rebuilds channel 0 as the foreground complement, because
interpolating spatial transforms otherwise leave fractional masks and an
inconsistent background channel. Pipelines therefore always return hard labels.

## Package Boundary

Dataset resolution, label-space rebuilding, training loops, and inference
runtimes live outside `fxr.augmentation`. Test-time augmentation lives in
`fxr.inference` (see `inference.md`); this package stays train-only.
