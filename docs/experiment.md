# `fxr.experiment`

Config-driven segmentation training experiments, built on the `fxr` data, model,
loss, metric, augmentation, and protocol surfaces.

WandB is the sole logging backend (installed by the `train` extra). The experiment
persists the EMA-selected `model` weights, raw resume weights when EMA is
enabled, the optimizer/scheduler/scaler state, and the run config; there is no
DataFrame/JSONL metric logging or validation-prediction disk capture.

## Layers

- `BaseExperiment` — owns the run directory: loads the immutable `config.yml`,
  tracks resumable `properties`, and `from_config(...)` stamps a new run under
  `log.root/{run_id}`.
- `TrainExperiment` — the generic lifecycle: `run() -> run_phase() -> run_step()`.
  It builds the optimizer and (optionally warmed-up) LR scheduler, checkpoints
  and resumes full training state, logs per-phase scalar means to WandB each
  epoch, and dispatches config-built callbacks by group (`epoch`, `step`,
  `wrapup`).
- `FleXrayTrainExperiment` — the concrete segmentation experiment. Native X-ray
  integer labels are projected into model channels (`TrainingLabelProjection`).
  Native CT labels are validated with the same projection, rendered as native
  foreground DRR masks, then collapsed into model-output channels. Rendered CT
  images are scaled per view into a finite `[0, 1]` range before augmentation
  so the X-ray intensity chain sees its expected input range. Each source is
  then augmented with its modality preset (`CT_base` for DRRs, `Xray_base` for
  real and generated X-rays), followed by the configured per-sample
  normalization, model, and loss tail.
  The loss may be single, combined (`_combo_class`), or dataset-routed.

## Scope

The experiment supports two input types: 2D X-ray segmentation and CT
segmentation rendered to DRRs. X-ray labels may be dense integer maps or named
channel masks. The latter includes partially labeled and generated FluXray data;
stored masks are reordered into model-output channels by label name via
`project_channel_mask`, with no DRR step. CT
batches carry `metadata.affine` and dense native integer labels; the experiment
passes a native-to-model collapse map into the DRR renderer. Label projection is
selected from storage/sample encoding metadata rather than from whether the
image was generated.

Mixed precision is opt-in via `train.amp_dtype` (`null` for fp32, `bfloat16`, or
`float16`): the model forward and loss run under `torch.autocast`, and a
`GradScaler` is engaged only for `float16`. A cluster config can set
`train.amp_dtype: bfloat16` as an override; the released weights were trained
with it.

EMA is configured under `train.ema`. Once initialized, its weights drive
validation, epoch/wrapup callbacks, and the `model` state in each
checkpoint; `model_raw` preserves exact optimizer continuation. The update
count and policy are restored on resume.

Checkpoint files commit through a same-directory temporary file and atomic
rename; only then does the run advance its persisted epoch property. Resume
treats `checkpoints/last.pt` as authoritative and restores Python, NumPy, Torch
CPU, and Torch CUDA random streams. WandB uses one persisted run ID with
`resume="allow"` so process restarts continue the same experiment record.

A top-level `initialization` block is model-only initialization for a new run,
not resume. `kind: pretrained` accepts a complete local/Hugging Face bundle;
`kind: run` accepts an explicitly trusted local PyTorch checkpoint. Bundle model
metadata and ordered labels are required and must exactly match the target
(excluding runtime-only `compile_cfg`). A standalone safetensors file is rejected
unless `allow_unverified_label_order: true` explicitly records that its label
order cannot be checked. `replace_head: true` loads every tensor except the
`output_conv` head (so the target may use a different protocol; `out_channels`
is excluded from the architecture check) and `freeze_backbone: true` leaves
only that head trainable.

Optional single-process compilation is configured under `model.compile_cfg`.
Only the forward callable is compiled, leaving the raw model responsible for
optimizer parameters, EMA tracking, and portable checkpoint keys. Supported
arguments are `backend`, `mode`, `options`, `fullgraph`, and `dynamic`, plus the
required boolean opt-in `enabled`.

## Running

```python
from fxr.experiment import FleXrayTrainExperiment

experiment = FleXrayTrainExperiment.from_config(config)  # dict or Config
experiment.run()
```

Construction failures close resources that were already created. `run()` closes
the shared training/validation ThunderDB readers and any lazily built callback
eval readers on success or failure. For build-only programmatic
use, call `experiment.close()` or use the experiment as a context manager.

The packaged config at `fxr/configs/training/base.yml` defines the supported
multimodal recipe: real X-ray sources and channel-mask `FluXray` live under
`data.Xray`, while CT sources render to DRRs. `DatasetRoutedLoss` applies the
partially-labeled route where configured. The recipe includes the released
normalization, augmentation, and DRR defaults, EMA, and callback-local
X-ray eval sets whose coverage is independent of the train mix.
Values marked `?` (e.g. `dataloader.batch_size`, `dataloader.num_workers`,
`log.root`) must be supplied at launch. The model's
`out_channels` is injected from `protocol.name`, so it is not set by hand.

### Config shape

The block below is an abbreviated schema illustration; the packaged `base.yml`
supplies the source selections and a `drr_model` profile for every active CT
dataset.

```yaml
experiment: {_class: fxr.experiment.FleXrayTrainExperiment, seed: 40}
train:
  epochs: 2000
  eval_freq: 50
  normalization: {scheme: percentile_minmax, percentiles: [0.5, 99.5]}
  ema: {enabled: true, decay: 0.9999, start_after_steps: 0, update_every: 1}
protocol:   {name: all_structures_flexray_v4}
model:
  _class: fxr.models.UNet
  in_channels: 1
  filters: [...]
  compile_cfg: {enabled: false}
data:       {Xray: {HipRay: {}}, CT: {MOOSE: {version: "2.0"}}}
dataloader:
  batch_size: 8
  num_workers: 4
  iters_per_epoch: 1000        # optional fixed train batches per epoch
  CT: {batch_size: 1}
  Xray: {batch_size: 8}
  proportions: {HipRay: 1}
optim:      {_class: torch.optim.AdamW, lr: 3.0e-4, lr_scheduler: null}
loss_func:  {_class: DatasetRoutedLoss, losses: {...}, dataset_losses: {HipRay: partial_labeled_seg}}
callbacks:
  epoch:
    samples: {_class: fxr.experiment.WandbSamplePredictionLogger, every: 250}
    eval_sets: {_class: fxr.experiment.EvalSetMetricLogger, every: 50, data: {Xray: {...}}}
log:        {root: ..., model_weights: {save_freq: 0}, wandb: {project: flexray}}
```

`FleXrayTrainExperiment` augments training batches with the preset of each
source's modality, selected under the optional top-level `augmentation.presets`
block (`{CT: CT_base, Xray: Xray_base}` by default; generated sources use the
X-ray preset). The resolved chains are snapshotted to `<run>/augmentations/`.
Validation batches are not augmented. See [augmentation.md](augmentation.md).

Crop-backed CT sources (`_attrs.storage_layout: crops`) can oversample crops with
rare anatomy through `data.CT.<name>.sample_weighting`
(`{scheme: inverse_label_frequency, tau: 0.5, class_aggregation: max}` in the
base recipe): each epoch draws `len(dataset)` crops with replacement from a
seeded `WeightedRandomSampler`. Stored crop centroids are filtered to the labels
the protocol supervises before they are offered as `random_label` isocenters; a
crop without supervised foreground falls back to the volume centre.

CT training adds a `data.CT` section and a matching `drr_model` block whose
`datasets` keys exactly match the configured CT dataset names. See
[camera.md](camera.md) for the CT render profile schema.

### Train mixing weights

`dataloader.proportions` maps active dataset names to non-negative numeric
sampling weights. When this mapping is present, it acts as an allowlist: datasets
listed with positive weights are active, datasets omitted from the mapping are
inactive, and explicit `0` entries may still be used to document a disabled
source. Remaining positive weights are forwarded to the mixed loader as the
relative sampling schedule (e.g. `MOOSE: 0.4` is sampled 16x as often as a
source with weight `0.025`). When `dataloader.iters_per_epoch` is set, active
loaders cycle until exactly that many train batches have emitted; otherwise each
source is consumed once per epoch.
`dataloader.CT` and `dataloader.Xray` may override `batch_size`, `num_workers`,
`pin_memory`, `prefetch_factor`, and `persistent_workers`; CT batch size must be
`1` because one CT volume renders into `num_views` DRRs. Unknown global or
modality keys are rejected. Count fields must be actual integers (not booleans
or floats), boolean fields must be actual booleans, `prefetch_factor` must be at
least one, and prefetch/persistent workers require positive `num_workers`.

### Train/val metrics

`TrainExperiment.run_phase(...)` returns sample-weighted scalar means. The
standard segmentation experiment computes Dice in independent-channel binary
mode and reports overall keys such as `dice` and `loss`. When step outputs include
`dataset_name`, per-dataset keys such as
`HipRay/dice` and `HipRay/loss`. Epoch WandB logging prefixes these with the
phase, producing keys like `train/dice`, `train/HipRay/dice`, `val/loss`, and
`val/JRST/loss`. The sample weight is inferred from `outputs["y_pred"].shape[0]`;
steps without a batch-shaped prediction fall back to weight `1`. Dice uses the
number of images retained by its empty-label filtering instead. Images with no
positive target in any channel contribute neither Dice nor count, while
background-only images remain eligible. A phase or dataset with no eligible
images omits its Dice key. Loss continues to use the full batch size.

## Callbacks

Experiment callbacks are constructed with the experiment as the first argument
and invoked with keyword context (`epoch=...`), grouped under `callbacks.<group>`
in the config. `WandbSamplePredictionLogger` periodically logs per-dataset
validation Dice and a few prediction overlays as `wandb.Image`.

`EvalSetMetricLogger` logs full-dataset benchmark Dice for callback-local X-ray
eval sets. It accepts only a `data: {Xray: ...}` block, builds the configured
`split` lazily on first use, evaluates without shuffling or dropping partial
batches, and logs only Dice metrics:

```yaml
callbacks:
  epoch:
    eval_sets:
      _class: fxr.experiment.EvalSetMetricLogger
      every: 50
      split: val
      batch_size: 4
      num_workers: 0
      min_ground_truth_area_fraction: 0.001
      data:
        Xray:
          HipRay: {}
```

`min_ground_truth_area_fraction` (default `0.0`) excludes, per image, foreground
labels whose ground-truth mask covers less than that fraction of the pixels, and
skips images that keep no scoreable label; the base recipe uses `0.001`.
With the default zero area threshold, batch Dice means are weighted by their
eligible image counts, consistently with training/validation phase metrics.

`every: N` runs on epoch `0` and every `N` epochs after that. If `data.Xray` is
empty or builds no datasets, the callback is a no-op. `data.CT` raises a clear
error because this callback supports X-ray eval sets only. The callback logs
`evalset/{dataset}/dice` for each eval dataset and `evalset/dice` as the mean
of those per-dataset Dice values.
