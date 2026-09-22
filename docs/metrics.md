# FleXray Segmentation Metrics

`fxr.metrics` is the tensor segmentation metrics surface for FleXray. It
contains pure metric functions for predictions and targets that are already in
memory. It does not include evaluation runners, config loading, logging,
aggregation, inference runtimes, or rendering.

## Public API

`fxr.metrics` exports:

- `dice_score(...)`
- `soft_dice_score(...)`
- `hd95(...)`

All functions return `torch.Tensor` values. Metrics keep results on the
prediction device where practical; `hd95` computes SciPy distance transforms on
CPU and returns the reduced tensor on the prediction device.

## Segmentation Modes

`dice_score`, `soft_dice_score`, and `hd95` accept
`mode="binary"`, `"onehot"`, `"multiclass"`, or `"auto"`.

- `binary`: predictions and targets have the same shape `(B, C, ...)`. Each
  channel is treated as an independent binary mask, including multi-channel
  overlapping X-ray masks.
- `onehot`: predictions and targets have the same shape `(B, C, ...)`, and the
  class axis is treated as a categorical distribution.
- `multiclass`: predictions have shape `(B, C, ...)`; targets are dense integer
  labels with shape `(B, ...)` or `(B, 1, ...)`.
- `auto`: same-shaped single-channel inputs resolve to `binary`, same-shaped
  multi-channel inputs resolve to `onehot`, and dense-target inputs resolve to
  `multiclass`.

Set `from_logits=True` when passing raw logits. Binary mode applies a sigmoid;
categorical modes apply softmax before scoring or discretization.

## Dice Metrics

`dice_score` rejects non-finite predictions with a `ValueError` (`NaN > 0.5`
is `False`, so a diverged model would otherwise be scored as predicting
background everywhere). It thresholds binary predictions and argmaxes
categorical predictions before scoring. In `mode="binary"`, `threshold`
applies independently to every channel, including overlapping multi-channel
masks. In categorical modes, multi-channel inputs use argmax regardless of
`threshold`, including two-channel inputs. `soft_dice_score` scores probabilities
directly and supports `pixel_weights` with batch/spatial shape or
batch/channel/spatial shape.

Both Dice functions support:

- `reduction="mean"`, `"sum"`, `"none"`, or `None` across channels.
- `batch_reduction="mean"`, `"sum"`, `"none"`, or `None` across samples.
- Class `weights` as a class vector or batch/class matrix.
- Scalar or sequence `ignore_index` for class channels.
- `ignore_empty_labels=True` to zero target-empty class weights.
- `ignore_background=True` to drop channel zero when foreground labels remain.

With `reduction="mean"`, the channel mean divides by the surviving weight sum,
so scaling every weight by a positive constant leaves the score unchanged. A
sample whose weights all vanish (everything ignored, or every target channel
empty) is left out of the batch mean rather than contributing a zero to it. An entirely
ignored batch returns zero; callers aggregating multiple batches must exclude
it and weight other batch means by their eligible image counts. Unreduced
entries retain their shape and use zero for ignored scores; `sum` and `none`
channel reductions keep their existing batch reduction behavior.

## HD95

`hd95` computes the symmetric 95th percentile Hausdorff distance between hard
binary surfaces for each batch item and class: the larger of the two
directional 95th percentiles (prediction-to-target and target-to-prediction),
matching the definition in the FleXray paper's Evaluation Metrics section.
This differs from taking a single percentile over pooled directional distances.
Both-empty masks score `0.0`.
When exactly one mask is empty, the penalty is the spatial diagonal of the
image grid. The metric supports the same class weighting, ignore, and reduction
arguments as `dice_score`.

The paper's evaluation additionally excludes labels below 0.1% ground-truth
area and labels with empty predictions from each image's HD95 average. Those
evaluation filters are separate from this tensor metric's empty-mask penalty;
matching the distance formula alone does not reproduce the paper's full
evaluation procedure.
