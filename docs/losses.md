# FleXray Training Losses

`fxr.losses` is the focused training-loss surface for FleXray. It contains only
the loss functions and containers used by the packaged training recipe.
It does not include training loops, metrics modules, inference code, rendering,
or cache-only/eval-only losses.

It has no experiment or config imports. FleXray
implements the small local helpers needed for wrapping loss functions,
constructing the supported config dictionaries, converting segmentation input
modes, and applying reductions.

## Public API

`fxr.losses` exports:

- `SoftDiceLoss`
- `PixelCELoss`
- `CombinedLoss`
- `DatasetRoutedLoss`
- `iter_leaf_loss_modules(...)`
- `set_batch_reduction(...)`

`SoftDiceLoss` and `PixelCELoss` are `torch.nn.Module` wrappers around validated
loss functions. Constructor keyword arguments become default loss arguments and
call-time keyword arguments may override them.

`PixelCELoss(..., ignore_empty_labels=True, mode="binary")` accepts a per-call
`supervise_empty_label_ids` sequence: those channels stay supervised as known
negatives even when their target is empty, while other empty channels are
still ignored. `CombinedLoss.forward(..., component_kwargs={name: {...}})`
routes such per-call arguments to one component only.

## Segmentation Modes

Both segmentation losses accept `mode="binary"`, `"onehot"`, `"multiclass"`, or
`"auto"`.

- `binary`: predictions and targets have the same shape `(B, C, ...)`. Each
  channel is treated as an independent binary mask. This is the mode used for
  FleXray X-ray model-label masks.
- `onehot`: predictions and targets have the same shape `(B, C, ...)`, but the
  class axis is treated as a categorical distribution.
- `multiclass`: predictions have shape `(B, C, ...)`; targets are dense integer
  labels with shape `(B, ...)` or `(B, 1, ...)`.
- `auto`: same-shaped single-channel inputs resolve to `binary`, same-shaped
  multi-channel inputs resolve to `onehot`, and dense-target inputs resolve to
  `multiclass`.

Set `from_logits=True` when passing raw model logits. Binary mode applies a
sigmoid; one-hot and multiclass modes apply softmax or log-softmax as needed.

## Reductions

`reduction` reduces spatial dimensions first:

- `"mean"` averages each sample or channel over pixels.
- `"sum"` sums each sample or channel over pixels.
- `"none"` or `None` keeps per-pixel values where the loss supports them.

`batch_reduction` then reduces the batch axis:

- `"mean"` averages over the batch.
- `"sum"` sums over the batch.
- `"none"` or `None` keeps the batch axis.

For `SoftDiceLoss`, `reduction` applies across channels and `batch_reduction`
applies across samples because dice scores are already spatially aggregated.
For `PixelCELoss`, `reduction` applies across pixels and `batch_reduction`
applies across samples or flattened binary sample/channel pairs.

`set_batch_reduction(loss_module, value)` recursively updates every wrapped leaf
loss under a `CombinedLoss` or `DatasetRoutedLoss`. It raises if a leaf module
does not expose a configurable `batch_reduction`.

## Combined Loss Config

Use `_combo_class` for weighted sums:

```yaml
_combo_class:
  SoftDiceLoss:
    _class: fxr.losses.SoftDiceLoss
    mode: binary
    from_logits: true
    ignore_empty_labels: true
    weight: 1.0
  PixelCELoss:
    _class: fxr.losses.PixelCELoss
    mode: binary
    from_logits: true
    weight: 0.5
```

`weight` defaults to `1.0`. `label_type` may be set on a component to route that
component to `outputs[label_type]` and `targets[label_type]`.

`CombinedLoss.get_last_loss_breakdown()` returns detached raw component losses,
not weighted losses. Returned tensors are cloned so logging code cannot mutate
the stored breakdown.

## Dataset-Routed Config

`DatasetRoutedLoss` selects one named profile by `dataset_name`:

```yaml
_class: fxr.losses.DatasetRoutedLoss
losses:
  standard_seg:
    _combo_class:
      SoftDiceLoss:
        _class: fxr.losses.SoftDiceLoss
        mode: binary
        from_logits: true
      PixelCELoss:
        _class: fxr.losses.PixelCELoss
        mode: binary
        from_logits: true
  partial_labeled_seg:
    _class: fxr.losses.SoftDiceLoss
    mode: binary
    from_logits: true
    ignore_background: true
    ignore_empty_labels: true
dataset_losses:
  HipRay: standard_seg
  FluXray: partial_labeled_seg
```

Each profile body must contain either `_combo_class` or `_class`. `_class`
accepts FleXray loss class names such as `fxr.losses.SoftDiceLoss` and import
strings for test doubles or local extensions.

`DatasetRoutedLoss.validate_routing(configured_dataset_names)` raises if any
configured dataset is missing a route or if the route table contains datasets
not present in the run.

`DatasetRoutedLoss.configure_supervise_empty_label_ids({dataset: [ids]})`
registers known-negative model channels per dataset (resolved by
`FleXrayTrainExperiment` from each packaged dataset spec's
`supervise_empty_labels`); every `PixelCELoss` component of the routed profile
that ignores empty labels receives them and must run in binary mode.

`DatasetRoutedLoss.get_last_loss_breakdown()` returns stable profile-prefixed
keys. Components from inactive profiles are present as zeros matching the active
loss tensor shape, device, and dtype.
