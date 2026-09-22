# FleXray Inference Utilities

`fxr.inference` contains the public pretrained-model API, image-file CLI,
framework-neutral tensor runner, logits/probability helpers, and postprocessing
utilities. Public inference loads Hugging Face model bundles. It does not read
FleXray training run directories directly.

## Public Model API

The one-call API is `FleXraySegmenter.from_pretrained`. The public model id is
`VictorButoi/flexray`, one repository holding the flagship and its four
ensemble members.

```python
from fxr.inference import FleXraySegmenter

segmenter = FleXraySegmenter.from_pretrained()                   # flagship model
segmenter = FleXraySegmenter.from_pretrained(ensemble=True)      # five-model ensemble
segmenter = FleXraySegmenter.from_pretrained(subfolder="members/flux000")  # one member
prediction = segmenter.predict("./image.png", tta_samples=16)    # tta_samples=1: one pass

logits = prediction.logits
probabilities = prediction.probabilities
masks = prediction.masks
```

`prediction.logits`, `prediction.probabilities`, and `prediction.masks` are
batched tensors with shape `BxCxHxW`. `masks` is `uint8`. Logits and
probabilities stay on the segmenter device.

`predict(image, *, label=None, threshold=0.5, tta_samples=1)` returns every
model channel by default. Pass `label="femurs"` to restrict logits,
probabilities, and masks to that single channel (`Bx1xHxW`). Label names come
from the bundle `label_schema.json` and unknown names raise a `ValueError`
listing the valid labels.

`from_pretrained(repo_id="VictorButoi/flexray", revision=None, device=None,
*, subfolder=None, ensemble=False)` loads one bundle, a directory holding:

- `model.safetensors` - model weights.
- `config.yml`, `config.yaml`, or `config.json` - model architecture config.
- `label_schema.json` - ordered output labels.
- `preprocessing.json` - input preprocessing contract.

Published release bundles also include `checksums.json`; the runtime does not
require or verify it.

`device=None` (the default) places the models on CUDA when it is available,
like `flexify --device auto`; `CUDA_VISIBLE_DEVICES` picks which GPUs that may
use and `CUDA_VISIBLE_DEVICES=""` forces CPU.

### Repository layout and ensembles

A repository is either one bundle at its root, or several bundles in
subfolders described by a root `ensemble.json`:

```json
{
  "flagship": "members/flux0375",
  "aggregation": "mean_probability",
  "members": [
    {"subfolder": "members/flux0375", "fluxray_proportion": 0.375},
    {"subfolder": "members/flux000", "fluxray_proportion": 0.0}
  ]
}
```

With no `subfolder`, a repository that ships `ensemble.json` loads its
`flagship`; `ensemble=True` loads every listed member; `subfolder="members/…"`
loads exactly that bundle and skips the manifest. A repository without the
file is loaded from its root, and `ensemble=True` on it raises
`FileNotFoundError`. `ENSEMBLE_MEMBER_SUBFOLDERS` mirrors the published
repository's manifest for offline consumers (CLI help, the MCP registry);
`FLAGSHIP_SUBFOLDER` is its first entry.

`repo_id` also accepts a sequence of repository ids (each resolved as above,
without `subfolder`/`ensemble`) for ensembles that span repositories. Every
member bundle is loaded (the one `revision` applies to all of them) and must
declare the same `label_names`, `preprocessing.json` contract, and
`probability_mode`; a mismatch, an empty sequence, or a repeated member raises
`ValueError`. `predict` runs each member and averages their probabilities (the
logits are the re-logit-ized mean, exactly as for TTA). The loaded modules are
available as `segmenter.models` with their bundle directories in
`segmenter.subfolders`; `segmenter.model` is only defined for single-model
segmenters and raises for ensembles. The published members are listed in
[`MODEL_ZOO.md`](../MODEL_ZOO.md).

Portable bundle configs allow only `fxr.models.UNet` and its documented
scalar/list constructor fields. FleXray does not pass bundle `_class` values to
a generic importer; unknown constructors, unknown arguments, and nested `_class`
or `_fn` directives are rejected before model construction. Runtime-only
`model.compile_cfg` is validated for hidden directives and omitted from the
portable config.

FleXray does not support exporting training checkpoints to inference bundles
or ONNX models. To load an existing bundle from disk, use
`flexify --model-id /path/to/bundle` or
`FleXraySegmenter.from_pretrained("/path/to/bundle")`. The directory must contain
the bundle files listed above. `fxr-train --init-from` also accepts an existing
bundle for training initialization.

Image-file preprocessing honors EXIF orientation and composites transparency
over black, including palette and grayscale-alpha images. Grayscale inputs
retain their bit depth. All input types receive the configured zero-padding to
square, bilinear resize, and `float32` output.

With `scale="zero_one"`, integers up to 16 bits use their source dtype bounds.
Wider integers (`int32`, `uint32`, `int64`, and `uint64`) use the image's own
min/max range, independently for each tensor batch member. Large integer
offsets are removed before floating-point conversion so narrow grayscale
contrast survives. Floating-point intensities are preserved.
NumPy and tensor RGB/RGBA inputs are converted to luminance before pixel
scaling. Alpha is clamped to `[0, 1]`: floating values already use that range,
8/16-bit integers divide by the dtype maximum, and wider integers divide by
the image's largest alpha value (at least 1). This compositing also applies
with `scale="none"`. Pillow color conversion can differ slightly due to its
8-bit luminance and alpha rounding.

Tensor inputs receive the same padding and antialiased bilinear resize as
files and arrays, so a `512x512` tensor produces `256x256` predictions when the
bundle declares the default `image_size`. Use `Bx1xHxW` for grayscale
batches and `1x3xHxW` / `1x4xHxW` for single RGB/RGBA images. A three-dimensional
tensor with leading dimension 3 or 4 is rejected as ambiguous. Channel-last
tensors are accepted when the leading dimension is not 1, 3, or 4; convert
other channel-last inputs explicitly to `1xCxHxW`.

Non-finite grayscale values are replaced with `0` before padding and resizing,
with a `RuntimeWarning`. Constant images are accepted with a warning, and empty
images are rejected. The bundle may declare per-sample `standardize`, `minmax`,
or `percentile_minmax` normalization; legacy scalar mean/std metadata remains
readable. Defaults are `256x256`, square padding, `zero_one` scaling, and no
normalization when metadata is constructed directly.

## Predict CLI

`flexify` is the base-install image-file command. It downloads or reuses the
Hugging Face cache for a FleXray model bundle, runs inference on one image file or
a directory of images, and writes NumPy artifacts. Directory discovery is
non-recursive and uses a deterministic sort over BMP, JPEG, PNG, TIFF, and DICOM
files (`.dcm`/`.dicom`, or extension-less files with the `DICM` header). DICOM
support needs `pip install flexray[dicom]` (pydicom): pixels are rescaled with
`RescaleSlope`/`RescaleIntercept`, `MONOCHROME1` images are inverted, and the
image is min-max scaled to `[0, 1]` over its own range before the usual
padding/resize, since DICOM `BitsStored` rarely fills the container dtype.
Existing artifacts with the same output filename are replaced. `--device`
selects the inference device; the default `auto` uses CUDA when
`torch.cuda.is_available()` and falls back to CPU, and explicit values such as
`cuda:1` or `cpu` are passed to `torch.device` (asking for CUDA on a machine
without it is an error). The first
`--ensemble` run downloads about 1.9 GiB of `model.safetensors` weights plus
small metadata files into the standard Hugging Face cache (controlled by
`HF_HOME`); later CLI processes reuse those files from disk, although each
process still loads all five models into memory. TTA adds forward passes, not
additional weight downloads.

```bash
flexify \
  --model-id VictorButoi/flexray \
  --input /path/to/image-or-directory \
  --output-dir /tmp/fxr-predictions
```

`--model-id` defaults to `VictorButoi/flexray`, so the minimal command is:

```bash
flexify --input ./image.png --output-dir ./predictions
```

`--ensemble` averages every member the repository's `ensemble.json` declares
(the published five-model FleXray ensemble) instead of the flagship alone;
`--subfolder` runs one member. Both apply to a single `--model-id` and exclude
each other. To average bundles from several repositories, repeat `--model-id`
(see [Repository layout and ensembles](#repository-layout-and-ensembles)):

```bash
flexify --ensemble --input ./image.png --output-dir ./predictions
flexify --device cpu --input ./image.png --output-dir ./predictions
flexify --subfolder members/flux000 --input ./image.png --output-dir ./predictions
flexify --model-id VictorButoi/flexray --model-id someone/other-flexray-bundle \
  --input ./image.png --output-dir ./predictions
```

Together with `--tta-samples`, this gives the four quality levels of the
browser demo: Low (`flexify`), Normal (`--tta-samples 8`), High
(`--ensemble`), and X-High (`--ensemble --tta-samples 8`).

Each input image writes three channel-first arrays at the bundle's configured
preprocessing size:

- `*_logits.npy` - raw model logits as `float32`, shape `CxHxW`.
- `*_probabilities.npy` - probabilities as `float32`, shape `CxHxW`.
- `*_masks.npy` - per-channel `uint8` masks, shape `CxHxW`.

One `label_names.json` per output directory names the `C` channels in order;
it holds the selected name under `--binary` and is omitted when the bundle
declares no label schema. It is written only once a prediction succeeds.
Reusing a directory requires the same channel names in the same order. A
different `--binary` selection, different model labels, malformed sidecar,
or existing arrays without label metadata requires a new output directory;
the command rejects these cases before inference or writes.

By default every model label is written. `--binary LABEL` restricts the
artifacts to one label channel, for example `--binary femurs`. Masks use
`--threshold 0.5` by default. `--tta-samples N` enables test-time augmentation
with `N` total forward passes per image (see below). `0` or the default of `1`
is just a normal single forward pass with no TTA; a negative value is rejected
before any model is downloaded.

Unreadable images, including DICOM decoding failures, are reported on stderr
and skipped so the remaining images can be processed. The CLI returns `0` on
complete success, `1` when any images were skipped, and `2` for command, model,
dependency, or output errors. Invalid labels and model execution failures abort
the command.

The Python helper `fxr.inference.cli.predict_from_paths(...)` still returns a
list of artifact-path dictionaries and raises on unreadable input by default.
Pass `on_error(path, exception)` as a callback to report and skip decoding
failures; all preprocessing, model, argument, and output errors still propagate.

The probability conversion mode is not a CLI concern. It is bundle metadata
(`probability_mode` in `preprocessing.json`, `multilabel` for the public
model) applied internally by the segmenter.

## Test-Time Augmentation

`FleXraySegmenter.predict(..., tta_samples=N)`, `InferenceRunner.predict(...,
tta_samples=N)`, and `flexify --tta-samples N` run test-time augmentation
with `N` total forward passes: one un-augmented pass plus `N - 1` randomly
augmented passes. Values `<= 1` disable TTA and reproduce the plain single-pass
output exactly.

The augmentation chain is the released `tta_v3` preset. Each augmented
pass independently samples, in order:

- horizontal flip (`p=0.5`) - the only geometric transform, exactly inverted
  on the logits before merging
- invert `1 - x` (`p=0.5`)
- mutually exclusive CLAHE (clip limit `[1.0, 2.0]`, `p=0.1`) or gamma
  `[0.9, 1.1]` with gain `[0.9, 1.1]` (`p=0.25`); the remaining `0.65`
  probability leaves this tone-transform stage unchanged
- multiplicative contrast `[0.7, 1.3]` (`p=0.25`)
- sharpness `[0.7, 1.3]` (`p=0.5`)
- Gaussian noise (`std=0.01`, `p=0.25`)

Intensity transforms do not move pixels and are never inverted. Every augmented
view is re-normalized with the loaded bundle preprocessing contract before it is
run through the model. The inference surface includes its own dependency-light
CLAHE implementation so the base `flexray` install does not require Kornia.

Per-pass logits are converted to probabilities, averaged, and the mean is
converted back to logits (`log(p) - log1p(-p)` for sigmoid modes, `log(p)` for softmax modes) so
callers always receive a consistent logits/probabilities pair. Sampling draws
from the global torch RNG, so repeated runs differ; `flexify --seed 0` and the
`seed=` argument of `predict`, `InferenceRunner.predict`, and
`predict_with_tta` use a forked, seeded CPU RNG instead. Augmentation preserves
the caller's CPU and CUDA random states; it does not reseed accelerator generators.
TTA expects prepared inputs in `[0, 1]` with 1 or 3 channels. Use a single
plain pass for custom preprocessing metadata that moves values outside this
range.

With an ensemble, every pass (the plain view and each augmented view) is drawn
once and run through every member, so `M` members with `tta_samples=N` cost
`M x N` forward passes and the mean spans all of them.

The functional core is exported as `predict_with_tta(forward_fn, images, *,
mode, tta_samples, seed=None)` for callers who own their model loop; `forward_fn` may be
one callable or a sequence of member callables.

## Model Runner

`InferenceRunner(model, probability_mode="multilabel")` owns one PyTorch model
(or a sequence of ensemble members, exposed as `runner.models`) and its
logits-to-probability conversion mode, and exposes
`predict(images, *, tta_samples=1, seed=None)`. The method accepts only a `torch.Tensor`
with shape `BxCxHxW`. Callers own preprocessing and normalization before
calling it. The runner infers the model device from the first parameter,
falling back to the first buffer (every member must share one device), moves
`images` to that device, temporarily sets the models to eval mode, and runs
the forward passes under `torch.inference_mode()`. The previous training mode
is restored after prediction.

```python
import torch
from fxr.inference import InferenceRunner

model = MySegmentationModel()
images = torch.zeros((2, 1, 256, 256), dtype=torch.float32)
result = InferenceRunner(model).predict(images)
```

The runner does not download models, walk datasets, postprocess predictions,
quantize probabilities, or save files. Use `FleXraySegmenter` or `flexify` for
the built-in pretrained image workflow.

## Logits to Probabilities

`probabilities_from_logits(logits, *, mode)` accepts tensors with batch and
channel dimensions, conventionally `BxCx...`.

- `mode="binary"`, `"sigmoid"`, or `"multilabel"` applies sigmoid to every
  channel.
- `mode="onehot"`, `"multiclass"`, or `"softmax"` applies softmax over channel
  dimension.
- Any known mode with a single output channel uses sigmoid, because one-channel
  softmax would always return one.
- Unknown modes raise `ValueError`.

## Probability Sidecars

`quantized_probabilities(probs)` scales probabilities by 255, rounds, clips to
`[0, 255]`, and returns a CPU `np.uint8` array with the same shape. The helper
does not perform file I/O. Callers decide where or whether to persist the array.

`dequantize_saved_probabilities(probs)` reads one saved sidecar array in `CxHxW`
shape and returns `float32` probabilities in `[0, 1]`.

## Dataset Postprocessing

Dataset-specific mask cleanup is a Python-only helper with no CLI flag.
`apply_inference_postprocessing(...)` accepts in-memory logits and probabilities
with shape `BxCxHxW`, ordered `label_names`, a `dataset_name`, and an optional
prepared ground-truth `label` tensor with shape `BxCxHxW`. The return value is an
`InferencePostprocessResult` containing `logits`, `probabilities`, and metadata.

Registered postprocessors are:

| Dataset | Required labels | Changed labels | Behavior |
| --- | --- | --- | --- |
| `HipRay` | `femurs`, `hips` | `hips` | Removes thresholded hip predictions within four pixels of thresholded femurs. |
| `MendeleyCXR` | `liver`, `lungs` | `lungs` | Zeros lungs where liver probability exceeds the threshold. |
| `ShoulderMonch` | `clavicles`, `humeri` | `clavicles` | Uses GT humeri, when present, to keep the closest clavicle component among the two largest non-tiny components. |

## Runtime Boundary

Public pretrained inference is Hugging Face bundle based. `flexify` and
`FleXraySegmenter.from_pretrained` do not load local bundles or raw training run
directories. Dataloader iteration, experiment callbacks, rendering, and
eval-dataset construction remain outside `fxr.inference`.
