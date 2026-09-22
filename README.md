<p align="center">
  <img
    src="https://flexray.csail.mit.edu/assets/hero-mosaic/light/hero-mosaic-v3-desktop-prediction.webp"
    alt="Mosaic of full-body radiographs with FleXray anatomy segmentation overlays"
    width="100%"
  >
</p>

# FleXray: Universal Clinical X-ray Segmentation 💪

### [Project Page](https://flexray.csail.mit.edu/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jMBoOyV8PkRThHi3i6QIMjolmNoRE0cD)
[![PyPI version](https://img.shields.io/pypi/v/flexray.svg)](https://pypi.org/project/flexray/)
[![Total PyPI downloads](https://static.pepy.tech/personalized-badge/flexray?period=total&units=none&left_color=grey&right_color=blue&left_text=downloads)](https://pepy.tech/project/flexray)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/VictorButoi/FleXray/blob/main/LICENSE)

[Victor Ion Butoi](https://victorbutoi.github.io/),
[Vivek Gopalakrishnan](https://vivekg.dev/),
[John V. Guttag](https://people.csail.mit.edu/guttag/),
[Adrian V. Dalca](https://www.mit.edu/~adalca/),
[Neel Dey](https://www.neeldey.com)

FleXray (pip package `flexray`, imported as `fxr`) is a command-line tool and
Python package for segmenting
anatomy in X-ray images. The quickest path is to install the package, run
`flexify` on an image or folder of images, and inspect the saved NumPy
arrays.

FleXray can also train new full-body X-ray segmentation models. Training
combines **2D X-ray segmentation data** (including generated/FluXray samples)
with **CT segmentation volumes rendered to DRRs** during training.

## What FleXray Does

FleXray is a segmentation model that outputs anatomical segmentations for
arbitrary X-rays in common image formats such as PNG, JPEG, TIFF, and BMP, plus
DICOM (`pip install flexray[dicom]`). The
pretrained weights live on our Hugging Face page,
[`VictorButoi/flexray`](https://huggingface.co/VictorButoi/flexray).

For every input image, FleXray writes three `.npy` arrays:

- Thresholded segmentation masks for each anatomical label.
- Per-channel probabilities that show model confidence.
- Raw logits for developers who need the unprocessed model scores.

It also writes one `label_names.json` naming those channels in order.

## Install FleXray

FleXray targets Python 3.10+. The PyPI package name is `flexray` (the import
package is `fxr`) and the default install is inference-focused:

```bash
python -m pip install flexray
```

## Predict From The Command Line

From the terminal, run prediction at one of four quality levels. These are the
same Low / Normal / High / X-High modes as the
[browser demo](https://flexray.csail.mit.edu/#demo):

```bash
# Low: flagship model, one forward pass
flexify --input ./image.png --output-dir ./predictions

# Normal: flagship model with 8-pass test-time augmentation
flexify --tta-samples 8 --input ./image.png --output-dir ./predictions

# High: five-model FleXray ensemble, one pass per model
flexify --ensemble --input ./image.png --output-dir ./predictions

# X-High: five-model ensemble with 8-pass test-time augmentation
flexify --ensemble --tta-samples 8 --input ./image.png --output-dir ./predictions
```

`--input` also accepts a directory, in which case every supported image in it
is predicted. `--model-id` defaults to `VictorButoi/flexray`, one Hugging
Face repository that holds the flagship and its four ensemble members under
`members/` (listed in its `ensemble.json`). Pass `--subfolder` to run one
member on its own, `--revision` to pin a repository commit, or repeat
`--model-id` to average bundles from several repositories:

```bash
flexify --subfolder members/flux000 --input ./image.png --output-dir ./predictions
```

By default, every model label is written. Pass `--binary LABEL` to restrict the
output to one anatomical label, for example `--binary femurs`. Masks use
`--threshold 0.5` by default. For more documentation around command-line usage,
see [`docs/inference.md`](https://github.com/VictorButoi/FleXray/blob/main/docs/inference.md).

## Understand The Output Files

For an input named `image.png`, `flexify` writes:

- `image_masks.npy`: thresholded segmentation masks. These are `uint8` arrays
  where each channel is a predicted anatomical mask.
- `image_probabilities.npy`: model confidence per channel. These are `float32`
  arrays with values after the selected probability conversion mode.
- `image_logits.npy`: raw model scores before probability conversion. These are
  mostly useful for developers and debugging.

All three arrays are channel-first with shape `CxHxW`, where `C` is the number
of model output labels and `H`/`W` come from the bundle's preprocessing size.
With `--binary LABEL`, `C` is 1. The CLI writes `.npy` arrays and does not write
PNG overlays.

## Optional Python API

We also support Python. The API runs the same Hugging Face-backed workflow,
with the same quality levels, and returns logits, probabilities, and masks in
memory:

```python
from fxr.inference import FleXraySegmenter

segmenter = FleXraySegmenter.from_pretrained()                   # flagship model
segmenter = FleXraySegmenter.from_pretrained(ensemble=True)      # five-model ensemble
prediction = segmenter.predict("./image.png", tta_samples=16)    # tta_samples=1: one pass

logits = prediction.logits
probabilities = prediction.probabilities
masks = prediction.masks
```

For more details on the inference code, see
[`docs/inference.md`](https://github.com/VictorButoi/FleXray/blob/main/docs/inference.md).

## Use FleXray From AI Clients (MCP)

FleXray ships an MCP server, so AI clients such as Claude Code and Claude
Desktop can segment X-rays and inspect label protocols through natural
language:

```bash
python -m pip install "flexray[mcp]"
claude mcp add flexray -- fxr-mcp
```

The server exposes `segment_image` (with per-label statistics), model listing
and description, and protocol/dataset introspection tools over local stdio.
See [`docs/mcp.md`](https://github.com/VictorButoi/FleXray/blob/main/docs/mcp.md)
for the full tool reference and client configuration.

## Training Your Own Model

The training release has three parts: the **model CLI**, the **data engine
CLI**, and the **label harmonizer and config system**. Install them with:

```bash
python -m pip install "flexray[train]"
```

For repository development on Linux, `uv sync --extra train --extra test` uses
PyTorch's official CUDA 12.6 index so Volta/V100 cards remain supported. The
packaged `base` recipe reproduces the training run behind the
released FleXray weights; see
[`docs/training.md`](https://github.com/VictorButoi/FleXray/blob/main/docs/training.md).

### Model CLI

- `flexify` segments images with a published bundle (above).
- `fxr-train` validates and runs training; `fxr-submit` submits config-driven
  sweeps to a cluster. Local training accepts `--device {auto,cpu,cuda}` and
  `--gpu N`.

Initialize from the released model with `--init-from VictorButoi/flexray`;
add `--replace-head` (and optionally `--freeze-backbone`) to fine-tune onto a
different label protocol. `--init-from-run RUN_DIR` seeds a new run from a
trusted local run and `--resume RUN_DIR` continues an interrupted run with its
optimizer, scheduler, EMA, and random state. Training is single-process with at
most one GPU.

### Data engine CLI

`fxr-dataset` turns images/masks or CT volumes into the ThunderDB packages
training reads, and `fxr-render` renders DRR training samples from packed CT:

```bash
fxr-dataset scaffold xray-seg --images ./images --masks ./masks --name MyXrays --output dataset.yml
fxr-dataset validate dataset.yml
fxr-dataset pack dataset.yml /data/MyXrays        # optional preprocessing/crops blocks
fxr-dataset check /data/MyXrays                   # or `inspect` for any ThunderDB
fxr-render /data/MyCT --dataset-name MyCT --profile MOOSE --output ./drr
```

CT manifests accept NIfTI volumes, an HU window, and the `512x512x256` axial
crop layout the released model trains on; X-ray manifests can reproduce its
pad-to-square/area-resize preprocessing. See
[`docs/datasets.md`](https://github.com/VictorButoi/FleXray/blob/main/docs/datasets.md)
and [`docs/camera.md`](https://github.com/VictorButoi/FleXray/blob/main/docs/camera.md).

### Label harmonizer and config system

A dataset spec maps a dataset's native mask ids into the FleXray protocol
(merging finer labels, dropping structures the protocol lacks); `fxr-protocol`
inspects protocols and compiles those mappings, and `data.<modality>.<name>.dataset_spec`
wires a spec into a training config:

```bash
fxr-protocol list
fxr-protocol show all_structures_flexray_v4
fxr-protocol compile --dataset CustomHips.yml
```

The runnable walkthrough in
[`examples/custom_dataset`](https://github.com/VictorButoi/FleXray/blob/main/examples/custom_dataset/README.md)
builds a tiny synthetic dataset, harmonizes its labels, packs it, and dry-runs
training on CPU in seconds:

```bash
cd examples/custom_dataset
python make_synthetic_dataset.py --output work/data
fxr-protocol compile --dataset CustomHips.yml
fxr-dataset validate work/data/dataset.yml
fxr-dataset pack work/data/dataset.yml "$PWD/work/packed/CustomHips"
fxr-train train_custom.yml \
  --set data.Xray.CustomHips.path="$PWD/work/packed/CustomHips" \
  --set data.Xray.CustomHips.dataset_spec="$PWD/CustomHips.yml" \
  --set log.root="$PWD/work/runs" --device cpu --dry-run --smoke-data
```

Training configs are packaged YAML (`fxr/configs/training/base.yml`) that
inherit with `_base_:` and take `--set KEY=VALUE` overrides; see
[`docs/protocols.md`](https://github.com/VictorButoi/FleXray/blob/main/docs/protocols.md)
and [`docs/config.md`](https://github.com/VictorButoi/FleXray/blob/main/docs/config.md).

## Citation

If you find FleXray or any of its materials useful, please cite the software. See
[`CITATION.cff`](https://github.com/VictorButoi/FleXray/blob/main/CITATION.cff) for citation metadata.

```bibtex
@software{butoi2026flexray,
  title = {FleXray: Universal Clinical X-ray Segmentation},
  author = {Victor Ion Butoi and Vivek Gopalakrishnan and John V. Guttag and Adrian V. Dalca and Neel Dey},
  year = {2026},
  url = {https://github.com/VictorButoi/FleXray},
  license = {MIT}
}
```

## Licenses

* Code is released under the [MIT License](https://github.com/VictorButoi/FleXray/blob/main/LICENSE).
* Public model weights are released under `CC-BY-NC-4.0` unless a model card says otherwise.
