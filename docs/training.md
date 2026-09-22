# Training Your Own Model

Training is the advanced path for building or adapting a FleXray model.

## Install The Training Stack

Install the published training stack when you need datasets, DRR rendering,
WandB logging, or local training:

```bash
python -m pip install "flexray[train]"
```

For a repository checkout, create the repository-local environment with test
tools instead:

```bash
uv sync --extra train --extra test
```

`uv sync` installs every console script into the checkout environment. Run
`.venv/bin/flexify`, `.venv/bin/fxr-dataset`,
`.venv/bin/fxr-render`, `.venv/bin/fxr-protocol`, `.venv/bin/fxr-train`, and
`.venv/bin/fxr-submit` directly; the examples below omit the `.venv/bin/`
prefix for readability.

The core install provides inference dependencies. The `[train]` extra adds
dataset storage (`thunderpack`), logging (`wandb`), and the augmentation, DRR,
and launch stack, including `kornia`, `nanodrr`, and `submitit`.
Importing `fxr.experiment` requires the training stack. WandB is required for
training and remains the sole experiment-logging backend. The base config sets
a project and mode but does
not hardcode a WandB entity; use your WandB login/environment or an explicit
config override. Installing also puts `flexify` (image-file inference),
`fxr-dataset` (dataset scaffolding and
packaging), `fxr-render` (offline DRR rendering), `fxr-protocol` (label
harmonizer inspection), `fxr-train` (local training), and `fxr-submit`
(cluster submission) on your path. For a complete small example, see
[`examples/custom_dataset`](../examples/custom_dataset/README.md).

## Read The Training Configs

You only need three ideas to read the training configs:

**Protocol (label space).** A *protocol* is a fixed, ordered list of anatomical
channels -- `background` is always channel 0 -- that every dataset native label
space is remapped into. This lets heterogeneous datasets, which each store labels
differently, train one shared model head. The bundled protocol is
`all_structures_flexray_v4`, selected in a config via `protocol.name`. See
[`protocols.md`](protocols.md).

**Two data types.** A config `data` block selects `Xray` sources (real or
generated 2D images, including FluXray) and `CT` volumes rendered to DRR each
step. Dense maps and named channel masks are both valid X-ray labels.
`dataloader.proportions` sets how sources are mixed into each epoch. With a
fixed `dataloader.iters_per_epoch`, source counts and ordering are deterministic
per epoch and resume restores the matching loader epoch.

**Configs are packaged data.** Training configs live in `fxr/configs/training/`.
`base.yml` is the full multimodal config. A config can inherit another with a top-level `_base_:` key
and restate only the keys it overrides. See [`config.md`](config.md) and
[`launch.md`](launch.md).

## 1. Point At Your Data

Built-in datasets normally resolve below two root environment variables:

```bash
export CT_DATAPATH=/path/to/ct_datasets
export XRAY_DATAPATH=/path/to/xray_datasets
```

The packaged FluXray layout reads `GENERATED_DATAPATH` for storage
compatibility. FluXray is nevertheless an `Xray` source,
not a third runtime or package data type.

Set `train.eval_freq: 0` to disable the core validation phase. This permits
train-only source packages unless an enabled `WandbSamplePredictionLogger` also
uses source validation data. `EvalSetMetricLogger` owns its separate data and
requires only its configured callback split.

## 2. Dry-Run First

A dry-run validates config placeholders, runtime scalars, training and callback
dataset paths, protocol remaps, routed losses, and DRR profiles, then builds the experiment **without training**. Add
`--smoke-data` to also open one real sample per active training source and
callback-local eval set:

```bash
fxr-train --base base \
  --set log.root=/tmp/flexray-runs \
  --set dataloader.batch_size=2 \
  --set dataloader.num_workers=0 \
  --dry-run --smoke-data
```

Configs ship with `?` placeholders (`dataloader.batch_size`,
`dataloader.num_workers`, `log.root`) that you must fill in at launch with
`--set`. `--base NAME` loads `fxr/configs/training/NAME.yml`. Pass a path instead
to use your own file.

The packaged `base` recipe reproduces the training run behind the released
FleXray weights (2026-08-22): a seven-level residual U-Net with
three convolutions per block, per-modality `CT_base`/`Xray_base` augmentation,
tempered inverse-label-frequency sampling of crop-backed CT sources,
dataset-owned negative supervision (`supervise_empty_labels`), randomized
attenuation with probability `0.5`, 256 px real X-ray packages, and
twelve callback-local eval sets scored on labels covering at least 0.1% of the
image. `XRAY_DATAPATH` must point at the directory holding those packages.

For a package created by `fxr-dataset`, add its absolute path and make the
proportions mapping an allowlist for that source. This example uses the
partial-label loss, which is appropriate for named multichannel masks:

```bash
fxr-train --base base \
  --set data.Xray.MyDataset.path=/data/flexray/MyDataset \
  --set 'dataloader.proportions={MyDataset: 1}' \
  --set 'loss_func.dataset_losses={MyDataset: partial_labeled_seg}' \
  --set 'callbacks={}' \
  --set log.root=/tmp/flexray-runs \
  --set dataloader.batch_size=4 \
  --set dataloader.num_workers=0 \
  --dry-run --smoke-data
```

Use `data.CT.MyDataset.path` for a `ct-seg` package and add a matching
`drr_model.datasets.MyDataset` profile. Fully labeled dense maps may route to
`standard_seg`; partially labeled data should route to
`partial_labeled_seg`. Dense packages whose native names differ from the
protocol add `data.Xray.MyDataset.dataset_spec=/abs/MyDataset.yml` (see
[`protocols.md`](protocols.md)).

## 3. Train Locally

Omit `--init-from`, `--init-from-run`, and `--resume` to initialize the model
from scratch. Drop `--dry-run` to train. Use `--device cuda --gpu N` when CUDA
is required, `--device cpu` to hide/avoid CUDA, or the default `--device auto`
to select a compatible visible card and otherwise use CPU:

```bash
fxr-train --base base \
  --set log.root=/tmp/flexray-runs \
  --set dataloader.batch_size=8 \
  --set dataloader.num_workers=8 \
  --device cuda \
  --gpu 0
```

On Linux, repository `uv sync` environments resolve Torch and torchvision from
PyTorch's official CUDA 12.6 index. This preserves Volta/V100 support; CUDA 13
wheels do not support those cards. The device choice is launch-local and
is intentionally excluded from the immutable run/config identity.

Each run is written to `{log.root}/{run_id}/`, containing the resolved
`config.yml` and `checkpoints/last.pt`. Continuing a run is always an explicit
operation:

```bash
fxr-train --resume /tmp/flexray-runs/RUN_ID
```

Resume restores the raw/EMA model states, optimizer, scheduler, scaler, epoch,
global step, and random streams from the committed `last.pt` checkpoint. New
run metadata and checkpoints bind `run_id` and `config_digest`; resume rejects
cross-run or modified-config checkpoints while accepting legacy checkpoints
that predate these fields. Run checkpoints use Python pickle deserialization, so
resume only a run directory you created or explicitly trust.

## 4. Fine-Tune Existing Weights

Use `--init-from` to start a new run from a local pretrained bundle. A bare
`.safetensors` file has no label metadata and is rejected by default; use
`--allow-unverified-label-order` only when you have independently verified the
weights' ordered output-label contract:

```bash
fxr-train --base base \
  --init-from /models/flexray-bundle \
  --set log.root=/tmp/flexray-runs \
  --set dataloader.batch_size=8 \
  --set dataloader.num_workers=8
```

For a Hugging Face bundle, replace the path with its repository ID and use
`--init-revision COMMIT_OR_TAG` when you want a pinned revision.

A trusted local FleXray run can supply one named checkpoint instead:

```bash
fxr-train --base base \
  --init-from-run /tmp/flexray-runs/SOURCE_RUN \
  --init-checkpoint epoch_0050 \
  --set log.root=/tmp/flexray-runs \
  --set dataloader.batch_size=8 \
  --set dataloader.num_workers=8
```

Both forms load model weights only. Local and Hugging Face bundles must include
a model config and `label_schema.json`; their architecture and exact ordered
labels are checked before strict state loading. Missing metadata is an error, not
a shape-only fallback. Trusted-run initialization likewise requires enough saved
protocol metadata to verify a named target label order.

A standalone `.safetensors` file cannot prove architecture provenance or label
order and is rejected by default. It is available only as an explicit escape
hatch:

```bash
fxr-train --base base \
  --init-from /models/legacy-weights.safetensors \
  --allow-unverified-label-order \
  --set log.root=/tmp/flexray-runs \
  --set dataloader.batch_size=8 \
  --set dataloader.num_workers=8
```

That opt-in is persisted in the new-run config and provenance records
`label_order_verified: false`; it is invalid for bundles. Every initialization
starts with fresh optimizer, scheduler, scaler, epoch, and random state.

### Fine-tune to a new label protocol

To adapt the released FleXray model to your own protocol (a different set or order of
output channels), keep its backbone and replace the output head with
`--replace-head`. The source's ordered labels are then not required to match;
only the backbone architecture is checked. `--freeze-backbone` additionally
trains only the new head:

```bash
fxr-train my_protocol_config.yml \
  --init-from VictorButoi/flexray \
  --replace-head --freeze-backbone \
  --set log.root=/tmp/flexray-runs
```

Both flags persist as `initialization.replace_head` / `initialization.freeze_backbone`
and are recorded in the run provenance together with the source's channel count.

## Training Runtime

The packaged base enables per-sample percentile normalization and an
exponential moving average (EMA):

```yaml
train:
  normalization:
    scheme: percentile_minmax
    percentiles: [0.5, 99.5]
    eps: 1.0e-8
  ema:
    enabled: true
    decay: 0.9999
    start_after_steps: 0
    update_every: 1
```

EMA updates after optimizer steps. Validation and epoch/wrapup callbacks use
the averaged weights once initialized. Each checkpoint stores the EMA-selected
state under `model` and also stores `model_raw` plus EMA metadata so
`--resume` can continue from the optimizer-owned weights without losing the
average.

`torch.compile` is optional and disabled by default:

```yaml
model:
  compile_cfg:
    enabled: false
    # Optional torch.compile arguments:
    # backend: inductor
    # mode: default
    # fullgraph: false
    # dynamic: false
    # options: {}
```

When enabled, FleXray compiles only the forward wrapper; the raw model remains
the optimizer, EMA, and checkpoint owner, so saved state dictionaries keep
portable model keys. Compiled training is single-process; distributed and multi-GPU training are not
supported.

## Developer And Docs References

Each module has a focused reference in this directory.

| Module | Purpose | Reference |
| --- | --- | --- |
| `fxr.protocols` | Label-space contracts and dataset/eval label remaps | [protocols.md](protocols.md) |
| `fxr.datasets` | Training datasets, storage backends, source mixing | [datasets.md](datasets.md) |
| `fxr.experiment` | Config-driven training lifecycle (the train loop) | [experiment.md](experiment.md) |
| `fxr.launch` | Training launch harness (`fxr-train`) | [launch.md](launch.md) |
| `fxr.models` | Training architectures (`UNet`) and NN blocks | -- |
| `fxr.models.camera` | CT->DRR rendering runtime used during training | [camera.md](camera.md) |
| `fxr.drr` | Low-level nanoDRR rendering primitives | [drr.md](drr.md) |
| `fxr.augmentation` | Train-only 2D image/label augmentation presets | [augmentation.md](augmentation.md) |
| `fxr.losses` | Segmentation losses and dataset-routed loss containers | [losses.md](losses.md) |
| `fxr.metrics` | Tensor segmentation metrics (Dice, HD95) | [metrics.md](metrics.md) |
| `fxr.callbacks` | Framework-neutral training-callback dispatch | [callbacks.md](callbacks.md) |
| `fxr.inference` | HF pretrained loading, prediction CLI, tensor runner | [inference.md](inference.md) |
| `fxr.config` | Hierarchical configs and `_class`/`_fn` instantiation | [config.md](config.md) |

Run the test suite from the repository root:

```bash
.venv/bin/python -m pytest -q
```
