# `fxr.launch`

The launch harness turns a training config into a running experiment. It is a
thin driver over `fxr.experiment`: it loads and merges configs, validates them,
and runs one in-process.

`fxr-train` requires a training-capable install:

```bash
python -m pip install "flexray[train]"
# Repository checkout:
uv sync --extra train --extra test
```

The base inference install reports the missing extra without a traceback.

## CLI

The packaged console script is `fxr-train`:

```bash
# Local run on GPU 0, starting from the packaged base config.
fxr-train --base base \
  --set log.root=/path/to/runs \
  --set dataloader.batch_size=8 \
  --set dataloader.num_workers=8

# Local run from an explicit config file on GPU 2.
fxr-train my_config.yml --device cuda --gpu 2

# Build/readiness check: validates placeholders, runtime scalars, active data and
# callback paths, protocol remaps, routed-loss coverage, and DRR profiles, then builds without
# loading data or training. Add --smoke-data to open training and callback data.
fxr-train --base base \
  --set log.root=/tmp/x \
  --set dataloader.batch_size=2 \
  --set dataloader.num_workers=0 \
  --dry-run
```

Arguments:

- `config` / `--base NAME` — choose one to create a new run. `--base` loads the
  packaged `fxr/configs/training/{NAME}.yml`. A config may declare a top-level
  `_base_: NAME_OR_PATH` to inherit from another training config: the parent is
  loaded first and the child is deep-merged on top, so a variant need only
  restate the keys it overrides. The default source mix lives in `base.yml`.
- `--resume RUN_DIR` — continue one existing run from its immutable persisted
  config and `checkpoints/last.pt`; it cannot be combined with new-run overrides.
  A run checkpoint is a Python pickle and must come from a run you trust.
- `--init-from MODEL` — initialize a new run from a complete local or Hugging
  Face bundle. A standalone `.safetensors` file also requires the explicit
  unverified-label option below. `--init-revision` pins the remote revision.
- `--allow-unverified-label-order` — explicit opt-in for a standalone weights
  file whose metadata cannot prove output-label order. It is valid only with
  `--init-from`; prefer a complete pretrained bundle.
- `--init-from-run RUN_DIR` — initialize only model weights from an explicitly
  trusted FleXray checkpoint. Optimizer/scheduler state starts fresh.
- `--init-checkpoint NAME` — checkpoint stem under the trusted run's
  `checkpoints/` directory (default `last`); valid only with
  `--init-from-run`.
- `--set KEY=VALUE` — repeatable literal dotted-key override. `VALUE` is parsed
  as YAML, and lists/maps remain ordinary config values.
- `--sweep KEY=VALUES` — repeatable explicit Cartesian sweep axis. `VALUES`
  must be a non-empty YAML list and local runs reject multiple cells. The
  convenience axis `experiment.seed_range=N` instead takes a positive integer.
- `--device {auto,cpu,cuda}` — launch-only local device policy; it is not stored
  in the immutable training config. `auto` selects a compatible visible CUDA
  card and otherwise uses CPU. `cpu` hides CUDA before Torch-backed modules are
  imported. `cuda` fails before model construction if CUDA is unavailable or
  the installed Torch wheel does not support the card.
- `--gpu INT` — index into the GPUs already visible to the process (default `0`).
  With `CUDA_VISIBLE_DEVICES=4,7`, `--gpu 1` selects GPU 7. The inherited list is
  narrowed before training; device discovery runs in a separate process so a
  CUDA-runtime fallback cannot freeze the training process's device list. Invalid
  indices fail before training. Only `auto` with the default index falls back to
  CPU when no GPU is visible. FleXray runs one process on at most one GPU.
- `--dry-run` — run readiness validation and build the experiment without data
  or training. No run directory is created and no job is submitted. Readiness
  also compiles each path-configured package's label mapping, through its
  `dataset_spec` when one is configured.
- `--smoke-data` — with readiness validation, open one train sample per active
  source and one sample from each callback-local eval set at its configured
  split. CT samples are checked for render-required label, affine, and centroid
  fields.
- `--submitit` — submit resolved configs instead of running in-process. Use
  `--local` for submitit's local executor; `--partition`, `--cpus`, `--timeout`,
  and `--gpus 1` configure a Slurm submission. Multi-GPU processes are rejected.

## Cluster Groups With `fxr-submit`

`fxr-submit` combines a packaged or custom cluster config with an experiment
group spec. A cluster config owns machine-specific dataset roots, `scratch_root`,
submitit/Slurm settings, and requeue policy. An experiment spec contains
`base_cfgs` plus an `experiment_cfg.group`; list values under
`experiment_cfg.sweep` are Cartesian sweep axes, and every value outside that
block stays a literal.

CLI `--sweep` values replace YAML axes; cluster overrides apply next, and
`--set` wins last. Empty axes are rejected before submission.

```bash
fxr-submit --cluster example_slurm \
  --exp-config flexray_default \
  --dry-run --smoke-data
```

The packaged `example_slurm` config is an annotated template with placeholder
paths, partition, and account; copy it, fill it in for your machine, and pass it
with `--cluster-cfg PATH` instead of `--cluster NAME`.
`--set` remains literal, while `--sweep` adds explicit list-valued axes. A dry
run resolves every cell, validates data and callbacks, and performs a build-only
experiment construction without submitting jobs. Normal submission writes the
run group under the cluster `scratch_root` and uses stable run IDs for submitit
requeues. The group must be one safe directory component: `.`, `..`, path
separators, NUL, and other control characters are rejected.

Both launchers record effective CT `num_views` after applying batch-size and
profile overrides. Existing configs that omit the setting keep their previous
training batch-size fallback, including on resume. See
[training view counts](camera.md#training-view-counts) for explicit configuration.

## Programmatic API

```python
from fxr.launch import resume_run, run_config

new_run = run_config(config)
resumed_run = resume_run("/runs/20260724_120000-ABCD-digest")
```

When `run_config(..., run_id=...)` is used by scheduler integrations, the id must
contain exactly three nonempty hyphen-separated metadata parts. Generated IDs use
`YYYYmmdd_HHMMSS-NONCE-32hex`, where `NONCE` is uppercase alphanumeric.
Human-readable three-part IDs remain valid. Path separators, traversal names,
and control characters are rejected before `log.root` is joined.

`run_config` reads `experiment._class` from the config, calls its `from_config`
(which stamps a `log.root/{uuid}` run directory and writes `config.yml`), and
runs it. It never silently adopts an existing directory. `resume_run` is the
explicit interactive continuation API used by `--resume`. Scheduler integrations
use an explicit stable `run_id` and `resume_existing=True` only for requeues.
The submitted config is normalized before its ID is computed; a requeue must
match the exact persisted config digest or it is rejected.
