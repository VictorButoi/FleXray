# exp_configs

Experiment-group sweep specs for `fxr-submit`. These are project work — they are
**not** packaged in the wheel (unlike the cluster configs under
`src/fxr/configs/cluster/`).

Each spec has two top-level keys:

- `base_cfgs`: one or more training base configs to launch. Each entry is either
  the stem of a packaged config under `fxr/configs/training/` (e.g. `base`) or a
  path to a YAML file.
- `experiment_cfg`: a required `group` name (which names the run-group folder)
  plus overrides applied to every base. **Sweep axes are declared under a
  `sweep:` mapping** — every list value there is one axis and the launcher takes
  the Cartesian product across all axes and all bases, so the job count is
  `len(base_cfgs) × product(sweep axes)`. Everything outside `sweep:` is a
  literal, lists included, so a genuine list setting such as `model.filters`
  survives intact. `experiment.seed_range` remains a sweep convenience of its
  own. The `group` value must be one safe directory component; `.`, `..`,
  separators, NUL, and other control characters are rejected before joining the
  cluster scratch root.

```yaml
base_cfgs: [base]
experiment_cfg:
  group: lr_sweep
  model:
    filters: [32, 64, 128, 256]   # a literal list: one job, not four
  sweep:
    optim.lr: [1.0e-3, 1.0e-4]    # 2 cells
    train.epochs: [500, 1000]     # x 2 = 4 jobs
```

Submit a group with:

```bash
fxr-submit --cluster <name> --exp-config <name_or_path>      # submit
fxr-submit --cluster <name> --exp-config <name_or_path> --dry-run  # build + validate only
```

A bare `--exp-config <name>` resolves to `./exp_configs/<name>.yml`.

Precedence per job (lowest to highest): `base_cfg` < literal `experiment_cfg`
values < YAML `sweep` axes < CLI `--sweep` axes < **cluster config overrides**
< `--set`. A cluster's `dataloader.batch_size` still wins over an experiment
sweep of that key. A final `--set` replaces the value and removes the conflicting
axis, so it does not create duplicate jobs. Setting an entire mapping removes
axes beneath that mapping. `--set experiment.seed=N` also removes a seed-range
axis. Every explicit axis must be a non-empty list, except `experiment.seed_range`,
which takes a positive integer. The CLI range overrides the YAML range and starts
from the experiment's literal seed (or the base seed when omitted).

`log.root` is derived as `{scratch_root}/training/{MM_DD_YY_}{group}` unless the
experiment spec or `--set` supplies it, including a `log.root` sweep. Submission
logs live under the first config's `log.root/submitit`; each run uses its own root.

Set `drr_model.default.num_views` to choose CT views per volume independently
of the image batch size; see [training view counts](../docs/camera.md#training-view-counts).
