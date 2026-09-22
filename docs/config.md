# `fxr.config`

A small, dependency-free configuration layer for experiments, kept to the
minimum FleXray needs.

## `Config`

A read-only hierarchical view over a nested `dict` with dotted-key access:

```python
from fxr.config import Config

cfg = Config({"train": {"epochs": 10, "opt": {"lr": 3e-4}}})
cfg["train.epochs"]        # 10
cfg["train.opt"]           # Config({"lr": 0.0003})
cfg.get("train.missing", 1)  # 1
"train.opt.lr" in cfg      # True
cfg.to_dict()              # deep copy as a plain dict
```

`Config` is intentionally immutable; build a new one (or use `merge_configs`)
instead of mutating in place. `Config.from_file(path)` loads a YAML document.

## Dynamic construction

Configs describe objects with a `_class` (or `_fn`) dotted import string plus
keyword arguments. `eval_config` walks the config depth-first and instantiates
them:

```python
from fxr.config import eval_config

model = eval_config({"_class": "fxr.models.UNet", "in_channels": 1, "out_channels": 8,
                     "filters": [16, 32]})
```

`absolute_import("torch.optim.AdamW")` resolves a single dotted reference.
The `loss_func` subtree is the exception: `fxr.losses.build_loss_from_config`
builds it and also accepts bare FleXray loss names (`_class: SoftDiceLoss`)
and `_combo_class`, which `eval_config` rejects.

> **Security:** `_class` and `_fn` entries import and execute Python code. Treat
> training, cluster, experiment-group, and persisted run configs as executable
> input: load them only from this project or another source you explicitly trust.

## Identity, merging, validation

- `generate_tuid()` → `(timestamp, nonce)` for stamping run directories.
- `config_digest(config)` → stable, order-independent MD5 hash.
- `merge_configs(base, override)` → deep merge without mutating `base`.
- `check_missing(config)` → raise if any leaf is the placeholder `"?"`.
