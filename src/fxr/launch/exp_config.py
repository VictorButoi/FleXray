"""Exp-config sweep specs for the ``fxr-submit`` launcher.

An exp-config is a small YAML sweep spec with two keys: ``base_cfgs`` (one or
more training base configs to launch) and ``experiment_cfg`` (a ``group`` name
plus overrides applied to every base). Sweep axes are declared explicitly under
``experiment_cfg.sweep``: every list value there is one axis and the launcher
takes the Cartesian product across all axes and all bases, so the job count is
``len(base_cfgs) x product(sweep axes)``. Every value outside that block is a
literal, lists included, so genuine list settings such as ``model.filters`` stay
intact. ``experiment.seed_range`` remains a sweep convenience of its own.

Exp-configs live in a top-level ``exp_configs/`` folder (project work, not
packaged); :func:`load_exp_config` accepts either a path or a bare name resolved
against ``./exp_configs/``.
"""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fxr.config import (
    Config,
    check_missing,
    merge_configs,
    prune_zero_proportion_datasets,
)
from fxr.config.training import _materialize_ct_view_counts

from .sweep import apply_overrides, expand_configs, validate_sweep_axes


def load_exp_config(reference: str) -> tuple[str, list[str], dict[str, Any]]:
    """Load an exp-config sweep spec by path or bare name.

    Args:
        reference : str
            Path to a YAML file, or a bare name resolved against ``./exp_configs/``
            (an optional ``.yml``/``.yaml`` suffix is accepted).

    Returns:
        Tuple ``(group, base_cfgs, experiment_cfg)``.

    Raises:
        ValueError: If the spec lacks a non-empty ``base_cfgs`` list, an
            ``experiment_cfg`` mapping, or an ``experiment_cfg.group`` string.
    """

    path = Path(reference)
    if not path.is_file():
        stem = reference.removesuffix(".yml").removesuffix(".yaml")
        path = Path("exp_configs") / f"{stem}.yml"
    spec = Config.from_file(path).to_dict()

    base_cfgs = spec.get("base_cfgs")
    experiment_cfg = spec.get("experiment_cfg")
    if not isinstance(base_cfgs, list) or not base_cfgs:
        raise ValueError(f"{path}: 'base_cfgs' must be a non-empty list.")
    if not isinstance(experiment_cfg, dict):
        raise ValueError(f"{path}: 'experiment_cfg' must be a mapping.")
    group = experiment_cfg.get("group")
    if not isinstance(group, str) or not group:
        raise ValueError(f"{path}: 'experiment_cfg.group' must be a string.")
    group = _validate_group_name(group)
    return group, base_cfgs, experiment_cfg


def _validate_group_name(group: str) -> str:
    """Require an experiment group to be one safe directory component.

    Args:
        group: Experiment group supplied by an exp-config.

    Returns:
        The validated group name.

    Raises:
        ValueError: If the name is empty, a traversal name, or contains a path
            separator, NUL byte, or control character.
    """

    if group in {"", ".", ".."} or any(
        character in group for character in ("/", "\\", "\x00")
    ):
        raise ValueError("experiment_cfg.group must be one safe directory name.")
    if any(ord(character) < 32 for character in group):
        raise ValueError("experiment_cfg.group cannot contain control characters.")
    return group


def _group_log_root(scratch_root: str, group: str, add_date: bool) -> str:
    """Build the run-group log root ``{scratch_root}/training/{date_}{group}``."""
    folder = _validate_group_name(group)
    if add_date:
        folder = f"{datetime.datetime.now().strftime('%m_%d_%y')}_{group}"
    return str(Path(scratch_root) / "training" / folder)


def build_submission_configs(
    *,
    group: str,
    base_cfgs: list[str],
    experiment_cfg: dict[str, Any],
    cluster_overrides: dict[str, Any],
    scratch_root: str,
    add_date: bool,
    set_overrides: dict[str, Any] | None = None,
    sweep_overrides: dict[str, Any] | None = None,
    validate: bool = True,
) -> list[dict[str, Any]]:
    """Expand an exp-config + cluster into concrete, ready-to-submit run configs.

    For each base config the literal ``experiment_cfg`` values are assigned
    before its ``sweep`` axes are Cartesian-expanded via
    :func:`fxr.launch.expand_configs`. The cluster's generic overrides are then
    deep-merged on top, and ``set_overrides`` last of all so the most explicit
    input wins. ``log.root`` is set to the shared run-group directory unless the
    exp-config or ``--set`` already chose one.

    Args:
        group : str
            Run-group name; names the log-root folder.
        base_cfgs : list of str
            Training base configs (packaged names or paths) to launch.
        experiment_cfg : dict
            Overrides applied to every base. Lists under its ``sweep`` mapping are
            sweep axes; every other value is a literal. ``group`` is ignored here.
        cluster_overrides : dict
            Generic experiment overrides from the cluster config (cluster wins).
        scratch_root : str
            Root the run-group folder is created under.
        add_date : bool
            Whether to prefix the run-group folder with ``MM_DD_YY_``.
        set_overrides : dict or None, default=None
            Dotted-key literal CLI overrides applied after ``cluster_overrides``,
            so they win over both the exp-config and the cluster. Lists remain
            literal values rather than sweep axes.
        sweep_overrides : dict or None, default=None
            Additional dotted-key sweep axes supplied explicitly by the CLI.
        validate : bool, default=True
            Whether to reject configs that still contain ``"?"`` placeholders.

    Returns:
        One resolved config ``dict`` per (base, sweep-cell), in base-then-cell order.
    """

    overrides = {key: value for key, value in experiment_cfg.items() if key != "group"}
    sweep_axes = overrides.pop("sweep", {})
    if not isinstance(sweep_axes, Mapping):
        raise ValueError("experiment_cfg.sweep must be a mapping of sweep axes.")
    flat_literals = Config(overrides).flatten()
    flat_sweep = Config(sweep_axes).flatten()
    validate_sweep_axes(flat_sweep)
    seed_range = flat_literals.pop("experiment.seed_range", None)
    if seed_range is not None:
        validate_sweep_axes({"experiment.seed_range": seed_range})
        flat_sweep.setdefault("experiment.seed_range", seed_range)
    validate_sweep_axes(sweep_overrides or {})
    flat_sweep.update(sweep_overrides or {})
    literal_overrides = dict(set_overrides or {})
    for key in literal_overrides:
        for values in (flat_literals, flat_sweep):
            for overridden in list(values):
                if overridden == key or overridden.startswith(f"{key}."):
                    values.pop(overridden)
    if "experiment.seed" in literal_overrides:
        flat_sweep.pop("experiment.seed_range", None)

    log_root = _group_log_root(scratch_root, group, add_date)
    keep_configured_log_root = any(
        "log.root" in values for values in (
            flat_literals, flat_sweep,
            Config(apply_overrides({}, literal_overrides)).flatten(),
        )
    )

    from .cli import load_training_base

    configs: list[dict[str, Any]] = []
    for base_cfg in base_cfgs:
        base = apply_overrides(load_training_base(base_cfg), flat_literals)
        for cfg in expand_configs(base, flat_sweep):
            cfg = merge_configs(cfg, cluster_overrides)
            cfg = apply_overrides(cfg, literal_overrides)
            if not keep_configured_log_root:
                cfg = merge_configs(cfg, {"log": {"root": log_root}})
            cfg = prune_zero_proportion_datasets(cfg)
            if validate:
                check_missing(cfg)
            cfg = _materialize_ct_view_counts(cfg)
            configs.append(cfg)
    return configs
