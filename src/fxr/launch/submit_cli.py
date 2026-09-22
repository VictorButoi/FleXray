"""``fxr-submit`` command line entry point.

Submits a whole experiment group to a cluster the way the previous internal
training codebase did: pick a named cluster config, point at an exp-config
sweep spec, and the launcher expands every (base x sweep-cell) into a run config,
derives a shared ``log.root`` under the cluster ``scratch_root``, and submits each
through ``submitit`` (resuming from ``last`` on preemption).

Cluster ``slurm_args`` and requeue policy are honored on Slurm; the cluster's
clean dataset roots are exported inside each job. ``--dry-run`` builds and
validates the configs and prints a summary without submitting.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fxr.launch._extras import require_training_extra
from fxr.launch.cli import (
    _parse_overrides,
    _parse_sweep_overrides,
    _validate_build_only,
)
from fxr.launch.cluster import (
    cluster_submit_params,
    load_cluster_config,
    split_cluster_config,
)
from fxr.launch.exp_config import build_submission_configs, load_exp_config
from fxr.launch.submit import submit_configs


def _build_parser() -> argparse.ArgumentParser:
    """Build the ``fxr-submit`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="fxr-submit", description="Submit a FleXray experiment group to a cluster."
    )
    cluster_group = parser.add_mutually_exclusive_group(required=True)
    cluster_group.add_argument(
        "--cluster", help="Named cluster config under fxr/configs/cluster/."
    )
    cluster_group.add_argument(
        "--cluster-cfg", help="Path to a custom cluster YAML file."
    )
    parser.add_argument(
        "--exp-config",
        required=True,
        dest="exp_config",
        help="Exp-config sweep spec: a path or a name resolved against ./exp_configs/.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="overrides",
        help=(
            "Dotted-key literal override applied after the cluster config, so it "
            "wins over both it and the exp-config; VALUE is YAML (repeatable)."
        ),
    )
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        metavar="KEY=VALUES",
        dest="sweep_overrides",
        help="Additional explicit sweep axis; VALUES must be a non-empty YAML list.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Force submitit's local executor regardless of the cluster mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate configs and print a summary without submitting.",
    )
    parser.add_argument(
        "--smoke-data",
        action="store_true",
        help="During dry-run validation, open one sample per active dataset.",
    )
    return parser


def _summary(cluster_name: str, group: str, configs: list[dict[str, Any]]) -> str:
    """Format a one-glance summary of what will be submitted."""
    lines = [
        f"Cluster: {cluster_name}",
        f"Experiment group: {group}",
        f"Generated configs: {len(configs)}",
        f"Log root: {configs[0]['log']['root']}",
    ]
    return "\n".join(lines)


def _run(args: argparse.Namespace) -> int:
    """Parse arguments, expand the sweep, and submit (or dry-run) the group.

    Args:
        args: Parsed command-line namespace.

    Returns:
        Process exit code (``0`` on success).
    """

    if args.smoke_data and not args.dry_run:
        raise ValueError("--smoke-data requires --dry-run.")

    require_training_extra("submitit")

    cluster_cfg = load_cluster_config(name=args.cluster, path=args.cluster_cfg)
    submit_cfg, cluster_overrides = split_cluster_config(cluster_cfg)
    params = cluster_submit_params(submit_cfg)

    group, base_cfgs, experiment_cfg = load_exp_config(args.exp_config)
    set_overrides = _parse_overrides(args.overrides)
    sweep_overrides = _parse_sweep_overrides(args.sweep_overrides)

    configs = build_submission_configs(
        group=group,
        base_cfgs=base_cfgs,
        experiment_cfg=experiment_cfg,
        cluster_overrides=cluster_overrides,
        scratch_root=params.scratch_root,
        add_date=params.add_date,
        set_overrides=set_overrides,
        sweep_overrides=sweep_overrides,
    )

    cluster_name = args.cluster if args.cluster is not None else args.cluster_cfg
    print(_summary(cluster_name, group, configs))
    if args.dry_run:
        from fxr.launch.readiness import validate_training_config

        for config in configs:
            validate_training_config(
                config,
                smoke_data=args.smoke_data,
                ct_data_root=params.ct_data_path,
                xray_data_root=params.xray_data_path,
                generated_data_root=params.generated_data_path,
            )
            _validate_build_only(config)
        return 0

    folder = str(Path(configs[0]["log"]["root"]) / "submitit")
    submit_configs(
        configs,
        folder=folder,
        slurm_kwargs=params.slurm_kwargs,
        local=args.local or params.mode == "local",
        slurm_max_num_timeout=params.slurm_max_num_timeout,
        ct_data_path=params.ct_data_path,
        xray_data_path=params.xray_data_path,
        generated_data_path=params.generated_data_path,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the production submission command.

    Args:
        argv: Argument vector, or ``None`` to use ``sys.argv[1:]``.

    Returns:
        Process exit code (``0`` on success).
    """

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
