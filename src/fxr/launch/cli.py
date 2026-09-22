"""``fxr-train`` command line entry point.

Loads a training config (a YAML file or a packaged ``--base`` name), applies
dotted-key ``--set`` overrides, validates it, and either runs it in-process on
one selected device or submits a sweep to the cluster with ``submitit``.

Local device selection is launch-only. CPU hides CUDA, while auto/CUDA expose
one requested card through ``CUDA_VISIBLE_DEVICES`` before any Torch-backed
import. For that to hold, the runner is imported lazily inside :func:`main`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from fxr.config import (
    Config,
    check_missing,
    merge_configs,
    prune_zero_proportion_datasets,
)
from fxr.config.training import _materialize_ct_view_counts
from fxr.launch._extras import require_training_extra
from fxr.launch.sweep import apply_overrides, expand_configs, validate_sweep_axes


def _build_parser() -> argparse.ArgumentParser:
    """Build the ``fxr-train`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="fxr-train", description="Launch a FleXray training run."
    )
    parser.add_argument(
        "config", nargs="?", help="Path to a training config YAML file."
    )
    parser.add_argument(
        "--base", help="Name of a packaged config under fxr/configs/training/."
    )
    parser.add_argument(
        "--resume",
        help="Trusted existing run directory to continue with immutable full state.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="overrides",
        help="Dotted-key override; VALUE is parsed as YAML (repeatable).",
    )
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        metavar="KEY=VALUES",
        dest="sweep_overrides",
        help="Explicit sweep axis; VALUES must be a non-empty YAML list.",
    )
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument(
        "--init-from",
        metavar="MODEL",
        help=(
            "Initialize a new run from a standalone .safetensors file, "
            "local bundle, or Hugging Face repo."
        ),
    )
    initialization.add_argument(
        "--init-from-run",
        metavar="RUN_DIR",
        help="Initialize a new run from a trusted local FleXray run checkpoint.",
    )
    parser.add_argument(
        "--init-revision",
        help="Pinned Hugging Face revision used with --init-from.",
    )
    parser.add_argument(
        "--init-checkpoint",
        default="last",
        help="Safe checkpoint stem for --init-from-run (default last).",
    )
    parser.add_argument(
        "--allow-unverified-label-order",
        action="store_true",
        help=(
            "Allow standalone weights without metadata proving label order; "
            "valid only with --init-from."
        ),
    )
    parser.add_argument(
        "--replace-head",
        action="store_true",
        help=(
            "Load only the backbone from the initialization source and keep a fresh "
            "output head, for fine-tuning onto a different label protocol."
        ),
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Train only the replaced output head; requires --replace-head.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help=(
            "Local training device (default auto). Explicit cuda fails if the "
            "visible card is unavailable or incompatible."
        ),
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help=(
            "Index into the CUDA devices already visible to this process for "
            "local auto/cuda training (default 0)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the experiment and print the resolved config without training.",
    )
    parser.add_argument(
        "--smoke-data",
        action="store_true",
        help="During readiness validation, open one sample per active dataset.",
    )
    parser.add_argument(
        "--submitit", action="store_true", help="Submit the sweep via submitit."
    )
    parser.add_argument(
        "--local", action="store_true", help="Use submitit's local executor."
    )
    parser.add_argument("--partition", help="Slurm partition for submitit jobs.")
    parser.add_argument("--gpus", type=int, help="GPUs per node for submitit jobs.")
    parser.add_argument("--cpus", type=int, help="CPUs per task for submitit jobs.")
    parser.add_argument(
        "--timeout", type=int, help="Job timeout in minutes for submitit jobs."
    )
    return parser


def load_training_base(
    reference: str,
    *,
    _relative_to: Path | None = None,
    _seen: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Load a training base config, resolving ``_base_`` inheritance.

    A config may declare a top-level ``_base_: NAME_OR_PATH`` to inherit from
    another training config. The parent is loaded first (recursively) and the
    child is deep-merged on top, so a variant need only restate the keys it
    overrides (e.g. ``dataloader.proportions``). The ``_base_`` key is stripped
    from the returned config.

    Args:
        reference : str
            Either a path to a YAML file, or the stem of a packaged config under
            ``fxr/configs/training/`` (an optional ``"training/"`` prefix and
            ``.yml``/``.yaml`` suffix are stripped before the packaged lookup).
        _relative_to: Internal parent directory used to resolve inherited paths.
        _seen: Internal identity set used to reject inheritance cycles.

    Returns:
        The parsed, inheritance-resolved config as a plain ``dict``.
    """

    candidate = Path(reference)
    if _relative_to is not None and not candidate.is_absolute():
        relative_candidate = _relative_to / candidate
        if relative_candidate.is_file():
            candidate = relative_candidate
    if candidate.is_file():
        candidate = candidate.resolve()
        reference = str(candidate)
        identity = f"file:{candidate}"
        parent_dir: Path | None = candidate.parent
    else:
        stem = reference.removeprefix("training/").removesuffix(".yml").removesuffix(".yaml")
        identity = f"packaged:{stem}"
        parent_dir = None
    seen = _seen or frozenset()
    if identity in seen:
        raise ValueError(f"Training config inheritance cycle detected at {reference!r}.")

    config = _read_training_config(reference)
    base_reference = config.pop("_base_", None)
    if base_reference is None:
        return config
    parent = load_training_base(
        str(base_reference), _relative_to=parent_dir, _seen=seen | {identity}
    )
    return merge_configs(parent, config)


def _read_training_config(reference: str) -> dict[str, Any]:
    """Read one training config file or packaged config without resolving ``_base_``."""
    if Path(reference).is_file():
        return Config.from_file(Path(reference)).to_dict()
    stem = (
        reference.removeprefix("training/").removesuffix(".yml").removesuffix(".yaml")
    )
    packaged = resources.files("fxr.configs").joinpath("training", f"{stem}.yml")
    with resources.as_file(packaged) as path:
        return Config.from_file(path).to_dict()


def _load_base_config(args: argparse.Namespace) -> dict[str, Any]:
    """Load the base config from ``--base`` or the positional config path."""
    if (args.base is None) == (args.config is None):
        raise SystemExit("Provide exactly one of a config path or --base NAME.")
    return load_training_base(args.base if args.base is not None else args.config)


def _parse_overrides(raw_overrides: list[str]) -> dict[str, Any]:
    """Parse ``KEY=VALUE`` strings into a dotted-key override mapping."""
    overrides: dict[str, Any] = {}
    for item in raw_overrides:
        if "=" not in item:
            raise ValueError(f"Override {item!r} must be KEY=VALUE.")
        key, _, value = item.partition("=")
        key = key.strip()
        if not key:
            raise ValueError("Override keys cannot be empty.")
        overrides[key] = yaml.safe_load(value)
    return overrides


def _parse_sweep_overrides(raw_overrides: list[str]) -> dict[str, Any]:
    """Parse and validate explicit ``--sweep`` axes.

    Args:
        raw_overrides: Repeatable dotted-key sweep assignments.

    Returns:
        Mapping consumable by :func:`expand_configs`.
    """

    overrides = _parse_overrides(raw_overrides)
    validate_sweep_axes(overrides)
    return overrides


def _initialization_override(args: argparse.Namespace) -> dict[str, Any]:
    """Build the persisted model-initialization config from CLI options.

    Args:
        args: Parsed command-line namespace.

    Returns:
        Nested initialization override, or an empty mapping.
    """

    if args.init_revision and not args.init_from:
        raise ValueError("--init-revision requires --init-from.")
    if args.init_checkpoint != "last" and not args.init_from_run:
        raise ValueError("--init-checkpoint requires --init-from-run.")
    if args.allow_unverified_label_order and not args.init_from:
        raise ValueError("--allow-unverified-label-order requires --init-from.")
    if args.replace_head and not (args.init_from or args.init_from_run):
        raise ValueError("--replace-head requires --init-from or --init-from-run.")
    if args.freeze_backbone and not args.replace_head:
        raise ValueError("--freeze-backbone requires --replace-head.")
    head_options = {
        key: True
        for key, enabled in (
            ("replace_head", args.replace_head),
            ("freeze_backbone", args.freeze_backbone),
        )
        if enabled
    }
    if args.init_from:
        source = args.init_from
        local_source = Path(source).expanduser()
        if local_source.exists():
            source = str(local_source.resolve())
        config = {"kind": "pretrained", "source": source}
        if args.init_revision:
            config["revision"] = args.init_revision
        if args.allow_unverified_label_order:
            config["allow_unverified_label_order"] = True
        return {"initialization": {**config, **head_options}}
    if args.init_from_run:
        return {
            "initialization": {
                "kind": "run",
                "source": str(Path(args.init_from_run).expanduser().resolve()),
                "checkpoint": args.init_checkpoint,
                **head_options,
            }
        }
    return {}


def _validate_cli_args(args: argparse.Namespace) -> None:
    """Validate relationships between command-line options.

    Args:
        args: Parsed command-line namespace.

    Returns:
        ``None``.
    """

    source_count = sum(value is not None for value in (args.config, args.base, args.resume))
    if source_count != 1:
        raise ValueError("Provide exactly one of CONFIG, --base, or --resume.")
    if args.resume and (args.overrides or args.sweep_overrides):
        raise ValueError("A resumed run uses its immutable config; omit --set/--sweep.")
    if args.resume and (
        args.init_from
        or args.init_from_run
        or args.init_revision
        or args.init_checkpoint != "last"
        or args.allow_unverified_label_order
        or args.replace_head
        or args.freeze_backbone
    ):
        raise ValueError(
            "Initialization creates a new run and cannot be used with --resume."
        )
    if args.resume and args.submitit:
        raise ValueError("--resume is for an existing local run, not a new submission.")
    if args.gpu < 0:
        raise ValueError("--gpu must be non-negative.")
    if args.submitit and args.device != "auto":
        raise ValueError(
            "--device is a local launch option; submitted jobs use auto device "
            "selection with scheduler-provided CUDA visibility."
        )
    if args.gpus is not None and args.gpus != 1:
        raise ValueError("FleXray currently supports exactly one GPU per training process.")
    submit_options = (args.local, args.partition, args.gpus, args.cpus, args.timeout)
    if not args.submitit and any(value not in (None, False) for value in submit_options):
        raise ValueError(
            "--local/--partition/--gpus/--cpus/--timeout require --submitit."
        )
    if args.submitit and args.smoke_data and not args.dry_run:
        raise ValueError("--smoke-data requires --dry-run when using --submitit.")


def _resolve_new_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Resolve literal overrides and explicit sweep axes for a new run.

    Args:
        args: Parsed command-line namespace.

    Returns:
        Concrete pruned configs, one per explicit sweep cell.
    """

    base = _load_base_config(args)
    literal_overrides = _parse_overrides(args.overrides)
    base = apply_overrides(base, literal_overrides)
    base = merge_configs(base, _initialization_override(args))
    sweep_overrides = _parse_sweep_overrides(args.sweep_overrides)
    return [
        _materialize_ct_view_counts(prune_zero_proportion_datasets(config))
        for config in expand_configs(base, sweep_overrides)
    ]


def _slurm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Collect the provided submitit executor parameters."""
    mapping = {
        "slurm_partition": args.partition,
        "gpus_per_node": args.gpus,
        "cpus_per_task": args.cpus,
        "timeout_min": args.timeout,
    }
    return {key: value for key, value in mapping.items() if value is not None}


def _validate_build_only(config: dict[str, Any]) -> None:
    """Construct one config in a temporary run directory without training.

    Args:
        config: Fully resolved training configuration.

    Returns:
        ``None`` after model, recipe, optimizer, and callback construction.
    """

    from fxr.launch.run import run_config

    experiment = None
    with tempfile.TemporaryDirectory(prefix="flexray-dry-run-") as dry_root:
        dry_config = merge_configs(config, {"log": {"root": dry_root}})
        try:
            experiment = run_config(dry_config, train=False, load_data=False)
        finally:
            close = getattr(experiment, "close", None)
            if callable(close):
                close()


def _visible_gpu_count() -> int:
    """Count visible GPUs without initializing CUDA in the training process.

    Returns:
        The visible CUDA device count reported by an isolated Python process.

    Raises:
        ValueError: If device discovery fails or does not return a count.
    """

    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
            check=True, capture_output=True, text=True, timeout=60,
        )
        return int(result.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise ValueError("Could not inspect visible CUDA devices before launch.") from exc


def _select_visible_gpu(index: int, device: str) -> str | None:
    """Resolve ``--gpu`` to the single CUDA device the run should keep visible.

    ``--gpu`` indexes the devices this process can already see, so an inherited
    ``CUDA_VISIBLE_DEVICES`` (interactive ``srun``, a container, a shared-machine
    policy) is narrowed rather than replaced by a physical index. Discovery runs
    in a child process because Torch's NVML fallback initializes the CUDA runtime
    and can freeze visibility even before ``torch.cuda.is_initialized()`` is true.

    Args:
        index: Requested index into the visible CUDA devices.
        device: Local device policy, ``"auto"`` or ``"cuda"``.

    Returns:
        The ``CUDA_VISIBLE_DEVICES`` value to export, or ``None`` when no CUDA
        device is visible and ``auto`` therefore falls back to CPU.

    Raises:
        ValueError: If ``cuda`` was requested without a visible CUDA device, or
            the index is outside the visible device count.
    """

    if index < 0:
        raise ValueError("--gpu must be non-negative.")
    visible = _visible_gpu_count()
    if visible == 0:
        if device == "cuda" or index != 0:
            raise ValueError(
                "A CUDA device was requested, but no CUDA device is visible to "
                "this process."
            )
        return None
    if index >= visible:
        raise ValueError(
            f"--gpu {index} is out of range; {visible} CUDA device(s) are "
            "visible to this process."
        )
    inherited = [
        entry.strip()
        for entry in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if entry.strip()
    ]
    return inherited[index] if index < len(inherited) else str(index)


def _configure_local_device_environment(args: argparse.Namespace) -> None:
    """Set local device policy before any Torch-backed dependency is inspected.

    Args:
        args: Validated command-line namespace.

    Returns:
        ``None``. Submitted jobs retain scheduler-provided CUDA visibility.

    Raises:
        ValueError: If ``--gpu`` does not name a visible CUDA device.
    """

    if args.submitit:
        return
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        selected = _select_visible_gpu(args.gpu, args.device)
        if selected is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = selected
    os.environ["FXR_DEVICE"] = args.device


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and launch a local run or a submitit sweep.

    Args:
        argv : list of str or None, default=None
            Argument vector; defaults to ``sys.argv[1:]`` when ``None``.

    Returns:
        Process exit code (``0`` on success).
    """

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_cli_args(args)
        _configure_local_device_environment(args)
        require_training_extra("kornia")
        if args.resume:
            resume_path = Path(args.resume).expanduser()
            config_path = resume_path / "config.yml"
            if not config_path.is_file():
                raise FileNotFoundError(f"Run config does not exist: {config_path}.")
            configs = [Config.from_file(config_path).to_dict()]
        else:
            configs = _resolve_new_configs(args)
    except (ImportError, KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))

    if args.submitit:
        try:
            for config in configs:
                check_missing(config)
        except ValueError as exc:
            parser.error(str(exc))
        if args.dry_run:
            from fxr.launch.readiness import validate_training_config

            try:
                for config in configs:
                    validate_training_config(
                        config, smoke_data=args.smoke_data
                    )
                    _validate_build_only(config)
            except (
                ImportError,
                KeyError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                parser.error(str(exc))
            print(
                f"Dry run succeeded for {len(configs)} submission config(s); "
                "no jobs were submitted."
            )
            return 0

        from fxr.launch.submit import submit_configs

        folder = str(Path(configs[0]["log"]["root"]) / "submitit")
        submit_configs(
            configs, folder=folder, slurm_kwargs=_slurm_kwargs(args), local=args.local
        )
        return 0

    if len(configs) != 1:
        parser.error("Local runs take one config; use --submitit for --sweep axes.")
    config = configs[0]
    try:
        check_missing(config)
    except ValueError as exc:
        parser.error(str(exc))

    if args.device == "cuda":
        from fxr.experiment._device import resolve_training_device

        try:
            resolve_training_device("cuda")
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))
    from fxr.launch.readiness import validate_training_config

    try:
        validate_training_config(config, smoke_data=args.smoke_data)
    except ValueError as exc:
        parser.error(str(exc))

    from fxr.launch.run import resume_run, run_config

    if args.resume:
        if args.dry_run:
            print(yaml.safe_dump(config, sort_keys=False))
        experiment = resume_run(
            resume_path, train=not args.dry_run, load_data=not args.dry_run
        )
        try:
            action = "Validated" if args.dry_run else "Resumed"
            print(f"{action} run directory: {experiment.path}")
        finally:
            if args.dry_run:
                close = getattr(experiment, "close", None)
                if callable(close):
                    close()
        return 0

    if args.dry_run:
        print(yaml.safe_dump(config, sort_keys=False))
        try:
            _validate_build_only(config)
        except (
            ImportError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            parser.error(str(exc))
        print("Dry run succeeded; no run directory was created.")
        return 0

    experiment = run_config(config)
    if hasattr(experiment, "path"):
        print(f"Run directory: {experiment.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
