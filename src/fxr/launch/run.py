"""In-process launch and resume helpers for one training config.

``run_config`` creates a new run directory and ``resume_run`` opens an existing
one. Keeping those operations separate prevents an accidental fresh launch from
silently reusing a previous checkpoint. Submitit jobs use an explicit run id so
the first invocation creates the run and a requeued invocation resumes that same
directory.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fxr.config import (
    Config,
    absolute_import,
    config_digest,
    generate_tuid,
    prune_zero_proportion_datasets,
    validate_run_id,
)


def new_run_id(config: Mapping[str, Any]) -> str:
    """Return a time-ordered id tied to the resolved config.

    Args:
        config: Resolved configuration used to compute the identity digest.

    Returns:
        A ``TIMESTAMP-NONCE-DIGEST`` run id.
    """

    create_time, nonce = generate_tuid()
    return f"{create_time}-{nonce}-{config_digest(config)}"


def run_config(
    config: Mapping[str, Any],
    *,
    train: bool = True,
    load_data: bool = True,
    run_id: str | None = None,
    resume_existing: bool = False,
) -> Any:
    """Create the configured experiment and optionally run it to completion.

    Args:
        config : Mapping
            Fully resolved experiment config with no ``"?"`` placeholders. Must
            contain ``experiment._class`` (a dotted import string).
        train : bool, default=True
            Whether to call ``experiment.run()`` after construction. Set ``False``
            for a build-only dry run.
        load_data : bool, default=True
            Whether the experiment constructs its datasets and dataloaders. Set
            ``False`` to validate model/protocol wiring without clean dataset root env vars.
        run_id : str or None, default=None
            Explicit safe three-part run-directory id. A new id is generated
            when omitted.
        resume_existing : bool, default=False
            Open an existing explicit-id directory. This is reserved for
            scheduler requeues; interactive resumes use :func:`resume_run`.

    Returns:
        The constructed experiment instance.
    """

    config = prune_zero_proportion_datasets(config)
    selected_id = new_run_id(config) if run_id is None else run_id
    validate_run_id(selected_id)
    run_path = Path(config.get("log", {}).get("root", ".")) / selected_id
    if run_path.exists():
        if not resume_existing:
            raise FileExistsError(
                f"Run directory already exists: {run_path}. "
                "Use --resume to continue an existing run."
            )
        persisted_path = run_path / "config.yml"
        if not persisted_path.is_file():
            raise FileNotFoundError(
                f"Existing run is missing its persisted config: {persisted_path}."
            )
        persisted_config = Config.from_file(persisted_path)
        if config_digest(config) != config_digest(persisted_config):
            raise ValueError(
                "Scheduler requeue config does not match the immutable config "
                f"stored in {persisted_path}."
            )
        experiment_class = absolute_import(persisted_config["experiment._class"])
        experiment = experiment_class(run_path, load_data=load_data)
    else:
        experiment_class = absolute_import(config["experiment"]["_class"])
        experiment = experiment_class.from_config(
            config, uuid=selected_id, load_data=load_data
        )
    if train:
        experiment.run()
    return experiment


def resume_run(
    path: str | Path,
    *,
    train: bool = True,
    load_data: bool = True,
) -> Any:
    """Open an existing run and optionally continue it.

    Args:
        path: Run directory containing its persisted ``config.yml``.
        train: Whether to invoke ``experiment.run()`` after construction.
        load_data: Whether to construct datasets and dataloaders.

    Returns:
        The opened experiment instance.

    Raises:
        FileNotFoundError: If the directory or its config does not exist.
    """

    run_path = Path(path)
    config_path = run_path / "config.yml"
    checkpoint_path = run_path / "checkpoints" / "last.pt"
    if not run_path.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_path}.")
    if not config_path.is_file():
        raise FileNotFoundError(f"Run config does not exist: {config_path}.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Run checkpoint does not exist: {checkpoint_path}. "
            "Only checkpointed runs can be resumed."
        )
    config = Config.from_file(config_path)
    experiment_class = absolute_import(config["experiment._class"])
    experiment = experiment_class(run_path, load_data=load_data)
    if train:
        experiment.run()
    elif hasattr(experiment, "load"):
        experiment.load(tag="last")
    return experiment
