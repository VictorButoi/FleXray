"""Named cluster configs for the ``fxr-submit`` launcher.

A cluster config bundles everything machine-specific that a training sweep needs:
the submitit ``slurm_args``, the ``scratch_root`` runs are written under, the
clean dataset roots exported on the node, the requeue policy, and any
generic experiment overrides (e.g. a machine-appropriate ``dataloader.batch_size``)
that should be merged onto every job. Cluster configs are packaged under
``fxr/configs/cluster/`` and selected by name, or loaded from an arbitrary path.

``split_cluster_config`` separates the submit-only keys from the generic
experiment overrides; :func:`cluster_submit_params` turns the submit-only keys
into the concrete arguments :func:`fxr.launch.submit.submit_configs` consumes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from fxr.config import Config

# Keys consumed by the submission machinery rather than merged into a run config.
_SUBMIT_ONLY_KEYS = frozenset(
    {
        "mode",
        "scratch_root",
        "ct_data_path",
        "xray_data_path",
        "generated_data_path",
        "add_date",
        "slurm_args",
        "training_requeue",
        "jobs_per_gpu",
    }
)


def _expand_root(value: Any) -> str:
    """Expand ``~`` and shell variables in a cluster-config root path.

    Args:
        value : Any
            Raw ``scratch_root`` or dataset root as written in the cluster YAML.

    Returns:
        The path text with ``~`` and ``$VAR`` references resolved, so a root the
        user wrote for their shell does not become a directory of that name.
    """

    return os.path.expandvars(os.path.expanduser(str(value)))


@dataclass
class ClusterSubmitParams:
    """Concrete submission parameters derived from a cluster config.

    Attributes:
        mode: ``"slurm"`` to submit through ``AutoExecutor``, ``"local"`` to use
            submitit's local executor.
        scratch_root: Root directory all run-group folders are written under.
        ct_data_path: Value exported as ``CT_DATAPATH`` inside each job, or ``None``.
        xray_data_path: Value exported as ``XRAY_DATAPATH`` inside each job, or ``None``.
        generated_data_path: Value exported as ``GENERATED_DATAPATH`` inside each job, or ``None``.
        add_date: Whether the run-group folder name is prefixed with ``MM_DD_YY_``.
        slurm_kwargs: Parameters forwarded to ``executor.update_parameters``.
        slurm_max_num_timeout: Requeue cap for ``AutoExecutor``, or ``None``.
    """

    mode: str
    scratch_root: str
    ct_data_path: str | None
    xray_data_path: str | None
    generated_data_path: str | None
    add_date: bool
    slurm_kwargs: dict[str, Any]
    slurm_max_num_timeout: int | None


def load_cluster_config(name: str | None = None, path: str | None = None) -> dict[str, Any]:
    """Load a cluster config by packaged name or filesystem path.

    Args:
        name : str or None, default=None
            Stem of a packaged config under ``fxr/configs/cluster/``.
        path : str or None, default=None
            Path to a custom cluster YAML file.

    Returns:
        The parsed cluster config as a plain ``dict``.

    Raises:
        ValueError: If neither or both of ``name`` and ``path`` are provided.
    """

    if (name is None) == (path is None):
        raise ValueError("Provide exactly one of cluster name or path.")
    if path is not None:
        return Config.from_file(Path(path)).to_dict()
    packaged = resources.files("fxr.configs").joinpath("cluster", f"{name}.yml")
    with resources.as_file(packaged) as resolved:
        return Config.from_file(resolved).to_dict()


def split_cluster_config(cluster_cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Partition a cluster config into submit-only keys and experiment overrides.

    Args:
        cluster_cfg : dict
            The full cluster config as loaded from YAML.

    Returns:
        Tuple ``(submit_cfg, experiment_overrides)`` where ``submit_cfg`` holds
        only the submission keys and ``experiment_overrides`` holds everything
        else, ready to deep-merge onto each run config.
    """

    submit_cfg = {key: value for key, value in cluster_cfg.items() if key in _SUBMIT_ONLY_KEYS}
    overrides = {key: value for key, value in cluster_cfg.items() if key not in _SUBMIT_ONLY_KEYS}
    return submit_cfg, overrides


def cluster_submit_params(submit_cfg: dict[str, Any]) -> ClusterSubmitParams:
    """Derive concrete submission parameters from the submit-only cluster keys.

    ``None`` entries in ``slurm_args`` are dropped (e.g. ``gpus_per_node: null`` on
    CPU partitions), and an enabled ``training_requeue`` folds its
    ``slurm_signal_delay_s`` into ``slurm_kwargs`` and surfaces ``max_num_timeout``.
    ``scratch_root`` and the three dataset roots are expanded, so ``~`` and
    ``$VAR`` in a hand-written cluster config resolve instead of becoming
    directory names under the submitting shell's working directory.

    Args:
        submit_cfg : dict
            The submit-only keys returned by :func:`split_cluster_config`.

    Returns:
        The populated :class:`ClusterSubmitParams`.
    """

    slurm_kwargs = {
        key: value for key, value in dict(submit_cfg.get("slurm_args", {})).items() if value is not None
    }
    gpu_count = slurm_kwargs.get("gpus_per_node")
    if gpu_count is not None and int(gpu_count) != 1:
        raise ValueError(
            "FleXray currently supports exactly one GPU per training process; "
            f"got gpus_per_node={gpu_count}."
        )

    requeue = dict(submit_cfg.get("training_requeue", {}))
    max_num_timeout: int | None = None
    if requeue.get("enabled", False):
        max_num_timeout = int(requeue.get("max_num_timeout", 3))
        slurm_kwargs["slurm_signal_delay_s"] = int(requeue.get("slurm_signal_delay_s", 600))

    scratch_root = submit_cfg.get("scratch_root")
    if not scratch_root:
        raise ValueError("Cluster config must define a 'scratch_root'.")

    mode = str(submit_cfg.get("mode", "slurm"))
    if mode not in {"slurm", "local"}:
        raise ValueError("Cluster mode must be slurm or local.")

    ct_data_path = submit_cfg.get("ct_data_path")
    xray_data_path = submit_cfg.get("xray_data_path")
    generated_data_path = submit_cfg.get("generated_data_path")
    return ClusterSubmitParams(
        mode=mode,
        scratch_root=_expand_root(scratch_root),
        ct_data_path=None if ct_data_path is None else _expand_root(ct_data_path),
        xray_data_path=None if xray_data_path is None else _expand_root(xray_data_path),
        generated_data_path=(
            None if generated_data_path is None else _expand_root(generated_data_path)
        ),
        add_date=bool(submit_cfg.get("add_date", True)),
        slurm_kwargs=slurm_kwargs,
        slurm_max_num_timeout=max_num_timeout,
    )
