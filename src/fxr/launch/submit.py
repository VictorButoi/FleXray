"""Cluster submission of training configs via ``submitit``.

Each config becomes a checkpointable job that calls :func:`run_config`. On Slurm,
the scheduler allocates the GPU (``gpus_per_node``), so no ``CUDA_VISIBLE_DEVICES``
is set here. Because ``run_config`` -> ``from_config`` auto-resumes from the
``last`` checkpoint, the ``checkpoint`` hook lets preempted jobs requeue and
continue. This module deliberately supports only the ``submitit`` path.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from fxr.config import prune_zero_proportion_datasets, validate_run_id

from ._extras import require_training_extra
from .run import new_run_id, run_config


class _RunConfigJob:
    """Checkpointable submitit callable for one training config.

    Attributes:
        config: The config dict the job trains from.
        ct_data_path: Value to export as ``CT_DATAPATH`` before training.
        xray_data_path: Value to export as ``XRAY_DATAPATH`` before training.
        generated_data_path: Value to export as ``GENERATED_DATAPATH`` before training.
        run_id: Safe three-part run-directory identity retained across requeues.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        ct_data_path: str | None = None,
        xray_data_path: str | None = None,
        generated_data_path: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Store an owned copy of the config and root paths to export.

        Args:
            config: Resolved training config to own and submit.
            ct_data_path: Optional CT dataset root exported inside the job.
            xray_data_path: Optional X-ray dataset root exported inside the job.
            generated_data_path: Optional generated-X-ray root exported inside the job.
            run_id: Optional validated run id retained across requeues.

        Returns:
            ``None``.
        """
        self.config = prune_zero_proportion_datasets(config)
        self.ct_data_path = ct_data_path
        self.xray_data_path = xray_data_path
        self.generated_data_path = generated_data_path
        self.run_id = new_run_id(self.config) if run_id is None else run_id
        validate_run_id(self.run_id)

    def __call__(self) -> None:
        """Select auto device policy, set data roots, and run to completion.

        Returns:
            ``None``.
        """
        os.environ["FXR_DEVICE"] = "auto"
        if self.ct_data_path is not None:
            os.environ["CT_DATAPATH"] = self.ct_data_path
        if self.xray_data_path is not None:
            os.environ["XRAY_DATAPATH"] = self.xray_data_path
        if self.generated_data_path is not None:
            os.environ["GENERATED_DATAPATH"] = self.generated_data_path
        run_config(self.config, run_id=self.run_id, resume_existing=True)

    def checkpoint(self, *args: Any, **kwargs: Any) -> Any:
        """Requeue the same job on preemption; it resumes from ``last``.

        Args:
            *args: Positional submitit checkpoint context, accepted for compatibility.
            **kwargs: Keyword submitit checkpoint context, accepted for compatibility.

        Returns:
            Delayed submission wrapping this immutable job.
        """
        require_training_extra("submitit")
        import submitit

        return submitit.helpers.DelayedSubmission(self)


def submit_configs(
    configs: Sequence[Mapping[str, Any]],
    *,
    folder: str,
    slurm_kwargs: Mapping[str, Any] | None = None,
    local: bool = False,
    slurm_max_num_timeout: int | None = None,
    ct_data_path: str | None = None,
    xray_data_path: str | None = None,
    generated_data_path: str | None = None,
) -> list[Any]:
    """Submit one job per config and return the submitit jobs.

    Args:
        configs : Sequence of Mapping
            Resolved configs to train, typically from :func:`expand_configs`.
        folder : str
            Directory submitit writes its logs and pickles to.
        slurm_kwargs : Mapping or None, default=None
            Forwarded to ``executor.update_parameters`` (e.g. ``timeout_min``,
            ``slurm_partition``, ``gpus_per_node``, ``cpus_per_task``).
        local : bool, default=False
            Use submitit's local executor (debugging) instead of ``AutoExecutor``.
        slurm_max_num_timeout : int or None, default=None
            Maximum number of preemption/timeout requeues submitit allows before
            giving up. Ignored by the local executor.
        ct_data_path : str or None, default=None
            Exported as ``CT_DATAPATH`` inside each job before training.
        xray_data_path : str or None, default=None
            Exported as ``XRAY_DATAPATH`` inside each job before training.
        generated_data_path : str or None, default=None
            Exported as ``GENERATED_DATAPATH`` inside each job before training.

    Returns:
        The submitted submitit jobs, in config order.
    """

    require_training_extra("submitit")
    import submitit

    if local:
        executor: submitit.Executor = submitit.LocalExecutor(folder=folder)
    else:
        constructor_kwargs: dict[str, Any] = {}
        if slurm_max_num_timeout is not None:
            constructor_kwargs["slurm_max_num_timeout"] = int(slurm_max_num_timeout)
        executor = submitit.AutoExecutor(folder=folder, **constructor_kwargs)
    executor.update_parameters(**dict(slurm_kwargs or {}))

    jobs = [
        executor.submit(
            _RunConfigJob(
                config,
                ct_data_path=ct_data_path,
                xray_data_path=xray_data_path,
                generated_data_path=generated_data_path,
            )
        )
        for config in configs
    ]
    for job in jobs:
        print(f"Submitted job {job.job_id}.")
    return jobs
