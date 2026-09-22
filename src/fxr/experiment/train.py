"""Generic training lifecycle for FleXray experiments.

``TrainExperiment`` reproduces the recognizable ``run() -> run_phase() ->
run_step()`` loop without DataFrame/meter machinery. It owns optimizer and
scheduler construction, full-state checkpointing and resume, WandB scalar
logging, and config-driven callback dispatch. Concrete build steps and the
per-step forward pass are provided by subclasses.
"""

from __future__ import annotations

from collections.abc import Mapping
import inspect
import os
import random
import secrets
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fxr.config import absolute_import, config_digest

from ._device import resolve_training_device
from ._wandb import require_wandb, wandb
from .base import BaseExperiment
from .ema import ExponentialMovingAverage, resolve_ema_policy


def _generate_wandb_id() -> str:
    """Generate a fresh WandB run id.

    Returns:
        A run id from ``wandb.sdk.lib.runid``, or a random hex string when that
        private helper is missing from the installed WandB.
    """

    try:
        from wandb.sdk.lib.runid import generate_id
    except ImportError:  # pragma: no cover - depends on the installed wandb
        return secrets.token_hex(4)
    return generate_id()


def _shutdown_dataloader_workers(composed_loader: Any) -> None:
    """Stop persistent PyTorch workers owned by one composed dataloader.

    PyTorch exposes no public shutdown method for persistent workers, so this
    guarded cleanup uses the iterator hook before database readers are closed.

    Args:
        composed_loader: Mixed or sequential loader containing a ``loaders`` map.

    Returns:
        ``None``. Loaders without a live persistent iterator are ignored.
    """

    loaders = getattr(composed_loader, "loaders", None)
    values = getattr(loaders, "values", None)
    if not callable(values):
        return
    for loader in values():
        iterator = getattr(loader, "_iterator", None)
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()
        if iterator is not None and hasattr(loader, "_iterator"):
            loader._iterator = None


def _build_grad_scaler(device_type: str, *, enabled: bool) -> Any:
    """Build a gradient scaler across the supported PyTorch versions.

    Args:
        device_type: Training device type such as ``"cuda"`` or ``"cpu"``.
        enabled: Whether fp16 scaling is active.

    Returns:
        A modern ``torch.amp`` scaler when available, otherwise the compatible
        CUDA scaler provided by older supported PyTorch releases.
    """

    scaler_class = getattr(getattr(torch, "amp", None), "GradScaler", None)
    if scaler_class is not None:
        return scaler_class(device_type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


class TrainExperiment(BaseExperiment):
    """Train a model for a fixed number of epochs with WandB logging.

    Attributes:
        device: Torch device the model and batches live on.
        amp_dtype: Optional autocast dtype selected by training config.
        grad_scaler: Gradient scaler used for fp16 CUDA training.
        lr_scheduler_interval: Whether the scheduler advances per epoch or step.
        _scheduler_steps_per_epoch: Cached training-loader length for schedules.
        _global_step: Number of completed optimizer steps.
        _epoch: Most recently started epoch index.
        state: Full serializable training and RNG state.
        model: The trainable model (set by ``build_model``).
        optim: The optimizer (set by ``build_optim``).
        lr_scheduler: Optional learning-rate scheduler.
        ema: Moving-average tracker governed by ``train.ema``.
        callbacks: Mapping from callback group name to a list of callables.
        _closed: Whether experiment-owned dataset and callback resources were
            released.
    """

    def __init__(
        self,
        path: str,
        set_seed: bool = True,
        load_data: bool = True,
        load_optim: bool = True,
    ) -> None:
        """Build all run components from the experiment config.

        Args:
            path: Run directory (handled by ``BaseExperiment.__init__``).
            set_seed: Whether to seed RNGs from the config.
            load_data: Whether to construct datasets and dataloaders.
            load_optim: Whether to construct the optimizer and scheduler.

        Returns:
            ``None``.
        """

        super().__init__(path, set_seed=set_seed)
        self.device = resolve_training_device()
        self.amp_dtype = self._resolve_amp_dtype()
        self.grad_scaler = _build_grad_scaler(
            self.device.type, enabled=self._amp_uses_scaler()
        )
        self.lr_scheduler = None
        self.lr_scheduler_interval = "epoch"
        self._scheduler_steps_per_epoch: int | None = None
        self._global_step = 0
        self._epoch = -1
        self._closed = False

        try:
            self.build_model()
            self.build_ema()
            self.build_augmentations()
            self.build_loss()
            self.build_data(load_data)
            self.build_optim(load_optim)
            self.build_callbacks()
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------------ builds
    def build_model(self) -> None:
        """Construct ``self.model``. Implemented by subclasses.

        Returns:
            ``None``.
        """
        raise NotImplementedError

    def build_ema(self) -> None:
        """Build the optional moving-average tracker from ``train.ema``.

        Args:
            None.

        Returns:
            ``None``.
        """

        policy = resolve_ema_policy(self.config)
        self.ema = ExponentialMovingAverage(self.model, policy)

    def build_augmentations(self) -> None:
        """Construct augmentation state. Optional; no-op by default.

        Returns:
            ``None``.
        """

    def build_loss(self) -> None:
        """Construct ``self.loss_func``. Implemented by subclasses.

        Returns:
            ``None``.
        """
        raise NotImplementedError

    def build_data(self, load_data: bool) -> None:
        """Construct datasets and dataloaders in a subclass.

        Args:
            load_data: Whether to open configured datasets.

        Returns:
            ``None``.
        """
        raise NotImplementedError

    def build_callbacks(self) -> None:
        """Instantiate callbacks grouped by dispatch point.

        Reads a ``callbacks`` config section shaped as
        ``{group: {name: {"_class": ..., **kwargs}}}`` and constructs each
        callback with the experiment as its first positional argument.

        Returns:
            ``None``.
        """

        self.callbacks: dict[str, list[Any]] = {}
        cfg = self.config.get("callbacks")
        if cfg is None:
            return
        cfg = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
        for group, group_cfg in cfg.items():
            self.callbacks[group] = []
            for name, spec in (group_cfg or {}).items():
                if spec is None:
                    continue
                if not isinstance(spec, dict) or "_class" not in spec:
                    raise ValueError(
                        f"callback {name!r} must be a dict with a '_class' key."
                    )
                kwargs = {key: value for key, value in spec.items() if key != "_class"}
                callback_class = absolute_import(spec["_class"])
                self.callbacks[group].append(callback_class(self, **kwargs))

    def close(self) -> None:
        """Release callback and dataset resources owned by this experiment.

        Returns:
            ``None``. Repeated calls are safe.

        Raises:
            BaseException: The first close failure after every owned resource
                has been attempted.
        """

        if getattr(self, "_closed", False):
            return
        self._closed = True
        resources: list[Any] = []
        first_error: BaseException | None = None
        for composed_loader in (
            getattr(self, "train_dl", None),
            getattr(self, "val_dl", None),
        ):
            try:
                _shutdown_dataloader_workers(composed_loader)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc

        callbacks = getattr(self, "callbacks", {})
        for group_callbacks in callbacks.values():
            resources.extend(group_callbacks)
        dataset_bundle = getattr(self, "dataset_bundle", None)
        if dataset_bundle is not None:
            resources.append(dataset_bundle)

        seen: set[int] = set()
        for resource in reversed(resources):
            identity = id(resource)
            if identity in seen:
                continue
            seen.add(identity)
            close = getattr(resource, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "TrainExperiment":
        """Return this experiment for context-managed use.

        Returns:
            This open experiment.
        """

        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Release owned resources when leaving a context manager.

        Args:
            exc_type: Exception type raised in the context, if any.
            exc_value: Exception value raised in the context, if any.
            traceback: Traceback raised in the context, if any.

        Returns:
            ``None``.
        """

        del exc_type, exc_value, traceback
        self.close()

    def build_optim(self, load_optim: bool) -> None:
        """Construct the optimizer and learning-rate scheduler from config.

        Args:
            load_optim: Whether optimizer state is needed for this construction.

        Returns:
            ``None``.
        """
        if not load_optim:
            return
        optim_cfg = self.config["optim"].to_dict()
        lr_scheduler_cfg = optim_cfg.pop("lr_scheduler", None)
        warmup_cfg = optim_cfg.pop("warmup", None)
        optim_cls = absolute_import(optim_cfg.pop("_class"))
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optim = optim_cls(params, **optim_cfg)
        self.lr_scheduler = self._create_lr_scheduler(lr_scheduler_cfg, warmup_cfg)
        self.optim.zero_grad(set_to_none=True)

    # --------------------------------------------------------------------- amp
    def _resolve_amp_dtype(self) -> "torch.dtype | None":
        """Resolve the autocast dtype from ``train.amp_dtype`` (``None`` = fp32).

        Returns:
            ``torch.bfloat16``/``torch.float16`` for mixed precision, or ``None``
            to run in full fp32.

        Raises:
            ValueError: If ``train.amp_dtype`` is set to an unsupported value.
        """

        raw = self.config.get("train.amp_dtype")
        if raw is None:
            return None
        dtypes = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
        }
        name = str(raw).lower()
        if name not in dtypes:
            raise ValueError(
                f"train.amp_dtype must be one of {sorted(dtypes)} or null; got {raw!r}."
            )
        return dtypes[name]

    def _amp_uses_scaler(self) -> bool:
        """Return whether gradient scaling is needed (only fp16 on CUDA).

        Returns:
            ``True`` only for float16 CUDA training.
        """
        return self.amp_dtype == torch.float16 and self.device.type == "cuda"

    # -------------------------------------------------------------- scheduler
    def _create_lr_scheduler(self, lr_scheduler_cfg: Any, warmup_cfg: Any):
        """Build an optional LR scheduler with optional linear warmup.

        Args:
            lr_scheduler_cfg: ``None``, a class-path string, or a config dict with
                ``_class`` and an optional ``interval`` of ``"epoch"``/``"step"``.
            warmup_cfg: ``None`` or a dict with exactly one of ``num_epochs`` /
                ``num_steps`` and an optional ``start_factor``.

        Returns:
            A scheduler instance, or ``None`` when no scheduler is configured.
        """

        path, kwargs, interval = self._parse_lr_scheduler_config(lr_scheduler_cfg)
        if interval not in {"epoch", "step"}:
            raise ValueError("optim.lr_scheduler.interval must be 'epoch' or 'step'.")
        self.lr_scheduler_interval = interval
        self._scheduler_steps_per_epoch = self._steps_per_epoch()

        if path is None:
            return None

        warmup_iters = self._resolve_warmup_iters(warmup_cfg, interval)
        scheduler_cls = absolute_import(path)
        signature = inspect.signature(scheduler_cls)
        if "T_max" in signature.parameters and "T_max" not in kwargs:
            main_iters = self._total_scheduler_steps(interval) - (warmup_iters or 0)
            if main_iters < 1:
                raise ValueError(
                    "Optimizer warmup must leave at least one scheduler step."
                )
            kwargs["T_max"] = main_iters
        main_scheduler = scheduler_cls(self.optim, **kwargs)

        if warmup_iters is None:
            return main_scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optim,
            start_factor=float((warmup_cfg or {}).get("start_factor", 0.01)),
            end_factor=1.0,
            total_iters=warmup_iters,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            self.optim,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_iters],
        )

    @staticmethod
    def _parse_lr_scheduler_config(cfg: Any) -> tuple[str | None, dict[str, Any], str]:
        """Normalize a scheduler config into ``(path, kwargs, interval)``.

        Args:
            cfg: Scheduler config as ``None``, a class path, or a mapping.

        Returns:
            Scheduler class path, constructor arguments, and step interval.

        Raises:
            TypeError: If ``cfg`` is not a supported scheduler config type.
        """
        if cfg is None:
            return None, {}, "epoch"
        if isinstance(cfg, str):
            return cfg, {}, "epoch"
        if isinstance(cfg, dict):
            kwargs = dict(cfg)
            interval = kwargs.pop("interval", "epoch")
            return kwargs.pop("_class", None), kwargs, interval
        raise TypeError("optim.lr_scheduler must be None, a string, or a dict.")

    def _steps_per_epoch(self) -> int | None:
        """Return ``len(train_dl)`` when available, else ``None``.

        Returns:
            Steps per training epoch, or ``None`` for an unsized loader.
        """
        train_dl = getattr(self, "train_dl", None)
        try:
            return len(train_dl) if train_dl is not None else None
        except TypeError:
            return None

    def _total_scheduler_steps(self, interval: str) -> int:
        """Return the scheduler horizon in its stepping interval units.

        Args:
            interval: Scheduler interval, either ``"epoch"`` or ``"step"``.

        Returns:
            Total number of scheduler advances for the training run.

        Raises:
            ValueError: If a step scheduler has no known loader length.
        """
        epochs = int(self.config["train.epochs"])
        if interval == "epoch":
            return epochs
        if self._scheduler_steps_per_epoch is None:
            raise ValueError("Step-interval scheduler requires an initialized train_dl.")
        return epochs * self._scheduler_steps_per_epoch

    def _resolve_warmup_iters(self, warmup_cfg: Any, interval: str) -> int | None:
        """Resolve warmup duration in scheduler-interval units, or ``None``.

        Args:
            warmup_cfg: Optional warmup configuration mapping.
            interval: Scheduler interval, either ``"epoch"`` or ``"step"``.

        Returns:
            Warmup scheduler advances, or ``None`` when warmup is disabled.

        Raises:
            ValueError: If duration fields conflict with each other or interval.
        """
        if warmup_cfg is None:
            return None
        warmup_cfg = dict(warmup_cfg)
        has_epochs = "num_epochs" in warmup_cfg
        has_steps = "num_steps" in warmup_cfg
        if has_epochs == has_steps:
            raise ValueError("optim.warmup needs exactly one of num_epochs/num_steps.")
        if interval == "epoch":
            if has_steps:
                raise ValueError("num_steps requires lr_scheduler.interval='step'.")
            return int(warmup_cfg["num_epochs"])
        if has_steps:
            return int(warmup_cfg["num_steps"])
        if self._scheduler_steps_per_epoch is None:
            raise ValueError("num_epochs warmup requires an initialized train_dl.")
        return int(warmup_cfg["num_epochs"]) * self._scheduler_steps_per_epoch

    def _step_scheduler_on_batch_end(self) -> None:
        """Advance a step-interval scheduler and the global step counter.

        Returns:
            ``None``.
        """
        self._global_step += 1
        if self.lr_scheduler is not None and self.lr_scheduler_interval == "step":
            self.lr_scheduler.step()

    def _step_scheduler_on_epoch_end(self) -> None:
        """Advance an epoch-interval scheduler.

        Returns:
            ``None``.
        """
        if self.lr_scheduler is not None and self.lr_scheduler_interval == "epoch":
            self.lr_scheduler.step()

    # ----------------------------------------------------------- checkpointing
    @property
    def state(self) -> dict[str, Any]:
        """Return the full resumable training state.

        Returns:
            Model, optimizer, scheduler, scaler, progress, EMA, and RNG state.
        """
        state = {
            "run_id": self.path.name,
            "config_digest": config_digest(self.config),
            "model": self.ema.deployment_model_state(),
            "optim": self.optim.state_dict(),
            "lr_scheduler": (
                None if self.lr_scheduler is None else self.lr_scheduler.state_dict()
            ),
            "grad_scaler": self.grad_scaler.state_dict(),
            "epoch": self._epoch,
            "global_step": self._global_step,
            "rng_state": _capture_rng_state(),
        }
        if self.ema.policy.enabled:
            state["model_raw"] = self.ema.raw_model_state()
            state["_ema_state"] = self.ema.checkpoint_metadata()
        return state

    def checkpoint(self, tag: str = "last") -> None:
        """Atomically save full training state under ``checkpoints/{tag}.pt``.

        Args:
            tag: Checkpoint filename stem.

        Returns:
            ``None``.
        """

        checkpoint_dir = self.path / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        safe_tag = _checkpoint_tag(tag)
        target_path = checkpoint_dir / f"{safe_tag}.pt"
        with tempfile.NamedTemporaryFile(
            dir=checkpoint_dir, prefix=f".{safe_tag}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            torch.save(self.state, temporary_path)
            with temporary_path.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary_path, target_path)
            _fsync_directory(checkpoint_dir)
        finally:
            temporary_path.unlink(missing_ok=True)
        self.properties["epoch"] = self._epoch
        self.properties.save()

    def load(self, tag: str = "last") -> "TrainExperiment":
        """Restore full training and RNG state from one checkpoint.

        Args:
            tag: Checkpoint filename stem.

        Returns:
            This experiment after in-place restoration.
        """

        checkpoint_path = self.path / "checkpoints" / f"{_checkpoint_tag(tag)}.pt"
        state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        _validate_checkpoint_identity(
            state,
            run_id=self.path.name,
            expected_config_digest=config_digest(self.config),
            checkpoint_path=checkpoint_path,
        )
        self._epoch = int(state["epoch"])
        self._global_step = int(state.get("global_step", 0))
        ema_state = state.get("_ema_state")
        raw_model_state = state.get("model_raw")
        if ema_state is not None and self.ema.policy.enabled:
            if raw_model_state is None:
                raise KeyError("EMA-aware checkpoint is missing model_raw state.")
            self.model.load_state_dict(raw_model_state)
            self.ema.load_checkpoint_state(state["model"], ema_state)
        elif ema_state is not None:
            if raw_model_state is None:
                raise KeyError("EMA-aware checkpoint is missing model_raw state.")
            self.model.load_state_dict(raw_model_state)
        else:
            self.model.load_state_dict(state["model"])
            if self.ema.policy.enabled:
                self.ema.seed_from_model(self._global_step)
        if getattr(self, "optim", None) is not None and state.get("optim") is not None:
            self.optim.load_state_dict(state["optim"])
        if self.lr_scheduler is not None and state.get("lr_scheduler") is not None:
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        if state.get("grad_scaler") is not None:
            self.grad_scaler.load_state_dict(state["grad_scaler"])
        if state.get("rng_state") is not None:
            _restore_rng_state(state["rng_state"])
        set_train_epoch = getattr(getattr(self, "train_dl", None), "set_epoch", None)
        if callable(set_train_epoch):
            set_train_epoch(self._epoch + 1)
        return self

    # --------------------------------------------------------------- training
    def to_device(self) -> None:
        """Move the model to the experiment device.

        Returns:
            ``None``.
        """
        self.model = self.model.to(self.device)

    def run_callbacks(self, group: str, **kwargs: Any) -> None:
        """Invoke callbacks, using EMA weights for epoch-level evaluation.

        Args:
            group: Configured callback dispatch group.
            **kwargs: Keyword context forwarded to every callback.

        Returns:
            ``None``.
        """

        callbacks = getattr(self, "callbacks", {}).get(group, [])
        if group in {"epoch", "wrapup"}:
            with self.ema.use_ema_weights():
                for callback in callbacks:
                    callback(**kwargs)
            return
        for callback in callbacks:
            callback(**kwargs)

    def _init_wandb(self) -> None:
        """Initialize or reconnect to the run's stable WandB identity.

        Every key of ``log.wandb`` is forwarded to ``wandb.init``. ``name`` and
        ``dir`` default to the run name and directory but may be overridden
        there, and a ``config`` mapping is merged over the run config rather
        than colliding with it.

        Returns:
            ``None``.

        Raises:
            ImportError: If WandB is not installed.
        """

        require_wandb()
        wandb_cfg = self.config.get("log.wandb")
        wandb_cfg = (
            wandb_cfg.to_dict()
            if hasattr(wandb_cfg, "to_dict")
            else dict(wandb_cfg or {})
        )
        run_id = self.properties.get("wandb.id")
        if run_id is None:
            configured_id = wandb_cfg.get("id")
            run_id = str(configured_id or _generate_wandb_id())
            self.properties["wandb.id"] = run_id
            self.properties.save()
        wandb_cfg["id"] = str(run_id)
        wandb_cfg["resume"] = "allow"
        init_kwargs: dict[str, Any] = {
            "config": self.config.to_dict(),
            "name": self.name,
            "dir": str(self.path),
        }
        extra_config = wandb_cfg.pop("config", None)
        if extra_config is not None:
            init_kwargs["config"] = {**init_kwargs["config"], **dict(extra_config)}
        init_kwargs.update(wandb_cfg)
        wandb.init(**init_kwargs)

    def run(self) -> None:
        """Run training to completion, logging scalars to WandB each epoch.

        Returns:
            ``None``.
        """
        epochs = int(self.config["train.epochs"])
        eval_freq = int(self.config.get("train.eval_freq", 1))
        save_freq = int(self.config.get("log.model_weights.save_freq", 0))

        last_epoch = -1
        last_checkpoint = self.path / "checkpoints" / "last.pt"
        try:
            if last_checkpoint.is_file():
                self.load(tag="last")
                last_epoch = self._epoch

            self._init_wandb()
            for epoch in range(last_epoch + 1, epochs):
                self._epoch = epoch
                epoch_metrics: dict[str, float] = {}
                for phase in ("train", "val"):
                    if phase == "val" and not self._should_eval(epoch, eval_freq, epochs):
                        continue
                    print(f"Start {phase} epoch {epoch}.")
                    phase_metrics = self.run_phase(phase, epoch)
                    epoch_metrics.update(
                        {f"{phase}/{name}": value for name, value in phase_metrics.items()}
                    )

                self._step_scheduler_on_epoch_end()
                wandb.log(epoch_metrics, step=epoch)

                self.run_callbacks("epoch", epoch=epoch)
                if save_freq > 0 and epoch % save_freq == 0:
                    self.checkpoint(tag=f"epoch_{epoch:04d}")
                self.checkpoint(tag="last")
            self.run_callbacks("wrapup")
        finally:
            try:
                if wandb is not None:
                    wandb.finish()
            finally:
                self.close()

    @staticmethod
    def _should_eval(epoch: int, eval_freq: int, epochs: int) -> bool:
        """Return whether a validation phase runs this epoch.

        Args:
            epoch: Zero-based epoch index.
            eval_freq: Validation interval; non-positive values disable validation.
            epochs: Total number of training epochs.

        Returns:
            Whether validation should run after this training epoch.
        """
        if eval_freq <= 0:
            return False
        return epoch % eval_freq == 0 or epoch == epochs - 1

    def run_phase(self, phase: str, epoch: int) -> dict[str, float]:
        """Run one train or validation phase and return sample-weighted metrics.

        Args:
            phase: Phase name identifying the dataloader and gradient mode.
            epoch: Current training epoch.

        Returns:
            Mean scalar metrics. When step outputs include ``dataset_name``, the
            result also includes per-dataset keys such as ``HipRay/dice``.
        """
        if (
            phase != "train"
            and self.ema.policy.enabled
            and self.ema.initialized
            and not self.ema.using_ema_weights
        ):
            with self.ema.use_ema_weights():
                return self.run_phase(phase, epoch)
        dataloader = getattr(self, f"{phase}_dl")
        grad_enabled = phase == "train"
        self.model.train(grad_enabled)

        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        with torch.set_grad_enabled(grad_enabled):
            for batch in dataloader:
                outputs = self.run_step(batch, phase=phase, backward=grad_enabled)
                if not outputs:
                    continue
                metrics = self.compute_metrics(outputs)
                weight = _infer_sample_weight(outputs)
                metric_counts = self._metric_sample_counts(outputs)
                dataset_name = outputs.get("dataset_name")
                for name, value in metrics.items():
                    metric_weight = metric_counts.get(name, weight)
                    _accumulate_weighted_metric(
                        totals, counts, name, value, metric_weight
                    )
                    if dataset_name:
                        _accumulate_weighted_metric(
                            totals,
                            counts,
                            f"{dataset_name}/{name}",
                            value,
                            metric_weight,
                        )
        return {name: totals[name] / counts[name] for name in totals}

    def _metric_sample_counts(self, outputs: dict[str, Any]) -> dict[str, int]:
        """Return overrides for metrics that exclude samples from their mean.

        Args:
            outputs: Detached values returned by ``run_step``.

        Returns:
            Metric-name to eligible-sample-count overrides. The default is
            empty, so every metric uses the full inferred batch size. A zero
            count excludes that metric for the current batch.
        """

        return {}

    def run_step(self, batch: Any, *, phase: str, backward: bool) -> dict[str, Any]:
        """Run one optimization or evaluation step in a subclass.

        Args:
            batch: Source-aware batch to process.
            phase: Training or validation phase name.
            backward: Whether to update trainable state.

        Returns:
            Detached step outputs used for metrics and callbacks.
        """
        raise NotImplementedError

    def compute_metrics(self, outputs: dict[str, Any]) -> dict[str, float]:
        """Compute scalar metrics for one subclass step.

        Args:
            outputs: Detached values returned by :meth:`run_step`.

        Returns:
            Scalar metric values keyed by display name.
        """
        raise NotImplementedError


def _validate_checkpoint_identity(
    state: Any,
    *,
    run_id: str,
    expected_config_digest: str,
    checkpoint_path: Path,
) -> None:
    """Validate optional identity fields in a resumable checkpoint.

    Checkpoints created before identity fields were introduced remain readable.

    Args:
        state: Deserialized checkpoint object.
        run_id: Run directory name expected to own the checkpoint.
        expected_config_digest: Digest of the run's immutable config.
        checkpoint_path: Path used in validation diagnostics.

    Returns:
        ``None`` when present identity fields match the opened run.

    Raises:
        TypeError: If the checkpoint is not a mapping.
        ValueError: If a present identity field names another run or config.
    """

    if not isinstance(state, Mapping):
        raise TypeError(f"Training checkpoint must be a mapping: {checkpoint_path}.")
    stored_run_id = state.get("run_id")
    if stored_run_id is not None and stored_run_id != run_id:
        raise ValueError(
            f"Checkpoint run_id {stored_run_id!r} does not match run {run_id!r}: "
            f"{checkpoint_path}."
        )
    stored_digest = state.get("config_digest")
    if stored_digest is not None and stored_digest != expected_config_digest:
        raise ValueError(
            "Checkpoint config_digest does not match the immutable run config: "
            f"{checkpoint_path}."
        )


def _checkpoint_tag(tag: str) -> str:
    """Validate a checkpoint stem as one safe filename component.

    Args:
        tag: Requested checkpoint stem.

    Returns:
        Validated stripped checkpoint stem.

    Raises:
        TypeError: If ``tag`` is not a string.
        ValueError: If ``tag`` is empty or contains path traversal.
    """

    if not isinstance(tag, str):
        raise TypeError("Checkpoint tag must be a string.")
    name = tag.strip()
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(
            "Checkpoint tag must be one non-empty filename component."
        )
    return name


def _capture_rng_state() -> dict[str, Any]:
    """Capture every random stream used by the single-process trainer.

    Returns:
        Python, NumPy, Torch CPU, and optional Torch CUDA RNG states.
    """

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    """Restore every available random stream from a checkpoint.

    Args:
        state: RNG state mapping returned by :func:`_capture_rng_state`.

    Returns:
        ``None``.
    """

    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    torch_cpu_state = state.get("torch_cpu")
    if torch_cpu_state is not None:
        torch.set_rng_state(
            torch.as_tensor(
                torch_cpu_state,
                dtype=torch.uint8,
                device="cpu",
            )
        )
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [
                torch.as_tensor(
                    value,
                    dtype=torch.uint8,
                    device="cpu",
                )
                for value in cuda_state
            ]
        )


def _fsync_directory(path: Path) -> None:
    """Flush a checkpoint-directory rename on supported platforms.

    Args:
        path: Directory whose metadata should be synchronized.

    Returns:
        ``None``.
    """

    if os.name == "nt":
        return
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _infer_sample_weight(outputs: dict[str, Any]) -> int:
    """Return the effective sample count represented by one step output.

    Args:
        outputs: Detached step outputs returned by ``run_step``.

    Returns:
        Batch size inferred from ``outputs["y_pred"]`` or ``1`` when no batch
        tensor with a leading dimension is available.
    """

    y_pred = outputs.get("y_pred")
    shape = getattr(y_pred, "shape", None)
    if shape is None or len(shape) == 0:
        return 1
    try:
        return max(1, int(shape[0]))
    except (TypeError, ValueError):
        return 1


def _accumulate_weighted_metric(
    totals: dict[str, float],
    counts: dict[str, int],
    name: str,
    value: float,
    weight: int,
) -> None:
    """Accumulate one scalar metric into weighted total and count mappings.

    Args:
        totals: Mutable mapping from metric key to weighted sum.
        counts: Mutable mapping from metric key to accumulated sample weight.
        name: Metric key to update.
        value: Scalar metric value for the current step.
        weight: Number of effective samples represented by ``value``.

    Returns:
        ``None``.
    """

    if weight == 0:
        return
    totals[name] = totals.get(name, 0.0) + float(value) * weight
    counts[name] = counts.get(name, 0) + weight
