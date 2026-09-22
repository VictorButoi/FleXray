"""Experiment training callbacks dispatched by the trainer's group runner.

These callbacks follow the trainer's callback contract: they are constructed with
the experiment as the first positional argument and invoked with keyword context
(``epoch=...``). ``WandbSamplePredictionLogger`` periodically logs per-dataset
validation Dice and a handful of prediction overlays to WandB.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import DataLoader

from fxr.datasets import NamedBatch, build_named_datasets
from fxr.metrics import dice_score

from ._collation import _ct_safe_collate, _metadata_safe_collate
from ._metrics import binary_dice_sample_count
from ._wandb import require_wandb, wandb


class WandbSamplePredictionLogger:
    """Log per-dataset validation Dice and prediction overlays to WandB.

    Attributes:
        experiment: The training experiment this callback observes.
        every: Epoch interval at which sample predictions are logged.
        max_samples: Maximum number of overlay images logged per dataset.
        batch_size: Batch size used to pull validation samples.
    """

    def __init__(
        self,
        experiment: Any,
        *,
        every: int = 1,
        max_samples: int = 4,
        batch_size: int = 4,
    ) -> None:
        """Store the experiment handle and logging cadence.

        Args:
            experiment: Training experiment exposing ``val_datasets`` and
                ``run_step``.
            every: Epoch interval between logging passes; ``<= 0`` disables it.
            max_samples: Maximum overlay images logged per dataset.
            batch_size: Batch size used when sampling validation data.

        Returns:
            ``None``.
        """

        self.experiment = experiment
        self.every = int(every)
        self.max_samples = int(max_samples)
        self.batch_size = int(batch_size)

    def __call__(self, *, epoch: int) -> None:
        """Log validation Dice and overlays for the current epoch.

        Args:
            epoch: Current training epoch.

        Returns:
            ``None``.

        Raises:
            ImportError: If logging is due and WandB is not installed.
        """

        if self.every <= 0 or epoch % self.every != 0:
            return
        datasets = getattr(self.experiment, "val_datasets", {})
        if not datasets:
            return

        require_wandb()
        payload: dict[str, Any] = {}
        for dataset_name, dataset in datasets.items():
            outputs = self._predict_one_batch(dataset_name, dataset)
            dice = dice_score(
                outputs["y_pred"],
                outputs["y_true"],
                mode="binary",
                from_logits=True,
                ignore_background=True,
            )
            payload[f"eval/{dataset_name}/dice"] = float(dice.item())
            payload[f"eval/{dataset_name}/samples"] = self._overlay_images(outputs)
        if payload:
            wandb.log(payload, step=epoch)

    def _predict_one_batch(self, dataset_name: str, dataset: Any) -> dict[str, Any]:
        """Run the model on one validation batch without gradients or loss.

        Args:
            dataset_name: Source identity attached to the batch.
            dataset: Validation dataset to sample.

        Returns:
            Detached inputs, logits, labels, and dataset identity.
        """
        modality = self._dataset_modality(dataset_name, dataset)
        loader = DataLoader(
            dataset,
            batch_size=1 if modality == "ct" else self.batch_size,
            shuffle=False,
            collate_fn=(
                _ct_safe_collate if modality == "ct" else _metadata_safe_collate
            ),
        )
        batch = NamedBatch(
            source_name=dataset_name, modality=modality, batch=next(iter(loader))
        )
        was_training = self.experiment.model.training
        self.experiment.model.eval()
        try:
            with torch.no_grad():
                prediction_step = getattr(
                    self.experiment, "_prediction_step", None
                )
                if callable(prediction_step):
                    return prediction_step(batch, phase="val")
                return self.experiment.run_step(batch, phase="val", backward=False)
        finally:
            self.experiment.model.train(was_training)

    def _dataset_modality(self, dataset_name: str, dataset: Any) -> str:
        """Return the runtime modality for a validation dataset.

        Args:
            dataset_name: Configured source dataset name.
            dataset: Validation dataset whose modality may be inspected.

        Returns:
            Lowercase runtime modality, defaulting to ``"xray"``.
        """
        modalities = getattr(self.experiment, "modalities", {})
        if isinstance(modalities, Mapping) and dataset_name in modalities:
            return str(modalities[dataset_name]).lower()
        dataset_modality = getattr(dataset, "modality", None)
        if dataset_modality is not None:
            return str(dataset_modality).lower()
        return "xray"

    def _overlay_images(self, outputs: dict[str, Any]) -> list[wandb.Image]:
        """Build independent multilabel WandB overlays for one step.

        Args:
            outputs: Step output containing input images, logits, and channel
                masks.

        Returns:
            WandB images whose foreground channels remain independently
            visible, including when multiple labels overlap.
        """
        images = outputs["x"]
        predictions = torch.sigmoid(outputs["y_pred"]) > 0.5
        ground_truth = outputs["y_true"] > 0.5
        label_names = self._overlay_label_names(int(predictions.shape[1]))
        first_channel = (
            1
            if len(label_names) > 1 and label_names[0].casefold() == "background"
            else 0
        )
        count = min(self.max_samples, int(images.shape[0]))
        overlays: list[wandb.Image] = []
        for index in range(count):
            masks: dict[str, dict[str, Any]] = {}
            for channel in range(first_channel, len(label_names)):
                label_name = label_names[channel]
                class_labels = {0: "background", 1: label_name}
                masks[f"prediction/{label_name}"] = {
                    "mask_data": predictions[index, channel]
                    .to(torch.uint8)
                    .cpu()
                    .numpy(),
                    "class_labels": class_labels,
                }
                masks[f"ground_truth/{label_name}"] = {
                    "mask_data": ground_truth[index, channel]
                    .to(torch.uint8)
                    .cpu()
                    .numpy(),
                    "class_labels": class_labels,
                }
            overlays.append(
                wandb.Image(
                    images[index, 0].float().cpu().numpy(),
                    masks=masks,
                )
            )
        return overlays

    def _overlay_label_names(self, num_channels: int) -> tuple[str, ...]:
        """Return one stable display name for each model output channel.

        Args:
            num_channels: Number of prediction channels being visualized.

        Returns:
            Protocol-derived names when available, otherwise positional names.
        """

        configured = getattr(self.experiment, "model_label_names", None)
        if configured is None:
            return tuple(f"channel_{index}" for index in range(num_channels))
        names = tuple(str(name) for name in configured)
        if len(names) != num_channels:
            return tuple(f"channel_{index}" for index in range(num_channels))
        return names


class EvalSetMetricLogger:
    """Log full X-ray eval-set Dice metrics to WandB.

    Attributes:
        experiment: Training experiment this callback evaluates.
        data: Callback-local X-ray data config keyed by dataset name.
        every: Epoch interval at which eval-set metrics are logged.
        split: Dataset split evaluated by this callback.
        min_ground_truth_area_fraction: Foreground labels whose ground-truth area
            fraction is below this value are excluded from an image's Dice;
            images with no scoreable label are skipped. ``0.0`` disables it.
        batch_size: Batch size used for eval-set dataloaders.
        num_workers: Number of worker processes used by eval-set dataloaders.
        datasets: Lazily built X-ray eval datasets, or ``None`` before use.
        _closed: Whether callback-owned dataset readers were released.
    """

    def __init__(
        self,
        experiment: Any,
        *,
        data: Mapping[str, Any] | None,
        every: int = 1,
        split: str = "val",
        batch_size: int = 4,
        num_workers: int = 0,
        min_ground_truth_area_fraction: float = 0.0,
    ) -> None:
        """Store callback-local eval-set configuration.

        Args:
            experiment: Training experiment exposing ``run_step`` and
                ``compute_metrics``.
            data: Callback-local data config shaped as ``{"Xray": {...}}``.
            every: Epoch interval between eval-set passes; ``<= 0`` disables it.
            split: Dataset split to build and evaluate.
            batch_size: Batch size used for each eval-set dataloader.
            num_workers: Number of workers used for each eval-set dataloader.
            min_ground_truth_area_fraction: Minimum ground-truth area fraction
                in ``[0, 1]`` for a foreground label to be scored.

        Returns:
            ``None``.

        Raises:
            NotImplementedError: If ``data.CT`` is configured.
            TypeError: If ``data`` or ``data.Xray`` is not mapping-like.
            ValueError: If unsupported modality sections are configured.
        """

        raw_data = _as_plain_mapping(data, context="EvalSetMetricLogger data")
        if "CT" in raw_data:
            raise NotImplementedError(
                "CT eval-set metrics are deferred; EvalSetMetricLogger currently "
                "supports callback data.Xray only."
            )
        unsupported = sorted(str(key) for key in raw_data if str(key) != "Xray")
        if unsupported:
            raise ValueError(
                "EvalSetMetricLogger only supports callback data.Xray; "
                f"unsupported data section(s): {unsupported}."
            )
        xray_data = raw_data.get("Xray") or {}
        if not isinstance(xray_data, Mapping):
            raise TypeError("EvalSetMetricLogger data.Xray must be a mapping or null.")

        self.experiment = experiment
        self.data = dict(xray_data)
        self.every = int(every)
        self.split = str(split)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        if self.batch_size <= 0:
            raise ValueError("EvalSetMetricLogger batch_size must be positive.")
        if self.num_workers < 0:
            raise ValueError("EvalSetMetricLogger num_workers must be non-negative.")
        fraction = min_ground_truth_area_fraction
        if isinstance(fraction, bool) or not 0.0 <= float(fraction) <= 1.0:
            raise ValueError(
                "EvalSetMetricLogger min_ground_truth_area_fraction must be in [0, 1]."
            )
        self.min_ground_truth_area_fraction = float(fraction)
        self.datasets: dict[str, Any] | None = None
        self._closed = False

    def __call__(self, *, epoch: int) -> None:
        """Evaluate configured X-ray eval sets and log Dice metrics.

        Args:
            epoch: Current training epoch.

        Returns:
            ``None``.

        Raises:
            ImportError: If logging is due and WandB is not installed.
        """

        if self._closed or self.every <= 0 or epoch % self.every != 0:
            return
        datasets = self._datasets()
        if not datasets:
            return

        require_wandb()
        payload: dict[str, float] = {}
        dice_values: list[float] = []
        for dataset_name, dataset in datasets.items():
            dice = self._evaluate_dataset(dataset_name, dataset)
            payload[f"evalset/{dataset_name}/dice"] = dice
            dice_values.append(dice)
        if dice_values:
            payload["evalset/dice"] = sum(dice_values) / len(dice_values)
            wandb.log(payload, step=epoch)

    def close(self) -> None:
        """Close lazily built eval-set dataset readers exactly once.

        Returns:
            ``None``. Repeated calls are safe.

        Raises:
            BaseException: The first reader-close failure after every dataset
                has been attempted.
        """

        if self._closed:
            return
        self._closed = True
        datasets = self.datasets or {}
        self.datasets = {}
        first_error: BaseException | None = None
        for dataset in reversed(tuple(datasets.values())):
            close = getattr(dataset, "close", None)
            if not callable(close):
                backend = getattr(dataset, "backend", None)
                database = getattr(backend, "db", None)
                close = getattr(database, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def _datasets(self) -> dict[str, Any]:
        """Build and cache the configured X-ray eval datasets on first use.

        Args:
            None.

        Returns:
            Mapping from dataset name to built eval-split dataset.
        """

        if self.datasets is None:
            if not self.data:
                self.datasets = {}
            else:
                self.datasets = build_named_datasets(
                    {"Xray": self.data}, split=self.split, modality="Xray"
                )
        return self.datasets

    def _evaluate_dataset(self, dataset_name: str, dataset: Any) -> float:
        """Return sample-weighted foreground Dice for one X-ray eval dataset.

        Args:
            dataset_name: Source dataset name passed into ``run_step``.
            dataset: Torch dataset to iterate without shuffling.

        Returns:
            Sample-weighted Dice in ``[0, 1]``, or ``0.0`` when no batches are
            emitted.
        """

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_metadata_safe_collate,
            drop_last=False,
        )
        total = 0.0
        count = 0
        was_training = self.experiment.model.training
        self.experiment.model.eval()
        try:
            with torch.no_grad():
                for raw_batch in loader:
                    batch = NamedBatch(
                        source_name=dataset_name, modality="xray", batch=raw_batch
                    )
                    prediction_step = getattr(
                        self.experiment, "_prediction_step", None
                    )
                    if callable(prediction_step):
                        outputs = prediction_step(batch, phase="val")
                    else:
                        outputs = self.experiment.run_step(
                            batch, phase="val", backward=False
                        )
                    if not outputs:
                        continue
                    if self.min_ground_truth_area_fraction > 0.0:
                        dice_sum, weight = _area_filtered_dice(
                            outputs["y_pred"],
                            outputs["y_true"],
                            self.min_ground_truth_area_fraction,
                        )
                        total += dice_sum
                        count += weight
                        continue
                    dice = dice_score(
                        outputs["y_pred"],
                        outputs["y_true"],
                        mode="binary",
                        from_logits=True,
                        ignore_background=True,
                    )
                    weight = binary_dice_sample_count(outputs["y_true"])
                    total += float(dice.item()) * weight
                    count += weight
        finally:
            self.experiment.model.train(was_training)
        if count == 0:
            return 0.0
        return total / count


def _area_filtered_dice(
    y_pred: torch.Tensor, y_true: torch.Tensor, min_fraction: float
) -> tuple[float, float]:
    """Sum per-image Dice over labels whose ground-truth area is large enough.

    Args:
        y_pred: Logits shaped ``(B, C, H, W)``.
        y_true: Binary channel targets shaped ``(B, C, H, W)``.
        min_fraction: Minimum fraction of image pixels a label must cover.

    Returns:
        ``(dice_sum, num_scored_images)``; images without any scoreable
        foreground label are excluded from both.
    """

    area = (y_true[:, 1:] > 0.5).flatten(2).float().mean(dim=-1)
    weights = torch.ones(y_true.shape[:2], dtype=torch.float32, device=y_true.device)
    weights[:, 1:] = (area >= min_fraction).float()
    keep = weights[:, 1:].any(dim=1)
    if not bool(keep.any()):
        return 0.0, 0.0
    scores = dice_score(
        y_pred[keep],
        y_true[keep],
        mode="binary",
        from_logits=True,
        ignore_background=True,
        weights=weights[keep],
        batch_reduction="none",
    )
    return float(scores.sum().item()), float(keep.sum().item())


def _as_plain_mapping(value: Any, *, context: str) -> dict[str, Any]:
    """Normalize an optional config mapping into a plain dictionary.

    Args:
        value: Mapping, Config-like object, or ``None`` to normalize.
        context: Human-readable config location used in errors.

    Returns:
        Plain dictionary.

    Raises:
        TypeError: If ``value`` is not mapping-like.
    """

    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping or null.")
    return dict(value)
