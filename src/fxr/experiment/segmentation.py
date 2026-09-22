"""X-ray segmentation training experiment.

``FleXrayTrainExperiment`` wires FleXray's data, model, loss, augmentation,
metric, and protocol surfaces into the generic :class:`TrainExperiment` lifecycle.
X-ray batches project native integer labels into model channels; CT batches are
rendered to DRRs on the fly through :class:`DrrForwardPipeline`. Both paths then
share the same augment, normalize, model, and loss tail.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler

from fxr.augmentation import (
    build_input_normalizer,
    build_segmentation_augmentation_pipeline,
    load_named_augmentation_preset,
    resolve_augmentation_presets,
    snapshot_augmentation_presets,
)
from fxr.config import eval_config, prune_zero_proportion_datasets
from fxr.config.training import requires_training_source_validation
from fxr.config._proportions import normalize_proportion_weight
from fxr.datasets import (
    MixedDataLoader,
    NamedBatch,
    SequentialDataLoader,
    build_multimodal_datasets,
    compile_training_label_remap_by_name,
    compile_package_label_remap,
    project_channel_mask,
)
from fxr.datasets.remap import normalize_training_channel_label_names
from fxr.losses import DatasetRoutedLoss
from fxr.losses._config import build_loss_from_config
from fxr.metrics import dice_score
from fxr.models.camera import scale_rendered_images

from ._collation import _ct_safe_collate, _metadata_safe_collate
from ._loader_config import loader_options
from ._metrics import binary_dice_sample_count
from .batch_inputs import BatchInputs, resolve_batch_inputs
from .compile import maybe_compile_model
from .drr_forward import DrrForwardPipeline
from .initialization import initialize_model
from .label_projection import TrainingLabelProjection, recompute_background_channel
from .protocol_resolve import (
    inject_protocol_derived_model_channels,
    resolve_run_protocol_spec,
    resolve_supervise_empty_label_ids,
)
from .train import TrainExperiment


class FleXrayTrainExperiment(TrainExperiment):
    """Train an X-ray segmentation model from a protocol-driven config.

    Attributes:
        model_label_names: Ordered model-output label names derived from the
            protocol, or ``None`` when no protocol is configured.
        model: Raw trainable segmentation model.
        forward_model: Raw or optionally compiled model callable.
        _projection_cache: Dataset label projections cached by source name.
        normalizer: Configured per-sample intensity normalizer.
        aug_pipelines: Train-time augmentation pipelines keyed by runtime
            modality (``"ct"`` and ``"xray"``).
        initialization_provenance: Model-only initialization source metadata,
            or ``None`` for from-scratch models. Existing resumable runs reuse
            the provenance persisted in ``properties.json``.
        dataset_bundle: Owned training and optional validation dataset bundle,
            present when ``load_data=True`` and released by :meth:`close`.
        loss_func: Configured segmentation loss module.
        train_datasets: Training datasets indexed by source name.
        val_datasets: Validation datasets indexed by source name.
        modalities: Runtime modality indexed by source name.
        train_dl: Source-aware composed training dataloader.
        val_dl: Optional sequential validation dataloader.
        drr_pipeline: Optional CT-to-DRR rendering pipeline.
        _channel_source_names: X-ray sources that already provide channel masks.
    """

    @classmethod
    def from_config(
        cls,
        config: Any,
        uuid: str | None = None,
        **kwargs: Any,
    ) -> "FleXrayTrainExperiment":
        """Create a run after removing explicitly inactive datasets.

        Args:
            config: Experiment config mapping. Datasets with
                ``dataloader.proportions`` weight ``0`` are removed before
                missing-value validation and persistence.
            uuid: Optional explicit run id passed to the base experiment.
            **kwargs: Constructor arguments forwarded to ``TrainExperiment``.

        Returns:
            Opened FleXray training experiment.
        """

        return super().from_config(
            prune_zero_proportion_datasets(config), uuid=uuid, **kwargs
        )

    # ------------------------------------------------------------------ builds
    def build_model(self) -> None:
        """Build, optionally initialize, and optionally compile the model.

        Returns:
            ``None``.
        """

        config_dict, model_label_names = inject_protocol_derived_model_channels(
            self.config.to_dict()
        )
        self.model_label_names = model_label_names
        model_cfg = deepcopy(config_dict["model"])
        compile_cfg = model_cfg.pop("compile_cfg", None)
        self.model = eval_config(model_cfg)
        initialization = config_dict.get("initialization")
        last_checkpoint = self.path / "checkpoints" / "last.pt"
        if initialization is not None and not last_checkpoint.is_file():
            self.initialization_provenance = initialize_model(
                self.model,
                initialization,
                expected_label_names=self.model_label_names,
                expected_model_config=model_cfg,
            )
            self.properties["initialization"] = self.initialization_provenance
            self.properties.save()
        else:
            self.initialization_provenance = self.properties.get("initialization")
        self.to_device()
        self.forward_model = maybe_compile_model(self.model, compile_cfg)
        self._projection_cache: dict[str, TrainingLabelProjection] = {}

    def build_augmentations(self) -> None:
        """Build the input normalizer and per-modality train augmentation pipelines.

        The resolved transform maps are snapshotted into the run directory.

        Returns:
            ``None``.
        """
        self.normalizer = build_input_normalizer(self.config).to(self.device)
        preset_names = resolve_augmentation_presets(self.config.to_dict())
        presets = {
            modality: load_named_augmentation_preset(name)
            for modality, name in preset_names.items()
        }
        snapshot_augmentation_presets(self.path, presets)
        self.aug_pipelines = {
            modality: build_segmentation_augmentation_pipeline(preset).to(self.device)
            for modality, preset in presets.items()
        }

    def build_loss(self) -> None:
        """Instantiate the configured single, combined, or routed loss.

        Returns:
            ``None``.
        """
        self.loss_func = build_loss_from_config(self.config["loss_func"].to_dict())
        self.loss_func = self.loss_func.to(self.device)
        if isinstance(self.loss_func, DatasetRoutedLoss):
            config_dict = prune_zero_proportion_datasets(self.config.to_dict())
            data_cfg = config_dict.get("data", {})
            names = [
                name
                for section in ("Xray", "CT")
                for name in (data_cfg.get(section) or {})
            ]
            self.loss_func.configure_supervise_empty_label_ids(
                resolve_supervise_empty_label_ids(config_dict, names)
            )

    def build_data(self, load_data: bool) -> None:
        """Build 2D X-ray/CT data, the DRR pipeline, and required loaders.

        Training-source validation datasets are omitted when ``train.eval_freq``
        is zero and no enabled callback consumes them.

        Args:
            load_data: Whether to construct datasets and loaders.

        Returns:
            ``None``.
        """

        if not load_data:
            return
        config_dict = prune_zero_proportion_datasets(self.config.to_dict())
        data_cfg = config_dict["data"]
        selected = {
            key: section
            for key, section in data_cfg.items()
            if key in ("Xray", "CT")
        }
        if not selected:
            raise ValueError(
                "FleXrayTrainExperiment requires a data.Xray and/or data.CT section."
            )
        include_validation = requires_training_source_validation(config_dict)
        bundle = build_multimodal_datasets(
            selected, include_validation=include_validation
        )
        self.dataset_bundle = bundle
        self.train_datasets = dict(bundle.train)
        self.val_datasets = dict(bundle.val)
        self.modalities = dict(bundle.modalities)
        self._channel_source_names = self._collect_channel_source_names()
        self.drr_pipeline = DrrForwardPipeline.from_config(
            config_dict, device=self.device
        )
        self.build_dataloader()

    def _collect_channel_source_names(self) -> dict[str, tuple[str, ...]]:
        """Collect stored label names declared by every 2D source.

        Names may describe dense native ids or mask channels; the payload
        dimensionality decides which projection is used. Names are read first
        from backend-wide metadata and then from record metadata.

        Returns:
            Ordered source label names keyed by dataset name.

        Raises:
            ValueError: If one dataset declares inconsistent label orders or
                labels outside the selected protocol.
        """

        config_dict = self.config.to_dict()
        protocol = resolve_run_protocol_spec(config_dict)
        protocol_cfg = config_dict.get("protocol")
        config_root = (
            protocol_cfg.get("config_root")
            if isinstance(protocol_cfg, Mapping)
            else None
        )
        source_names: dict[str, tuple[str, ...]] = {}
        for datasets in (self.train_datasets, self.val_datasets):
            for name, dataset in datasets.items():
                if self.modalities.get(name) == "ct":
                    continue
                names = _dataset_channel_label_names(dataset)
                if names is None:
                    continue
                if protocol is not None:
                    names = normalize_training_channel_label_names(
                        protocol.protocol_name,
                        name,
                        names,
                        config_root=config_root,
                    )
                previous = source_names.get(name)
                if previous is not None and previous != names:
                    raise ValueError(
                        f"Dataset {name!r} declares different train/val label "
                        f"orders: {previous!r} and {names!r}."
                    )
                source_names[name] = names
        return source_names

    def build_dataloader(self) -> None:
        """Build mixed train loaders and an optional sequential val loader.

        CT loaders use a batch size of 1 because each CT volume is rendered into
        ``num_views`` DRRs that form the effective training batch. A
        ``dataloader.proportions`` weight of ``0`` drops that dataset from both
        the train mix and the validation pass; remaining weights are passed to
        the mixed loader as positive sampling weights.

        Returns:
            ``None``.
        """
        dl_cfg = self.config["dataloader"].to_dict()
        proportions = dict(dl_cfg.get("proportions") or {})
        excluded = _excluded_datasets(proportions)
        train_datasets = {
            n: d for n, d in self.train_datasets.items() if n not in excluded
        }
        val_datasets = {
            n: d for n, d in self.val_datasets.items() if n not in excluded
        }
        kept_proportions = {
            n: w for n, w in proportions.items() if n not in excluded
        }
        configured = getattr(self, "modalities", {})
        all_names = (*train_datasets, *val_datasets)
        modalities = {name: configured.get(name, "xray") for name in all_names}

        seed = int(self.config.get("experiment.seed", 0))

        def make_loader(dataset: Any, name: str, *, shuffle: bool) -> DataLoader:
            options = loader_options(dl_cfg, modality=modalities[name])
            collate = (
                _ct_safe_collate
                if modalities[name] == "ct"
                else _metadata_safe_collate
            )
            sampler = self._weighted_sampler(dataset, name, seed=seed) if shuffle else None
            return DataLoader(
                dataset,
                shuffle=shuffle and sampler is None,
                sampler=sampler,
                drop_last=False,
                collate_fn=collate,
                **options,
            )

        self.train_dl = MixedDataLoader(
            {n: make_loader(d, n, shuffle=True) for n, d in train_datasets.items()},
            modalities={name: modalities[name] for name in train_datasets},
            proportions=kept_proportions or None,
            iters_per_epoch=dl_cfg.get("iters_per_epoch"),
            seed=int(self.config.get("experiment.seed", 0)),
        )
        self.val_dl = (
            SequentialDataLoader(
                {
                    n: make_loader(d, n, shuffle=False)
                    for n, d in val_datasets.items()
                },
                modalities={name: modalities[name] for name in val_datasets},
            )
            if val_datasets
            else None
        )

    # --------------------------------------------------------------- forward
    def _dataset_spec_path(self, dataset_name: str) -> str | None:
        """Return the configured ``dataset_spec`` path of one source, if any.

        Args:
            dataset_name: Source dataset name.

        Returns:
            Absolute spec path from ``data.<modality>.<name>.dataset_spec``, or
            ``None``.
        """

        data_cfg = self.config.to_dict().get("data") or {}
        for section in ("Xray", "CT"):
            dataset_cfg = (data_cfg.get(section) or {}).get(dataset_name)
            if isinstance(dataset_cfg, Mapping) and dataset_cfg.get("dataset_spec"):
                return str(dataset_cfg["dataset_spec"])
        return None

    def _stored_labels_for_dataset(
        self, dataset_name: str
    ) -> Mapping[int | str, str] | None:
        """Return package-owned dense label metadata for one source.

        Args:
            dataset_name: Source dataset name.

        Returns:
            Stored native-id/name mapping, or ``None`` for registered datasets.

        Raises:
            TypeError: If backend ``stored_labels`` metadata is not a mapping.
            ValueError: If train and validation metadata disagree.
        """

        found: dict[int | str, str] | None = None
        for datasets in (
            getattr(self, "train_datasets", {}),
            getattr(self, "val_datasets", {}),
        ):
            dataset = datasets.get(dataset_name)
            if dataset is None:
                continue
            attrs = getattr(getattr(dataset, "backend", None), "attrs", None)
            if not isinstance(attrs, Mapping):
                continue
            stored_labels = attrs.get("stored_labels")
            if stored_labels is None:
                continue
            if not isinstance(stored_labels, Mapping):
                raise TypeError("Backend attrs.stored_labels must be a mapping.")
            normalized = dict(stored_labels)
            if found is not None and found != normalized:
                raise ValueError(
                    f"Dataset {dataset_name!r} has different train/val "
                    "stored_labels metadata."
                )
            found = normalized
        return found

    def _label_projection(self, dataset_name: str) -> TrainingLabelProjection:
        """Return a cached native-to-model label projection for a dataset.

        Args:
            dataset_name: Source dataset whose native labels should be projected.

        Returns:
            Cached or newly compiled training-label projection.

        Raises:
            ValueError: If no protocol is configured for native label projection.
        """
        cached = self._projection_cache.get(dataset_name)
        if cached is not None:
            return cached
        config_dict = self.config.to_dict()
        protocol = resolve_run_protocol_spec(config_dict)
        if protocol is None:
            raise ValueError(
                "Native X-ray training labels require a top-level protocol.name to "
                "project labels into model channels."
            )
        protocol_cfg = config_dict.get("protocol")
        config_root = (
            protocol_cfg.get("config_root")
            if isinstance(protocol_cfg, Mapping)
            else None
        )
        stored_labels = self._stored_labels_for_dataset(dataset_name)
        if stored_labels is not None:
            remap = compile_package_label_remap(
                protocol.protocol_name,
                dataset_name,
                stored_labels,
                dataset_spec=self._dataset_spec_path(dataset_name),
                model_label_names=self.model_label_names,
                config_root=config_root,
            )
        else:
            remap = compile_training_label_remap_by_name(
                protocol.protocol_name,
                dataset_name,
                model_label_names=self.model_label_names,
                config_root=config_root,
            )
        projection = TrainingLabelProjection.from_label_remap(remap)
        self._projection_cache[dataset_name] = projection
        return projection

    def _project_channel_label(self, label: Tensor, dataset_name: str) -> Tensor:
        """Reorder a named channel mask into model-output channel order.

        X-ray datasets may emit either dense integer maps or channel-first masks.
        When source label names are present, the mask is projected by name and
        no native-id lookup table is used.

        Args:
            label: Channel-mask batch shaped ``(B, C_src, *spatial)``.
            dataset_name: Source dataset name used to look up stored label names.

        Returns:
            Channel-mask batch shaped ``(B, num_model_channels, *spatial)``.

        Raises:
            ValueError: If stored label names or model label names are missing.
        """

        source_names = self._channel_source_names_for(dataset_name)
        if source_names is None:
            raise ValueError(
                f"No stored channel label names for dataset {dataset_name!r}."
            )
        if self.model_label_names is None:
            raise ValueError(
                "Named channel-mask training requires protocol-derived model labels."
            )
        target_names = self.model_label_names
        projected = torch.empty(
            (int(label.shape[0]), len(target_names), *label.shape[2:]),
            dtype=torch.float32,
            device=label.device,
        )
        for index in range(int(label.shape[0])):
            projected[index] = project_channel_mask(
                label[index], source_names, target_names
            )
        # Dropped native channels project to background; rebuild it as the complement.
        return recompute_background_channel(projected)

    def _channel_source_names_for(
        self, dataset_name: str
    ) -> tuple[str, ...] | None:
        """Return stored channel names for one dataset.

        Args:
            dataset_name: Source dataset name.

        Returns:
            Ordered stored label names, or ``None`` for dense labels.
        """

        return getattr(self, "_channel_source_names", {}).get(dataset_name)

    def _apply_augmentation(
        self,
        image: Tensor,
        label: Tensor,
        phase: str,
        dataset_name: str,
    ) -> tuple[Tensor, Tensor]:
        """Apply the source modality's train augmentation pipeline when training.

        Args:
            image: Model-channel image batch.
            label: Model-channel label batch.
            phase: Current run phase; only ``"train"`` is augmented.
            dataset_name: Source dataset name selecting the CT or X-ray pipeline.

        Returns:
            Augmented ``(image, label)`` for train batches, otherwise the inputs.
        """
        if phase != "train":
            return image, label
        modality = "ct" if self.modalities[dataset_name] == "ct" else "xray"
        return self.aug_pipelines[modality](image, label)

    def _weighted_sampler(
        self, dataset: Any, dataset_name: str, *, seed: int
    ) -> WeightedRandomSampler | None:
        """Build a seeded weighted sampler for CT sources with sample weights.

        Args:
            dataset: Training dataset of the source.
            dataset_name: Source dataset name (used for the label LUT).
            seed: Experiment seed for the sampler generator.

        Returns:
            A replacement sampler drawing ``len(dataset)`` items per epoch, or
            ``None`` when the source samples uniformly.
        """

        if not hasattr(dataset, "sample_weights"):
            return None
        weights = dataset.sample_weights(self._label_projection(dataset_name).label_lut)
        if weights is None:
            return None
        generator = torch.Generator().manual_seed(seed)
        return WeightedRandomSampler(
            weights, num_samples=len(dataset), replacement=True, generator=generator
        )

    def _model_inputs(
        self,
        inputs: BatchInputs,
        *,
        dataset_name: str,
    ) -> tuple[Tensor, Tensor]:
        """Resolve a model-channel ``(image, label)`` pair for one batch.

        X-ray native integer labels are projected into model channels; CT volumes
        are rendered into DRR images and projected labels via the DRR pipeline.

        Args:
            inputs: Resolved batch inputs for the X-ray or CT path.
            dataset_name: Source dataset name for projection and DRR routing.

        Returns:
            Tuple ``(image, label)`` ready for augmentation and the model.
        """

        if inputs.modality == "ct":
            projection = self._label_projection(dataset_name)
            if self.drr_pipeline is None:
                raise ValueError("CT training requires configured drr_model profiles.")
            native_label = projection.validated_native_ids(
                inputs.label,
                source=f"CT dataset {dataset_name!r}",
                require_mapped=True,
            )
            rendered = self.drr_pipeline.render(
                volume=inputs.image,
                label=native_label,
                affine=inputs.affine,
                dataset_name=dataset_name,
                fg_centroids_ijk=supervised_foreground_centroids(inputs, projection),
                foreground_collapse_map=projection.foreground_collapse_map(),
                attenuated_label_ids=projection.native_foreground_label_ids(),
            )
            return scale_rendered_images(rendered.images), rendered.labels

        if (
            inputs.label.ndim == inputs.image.ndim
            and self._channel_source_names_for(dataset_name) is not None
        ):
            return inputs.image, self._project_channel_label(
                inputs.label, dataset_name
            )

        projection = self._label_projection(dataset_name)
        label = projection.project(inputs.label, source=f"dataset {dataset_name!r}")
        return inputs.image, label

    def forward_pass(
        self,
        image: Tensor,
        label: Tensor,
        *,
        phase: str,
        dataset_name: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Augment, normalize, run the model, and compute the loss.

        Args:
            image: Model-channel image batch shaped ``(B, 1, H, W)``.
            label: Model-channel label batch shaped ``(B, C, H, W)``.
            phase: ``"train"`` or ``"val"``.
            dataset_name: Source dataset name for augmentation and loss routing.

        Returns:
            Tuple ``(loss, logits, model_input, model_label)``.
        """

        logits, image, label = self._forward_without_loss(
            image,
            label,
            phase=phase,
            dataset_name=dataset_name,
        )
        loss_kwargs: dict[str, Any] = {}
        if isinstance(self.loss_func, DatasetRoutedLoss):
            loss_kwargs["dataset_name"] = dataset_name
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype or torch.bfloat16,
            enabled=self.amp_dtype is not None,
        ):
            loss = self.loss_func(logits, label, **loss_kwargs)
        return loss, logits, image, label

    def _forward_without_loss(
        self,
        image: Tensor,
        label: Tensor,
        *,
        phase: str,
        dataset_name: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Prepare inputs and return logits without invoking the routed loss.

        Args:
            image: Model-channel image batch shaped ``(B, 1, H, W)``.
            label: Model-channel label batch shaped ``(B, C, H, W)``.
            phase: ``"train"`` or ``"val"``.
            dataset_name: Source dataset name for augmentation routing.

        Returns:
            Tuple ``(logits, model_input, model_label)``.
        """

        image, label = self._apply_augmentation(image, label, phase, dataset_name)
        image = self.normalizer(image)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype or torch.bfloat16,
            enabled=self.amp_dtype is not None,
        ):
            logits = self.forward_model(image)
        return logits, image, label

    def _prediction_step(self, batch: NamedBatch, *, phase: str) -> dict[str, Any]:
        """Run a loss-free prediction step for experiment-coupled callbacks.

        Args:
            batch: Named X-ray or CT batch.
            phase: Phase used for augmentation selection.

        Returns:
            Detached inputs, logits, labels, and dataset identity.
        """

        dataset_name = batch.source_name
        inputs = _to_device(
            resolve_batch_inputs(batch.batch, modality=batch.modality), self.device
        )
        image, label = self._model_inputs(inputs, dataset_name=dataset_name)
        logits, model_input, model_label = self._forward_without_loss(
            image,
            label,
            phase=phase,
            dataset_name=dataset_name,
        )
        return {
            "x": model_input.detach(),
            "y_pred": logits.detach(),
            "y_true": model_label.detach(),
            "dataset_name": dataset_name,
        }

    def run_step(
        self, batch: NamedBatch, *, phase: str, backward: bool
    ) -> dict[str, Any]:
        """Run one forward/backward step for an X-ray or CT batch.

        Args:
            batch: Named batch produced by the mixed/sequential loaders.
            phase: ``"train"`` or ``"val"``.
            backward: Whether to backpropagate and step the optimizer.

        Returns:
            Detached step outputs used for metric computation.
        """

        dataset_name = batch.source_name
        inputs = _to_device(
            resolve_batch_inputs(batch.batch, modality=batch.modality), self.device
        )
        image, label = self._model_inputs(inputs, dataset_name=dataset_name)

        loss, logits, model_input, model_label = self.forward_pass(
            image, label, phase=phase, dataset_name=dataset_name
        )
        if backward:
            scale_before_step = self.grad_scaler.get_scale()
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optim)
            self.grad_scaler.update()
            optimizer_updated = self.grad_scaler.get_scale() >= scale_before_step
            if optimizer_updated:
                self.ema.update_after_optimizer_step(
                    completed_optimizer_steps=self._global_step + 1
                )
                self._step_scheduler_on_batch_end()
            self.optim.zero_grad(set_to_none=True)

        outputs = {
            "loss": loss.detach(),
            "x": model_input.detach(),
            "y_pred": logits.detach(),
            "y_true": model_label.detach(),
            "dataset_name": dataset_name,
        }
        self.run_callbacks("step", batch=outputs)
        return outputs

    def compute_metrics(self, outputs: dict[str, Any]) -> dict[str, float]:
        """Compute scalar loss and foreground Dice for one step.

        Args:
            outputs: Detached loss, logits, and labels from :meth:`run_step`.

        Returns:
            Scalar ``loss`` and foreground ``dice`` values.
        """
        loss = outputs["loss"]
        loss_scalar = float(loss.mean().item() if loss.ndim > 0 else loss.item())
        dice = dice_score(
            outputs["y_pred"],
            outputs["y_true"],
            mode="binary",
            from_logits=True,
            ignore_background=True,
        )
        return {"loss": loss_scalar, "dice": float(dice.item())}

    def _metric_sample_counts(self, outputs: dict[str, Any]) -> dict[str, int]:
        """Count the target-bearing images represented by the Dice mean.

        Args:
            outputs: Detached predictions and targets returned by ``run_step``.

        Returns:
            Dice's eligible image count, or no override for custom step outputs
            without a target tensor. Loss retains its full batch weight.
        """

        y_true = outputs.get("y_true")
        if not isinstance(y_true, Tensor):
            return {}
        return {"dice": binary_dice_sample_count(y_true)}


def _dataset_channel_label_names(dataset: Any) -> tuple[str, ...] | None:
    """Read a 2D dataset's stored label-name order from storage metadata.

    Backend-wide names are preferred because packaged xray-seg datasets declare
    one label encoding for the entire package. The names may correspond to
    dense native ids or channel positions. Record names remain supported for
    existing storage.

    Args:
        dataset: Runtime training dataset.

    Returns:
        Ordered source label names, or ``None`` when no names are declared.

    Raises:
        ValueError: If records declare more than one label order.
    """

    backend = getattr(dataset, "backend", None)
    names = getattr(backend, "label_names", None)
    if names is not None:
        return tuple(str(name) for name in names)

    record_orders: set[tuple[str, ...]] = set()
    for record in getattr(dataset, "records", ()):
        raw_names = getattr(record, "label_names", None)
        if raw_names is not None:
            record_orders.add(tuple(str(name) for name in raw_names))
    if len(record_orders) > 1:
        raise ValueError("A 2D dataset must use one label order across all records.")
    if record_orders:
        return next(iter(record_orders))
    return None


def _excluded_datasets(proportions: dict[str, Any]) -> set[str]:
    """Return dataset names whose mixing weight is zero (dropped from train/val).

    Args:
        proportions: Mapping from dataset name to a numeric sampling weight.

    Returns:
        Set of dataset names configured with weight ``0``.

    Raises:
        ValueError: If any weight is negative, non-numeric, or non-finite.
    """

    excluded: set[str] = set()
    for name, raw_weight in proportions.items():
        weight = normalize_proportion_weight(
            raw_weight,
            f"dataloader.proportions[{name!r}]",
        )
        if weight == 0:
            excluded.add(name)
    return excluded


def _to_device(inputs: BatchInputs, device: torch.device) -> BatchInputs:
    """Move all populated tensors of a batch-input bundle onto ``device``."""

    def move(tensor: Tensor | None) -> Tensor | None:
        return None if tensor is None else tensor.to(device, non_blocking=True)

    return BatchInputs(
        modality=inputs.modality,
        image=inputs.image.to(device, non_blocking=True),
        label=inputs.label.to(device, non_blocking=True),
        affine=move(inputs.affine),
        fg_centroids_ijk=move(inputs.fg_centroids_ijk),
        fg_centroid_label_ids=move(inputs.fg_centroid_label_ids),
    )


def supervised_foreground_centroids(
    inputs: BatchInputs, projection: TrainingLabelProjection
) -> Tensor | None:
    """Keep only centroids whose native label the protocol supervises.

    Without label ids (runtime-computed centroids of an already-projected label)
    the centroids pass through unchanged.
    """

    centroids = inputs.fg_centroids_ijk
    if centroids is None or inputs.fg_centroid_label_ids is None:
        return centroids
    supervised = torch.as_tensor(
        projection.native_foreground_label_ids(),
        dtype=torch.long,
        device=inputs.fg_centroid_label_ids.device,
    )
    keep = torch.isin(inputs.fg_centroid_label_ids.long(), supervised)
    return centroids[keep]


