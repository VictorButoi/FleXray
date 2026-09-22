from __future__ import annotations

import operator
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from fxr.protocols import load_dataset_spec_by_name

from .remap import (
    apply_label_lut,
    channelize_integer_mask,
    compile_training_label_remap_by_name,
    normalize_training_channel_label_names,
    project_channel_mask,
)
from .sampling import SampleWeightingSpec, inverse_label_frequency_weights
from .schemas import DatasetRecord, LabelMode, TrainingLabelRemap, TrainingSample
from .storage import ManifestStorageBackend, StorageBackend, ThunderDBStorageBackend

AIR_HU = -1000.0
CTCropMode = Literal["none", "random", "random_slices"]


class _BaseTrainingDataset(Dataset):
    """Shared training dataset implementation for storage-backed records.

    Attributes:
        modality: Modality string used to filter compatible records.
        backend: Storage backend used to load image, label, and metadata
            payloads.
        protocol_name: Optional protocol name used for automatic label remaps.
        dataset_name: Optional dataset name used for filtering and remapping.
        label_mode: Whether emitted labels remain ``"native"`` or are remapped
            to ``"model"`` labels.
        return_native_label: Whether model-mode samples include the original
            native label tensor.
        return_data_id: Whether emitted samples include ``data_id``.
        return_metadata: Whether emitted samples include metadata dictionaries.
        require_seg: Whether records without labels are rejected.
        records: Filtered immutable record sequence backing this dataset.
        label_remap: Optional compiled remap used in model label mode.
        _config_root: Optional config root used when channel-mask labels
            need dataset-spec lookup by native label name.
    """

    modality = "unknown"

    def __init__(
        self,
        records: Iterable[DatasetRecord] | None = None,
        *,
        backend: StorageBackend | None = None,
        protocol_name: str | None = None,
        dataset_name: str | None = None,
        model_label_names: Sequence[str] | None = None,
        label_mode: LabelMode = "model",
        label_remap: TrainingLabelRemap | None = None,
        split: str | None = None,
        require_seg: bool = True,
        exclude_subjects: Sequence[str] = (),
        num_subjects: int | None = None,
        return_native_label: bool = False,
        return_data_id: bool = True,
        return_metadata: bool = True,
        config_root: str | Path | None = None,
    ) -> None:
        """Initialize common record filtering and label-remap state.

        Args:
            records: Optional explicit records. When omitted, records are read
                from ``backend.records``.
            backend: Storage backend that loads payloads referenced by records.
            protocol_name: Protocol used to compile automatic label remaps.
            dataset_name: Dataset name used for filtering and remap lookup.
            model_label_names: Optional model label subset or ordering for
                automatic remap compilation.
            label_mode: ``"model"`` to emit remapped labels or ``"native"`` to
                emit source labels unchanged.
            label_remap: Optional precompiled remap for model label mode.
            split: Optional split name used to filter records.
            require_seg: Whether records without labels are excluded or rejected.
            exclude_subjects: Subject keys to drop before sampling.
            num_subjects: Optional maximum number of distinct subjects to keep.
            return_native_label: Whether to include native labels alongside
                model labels.
            return_data_id: Whether to include sample data ids.
            return_metadata: Whether to include metadata dictionaries.
            config_root: Optional config root for protocol and dataset lookup.

        Returns:
            ``None``.

        Raises:
            ValueError: If label mode, backend, records, subject count, or
                remap settings are invalid.
        """

        if label_mode not in {"model", "native"}:
            raise ValueError("label_mode must be 'model' or 'native'.")
        if label_mode == "native" and label_remap is not None:
            raise ValueError("label_mode='native' rejects label_remap.")
        if records is None:
            if backend is None:
                raise ValueError("records or backend must be provided.")
            records = backend.records
        if backend is None:
            raise ValueError("backend must be provided.")
        self.backend = backend
        self.protocol_name = protocol_name
        self.dataset_name = dataset_name
        self.label_mode = label_mode
        self.return_native_label = return_native_label
        self.return_data_id = return_data_id
        self.return_metadata = return_metadata
        self.require_seg = require_seg
        self._config_root = config_root
        self.records = self._filter_records(
            tuple(records),
            split=split,
            dataset_name=dataset_name,
            exclude_subjects=exclude_subjects,
            num_subjects=num_subjects,
            require_seg=require_seg,
            config_root=config_root,
        )
        self.label_remap = self._resolve_remap(
            label_remap,
            protocol_name=protocol_name,
            dataset_name=dataset_name,
            model_label_names=model_label_names,
            config_root=config_root,
        )

    @classmethod
    def from_manifest(
        cls,
        manifest: str | Path | dict[str, Any] | list[dict[str, Any]],
        **kwargs: Any,
    ) -> "_BaseTrainingDataset":
        """Construct a training dataset from manifest-backed storage.

        Args:
            manifest: Manifest path, manifest mapping, or list of record
                mappings accepted by ``ManifestStorageBackend``.
            **kwargs: Additional dataset constructor arguments.

        Returns:
            Dataset instance backed by the manifest records.
        """

        backend = ManifestStorageBackend(manifest)
        return cls(records=backend.records, backend=backend, **kwargs)

    @classmethod
    def from_thunderdb(cls, db: Any, **kwargs: Any) -> "_BaseTrainingDataset":
        """Construct a training dataset from ThunderDB-backed storage.

        Args:
            db: Open ThunderDB-like object, compatible key/value object, or path
                accepted by ``ThunderDBStorageBackend``.
            **kwargs: Additional dataset constructor arguments.

        Returns:
            Dataset instance backed by ThunderDB records.
        """

        backend = ThunderDBStorageBackend(db)
        try:
            return cls(records=backend.records, backend=backend, **kwargs)
        except BaseException:
            backend.close()
            raise

    def __len__(self) -> int:
        """Return the number of filtered records in the dataset.

        Returns:
            Dataset sample count.
        """

        return len(self.records)

    def close(self) -> None:
        """Release storage resources owned by this dataset backend.

        Returns:
            ``None``. Repeated calls are safe when the backend supports close.
        """

        close = getattr(self.backend, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "_BaseTrainingDataset":
        """Return this dataset for context-managed use.

        Returns:
            This open dataset.
        """

        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Release owned storage when leaving a context manager.

        Args:
            exc_type: Exception type raised in the context, if any.
            exc_value: Exception value raised in the context, if any.
            traceback: Traceback raised in the context, if any.

        Returns:
            ``None``.
        """

        del exc_type, exc_value, traceback
        self.close()

    def __getitem__(self, index: int) -> TrainingSample:
        """Load one training sample from storage.

        Args:
            index: Filtered record index to load.

        Returns:
            Training sample containing image, source metadata, and optional
            label fields according to the dataset configuration.

        Raises:
            ValueError: If a required label payload is missing.
        """

        record = self.records[index]
        image = _image_tensor(self.backend.load(record, "image"))
        sample: TrainingSample = {
            "image": image,
            "dataset_name": record.dataset_name,
            "modality": record.modality,
        }
        if self.return_data_id:
            sample["data_id"] = record.data_id
        metadata = dict(record.metadata)
        self._add_modality_metadata(record, metadata)
        if self.return_metadata:
            sample["metadata"] = metadata
        if record.label is not None:
            native_label = self._native_label(record)
            if self.label_mode == "native":
                sample["label"] = native_label
            else:
                sample["label"] = self._model_label(record, native_label)
                if self.return_native_label:
                    sample["native_label"] = native_label
        elif self.require_seg:
            raise ValueError(f"Record {record.data_id!r} does not define a label.")
        return sample

    def _filter_records(
        self,
        records: tuple[DatasetRecord, ...],
        *,
        split: str | None,
        dataset_name: str | None,
        exclude_subjects: Sequence[str],
        num_subjects: int | None,
        require_seg: bool,
        config_root: str | Path | None,
    ) -> tuple[DatasetRecord, ...]:
        """Filter records by modality, dataset, split, labels, and subjects.

        Args:
            records: Candidate records to filter.
            split: Optional split name to keep.
            dataset_name: Optional dataset name or alias to keep.
            exclude_subjects: Subject keys to remove.
            num_subjects: Optional maximum number of distinct subjects to keep.
            require_seg: Whether to reject selections without labels.
            config_root: Optional config root for dataset spec lookup.

        Returns:
            Filtered record tuple.

        Raises:
            ValueError: If ``num_subjects`` is negative or dataset lookup fails.
        """

        filtered = [record for record in records if record.modality == self.modality]
        canonical_dataset_name = dataset_name
        dataset_spec = None
        if dataset_name is not None:
            try:
                dataset_spec = load_dataset_spec_by_name(
                    dataset_name, config_root=config_root
                )
            except FileNotFoundError:
                dataset_spec = None
            if dataset_spec is not None:
                canonical_dataset_name = dataset_spec.dataset_name
            allowed_names = {dataset_name, canonical_dataset_name}
            filtered = [
                record for record in filtered if record.dataset_name in allowed_names
            ]
        if split is not None:
            filtered = [record for record in filtered if record.split == split]
        if require_seg:
            missing_label_ids = [
                record.data_id for record in filtered if record.label is None
            ]
            filtered = [record for record in filtered if record.label is not None]
            if missing_label_ids and not filtered:
                raise ValueError(
                    "require_seg=True but all matching records lack labels: "
                    f"{missing_label_ids}."
                )
        excluded = set(str(subject) for subject in exclude_subjects)
        if dataset_spec is not None and dataset_spec.skip_subjects is not None:
            excluded.update(dataset_spec.skip_subjects)
        if excluded:
            filtered = [
                record for record in filtered if _subject_key(record) not in excluded
            ]
        if num_subjects is not None:
            if num_subjects < 0:
                raise ValueError("num_subjects must be non-negative.")
            keep_subjects: list[str] = []
            for record in filtered:
                subject = _subject_key(record)
                if subject not in keep_subjects:
                    keep_subjects.append(subject)
                if len(keep_subjects) >= num_subjects:
                    break
            keep = set(keep_subjects)
            filtered = [record for record in filtered if _subject_key(record) in keep]
        return tuple(filtered)

    def _resolve_remap(
        self,
        label_remap: TrainingLabelRemap | None,
        *,
        protocol_name: str | None,
        dataset_name: str | None,
        model_label_names: Sequence[str] | None,
        config_root: str | Path | None,
    ) -> TrainingLabelRemap | None:
        """Resolve the remap required for model label mode.

        Args:
            label_remap: Optional explicit remap supplied by the caller.
            protocol_name: Protocol name for automatic remap compilation.
            dataset_name: Dataset name for automatic remap compilation.
            model_label_names: Optional model label subset or ordering.
            config_root: Optional config root for protocol and dataset lookup.

        Returns:
            ``None`` for native label mode, otherwise a compiled training remap.

        Raises:
            ValueError: If model label mode lacks enough information to compile
                a remap.
        """

        if self.label_mode == "native":
            return None
        if label_remap is not None:
            return label_remap
        if protocol_name is None or dataset_name is None:
            raise ValueError(
                "label_mode='model' requires label_remap or protocol_name and dataset_name."
            )
        return compile_training_label_remap_by_name(
            protocol_name,
            dataset_name,
            model_label_names,
            config_root=config_root,
        )

    def _native_label(self, record: DatasetRecord) -> torch.Tensor:
        """Load a dense integer native label tensor for a record.

        Args:
            record: Record whose label payload should be loaded.

        Returns:
            Native label tensor with ``torch.long`` dtype.
        """

        return torch.as_tensor(
            _tensor_source(self.backend.load(record, "label")), dtype=torch.long
        )

    def _model_label(
        self,
        record: DatasetRecord,
        native_label: torch.Tensor,
    ) -> torch.Tensor:
        """Map a native dense label tensor into model-label ids.

        Args:
            record: Record associated with ``native_label``.
            native_label: Dataset-native integer label tensor.

        Returns:
            Dense label tensor containing model-label ids.

        Raises:
            ValueError: If model label mode has no compiled remap.
        """

        if self.label_remap is None:
            raise ValueError("Model label mode requires a label remap.")
        return apply_label_lut(native_label, self.label_remap)

    def _add_modality_metadata(
        self,
        record: DatasetRecord,
        metadata: dict[str, Any],
    ) -> None:
        """Allow subclasses to add modality-specific metadata in place.

        Args:
            record: Record being emitted.
            metadata: Mutable metadata dictionary copied from the record.

        Returns:
            ``None``. The default implementation leaves metadata unchanged.
        """

        return None


class XrayTrainingDataset(_BaseTrainingDataset):
    """Torch dataset for dense-map or named-channel X-ray segmentation.

    Attributes:
        modality: Fixed modality filter value ``"xray"``.
    """

    modality = "xray"

    def __getitem__(self, index: int) -> TrainingSample:
        """Load and spatially validate one channel-first X-ray sample.

        Args:
            index: Filtered record index to load.

        Returns:
            Training sample whose image has shape ``(C, H, W)`` and whose
            label uses either dense ``(H, W)`` or channel-first ``(C, H, W)``
            representation.

        Raises:
            ValueError: If image or label dimensionality is invalid, or if
                their spatial shapes differ.
        """

        sample = super().__getitem__(index)
        image = sample["image"]
        if image.ndim != 3:
            raise ValueError(
                f"X-ray record {self.records[index].data_id!r} image must be "
                f"2D or channel-first 3D, got shape {tuple(image.shape)}."
            )
        label = sample.get("native_label", sample.get("label"))
        if label is not None and tuple(image.shape[-2:]) != tuple(label.shape[-2:]):
            raise ValueError(
                f"X-ray record {self.records[index].data_id!r} image and label "
                f"spatial shapes differ: {tuple(image.shape[-2:])} versus "
                f"{tuple(label.shape[-2:])}."
            )
        return sample

    def _native_label(self, record: DatasetRecord) -> torch.Tensor:
        """Load a dense integer map or named channel-first X-ray mask.

        Args:
            record: X-ray record whose label payload should be loaded.

        Returns:
            Dense ``torch.long`` map for a two-dimensional payload, or a
            floating-point channel mask for a three-dimensional payload.

        Raises:
            ValueError: If dimensionality, channel count, values, or dense ids
                are invalid.
        """

        raw_label = torch.as_tensor(
            _tensor_source(self.backend.load(record, "label"))
        )
        if raw_label.is_complex():
            raise ValueError("X-ray labels must use real numeric values.")
        declared = getattr(self.backend, "seg_storage", None)
        if declared is not None:
            expected_ndim = 3 if declared == "overlapping_channel_mask" else 2
            if raw_label.ndim != expected_ndim:
                raise ValueError(
                    f"X-ray record {record.data_id!r} declares seg_storage {declared!r} "
                    f"but its label has rank {raw_label.ndim} (expected {expected_ndim})."
                )
        if raw_label.ndim == 3:
            if record.label_names is None:
                raise ValueError(
                    f"X-ray record {record.data_id!r} with a channel-first mask "
                    "must declare label_names."
                )
            label = raw_label.to(dtype=torch.float32)
            if label.shape[0] != len(record.label_names):
                raise ValueError(
                    f"X-ray record {record.data_id!r} has {label.shape[0]} label "
                    f"channels but {len(record.label_names)} label names."
                )
            if not torch.isfinite(label).all() or torch.any(
                (label < 0) | (label > 1)
            ):
                raise ValueError(
                    "X-ray channel masks must contain finite values in [0, 1]."
                )
            return label
        if raw_label.ndim != 2:
            raise ValueError(
                f"X-ray record {record.data_id!r} label must use a dense 2D "
                "integer map or a named channel-first 3D mask."
            )
        if raw_label.is_floating_point() and not torch.equal(
            raw_label, torch.round(raw_label)
        ):
            raise ValueError("X-ray dense labels must contain integer-valued ids.")
        if torch.any(raw_label < 0):
            raise ValueError("X-ray dense labels cannot contain negative ids.")
        if record.label_names is not None and raw_label.numel() > 0:
            max_id = int(raw_label.max().item())
            if max_id >= len(record.label_names):
                raise ValueError(
                    f"X-ray record {record.data_id!r} contains dense label id "
                    f"{max_id}, but only {len(record.label_names)} ordered label "
                    "names were declared."
                )
        return raw_label.to(dtype=torch.long)

    def _model_label(
        self,
        record: DatasetRecord,
        native_label: torch.Tensor,
    ) -> torch.Tensor:
        """Project an X-ray dense map or named channels into model masks.

        Args:
            record: X-ray record associated with ``native_label``.
            native_label: Dataset-native dense map or channel-first mask.

        Returns:
            Channel-first mask tensor with one channel per model class.

        Raises:
            ValueError: If model label mode has no compiled remap.
        """

        if self.label_remap is None:
            raise ValueError("Model label mode requires a label remap.")
        if native_label.ndim == 3:
            if record.label_names is None:
                raise ValueError(
                    "Channel-first X-ray labels require ordered label_names."
                )
            source_label_names = record.label_names
            if self.protocol_name is not None and self.dataset_name is not None:
                source_label_names = normalize_training_channel_label_names(
                    self.protocol_name,
                    self.dataset_name,
                    source_label_names,
                    config_root=self._config_root,
                )
            return project_channel_mask(
                native_label,
                source_label_names,
                self.label_remap.label_names,
            )
        dense = super()._model_label(record, native_label)
        return channelize_integer_mask(dense, self.label_remap.num_classes)


class CTTrainingDataset(_BaseTrainingDataset):
    """Torch dataset for trainable CT segmentation volumes.

    Attributes:
        modality: Fixed modality filter value ``"ct"``.
        crop_mode: Validated CT crop mode.
        crop_size: Optional ``(x, y, z)`` crop size used for random crops.
        generator: Optional torch random generator used for deterministic crop
            offsets.
        hu_min: Minimum HU value applied to emitted image tensors.
        air_clamp_hu: Optional clamp value applied to air-like voxels.
        compute_fg_centroids: Whether emitted metadata includes foreground
            centroids computed from the returned label tensor. When false,
            stored crop metadata (``fg_centroids_ijk`` aligned with
            ``crop_foreground_label_ids``) is emitted instead when present.
        sample_weighting: Resolved per-crop sampling weighting spec.
    """

    modality = "ct"

    def __init__(
        self,
        *args: Any,
        crop_mode: CTCropMode = "none",
        crop_size: Sequence[int] | None = None,
        generator: torch.Generator | None = None,
        hu_min: float = AIR_HU,
        air_clamp_hu: float | None = None,
        compute_fg_centroids: bool = False,
        sample_weighting: str | Mapping[str, Any] = "uniform",
        **kwargs: Any,
    ) -> None:
        """Initialize CT-specific cropping and HU preprocessing settings.

        Args:
            *args: Positional arguments forwarded to ``_BaseTrainingDataset``.
            crop_mode: ``"none"``, ``"random"``, or ``"random_slices"``.
            crop_size: Required ``(x, y, z)`` crop size when cropping is enabled.
            generator: Optional torch generator for random crop offsets.
            hu_min: Minimum HU value after preprocessing.
            air_clamp_hu: Optional HU threshold below which voxels are clamped
                to ``hu_min``.
            compute_fg_centroids: Whether to add foreground centroids to sample
                metadata when labels are present.
            sample_weighting: ``"uniform"`` or an inverse-label-frequency spec
                (see :class:`fxr.datasets.SampleWeightingSpec`).
            **kwargs: Keyword arguments forwarded to ``_BaseTrainingDataset``.

        Returns:
            ``None``.

        Raises:
            ValueError: If crop settings are invalid or CT records lack affine
                or spacing payload references.
        """

        self.crop_mode = _validate_ct_crop_mode(crop_mode)
        self.crop_size = _validate_ct_crop_size(self.crop_mode, crop_size)
        self.generator = generator
        self.hu_min = float(hu_min)
        self.air_clamp_hu = None if air_clamp_hu is None else float(air_clamp_hu)
        self.compute_fg_centroids = bool(compute_fg_centroids)
        super().__init__(*args, **kwargs)
        self.sample_weighting = SampleWeightingSpec.parse(
            sample_weighting, dataset_name=str(self.dataset_name)
        )
        for record in self.records:
            if record.affine is None or record.spacing is None:
                raise ValueError(
                    f"CT record {record.data_id!r} must define affine and spacing payloads."
                )

    def sample_weights(self, label_lut: Sequence[int]) -> torch.Tensor | None:
        """Return per-record sampling weights for the configured scheme.

        Args:
            label_lut: Native-id to model-channel lookup table of the active
                protocol, used to map stored ``crop_foreground_label_ids``.

        Returns:
            ``None`` for uniform sampling, otherwise a float tensor with one
            weight per record (mean ``1.0``).
        """

        if self.sample_weighting.scheme == "uniform":
            return None
        attrs = getattr(self.backend, "attrs", {})
        assert attrs.get("storage_layout") == "crops", (
            f"CT dataset {self.dataset_name!r} sample_weighting requires a crop-backed "
            "ThunderDB (_attrs.storage_layout == 'crops') with crop label metadata."
        )
        fg_label_ids = [
            _required_crop_metadata(record, "crop_foreground_label_ids")
            for record in self.records
        ]
        return inverse_label_frequency_weights(
            fg_label_ids,
            label_lut,
            tau=self.sample_weighting.tau,
            class_aggregation=self.sample_weighting.class_aggregation,
        )

    def __getitem__(self, index: int) -> TrainingSample:
        """Load one CT sample with optional cropping and HU preprocessing.

        Args:
            index: Filtered record index to load.

        Returns:
            Training sample containing a CT image tensor, source metadata, and
            optional label fields according to dataset configuration.

        Raises:
            ValueError: If a required label is missing or CT image and label
                spatial shapes differ.
        """

        record = self.records[index]
        image = _ct_tensor(
            self.backend.load(record, "image"),
            dtype=torch.float32,
            record=record,
            field="image",
        )
        offset = (0, 0, 0)
        native_label: torch.Tensor | None = None
        if record.label is not None:
            native_label = _ct_tensor(
                self.backend.load(record, "label"),
                dtype=torch.long,
                record=record,
                field="label",
            )
            _validate_ct_label_shape(record, image, native_label)
        elif self.require_seg:
            raise ValueError(f"Record {record.data_id!r} does not define a label.")

        if self.crop_size is not None:
            offset = _random_ct_crop_offset(
                image.shape[-3:],
                self.crop_size,
                generator=self.generator,
            )
            image = _crop_or_pad_ct_tensor(
                image,
                crop_size=self.crop_size,
                offset=offset,
                fill_value=AIR_HU,
            )
            if native_label is not None:
                native_label = _crop_or_pad_ct_tensor(
                    native_label,
                    crop_size=self.crop_size,
                    offset=offset,
                    fill_value=0,
                )

        image = _apply_ct_hu_bounds(
            image,
            hu_min=self.hu_min,
            air_clamp_hu=self.air_clamp_hu,
        )
        metadata = dict(record.metadata)
        self._add_modality_metadata(record, metadata, offset=offset)

        sample: TrainingSample = {
            "image": image,
            "dataset_name": record.dataset_name,
            "modality": record.modality,
        }
        if self.return_data_id:
            sample["data_id"] = record.data_id
        if native_label is not None:
            if self.label_mode == "native":
                label = native_label
            else:
                label = self._model_label(record, native_label)
                if self.return_native_label:
                    sample["native_label"] = native_label
            sample["label"] = label
            if self.compute_fg_centroids:
                metadata["fg_centroids_ijk"] = _foreground_centroids_ijk(label)
            elif "fg_centroids_ijk" in record.metadata:
                assert self.crop_size is None, (
                    f"CT record {record.data_id!r} carries stored crop centroids; "
                    "runtime cropping cannot be combined with a crop-backed layout."
                )
                metadata["fg_centroids_ijk"] = _stored_centroids_ijk(record)
        if self.return_metadata:
            sample["metadata"] = metadata
        return sample

    def _add_modality_metadata(
        self,
        record: DatasetRecord,
        metadata: dict[str, Any],
        *,
        offset: Sequence[int] = (0, 0, 0),
    ) -> None:
        """Add CT affine and spacing metadata, offset for any crop.

        Args:
            record: CT record being emitted.
            metadata: Mutable metadata dictionary copied from the record.
            offset: Applied crop offset in ``(x, y, z)`` order.

        Returns:
            ``None``. ``metadata`` is modified in place.
        """

        metadata["affine"] = _offset_ct_affine(
            self.backend.load(record, "affine"), offset
        )
        metadata["spacing"] = self.backend.load(record, "spacing")


def _tensor_source(raw_tensor: Any) -> Any:
    """Return an array-like object suitable for warning-free tensor conversion.

    Args:
        raw_tensor: Array-like payload loaded from storage.

    Returns:
        ``raw_tensor`` unless it is a non-writable NumPy array, in which case a
        writable copy is returned.
    """

    if isinstance(raw_tensor, np.ndarray) and not raw_tensor.flags.writeable:
        return raw_tensor.copy()
    return raw_tensor


def _image_tensor(raw_image: Any) -> torch.Tensor:
    """Convert an image payload to a float tensor with an explicit channel.

    Args:
        raw_image: Array-like image payload loaded from storage.

    Returns:
        Float tensor. Two-dimensional inputs are promoted to ``(1, H, W)``.
    """

    image = torch.as_tensor(_tensor_source(raw_image), dtype=torch.float32)
    if image.ndim == 2:
        image = image.unsqueeze(0)
    return image


def _validate_ct_crop_mode(crop_mode: str) -> CTCropMode:
    """Validate a CT crop mode string.

    Args:
        crop_mode: Requested crop mode.

    Returns:
        Crop mode narrowed to the supported literal type.

    Raises:
        ValueError: If ``crop_mode`` is unsupported.
    """

    if crop_mode not in {"none", "random", "random_slices"}:
        raise ValueError(
            "crop_mode must be one of 'none', 'random', or 'random_slices'."
        )
    return crop_mode  # type: ignore[return-value]


def _validate_ct_crop_size(
    crop_mode: CTCropMode,
    crop_size: Sequence[int] | None,
) -> tuple[int, int, int] | None:
    """Validate CT crop size settings for the selected crop mode.

    Args:
        crop_mode: Validated CT crop mode.
        crop_size: Optional requested ``(x, y, z)`` crop size.

    Returns:
        Three positive integer crop sizes when cropping is enabled, otherwise
        ``None``.

    Raises:
        ValueError: If cropping is enabled without exactly three positive
            integer sizes.
    """

    if crop_mode == "none":
        return None
    if crop_size is None:
        raise ValueError(f"crop_mode={crop_mode!r} requires crop_size.")
    try:
        values = tuple(crop_size)
    except TypeError as exc:
        raise ValueError(
            "CT crop_size must contain exactly 3 values: (x, y, z)."
        ) from exc
    if len(values) != 3:
        raise ValueError("CT crop_size must contain exactly 3 values: (x, y, z).")
    size: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("CT crop_size values must be positive integers.")
        try:
            integer = operator.index(value)
        except TypeError as exc:
            raise ValueError("CT crop_size values must be positive integers.") from exc
        if integer <= 0:
            raise ValueError("CT crop_size values must be positive integers.")
        size.append(integer)
    return (size[0], size[1], size[2])


def _ct_tensor(
    raw_tensor: Any,
    *,
    dtype: torch.dtype,
    record: DatasetRecord,
    field: str,
) -> torch.Tensor:
    """Convert a CT payload to a channel-first tensor.

    Args:
        raw_tensor: Array-like CT image or label payload.
        dtype: Torch dtype for the returned tensor.
        record: Record used for validation context.
        field: Payload field name used in validation errors.

    Returns:
        Tensor shaped as ``(C, X, Y, Z)``. Three-dimensional payloads are
        promoted to one channel.

    Raises:
        ValueError: If the payload is not 3D or channel-first 4D.
    """

    tensor = torch.as_tensor(_tensor_source(raw_tensor), dtype=dtype)
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    if tensor.ndim == 4:
        return tensor
    raise ValueError(
        f"CT record {record.data_id!r} {field} must be 3D or channel-first 4D, "
        f"got shape {tuple(tensor.shape)}."
    )


def _validate_ct_label_shape(
    record: DatasetRecord,
    image: torch.Tensor,
    label: torch.Tensor,
) -> None:
    """Validate that CT image and label spatial shapes match.

    Args:
        record: Record used for validation context.
        image: Channel-first CT image tensor.
        label: Channel-first CT label tensor.

    Returns:
        ``None``.

    Raises:
        ValueError: If image and label spatial shapes differ.
    """

    if image.shape[-3:] != label.shape[-3:]:
        raise ValueError(
            f"CT record {record.data_id!r} image and label spatial shapes must match, "
            f"got {tuple(image.shape[-3:])} and {tuple(label.shape[-3:])}."
        )


def _random_ct_crop_offset(
    spatial_shape: Sequence[int],
    crop_size: Sequence[int],
    *,
    generator: torch.Generator | None,
) -> tuple[int, int, int]:
    """Sample a CT crop or pad offset for each spatial axis.

    Args:
        spatial_shape: Source spatial shape in ``(x, y, z)`` order.
        crop_size: Target crop size in ``(x, y, z)`` order.
        generator: Optional torch generator for deterministic sampling.

    Returns:
        Offset tuple. Negative values indicate padding before the source volume.
    """

    offsets: list[int] = []
    for dim, size in zip(spatial_shape, crop_size, strict=True):
        dim = int(dim)
        size = int(size)
        if size <= dim:
            choices = dim - size + 1
            start = int(torch.randint(0, choices, (), generator=generator).item())
            offsets.append(start)
        else:
            pad_before_choices = size - dim + 1
            pad_before = int(
                torch.randint(0, pad_before_choices, (), generator=generator).item()
            )
            offsets.append(-pad_before)
    return (offsets[0], offsets[1], offsets[2])


def _crop_or_pad_ct_tensor(
    tensor: torch.Tensor,
    *,
    crop_size: Sequence[int],
    offset: Sequence[int],
    fill_value: float | int,
) -> torch.Tensor:
    """Crop or pad a channel-first CT tensor to a target size.

    Args:
        tensor: Channel-first CT tensor with three trailing spatial dimensions.
        crop_size: Target ``(x, y, z)`` shape.
        offset: Source offset where positive values crop from the source and
            negative values pad before the source.
        fill_value: Value used for padded voxels.

    Returns:
        Tensor with trailing spatial shape equal to ``crop_size``.
    """

    target = tensor.new_full((*tensor.shape[:-3], *crop_size), fill_value)
    src_slices: list[slice] = [slice(None)] * (tensor.ndim - 3)
    dst_slices: list[slice] = [slice(None)] * (tensor.ndim - 3)
    for dim, size, axis_offset in zip(
        tensor.shape[-3:], crop_size, offset, strict=True
    ):
        src_start = max(int(axis_offset), 0)
        dst_start = max(-int(axis_offset), 0)
        length = min(int(dim) - src_start, int(size) - dst_start)
        src_slices.append(slice(src_start, src_start + length))
        dst_slices.append(slice(dst_start, dst_start + length))
    target[tuple(dst_slices)] = tensor[tuple(src_slices)]
    return target


def _offset_ct_affine(raw_affine: Any, offset: Sequence[int]) -> Any:
    """Shift a CT affine origin by an applied crop offset.

    Args:
        raw_affine: Original ``(4, 4)`` voxel-to-world affine.
        offset: Crop offset in ``(x, y, z)`` order.

    Returns:
        Affine with translated origin, preserving tensor or NumPy return type
        where possible. Zero offsets return ``raw_affine`` unchanged.

    Raises:
        ValueError: If a non-zero-offset affine is not shaped ``(4, 4)``.
    """

    if all(int(axis_offset) == 0 for axis_offset in offset):
        return raw_affine
    affine = torch.as_tensor(_tensor_source(raw_affine), dtype=torch.float32).clone()
    if affine.shape != (4, 4):
        raise ValueError(
            f"CT affine must have shape (4, 4), got {tuple(affine.shape)}."
        )
    offset_vector = torch.tensor(
        [int(offset[0]), int(offset[1]), int(offset[2]), 1.0],
        dtype=affine.dtype,
        device=affine.device,
    )
    affine[:3, 3] = (affine @ offset_vector)[:3]
    if isinstance(raw_affine, torch.Tensor):
        return affine.to(dtype=raw_affine.dtype)
    if isinstance(raw_affine, np.ndarray):
        return affine.cpu().numpy().astype(raw_affine.dtype, copy=False)
    return affine


def _apply_ct_hu_bounds(
    image: torch.Tensor,
    *,
    hu_min: float,
    air_clamp_hu: float | None,
) -> torch.Tensor:
    """Apply CT HU lower bounds and optional air restoration.

    Args:
        image: CT image tensor in HU-like units.
        hu_min: Minimum HU value after preprocessing.
        air_clamp_hu: Optional threshold whose voxels are restored to air HU
            after lower-bound clamping.

    Returns:
        Preprocessed CT image tensor.
    """

    air_mask = None if air_clamp_hu is None else image <= air_clamp_hu
    image = torch.clamp_min(image, hu_min)
    if air_mask is not None:
        image = image.masked_fill(air_mask, AIR_HU)
    return image


def _required_crop_metadata(record: DatasetRecord, key: str) -> Any:
    """Return one required crop-layout metadata entry of a record."""
    assert key in record.metadata, (
        f"CT record {record.data_id!r} metadata lacks {key!r}; crop-backed layouts "
        "must store per-crop label metadata."
    )
    return record.metadata[key]


def _stored_centroids_ijk(record: DatasetRecord) -> dict[int, tuple[float, float, float]]:
    """Normalize stored crop centroids into a native-id keyed mapping.

    Accepts a mapping ``{native_id: (i, j, k)}`` or an ``(N, 3)`` list aligned
    with ``crop_foreground_label_ids``.

    Args:
        record: CT record whose metadata holds ``fg_centroids_ijk``.

    Returns:
        Mapping from positive native label id to voxel centroid.
    """

    stored = record.metadata["fg_centroids_ijk"]
    if isinstance(stored, Mapping):
        items = [(int(key), value) for key, value in stored.items()]
    else:
        label_ids = _required_crop_metadata(record, "crop_foreground_label_ids")
        assert len(label_ids) == len(stored), (
            f"CT record {record.data_id!r} has {len(stored)} centroids for "
            f"{len(label_ids)} crop_foreground_label_ids."
        )
        items = [(int(key), value) for key, value in zip(label_ids, stored)]
    centroids = {}
    for label_id, value in items:
        coordinates = tuple(float(v) for v in value)
        assert label_id > 0 and len(coordinates) == 3, (
            f"CT record {record.data_id!r} centroid for label {label_id} must be a 3-vector."
        )
        centroids[label_id] = coordinates
    return centroids


def _foreground_centroids_ijk(
    label: torch.Tensor,
) -> dict[int, tuple[float, float, float]]:
    """Compute foreground centroids from a dense CT label tensor.

    Args:
        label: Dense label tensor, optionally channel-first, whose foreground
            ids are positive values.

    Returns:
        Mapping from foreground label id to centroid coordinates in ``(i, j, k)``
        order over the trailing spatial dimensions.

    Raises:
        ValueError: If the label tensor contains only background.
    """

    foreground = label > 0
    if not torch.any(foreground):
        raise ValueError(
            "Cannot compute foreground centroids for background-only labels."
        )
    centroids: dict[int, tuple[float, float, float]] = {}
    for label_id in torch.unique(label[foreground]).tolist():
        mask = label == int(label_id)
        coordinates = torch.nonzero(mask, as_tuple=False).to(dtype=torch.float32)
        spatial_coordinates = coordinates[:, -3:]
        centroid = spatial_coordinates.mean(dim=0)
        centroids[int(label_id)] = (
            float(centroid[0].item()),
            float(centroid[1].item()),
            float(centroid[2].item()),
        )
    return centroids


def _subject_key(record: DatasetRecord) -> str:
    """Return the subject key used by subject-level filters.

    Args:
        record: Dataset record being filtered.

    Returns:
        ``record.subject_id`` when present, otherwise ``record.data_id``.
    """

    return record.subject_id or record.data_id
