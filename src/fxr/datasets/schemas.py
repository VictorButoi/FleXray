from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

import torch

LabelMode = Literal["model", "native"]
Modality = Literal["xray", "ct"]


@dataclass(frozen=True)
class TrainingLabelRemap:
    """Torch-ready mapping from dataset-native labels to model labels.

    Attributes:
        dataset_name: Dataset name used to compile the remap.
        protocol_name: Protocol name used as the model-label target space.
        label_lut: Dense tensor indexed by dataset-native label id with model
            label ids as values.
        label_names: Ordered model label names produced by the remap.
        native_id_to_model_id: Sparse mapping from each declared native id to
            the remapped model id.
        native_id_to_model_label: Sparse mapping from each declared native id
            to the remapped model label name.
        num_classes: Number of model labels produced by the remap.
    """

    dataset_name: str
    protocol_name: str
    label_lut: torch.Tensor
    label_names: tuple[str, ...]
    native_id_to_model_id: dict[int, int]
    native_id_to_model_label: dict[int, str]

    @property
    def num_classes(self) -> int:
        """Return the number of model-label classes in the remap.

        Returns:
            Count of labels in ``label_names``.
        """

        return len(self.label_names)


class TrainingSample(TypedDict, total=False):
    """Common sample dictionary returned by FleXray training datasets.

    Attributes:
        image: Channel-first image or volume tensor.
        label: Training label tensor in the selected label mode.
        native_label: Optional unremapped dataset-native label tensor.
        dataset_name: Source dataset name for the sample.
        modality: Sample modality, ``"xray"`` or ``"ct"``.
        data_id: Stable sample identifier from the source manifest.
        metadata: Additional source metadata, including CT affine and spacing
            when available.
    """

    image: torch.Tensor
    label: torch.Tensor
    native_label: torch.Tensor
    dataset_name: str
    modality: str
    data_id: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DatasetRecord:
    """Storage-neutral description of one trainable image sample.

    Attributes:
        dataset_name: Source dataset name.
        modality: Sample modality used to route records to training dataset
            classes.
        data_id: Stable sample identifier.
        image: Storage key or path for the image payload.
        label: Optional storage key or path for the label payload.
        split: Optional dataset split name.
        subject_id: Optional subject identifier used for subject filtering.
        sample_id: Optional within-subject sample identifier.
        affine: Optional storage key or path for a CT affine matrix payload.
        spacing: Optional storage key or path for CT voxel spacing metadata.
        label_names: Optional ordered native-id names for dense labels or
            channel names for channel-mask labels.
        metadata: Additional storage-neutral record metadata.
    """

    dataset_name: str
    modality: str
    data_id: str
    image: str
    label: str | None = None
    split: str | None = None
    subject_id: str | None = None
    sample_id: str | None = None
    affine: str | None = None
    spacing: str | None = None
    label_names: tuple[str, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any],
        *,
        default_dataset_name: str | None = None,
        default_modality: str | None = None,
    ) -> "DatasetRecord":
        """Build a normalized record from manifest-style mapping data.

        Args:
            data: Raw record mapping from a manifest or database.
            default_dataset_name: Dataset name to use when ``data`` omits
                ``dataset_name``.
            default_modality: Modality to use when ``data`` omits ``modality``.

        Returns:
            Normalized immutable ``DatasetRecord``.

        Raises:
            TypeError: If ``data`` is not a dictionary or nested metadata fields
                have incompatible types.
            ValueError: If required string fields are missing or invalid.
        """

        if not isinstance(data, dict):
            raise TypeError(f"Dataset record must be a mapping, got {data!r}.")
        dataset_name = _required_str(
            data,
            "dataset_name",
            default=default_dataset_name,
        )
        modality = _required_str(data, "modality", default=default_modality)
        data_id = _required_str(data, "data_id")
        image = _required_str(data, "image")
        label_names = data.get("label_names")
        if label_names is None:
            raw_metadata = data.get("metadata") or {}
            if isinstance(raw_metadata, dict):
                label_names = raw_metadata.get("label_names") or raw_metadata.get(
                    "labels"
                )
        return cls(
            dataset_name=dataset_name,
            modality=modality,
            data_id=data_id,
            image=image,
            label=_optional_str(data.get("label"), "label"),
            split=_optional_str(data.get("split"), "split"),
            subject_id=_optional_str(data.get("subject_id"), "subject_id"),
            sample_id=_optional_str(data.get("sample_id"), "sample_id"),
            affine=_optional_str(data.get("affine"), "affine"),
            spacing=_optional_str(data.get("spacing"), "spacing"),
            label_names=_load_label_names(label_names),
            metadata=_load_metadata(data.get("metadata")),
        )


def _required_str(
    data: dict[str, Any],
    key: str,
    *,
    default: str | None = None,
) -> str:
    """Read and validate a required string record field.

    Args:
        data: Source mapping containing record fields.
        key: Field name to read.
        default: Optional fallback value when ``key`` is absent.

    Returns:
        Stripped string value.

    Raises:
        ValueError: If the resolved value is missing, not a string, or empty.
    """

    raw_value = data.get(key, default)
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"Dataset record must define non-empty {key!r}.")
    return raw_value.strip()


def _optional_str(raw_value: Any, key: str) -> str | None:
    """Normalize an optional string record field.

    Args:
        raw_value: Value to validate.
        key: Field name used in validation errors.

    Returns:
        Stripped string value, or ``None`` when ``raw_value`` is ``None``.

    Raises:
        ValueError: If a provided value is not a non-empty string.
    """

    if raw_value is None:
        return None
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"Dataset record field {key!r} must be a non-empty string.")
    return raw_value.strip()


def _load_label_names(raw_value: Any) -> tuple[str, ...] | None:
    """Normalize optional channel-mask label names.

    Args:
        raw_value: Optional list or tuple of label names.

    Returns:
        Tuple of stripped label names, or ``None`` when no names were supplied.

    Raises:
        TypeError: If names are supplied in a non-sequence value.
        ValueError: If any label name is empty or duplicated.
    """

    if raw_value is None:
        return None
    if not isinstance(raw_value, (list, tuple)):
        raise TypeError("Dataset record label_names must be a list of strings.")
    labels = tuple(str(label).strip() for label in raw_value)
    if any(not label for label in labels):
        raise ValueError("Dataset record label_names cannot contain empty names.")
    if len(set(labels)) != len(labels):
        raise ValueError("Dataset record label_names cannot contain duplicates.")
    return labels


def _load_metadata(raw_value: Any) -> dict[str, Any]:
    """Normalize optional record metadata.

    Args:
        raw_value: Optional metadata mapping.

    Returns:
        Shallow metadata dictionary, or an empty dictionary when omitted.

    Raises:
        TypeError: If metadata is supplied as a non-mapping value.
    """

    if raw_value is None:
        return {}
    if not isinstance(raw_value, dict):
        raise TypeError("Dataset record metadata must be a mapping.")
    return dict(raw_value)


@dataclass(frozen=True)
class NamedBatch:
    """Batch emitted by named-loader composition helpers.

    Attributes:
        source_name: Name of the loader or dataset source that produced the
            batch.
        modality: Optional modality associated with ``source_name``.
        batch: Original batch object yielded by the underlying loader.
    """

    source_name: str
    modality: str | None
    batch: Any
