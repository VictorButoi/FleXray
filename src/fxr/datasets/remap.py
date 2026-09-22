from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from fxr.protocols import (
    compile_training_lut,
    compile_training_lut_by_name,
    load_dataset_spec,
    load_dataset_spec_by_name,
    load_protocol_by_name,
)
from fxr.protocols.schemas import CompiledTrainingLut

from .schemas import TrainingLabelRemap

_BACKGROUND_LABEL = "background"


def compile_training_label_remap_by_name(
    protocol_name: str,
    dataset_name: str,
    model_label_names: Sequence[str] | None = None,
    *,
    config_root: str | Path | None = None,
) -> TrainingLabelRemap:
    """Compile a torch LUT from dataset-native ids to model label ids.

    Args:
        protocol_name: Protocol config name used as the target label space.
        dataset_name: Dataset config name or alias used as the native source.
        model_label_names: Optional model-label subset or ordering. Labels
            outside this set are projected to background.
        config_root: Optional config root for protocol and dataset lookup.

    Returns:
        Training label remap with a dense tensor LUT and sparse id/name maps.

    Raises:
        ValueError: If named configs or requested model labels are invalid.
    """

    compiled = compile_training_lut_by_name(
        protocol_name,
        dataset_name,
        config_root=config_root,
    )
    return _remap_from_compiled_lut(compiled, model_label_names)


def compile_package_label_remap(
    protocol_name: str,
    dataset_name: str,
    stored_labels: Mapping[int | str, str],
    *,
    dataset_spec: str | Path | None = None,
    model_label_names: Sequence[str] | None = None,
    config_root: str | Path | None = None,
) -> TrainingLabelRemap:
    """Compile the remap of a user package, optionally through a dataset spec.

    Without ``dataset_spec`` the package's ``stored_labels`` must already use
    protocol names. With a spec path, the spec supplies aliases and drops; its
    ``dataset_name`` and ``stored_labels`` must match the package exactly.

    Args:
        protocol_name: Protocol config name used as the target label space.
        dataset_name: Package dataset identity (config key).
        stored_labels: Package-owned native id to name mapping.
        dataset_spec: Optional path to a dataset spec YAML.
        model_label_names: Optional model-label subset or ordering.
        config_root: Optional config root for protocol lookup.

    Returns:
        Training remap whose LUT maps stored ids into model label ids.
    """

    if dataset_spec is None:
        return compile_training_label_remap_from_stored_labels(
            protocol_name,
            dataset_name,
            stored_labels,
            model_label_names=model_label_names,
            config_root=config_root,
        )
    spec = load_dataset_spec(dataset_spec)
    assert spec.dataset_name == dataset_name, (
        f"dataset_spec {dataset_spec} declares dataset_name {spec.dataset_name!r}, "
        f"but the package is configured as {dataset_name!r}."
    )
    normalized = _normalize_stored_labels(stored_labels)
    assert normalized == spec.stored_labels, (
        f"package stored_labels differ from dataset_spec {dataset_spec}: "
        f"package={normalized}, spec={spec.stored_labels}."
    )
    protocol = load_protocol_by_name(protocol_name, config_root=config_root)
    return _remap_from_compiled_lut(compile_training_lut(protocol, spec), model_label_names)


def _remap_from_compiled_lut(
    compiled: CompiledTrainingLut,
    model_label_names: Sequence[str] | None,
) -> TrainingLabelRemap:
    """Turn a compiled protocol LUT into a model-channel training remap.

    Args:
        compiled: Native-id to protocol-channel lookup table.
        model_label_names: Optional model-label subset or ordering; labels
            outside it project to background.

    Returns:
        Training remap with a dense tensor LUT and sparse id/name maps.
    """

    label_names = _resolve_model_label_names(
        compiled.protocol_labels, model_label_names
    )
    model_label_to_id = {label_name: idx for idx, label_name in enumerate(label_names)}
    protocol_id_to_label = {
        idx: label_name for idx, label_name in enumerate(compiled.protocol_labels)
    }
    native_id_to_model_id: dict[int, int] = {}
    native_id_to_model_label: dict[int, str] = {}
    dense_lut: list[int] = []
    for native_id, protocol_id in enumerate(compiled.label_lut):
        if protocol_id < 0:
            dense_lut.append(-1)
            continue
        protocol_label = protocol_id_to_label[protocol_id]
        model_label = (
            protocol_label if protocol_label in model_label_to_id else _BACKGROUND_LABEL
        )
        model_id = model_label_to_id[model_label]
        dense_lut.append(model_id)
        if native_id in compiled.native_id_to_protocol_id:
            native_id_to_model_id[native_id] = model_id
            native_id_to_model_label[native_id] = model_label
    return TrainingLabelRemap(
        dataset_name=compiled.dataset_name,
        protocol_name=compiled.protocol_name,
        label_lut=torch.tensor(dense_lut, dtype=torch.long),
        label_names=label_names,
        native_id_to_model_id=native_id_to_model_id,
        native_id_to_model_label=native_id_to_model_label,
    )


def compile_training_label_remap_from_stored_labels(
    protocol_name: str,
    dataset_name: str,
    stored_labels: Mapping[int | str, str],
    model_label_names: Sequence[str] | None = None,
    *,
    config_root: str | Path | None = None,
) -> TrainingLabelRemap:
    """Compile a dense-label remap from package-owned label metadata.

    This supports user-packaged datasets without requiring a bundled dataset
    specification. Stored foreground names must already use the selected
    protocol names; dataset-specific aliases remain the responsibility of named
    dataset specifications.

    Args:
        protocol_name: Protocol config name used as the target label space.
        dataset_name: Dataset identity recorded on the returned remap.
        stored_labels: Contiguous native ids mapped to canonical label names.
        model_label_names: Optional model-label subset or ordering. Labels
            outside this set are projected to background.
        config_root: Optional config root for protocol lookup.

    Returns:
        Training remap whose LUT maps stored ids into model label ids.

    Raises:
        TypeError: If ``stored_labels`` is not a mapping or has invalid values.
        ValueError: If ids, names, background, protocol membership, or requested
            model labels are invalid.
    """

    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("dataset_name must be a non-empty string.")
    normalized = _normalize_stored_labels(stored_labels)
    protocol = load_protocol_by_name(protocol_name, config_root=config_root)
    label_names = _resolve_model_label_names(protocol.labels, model_label_names)
    unknown = sorted(set(normalized.values()).difference(protocol.labels))
    if unknown:
        raise ValueError(
            f"stored_labels contains labels outside protocol "
            f"{protocol.protocol_name!r}: {unknown}."
        )

    model_label_to_id = {name: index for index, name in enumerate(label_names)}
    native_id_to_model_id: dict[int, int] = {}
    native_id_to_model_label: dict[int, str] = {}
    dense_lut: list[int] = []
    for native_id, source_label in normalized.items():
        model_label = (
            source_label if source_label in model_label_to_id else _BACKGROUND_LABEL
        )
        model_id = model_label_to_id[model_label]
        dense_lut.append(model_id)
        native_id_to_model_id[native_id] = model_id
        native_id_to_model_label[native_id] = model_label
    return TrainingLabelRemap(
        dataset_name=dataset_name.strip(),
        protocol_name=protocol.protocol_name,
        label_lut=torch.tensor(dense_lut, dtype=torch.long),
        label_names=label_names,
        native_id_to_model_id=native_id_to_model_id,
        native_id_to_model_label=native_id_to_model_label,
    )


def normalize_training_channel_label_names(
    protocol_name: str,
    dataset_name: str,
    source_label_names: Sequence[str],
    *,
    config_root: str | Path | None = None,
) -> tuple[str, ...]:
    """Normalize named mask channels through a registered dataset contract.

    Dataset-native names take the same alias and drop rules as dense native
    ids. Canonical protocol names that are not part of the registered native
    inventory pass through unchanged, allowing directly packaged channel
    subsets. Multiple native names may normalize to one protocol name; callers
    can then merge those channels with :func:`project_channel_mask`.

    Args:
        protocol_name: Protocol config name used as the target label space.
        dataset_name: Dataset config name or alias associated with the package.
        source_label_names: Ordered names aligned with source mask channels.
        config_root: Optional config root for protocol and dataset lookup.

    Returns:
        Ordered protocol-normalized channel names.

    Raises:
        ValueError: If any source name is neither declared by the registered
            dataset nor present in the protocol.
    """

    protocol = load_protocol_by_name(protocol_name, config_root=config_root)
    native_to_protocol: dict[str, str] = {}
    try:
        dataset_spec = load_dataset_spec_by_name(
            dataset_name,
            config_root=config_root,
        )
    except FileNotFoundError:
        dataset_spec = None
    if dataset_spec is not None:
        compiled = compile_training_lut_by_name(
            protocol_name,
            dataset_name,
            config_root=config_root,
        )
        native_to_protocol = {
            native_name: compiled.native_id_to_protocol_label[native_id]
            for native_id, native_name in dataset_spec.stored_labels.items()
        }

    protocol_names = set(protocol.labels)
    normalized: list[str] = []
    unknown: set[str] = set()
    for raw_name in source_label_names:
        source_name = str(raw_name).strip()
        if source_name in native_to_protocol:
            normalized.append(native_to_protocol[source_name])
        elif source_name in protocol_names:
            normalized.append(source_name)
        else:
            unknown.add(source_name)
    if unknown:
        raise ValueError(
            f"Dataset {dataset_name!r} channel labels are outside protocol "
            f"{protocol.protocol_name!r}: {sorted(unknown)}."
        )
    return tuple(normalized)


def _normalize_stored_labels(
    stored_labels: Mapping[int | str, str],
) -> dict[int, str]:
    """Validate and normalize package-owned dense label metadata.

    Args:
        stored_labels: Raw native-id to canonical-name mapping.

    Returns:
        Insertion-ordered mapping with integer keys sorted from zero.

    Raises:
        TypeError: If the mapping, ids, or names have incompatible types.
        ValueError: If ids are negative, sparse, duplicated after normalization,
            labels are empty or duplicated, or id zero is not background.
    """

    if not isinstance(stored_labels, Mapping):
        raise TypeError("stored_labels must be a mapping from ids to names.")
    normalized: dict[int, str] = {}
    for raw_id, raw_name in stored_labels.items():
        if isinstance(raw_id, bool):
            raise TypeError("stored label ids must be integers, not booleans.")
        try:
            native_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"stored label id {raw_id!r} must be an integer."
            ) from exc
        if native_id < 0:
            raise ValueError("stored label ids must be non-negative.")
        if native_id in normalized:
            raise ValueError(
                f"stored label id {native_id} is declared more than once."
            )
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise TypeError(
                f"stored label name for id {native_id} must be a non-empty string."
            )
        normalized[native_id] = raw_name.strip()

    expected_ids = list(range(len(normalized)))
    actual_ids = sorted(normalized)
    if actual_ids != expected_ids:
        raise ValueError(
            f"stored label ids must be contiguous from 0; got {actual_ids}."
        )
    normalized = {native_id: normalized[native_id] for native_id in actual_ids}
    if not normalized or normalized.get(0) != _BACKGROUND_LABEL:
        raise ValueError("stored_labels must define id 0 as background.")
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("stored label names must not contain duplicates.")
    return normalized


def apply_label_lut(label: torch.Tensor, remap: TrainingLabelRemap) -> torch.Tensor:
    """Apply a compiled native-id LUT to an integer label tensor.

    Args:
        label: Tensor containing dataset-native integer ids.
        remap: Compiled training remap whose LUT indexes native ids.

    Returns:
        Tensor with the same shape as ``label`` and model-label ids as values.

    Raises:
        ValueError: If ``label`` contains ids outside the LUT or ids marked as
            missing from the remap.
    """

    label = label.to(dtype=torch.long)
    if label.numel() == 0:
        return label
    min_id = int(label.min().item())
    max_id = int(label.max().item())
    if min_id < 0 or max_id >= remap.label_lut.numel():
        raise ValueError(
            f"Label ids must be in [0, {remap.label_lut.numel() - 1}], got "
            f"range [{min_id}, {max_id}]."
        )
    mapped = remap.label_lut[label]
    if torch.any(mapped < 0):
        bad_ids = torch.unique(label[mapped < 0]).tolist()
        raise ValueError(f"Label contains ids missing from the remap LUT: {bad_ids}.")
    return mapped


def channelize_integer_mask(label: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert an integer mask to channel-first binary float masks.

    Args:
        label: Integer mask whose values are class ids.
        num_classes: Number of output channels to produce.

    Returns:
        Float tensor shaped as ``(num_classes, *label.shape)`` with one-hot
        class channels.
    """

    label = label.to(dtype=torch.long)
    channels = torch.nn.functional.one_hot(label, num_classes=num_classes)
    order = (label.ndim, *range(label.ndim))
    return channels.permute(order).to(dtype=torch.float32)


def project_channel_mask(
    mask: torch.Tensor,
    source_label_names: Sequence[str],
    target_label_names: Sequence[str],
) -> torch.Tensor:
    """Project channel-first masks from stored label names to target labels.

    Args:
        mask: Channel-first binary or soft mask tensor.
        source_label_names: Label names aligned with ``mask`` channels.
        target_label_names: Desired output label-channel order.

    Returns:
        Float tensor with channels ordered according to ``target_label_names``.
        Duplicate source labels are merged with a channel-wise maximum.

    Raises:
        ValueError: If ``mask`` is not channel-first or its channel count does
            not match ``source_label_names``.
    """

    if mask.ndim < 3:
        raise ValueError("Channel-mask labels must be channel-first.")
    source_names = tuple(str(label) for label in source_label_names)
    target_names = tuple(str(label) for label in target_label_names)
    if mask.shape[0] != len(source_names):
        raise ValueError(
            f"Channel-mask label has {mask.shape[0]} channels but metadata lists "
            f"{len(source_names)} labels."
        )
    target = torch.zeros(
        (len(target_names), *mask.shape[1:]),
        dtype=torch.float32,
        device=mask.device,
    )
    target_index = {label_name: idx for idx, label_name in enumerate(target_names)}
    for source_idx, source_name in enumerate(source_names):
        target_idx = target_index.get(source_name)
        if target_idx is None:
            continue
        target[target_idx] = torch.maximum(
            target[target_idx],
            mask[source_idx].to(dtype=torch.float32),
        )
    return target


def _resolve_model_label_names(
    protocol_labels: tuple[str, ...],
    model_label_names: Sequence[str] | None,
) -> tuple[str, ...]:
    """Validate and resolve model-label names for a training remap.

    Args:
        protocol_labels: Complete protocol label order.
        model_label_names: Optional requested model-label subset or ordering.

    Returns:
        Resolved model-label tuple. When ``model_label_names`` is omitted, this
        is exactly ``protocol_labels``.

    Raises:
        ValueError: If the requested model-label set is empty, lacks background
            at channel 0, contains duplicates, or names labels outside the
            protocol.
    """

    if model_label_names is None:
        return protocol_labels
    labels = tuple(str(label).strip() for label in model_label_names)
    if not labels:
        raise ValueError("model_label_names must not be empty.")
    if labels[0] != _BACKGROUND_LABEL:
        raise ValueError("model_label_names must put 'background' at channel 0.")
    if len(set(labels)) != len(labels):
        raise ValueError("model_label_names contains duplicate labels.")
    unknown = sorted(label for label in labels if label not in protocol_labels)
    if unknown:
        raise ValueError(
            f"model_label_names contains labels outside the protocol: {unknown}."
        )
    return labels
