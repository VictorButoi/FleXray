"""Training-config normalization helpers.

The helpers in this module are intentionally dependency-free so launch code can
normalize configs before importing Torch-backed experiment modules.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from typing import Any

from ._proportions import normalize_proportion_weight


def resolve_num_views(
    profile: Mapping[str, Any], config: Mapping[str, Any] | None = None,
) -> int:
    """Resolve a profile's view count, preserving legacy training configs.

    Args:
        profile: Effective DRR profile with an optional ``num_views``.
        config: Training config used for the legacy ``dataloader.batch_size``
            fallback. Omit for a standalone profile whose default is one view.

    Returns:
        The explicit view count, legacy batch size, or one when neither is set.
    """

    value = profile.get("num_views")
    if value is None and config is not None:
        dataloader = config.get("dataloader")
        if isinstance(dataloader, Mapping):
            value = dataloader.get("batch_size")
    return 1 if value is None else int(value)


def _materialize_ct_view_counts(config: Mapping[str, Any]) -> dict[str, Any]:
    """Record effective CT view counts in a new launch config without mutation.

    Args:
        config: Training config after all batch-size and profile overrides.

    Returns:
        An owned config with explicit view counts on its dataset profiles.
        Legacy persisted configs are resolved at runtime, never rewritten.
    """

    source = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    resolved = deepcopy(source)
    drr = resolved.get("drr_model")
    if not isinstance(drr, Mapping):
        return resolved
    default = drr.get("default")
    datasets = drr.get("datasets")
    if not isinstance(default, Mapping) or not isinstance(datasets, MutableMapping):
        return resolved
    for name, profile in datasets.items():
        if profile is not None and not isinstance(profile, Mapping):
            continue  # The profile validator reports malformed entries.
        override = dict(profile or {})
        effective = {**default, **override}
        dataloader = resolved.get("dataloader")
        fallback = dataloader.get("batch_size") if isinstance(dataloader, Mapping) else None
        if effective.get("num_views", fallback) == "?" or (
            effective.get("num_views") is None and fallback == "?"
        ):
            continue  # Leave required-value diagnostics to check_missing.
        override["num_views"] = resolve_num_views(effective, resolved)
        datasets[name] = override
    return resolved


def prune_zero_proportion_datasets(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a config copy without proportion-inactive training datasets.

    Args:
        config: Training config mapping. When ``dataloader.proportions`` is
            present, datasets listed with positive weights are active. Datasets
            omitted from the mapping or listed with weight ``0`` are inactive.

    Returns:
        Deep-copied config with inactive datasets removed from training data,
        DatasetRoutedLoss routes, and DRR profiles. Callback data sections
        (e.g. eval sets) are left untouched: eval coverage is independent of
        the train mix.

    Raises:
        ValueError: If a configured dataset proportion is negative, non-numeric,
            or non-finite.
    """

    source = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    normalized = deepcopy(source)
    inactive = _inactive_proportion_dataset_names(normalized)
    if not inactive:
        return normalized

    _prune_data_mapping(normalized.get("data"), inactive)

    loss_func = normalized.get("loss_func")
    if isinstance(loss_func, MutableMapping):
        _prune_routed_loss(loss_func, inactive)

    drr_model = normalized.get("drr_model")
    if isinstance(drr_model, MutableMapping):
        _prune_named_mapping(drr_model.get("datasets"), inactive)

    return normalized


def zero_proportion_dataset_names(config: Mapping[str, Any]) -> set[str]:
    """Return dataset names explicitly assigned a zero sampling weight.

    Args:
        config: Training config mapping.

    Returns:
        Set of dataset names whose ``dataloader.proportions`` entry is ``0``.

    Raises:
        ValueError: If any proportion cannot be treated as a non-negative
            numeric sampling weight.
    """

    dataloader = config.get("dataloader")
    if not isinstance(dataloader, Mapping):
        return set()
    proportions = dataloader.get("proportions")
    if proportions is None:
        return set()
    if not isinstance(proportions, Mapping):
        raise ValueError("dataloader.proportions must be a mapping.")

    inactive: set[str] = set()
    for dataset_name, raw_weight in proportions.items():
        weight = normalize_proportion_weight(
            raw_weight,
            f"dataloader.proportions[{str(dataset_name)!r}]",
        )
        if weight == 0:
            inactive.add(str(dataset_name))
    return inactive


def requires_training_source_validation(config: Mapping[str, Any]) -> bool:
    """Return whether any configured feature consumes training-source ``val``.

    Core evaluation consumes validation data when ``train.eval_freq`` is
    positive. ``WandbSamplePredictionLogger`` also consumes the experiment
    validation datasets when its ``every`` interval is positive. Malformed
    values conservatively require validation and are rejected by their owning
    config or runtime validator.

    Args:
        config: Resolved training config mapping.

    Returns:
        ``True`` when the run must build validation splits for training sources.
    """

    train = config.get("train")
    raw_frequency = train.get("eval_freq", 1) if isinstance(train, Mapping) else 1
    if isinstance(raw_frequency, bool):
        return True
    if isinstance(raw_frequency, float) and not raw_frequency.is_integer():
        return True
    try:
        frequency = int(raw_frequency)
    except (TypeError, ValueError, OverflowError):
        return True
    if frequency != 0:
        return True

    callbacks = config.get("callbacks")
    if not isinstance(callbacks, Mapping):
        return False
    for raw_group in callbacks.values():
        if not isinstance(raw_group, Mapping):
            continue
        for raw_spec in raw_group.values():
            if not isinstance(raw_spec, Mapping):
                continue
            class_name = str(raw_spec.get("_class", "")).split(".")[-1]
            if class_name != "WandbSamplePredictionLogger":
                continue
            raw_every = raw_spec.get("every", 1)
            if isinstance(raw_every, bool):
                return True
            if isinstance(raw_every, float) and not raw_every.is_integer():
                return True
            try:
                if int(raw_every) > 0:
                    return True
            except (TypeError, ValueError, OverflowError):
                return True
    return False


def _inactive_proportion_dataset_names(config: Mapping[str, Any]) -> set[str]:
    """Return dataset names made inactive by ``dataloader.proportions``.

    A configured ``proportions`` mapping acts as an allowlist: positive entries
    remain active, explicit zeros are inactive, and configured datasets omitted
    from the mapping are inactive. When no proportions mapping is present, all
    configured datasets remain active.
    """

    dataloader = config.get("dataloader")
    if not isinstance(dataloader, Mapping):
        return set()
    proportions = dataloader.get("proportions")
    if proportions is None:
        return set()
    if not isinstance(proportions, Mapping):
        raise ValueError("dataloader.proportions must be a mapping.")

    active: set[str] = set()
    inactive: set[str] = set()
    for dataset_name, raw_weight in proportions.items():
        name = str(dataset_name)
        weight = normalize_proportion_weight(
            raw_weight,
            f"dataloader.proportions[{name!r}]",
        )
        if weight == 0:
            inactive.add(name)
        else:
            active.add(name)

    configured = _configured_dataset_names(config.get("data"))
    return inactive | (configured - active)


def _prune_routed_loss(loss_func: Any, inactive: set[str]) -> None:
    """Remove inactive dataset routes from a routed loss config."""

    class_name = str(loss_func.get("_class", ""))
    if class_name.split(".")[-1] != "DatasetRoutedLoss":
        return
    _prune_named_mapping(loss_func.get("dataset_losses"), inactive)


def _prune_data_mapping(data: Any, inactive: set[str]) -> None:
    """Remove inactive dataset keys from one multimodal data mapping."""

    if not isinstance(data, MutableMapping):
        return
    for modality_key in ("CT", "Xray", "ct", "xray"):
        section = data.get(modality_key)
        if isinstance(section, MutableMapping):
            _prune_named_mapping(section, inactive)
    if not any(
        str(key) in {"CT", "Xray", "ct", "xray"}
        for key in data
    ):
        _prune_named_mapping(data, inactive)


def _prune_named_mapping(mapping: Any, inactive: set[str]) -> None:
    """Drop inactive dataset keys from a mutable mapping in place."""

    if not isinstance(mapping, MutableMapping):
        return
    for dataset_name in inactive:
        mapping.pop(dataset_name, None)


def _configured_dataset_names(data: Any) -> set[str]:
    """Return dataset names configured under one data mapping."""

    if not isinstance(data, Mapping):
        return set()
    names: set[str] = set()
    modality_keys = {"CT", "Xray", "ct", "xray"}
    has_modality_sections = any(str(key) in modality_keys for key in data)
    if has_modality_sections:
        for modality_key in modality_keys:
            section = data.get(modality_key)
            if isinstance(section, Mapping):
                names.update(str(name) for name in section)
        return names
    return {str(name) for name in data}
