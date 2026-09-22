"""Inject protocol-derived model channels into an experiment config.

An experiment declares its label space through a top-level ``protocol`` block
(``protocol.name`` plus an optional ``protocol.model_labels.names`` subset). The
label-space resolution lives in :mod:`fxr.protocols.run_config`; this module adds
the experiment-facing step of writing the derived ``model.out_channels`` back into
the config so the model config is self-describing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

from fxr.protocols import (
    load_dataset_spec,
    load_dataset_spec_by_name,
    resolve_run_output_label_names,
    resolve_run_protocol_spec,
)

__all__ = [
    "inject_protocol_derived_model_channels",
    "resolve_run_output_label_names",
    "resolve_run_protocol_spec",
    "resolve_supervise_empty_label_ids",
]


def _configured_dataset_spec_path(
    config: Mapping[str, Any],
    dataset_name: str,
) -> str | None:
    """Return the ``dataset_spec`` path configured for one source, if any.

    Args:
        config: Full experiment config mapping.
        dataset_name: Source dataset name.

    Returns:
        Absolute spec path from ``data.<modality>.<name>.dataset_spec``, or
        ``None``.
    """

    data_cfg = config.get("data") or {}
    for section in ("Xray", "CT"):
        dataset_cfg = (data_cfg.get(section) or {}).get(dataset_name)
        if isinstance(dataset_cfg, Mapping) and dataset_cfg.get("dataset_spec"):
            return str(dataset_cfg["dataset_spec"])
    return None


def resolve_supervise_empty_label_ids(
    config: Mapping[str, Any],
    dataset_names: Iterable[str],
) -> dict[str, tuple[int, ...]]:
    """Resolve each dataset's ``supervise_empty_labels`` into model channel ids.

    Specs come from ``data.<modality>.<name>.dataset_spec`` when the config
    names one and from the packaged registry otherwise. Datasets with neither,
    and labels outside a model-label subset, are skipped silently; labels
    outside the protocol are an error.

    Args:
        config: Full experiment config mapping.
        dataset_names: Active training dataset names.

    Returns:
        Mapping from dataset name to ordered model channel ids (only datasets
        with at least one resolvable label are included).
    """

    protocol = resolve_run_protocol_spec(config)
    output_labels = resolve_run_output_label_names(config)
    if protocol is None or output_labels is None:
        return {}
    config_root = config.get("protocol", {}).get("config_root")
    resolved: dict[str, tuple[int, ...]] = {}
    for dataset_name in dataset_names:
        spec_path = _configured_dataset_spec_path(config, dataset_name)
        if spec_path is not None:
            spec = load_dataset_spec(spec_path)
        else:
            try:
                spec = load_dataset_spec_by_name(dataset_name, config_root=config_root)
            except FileNotFoundError:
                continue
        unknown = sorted(set(spec.supervise_empty_labels) - set(protocol.labels))
        assert not unknown, (
            f"Dataset {dataset_name!r} supervise_empty_labels are outside protocol "
            f"{protocol.protocol_name!r}: {unknown}."
        )
        ids = tuple(
            output_labels.index(label)
            for label in spec.supervise_empty_labels
            if label in output_labels
        )
        if ids:
            resolved[dataset_name] = ids
    return resolved


def inject_protocol_derived_model_channels(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...] | None]:
    """Return a config copy with ``model.out_channels`` set from the protocol.

    Args:
        config: Full experiment config mapping.

    Returns:
        Tuple of ``(config_copy, model_label_names)``. When the config has no
        protocol, ``model_label_names`` is ``None`` and the config is returned
        unchanged.

    Raises:
        TypeError: If the ``model`` config section is present but not a mapping.
    """

    config_copy = deepcopy(dict(config))
    model_label_names = resolve_run_output_label_names(config_copy)
    if model_label_names is None:
        return config_copy, None
    model_cfg = config_copy.setdefault("model", {})
    if not isinstance(model_cfg, dict):
        raise TypeError("model config section must be a mapping.")
    model_cfg["out_channels"] = len(model_label_names)
    return config_copy, model_label_names
