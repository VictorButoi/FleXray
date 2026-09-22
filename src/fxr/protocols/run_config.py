"""Resolve protocol-derived label spaces from an experiment config.

An experiment declares its label space through a top-level ``protocol`` block
(``protocol.name`` plus an optional ``protocol.model_labels.names`` subset). These
helpers turn that declaration into the ordered model-output label names and the
derived id lists that the X-ray and CT->DRR training paths consume. They live in
``fxr.protocols`` (the lowest layer) so both ``fxr.experiment`` and the CT->DRR
rendering runtime in ``fxr.models.camera`` can depend on them without inverting
the package layering.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fxr.protocols.registry import (
    load_model_label_space_by_name,
    load_protocol_by_name,
)
from fxr.protocols.schemas import ProtocolSpec

_BACKGROUND_LABEL = "background"


def resolve_run_protocol_spec(config: Mapping[str, Any] | None) -> ProtocolSpec | None:
    """Resolve the run protocol from ``protocol.name``.

    Args:
        config: Full experiment config mapping, or ``None``.

    Returns:
        The loaded ``ProtocolSpec``, or ``None`` if no ``protocol`` block exists.

    Raises:
        TypeError: If the ``protocol`` block is present but not a mapping.
        ValueError: If the ``protocol`` block omits ``name``.
    """

    if config is None:
        return None
    protocol_cfg = config.get("protocol")
    if protocol_cfg is None:
        return None
    if not isinstance(protocol_cfg, Mapping):
        raise TypeError("protocol must be a mapping with key 'name'.")
    name = protocol_cfg.get("name")
    if not name:
        raise ValueError("protocol.name is required when a protocol block is present.")
    config_root = protocol_cfg.get("config_root")
    if config_root is not None and not isinstance(config_root, (str, Path)):
        raise TypeError("protocol.config_root must be a path string or null.")
    return load_protocol_by_name(str(name), config_root=config_root)


def resolve_run_output_label_names(
    config: Mapping[str, Any] | None,
) -> tuple[str, ...] | None:
    """Resolve the ordered model-output label names for a run.

    Uses ``protocol.model_labels.names`` when provided. Otherwise, it loads the
    named model label space matching ``protocol.name``; registry fallback keeps
    custom protocols without a model resource on their full protocol labels.
    Either result is validated as a subset of the protocol labels with
    ``background`` first.

    Args:
        config: Full experiment config mapping, or ``None``.

    Returns:
        Ordered model label names including ``background``, or ``None`` if the
        config has no protocol.

    Raises:
        ValueError: If requested model labels are unknown or omit ``background``
            at channel 0.
    """

    protocol = resolve_run_protocol_spec(config)
    if protocol is None:
        return None

    protocol_labels = tuple(protocol.labels)
    label_names = _explicit_model_label_names(config)
    if label_names is None:
        protocol_cfg = config.get("protocol")
        assert isinstance(protocol_cfg, Mapping)
        label_names = load_model_label_space_by_name(
            protocol.protocol_name,
            config_root=protocol_cfg.get("config_root"),
        ).labels

    unknown = sorted(set(label_names) - set(protocol_labels))
    if unknown:
        raise ValueError(
            f"Model output labels contain labels absent from protocol "
            f"{protocol.protocol_name!r}: {unknown}."
        )
    if not label_names or label_names[0] != _BACKGROUND_LABEL:
        raise ValueError(
            "Model output labels must list 'background' as channel 0."
        )
    return label_names


def resolve_run_render_label_names(
    config: Mapping[str, Any] | None,
) -> tuple[str, ...] | None:
    """Resolve model labels for CT->DRR rendering.

    Clean training renders directly into the model label space, so the render
    labels match the output labels.

    Args:
        config: Full experiment config mapping, or ``None``.

    Returns:
        Ordered render label names, or ``None`` if the config has no protocol.
    """

    return resolve_run_output_label_names(config)


def resolve_run_attenuated_label_ids(
    config: Mapping[str, Any] | None,
) -> list[int] | None:
    """Resolve non-background model ids eligible for per-label DRR attenuation.

    Args:
        config: Full experiment config mapping, or ``None``.

    Returns:
        Foreground model ids ``1 .. num_labels - 1``, or ``None`` if the config
        has no protocol.
    """

    label_names = resolve_run_output_label_names(config)
    if label_names is None:
        return None
    return list(range(1, len(label_names)))


def resolve_run_render_label_collapse_index(
    config: Mapping[str, Any] | None,
) -> list[int] | None:
    """Resolve the render->output channel-collapse index.

    Clean training shares one render/output label space, so no collapse is ever
    required; this always returns ``None`` when a protocol is configured.

    Args:
        config: Full experiment config mapping, or ``None``.

    Returns:
        Always ``None`` (kept for API parity with dataset-routed render labels).
    """

    _ = resolve_run_protocol_spec(config)
    return None


def _explicit_model_label_names(config: Mapping[str, Any]) -> tuple[str, ...] | None:
    """Return ``protocol.model_labels.names`` as a tuple, or ``None`` if unset."""
    protocol_cfg = config.get("protocol")
    if not isinstance(protocol_cfg, Mapping):
        return None
    model_labels = protocol_cfg.get("model_labels")
    if not isinstance(model_labels, Mapping):
        return None
    names = model_labels.get("names")
    if names is None:
        return None
    return tuple(str(name) for name in names)
