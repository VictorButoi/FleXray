from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

_DEFAULT_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys.

    Attributes:
        No public attributes beyond the ``yaml.SafeLoader`` parsing state.
    """


def _construct_unique_mapping(
    loader: yaml.SafeLoader,
    node: yaml.Node,
    deep: bool = False,
) -> dict[object, object]:
    """Construct a YAML mapping while rejecting repeated keys.

    Args:
        loader: PyYAML loader instance constructing ``node``.
        node: YAML mapping node to load.
        deep: Whether nested nodes should be constructed eagerly.

    Returns:
        Mapping represented by the YAML node.

    Raises:
        yaml.constructor.ConstructorError: If the YAML mapping repeats a key.
    """

    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_named_augmentation_preset(
    name: str,
    config_root: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load a packaged train augmentation preset by flat name.

    Args:
        name: Preset stem or flat YAML filename under ``augmentations/``.
        config_root: Optional root containing an ``augmentations/`` directory.
            Defaults to the packaged ``fxr.configs`` resource directory.

    Returns:
        Ordered transform mapping loaded from the preset YAML.

    Raises:
        TypeError: If ``name`` is not a string or the preset body is not a
            mapping.
        ValueError: If ``name`` is empty, path-like, has an unsupported suffix,
            or any transform config is not a mapping.
        FileNotFoundError: If the named preset file is missing.
    """

    requested_name = _normalize_preset_name(name)
    preset_path = _resolve_preset_path(requested_name, config_root=config_root)
    loaded = yaml.load(
        preset_path.read_text(encoding="utf-8"),
        Loader=_UniqueKeySafeLoader,
    )
    return _normalize_transform_map(
        loaded,
        context=f"augmentation preset {preset_path.name}",
    )


def _normalize_preset_name(raw_name: object) -> str:
    """Validate and normalize a preset lookup name.

    Args:
        raw_name: Preset name supplied by the caller.

    Returns:
        Trimmed preset name.

    Raises:
        TypeError: If ``raw_name`` is not a string.
        ValueError: If the name is empty, path-like, or has an unsupported
            extension.
    """

    if not isinstance(raw_name, str):
        raise TypeError(f"augmentation preset name must be a string, got {raw_name!r}.")
    name = raw_name.strip()
    if not name:
        raise ValueError("augmentation preset name cannot be empty.")
    if "/" in name or "\\" in name:
        raise ValueError(
            f"augmentation preset name must be a flat filename or stem, got {raw_name!r}."
        )
    path = Path(name)
    if path.is_absolute() or len(path.parts) != 1:
        raise ValueError(
            f"augmentation preset name must be a flat filename or stem, got {raw_name!r}."
        )
    if path.suffix and path.suffix not in {".yml", ".yaml"}:
        raise ValueError(
            "augmentation preset name must use .yml, .yaml, or no extension; "
            f"got {raw_name!r}."
        )
    return name


def _resolve_preset_path(
    requested_name: str,
    *,
    config_root: str | Path | None,
) -> Path:
    """Resolve a validated preset name to one YAML file.

    Args:
        requested_name: Validated flat preset name.
        config_root: Optional config root containing ``augmentations/``.

    Returns:
        Existing preset path.

    Raises:
        FileNotFoundError: If no candidate exists.
        ValueError: If both ``.yml`` and ``.yaml`` candidates exist.
    """

    root = _DEFAULT_CONFIG_ROOT if config_root is None else Path(config_root)
    directory = root / "augmentations"
    path = Path(requested_name)
    if path.suffix:
        candidates = [directory / path.name]
    else:
        candidates = [
            directory / f"{requested_name}.yml",
            directory / f"{requested_name}.yaml",
        ]
    existing = [candidate for candidate in candidates if candidate.exists()]
    if not existing:
        expected = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f"Could not find FleXray augmentation preset {requested_name!r}; "
            f"expected one of: {expected}."
        )
    if len(existing) > 1:
        matches = ", ".join(str(path) for path in existing)
        raise ValueError(
            f"Ambiguous FleXray augmentation preset {requested_name!r}; "
            f"multiple files exist: {matches}."
        )
    return existing[0]


def _normalize_transform_map(
    raw_map: object,
    *,
    context: str,
) -> dict[str, dict[str, Any]]:
    """Validate a raw YAML transform mapping.

    Args:
        raw_map: YAML object expected to be a transform-name mapping.
        context: Human-readable config location for validation errors.

    Returns:
        Ordered dictionary copy keyed by transform name.

    Raises:
        TypeError: If the top-level object is not a mapping.
        ValueError: If a transform name is invalid or a transform config is not
            a mapping.
    """

    if not isinstance(raw_map, Mapping):
        raise TypeError(f"{context} must be a mapping.")
    normalized: dict[str, dict[str, Any]] = {}
    for raw_transform_name, raw_config in raw_map.items():
        transform_name = _normalize_transform_name(
            raw_transform_name,
            context=context,
        )
        if transform_name in normalized:
            raise ValueError(f"{context} repeats transform {transform_name!r}.")
        if not isinstance(raw_config, Mapping):
            raise ValueError(
                f"{context}.{transform_name} must be a mapping, "
                f"got {type(raw_config).__name__}."
            )
        normalized[transform_name] = deepcopy(dict(raw_config))
    return normalized


def _normalize_transform_name(raw_name: object, *, context: str) -> str:
    """Validate one transform name from a preset.

    Args:
        raw_name: YAML mapping key expected to be a non-empty string.
        context: Human-readable config location for validation errors.

    Returns:
        Trimmed transform name.

    Raises:
        TypeError: If the name is not a string.
        ValueError: If the name is empty.
    """

    if not isinstance(raw_name, str):
        raise TypeError(f"{context} transform names must be strings, got {raw_name!r}.")
    name = raw_name.strip()
    if not name:
        raise ValueError(f"{context} transform names cannot be empty.")
    return name


_DEFAULT_MODALITY_PRESETS = {"ct": "CT_base", "xray": "Xray_base"}
_MODALITY_CONFIG_KEYS = {"CT": "ct", "Xray": "xray"}


def resolve_augmentation_presets(config: Mapping[str, Any] | None) -> dict[str, str]:
    """Resolve the train augmentation preset name for each runtime modality.

    The optional ``augmentation.presets`` block maps ``CT`` and/or ``Xray`` to
    preset names; omitted entries fall back to ``CT_base`` and ``Xray_base``.
    Generated sources share the X-ray preset.

    Args:
        config: Full experiment config mapping, or ``None``.

    Returns:
        Mapping from runtime modality (``"ct"``, ``"xray"``) to preset name.

    Raises:
        TypeError: If the augmentation block or a preset name has the wrong type.
        ValueError: If the block declares unknown keys.
    """

    presets = dict(_DEFAULT_MODALITY_PRESETS)
    augmentation = (config or {}).get("augmentation")
    if augmentation is None:
        return presets
    if not isinstance(augmentation, Mapping):
        raise TypeError("augmentation config must be a mapping.")
    unknown = sorted(set(augmentation) - {"presets"})
    if unknown:
        raise ValueError(f"augmentation config has unknown key(s): {unknown}.")
    routed = augmentation.get("presets") or {}
    if not isinstance(routed, Mapping):
        raise TypeError("augmentation.presets must be a mapping of CT/Xray to preset names.")
    unknown = sorted(set(routed) - set(_MODALITY_CONFIG_KEYS))
    if unknown:
        raise ValueError(f"augmentation.presets has unknown modality key(s): {unknown}.")
    for config_key, modality in _MODALITY_CONFIG_KEYS.items():
        if config_key in routed:
            presets[modality] = _normalize_preset_name(routed[config_key])
    return presets


def snapshot_augmentation_presets(
    run_dir: str | Path,
    presets: Mapping[str, Mapping[str, Any]],
) -> None:
    """Write the resolved per-modality transform maps into a run directory.

    Snapshots make the exact augmentation chain of a run inspectable after
    packaged presets change. Existing snapshots are never overwritten, so a
    resumed run keeps its original record.

    Args:
        run_dir: Run directory receiving an ``augmentations/`` subdirectory.
        presets: Mapping from runtime modality to its resolved transform map.

    Returns:
        ``None``.
    """

    directory = Path(run_dir) / "augmentations"
    directory.mkdir(parents=True, exist_ok=True)
    for modality, transform_map in presets.items():
        path = directory / f"{modality}.yml"
        if path.exists():
            continue
        path.write_text(
            yaml.safe_dump(dict(transform_map), sort_keys=False), encoding="utf-8"
        )
