"""Pure dataloader-option validation shared by runtime and launch checks."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

_DATALOADER_OPTION_KEYS = frozenset(
    {
        "batch_size",
        "num_workers",
        "pin_memory",
        "prefetch_factor",
        "persistent_workers",
    }
)
_DATALOADER_CONTROL_KEYS = frozenset(
    {"iters_per_epoch", "proportions", "CT", "Xray"}
)


def loader_options(
    dataloader_config: Mapping[str, Any],
    *,
    modality: str,
) -> dict[str, Any]:
    """Return validated ``DataLoader`` options for one source modality.

    Args:
        dataloader_config: Top-level ``dataloader`` config mapping.
        modality: Runtime modality such as ``"ct"`` or ``"xray"``.

    Returns:
        Keyword arguments accepted by ``torch.utils.data.DataLoader``.

    Raises:
        TypeError: If a modality override is not mapping-like.
        ValueError: If keys, counts, or boolean worker settings are invalid.
    """

    unknown_global = sorted(
        set(dataloader_config) - _DATALOADER_OPTION_KEYS - _DATALOADER_CONTROL_KEYS
    )
    if unknown_global:
        raise ValueError(f"Unexpected dataloader keys: {unknown_global}.")

    option_keys = set(_DATALOADER_OPTION_KEYS)
    if modality == "ct":
        option_keys.discard("batch_size")
    base = {
        key: dataloader_config[key]
        for key in option_keys
        if key in dataloader_config
    }
    override_key = modality_loader_key(modality)
    override = dataloader_config.get(override_key)
    if override is not None:
        if not isinstance(override, Mapping):
            raise TypeError(f"dataloader.{override_key} must be a mapping.")
        unknown_override = sorted(set(override) - _DATALOADER_OPTION_KEYS)
        if unknown_override:
            raise ValueError(
                f"Unexpected dataloader.{override_key} keys: {unknown_override}."
            )
        base.update(
            {key: override[key] for key in _DATALOADER_OPTION_KEYS if key in override}
        )

    if modality == "ct":
        batch_size = _integer_option(
            base.get("batch_size", 1),
            name="dataloader.CT.batch_size",
            minimum=1,
        )
        if batch_size != 1:
            raise ValueError(
                "dataloader.CT.batch_size must be 1 for CT volume loaders."
            )
    else:
        batch_size = _integer_option(
            base.get("batch_size", dataloader_config.get("batch_size", 1)),
            name=f"dataloader.{override_key}.batch_size",
            minimum=1,
        )

    num_workers = _integer_option(
        base.get("num_workers", 0),
        name=f"dataloader.{override_key}.num_workers",
        minimum=0,
    )

    options: dict[str, Any] = {"batch_size": batch_size, "num_workers": num_workers}
    if "pin_memory" in base:
        options["pin_memory"] = _boolean_option(
            base["pin_memory"], name=f"dataloader.{override_key}.pin_memory"
        )
    if "prefetch_factor" in base:
        if num_workers <= 0:
            raise ValueError("dataloader.prefetch_factor requires num_workers > 0.")
        options["prefetch_factor"] = _integer_option(
            base["prefetch_factor"],
            name=f"dataloader.{override_key}.prefetch_factor",
            minimum=1,
        )
    if "persistent_workers" in base:
        persistent = _boolean_option(
            base["persistent_workers"],
            name=f"dataloader.{override_key}.persistent_workers",
        )
        if persistent and num_workers <= 0:
            raise ValueError("dataloader.persistent_workers requires num_workers > 0.")
        options["persistent_workers"] = persistent
    return options


def modality_loader_key(modality: str) -> str:
    """Return the config override key for one runtime modality.

    Args:
        modality: Runtime modality string.

    Returns:
        ``"CT"`` for CT and ``"Xray"`` for all 2D image sources.
    """

    return "CT" if modality == "ct" else "Xray"


def _integer_option(value: Any, *, name: str, minimum: int) -> int:
    """Validate and return one integral dataloader option.

    Args:
        value: Candidate integer value.
        name: Config path used in validation errors.
        minimum: Smallest accepted integer.

    Returns:
        Normalized Python integer.

    Raises:
        ValueError: If ``value`` is boolean, non-integral, or below ``minimum``.
    """

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return normalized


def _boolean_option(value: Any, *, name: str) -> bool:
    """Validate and return one boolean dataloader option.

    Args:
        value: Candidate boolean value.
        name: Config path used in validation errors.

    Returns:
        The validated boolean.

    Raises:
        TypeError: If ``value`` is not an actual boolean.
    """

    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")
    return value
