"""Cartesian expansion of a sweep spec into concrete training configs.

A sweep is a base config plus a mapping of dotted-key overrides whose values may
be lists. ``expand_configs`` takes the Cartesian product of those lists and
deep-merges each combination onto the base, producing one ready config per cell.
``experiment.seed_range`` is a convenience that expands a single seed into a
contiguous range. Run identity is intentionally left to the launcher, so this
module stays free of any filesystem or framework concern.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from fxr.config import Config, merge_configs


def validate_sweep_axes(axes: Mapping[str, Any]) -> None:
    """Validate explicit YAML or CLI axes before Cartesian expansion.

    Args:
        axes: Dotted-key mapping of non-empty lists, with a positive integer
            allowed for the ``experiment.seed_range`` convenience axis.

    Returns:
        ``None`` when every axis is valid.

    Raises:
        ValueError: If an axis has the wrong type or is empty.
    """

    for key, value in axes.items():
        if key == "experiment.seed_range":
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("experiment.seed_range must be a positive integer.")
        elif not isinstance(value, list) or not value:
            raise ValueError(f"Sweep {key!r} must be assigned a non-empty YAML list.")


def nest_overrides(flat: Mapping[str, Any]) -> dict[str, Any]:
    """Expand dotted-key overrides into a nested mapping.

    Args:
        flat : Mapping
            Mapping from dotted key (``"optim.lr"``) to a single value.

    Returns:
        Nested ``dict`` equivalent of ``flat``.
    """

    nested: dict[str, Any] = {}
    for dotted_key, value in flat.items():
        *parents, leaf = dotted_key.split(".")
        cursor = nested
        for part in parents:
            cursor = cursor.setdefault(part, {})
        cursor[leaf] = value
    return nested


def apply_overrides(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Assign dotted-key overrides as literal leaf values.

    Args:
        base: Base mapping copied before assignment.
        overrides: Dotted keys mapped to exact replacement values.

    Returns:
        A new nested config mapping.

    Raises:
        ValueError: If a dotted key contains an empty segment.
    """

    result = deepcopy(dict(base))
    for dotted_key, value in overrides.items():
        parts = str(dotted_key).split(".")
        if any(not part for part in parts):
            raise ValueError(f"Invalid empty segment in override key {dotted_key!r}.")
        cursor = result
        for part in parts[:-1]:
            child = cursor.get(part)
            if not isinstance(child, dict):
                child = {}
                cursor[part] = child
            cursor = child
        cursor[parts[-1]] = deepcopy(value)
    return result


def expand_configs(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Expand a base config and dotted-key overrides into concrete configs.

    Args:
        base : Mapping
            Base config every expanded config is merged onto.
        overrides : Mapping or None, default=None
            Mapping from dotted key to a value or a list of values. List values
            are swept via their Cartesian product. ``experiment.seed_range`` (an
            int) expands ``experiment.seed`` into ``seed, seed + 1, ...``.

    Returns:
        One merged config ``dict`` per combination, in product order.
    """

    overrides = dict(overrides or {})
    seed_range = overrides.pop("experiment.seed_range", None)
    if seed_range is not None:
        seed = overrides.pop("experiment.seed", Config(base).get("experiment.seed", 40))
        if isinstance(seed, list):
            raise ValueError("experiment.seed must be scalar with seed_range.")
        if int(seed_range) <= 0:
            raise ValueError("experiment.seed_range must be a positive integer.")
        overrides["experiment.seed"] = [int(seed) + offset for offset in range(int(seed_range))]

    option_set = {
        key: value if isinstance(value, list) else [value]
        for key, value in overrides.items()
    }
    keys = list(option_set)
    if not keys:
        return [merge_configs(base, {})]

    configs: list[dict[str, Any]] = []
    for combination in itertools.product(*(option_set[key] for key in keys)):
        configs.append(apply_overrides(base, dict(zip(keys, combination))))
    return configs
