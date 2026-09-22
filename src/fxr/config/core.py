"""Hierarchical, read-only experiment configuration.

``Config`` is a thin, dependency-free substitute for the configuration object
FleXray experiments are parameterized by. It wraps a plain nested ``dict`` and
exposes dotted-key access (``config["train.epochs"]``) plus the read helpers the
training stack relies on. Unlike the upstream implementation it is intentionally
read-only: experiments rebuild a new ``Config`` rather than mutating one in place.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterator

import yaml

_SEP = "."

# Sentinel distinguishing "missing key" from an explicit ``None`` default.
_MISSING = object()


def _split_key(key: str) -> tuple[str, ...]:
    """Split a dotted key string into its component segments.

    Args:
        key: Either a single segment (``"data"``) or a dotted path
            (``"train.epochs"``).

    Returns:
        Tuple of key segments in order.

    Raises:
        TypeError: If ``key`` is not a string.
    """

    if not isinstance(key, str):
        raise TypeError(f"Config keys must be strings, got {type(key).__name__}.")
    return tuple(key.split(_SEP))


def _get_nested(data: Mapping[str, Any], key: str) -> Any:
    """Resolve a dotted key against a nested mapping.

    Args:
        data: Nested mapping to traverse.
        key: Dotted key path.

    Returns:
        The value stored at ``key``.

    Raises:
        KeyError: If any segment of ``key`` is absent.
    """

    value: Any = data
    seen: list[str] = []
    for segment in _split_key(key):
        seen.append(segment)
        if not isinstance(value, Mapping) or segment not in value:
            raise KeyError(_SEP.join(seen))
        value = value[segment]
    return value


def _contains_nested(data: Mapping[str, Any], key: str) -> bool:
    """Return whether a dotted key resolves within a nested mapping."""
    value: Any = data
    for segment in _split_key(key):
        if not isinstance(value, Mapping) or segment not in value:
            return False
        value = value[segment]
    return True


def _flatten(data: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a nested mapping into dotted-key leaves.

    Args:
        data: Nested mapping to flatten.

    Returns:
        Mapping from dotted key to leaf (non-mapping) value.
    """

    flat: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Mapping):
            for sub_key, sub_value in _flatten(value).items():
                flat[f"{key}{_SEP}{sub_key}"] = sub_value
        else:
            flat[key] = value
    return flat


class Config(Mapping):
    """Read-only hierarchical configuration backed by a nested ``dict``.

    Attributes:
        _data: The wrapped nested mapping. Treated as owned and never mutated.
    """

    def __init__(self, initial: Mapping[str, Any] | "Config" | None = None) -> None:
        """Wrap an existing mapping without copying caller-visible structure.

        Args:
            initial: Source mapping, another ``Config``, or ``None`` for empty.

        Returns:
            ``None``.
        """

        if isinstance(initial, Config):
            initial = initial._data
        self._data: dict[str, Any] = dict(initial) if initial is not None else {}

    def __getitem__(self, key: str) -> Any:
        """Return the value at a dotted key, wrapping sub-mappings as ``Config``.

        Args:
            key: Dotted key path to resolve.

        Returns:
            Stored leaf value or a ``Config`` wrapping the selected sub-mapping.
        """
        value = _get_nested(self._data, key)
        return Config(value) if isinstance(value, Mapping) else value

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value at ``key`` or ``default`` when it is absent.

        Args:
            key: Dotted key path to resolve.
            default: Value returned when ``key`` is absent.

        Returns:
            Stored value at ``key`` or ``default``.
        """
        if not _contains_nested(self._data, key):
            return default
        return self[key]

    def __contains__(self, key: object) -> bool:
        """Return whether a dotted key resolves in the config.

        Args:
            key: Candidate dotted key path.

        Returns:
            ``True`` when ``key`` is a string that resolves, otherwise ``False``.
        """
        return isinstance(key, str) and _contains_nested(self._data, key)

    def __iter__(self) -> Iterator[str]:
        """Iterate over the top-level keys.

        Returns:
            Iterator over top-level key strings.
        """
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of top-level keys.

        Returns:
            Number of top-level entries.
        """
        return len(self._data)

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy of the wrapped mapping as a plain ``dict``.

        Returns:
            Independent nested dictionary.
        """
        return copy.deepcopy(self._data)

    def flatten(self) -> dict[str, Any]:
        """Return the config flattened to dotted-key leaves.

        Returns:
            Mapping from dotted paths to leaf values.
        """
        return _flatten(self._data)

    def __repr__(self) -> str:
        """Return a concise constructor-style config representation.

        Returns:
            Debug representation of this config.
        """
        return f"{type(self).__name__}({self._data!r})"

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        """Load a ``Config`` from a YAML file.

        Args:
            path: Path to a YAML document.

        Returns:
            ``Config`` wrapping the parsed mapping.
        """

        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(yaml.safe_load(handle))

    # Alias matching the upstream loader name used by callers.
    load = from_file
