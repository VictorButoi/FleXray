"""Config identity, merging, and validation helpers.

These functions support the experiment bookkeeping that the trainer performs at
construction time: stamping a run with a time-ordered unique id, hashing the
config for reproducibility, deep-merging overrides onto a base config, and
rejecting unfilled placeholder values.
"""

from __future__ import annotations

import datetime
import hashlib
import random
import string
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import yaml

from .core import Config

# Placeholder value marking a config entry that must be supplied at launch.
_MISSING_PLACEHOLDER = "?"


def generate_tuid(nonce_length: int = 4) -> tuple[str, str]:
    """Generate a time-ordered unique id pair for a new run.

    Args:
        nonce_length: Number of random alphanumeric characters in the nonce.

    Returns:
        Tuple of ``(timestamp, nonce)`` where ``timestamp`` is
        ``YYYYmmdd_HHMMSS`` and ``nonce`` is an uppercase alphanumeric string.
    """

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    alphabet = string.ascii_uppercase + string.digits
    nonce = "".join(random.choices(alphabet, k=nonce_length))
    return timestamp, nonce


def validate_run_id(run_id: str) -> tuple[str, str, str]:
    """Validate and split a safe explicit run-directory identifier.

    Explicit ids retain the historical three-part contract while path
    separators and traversal names are rejected before joining ``log.root``.

    Args:
        run_id: Candidate identifier with exactly three nonempty
            hyphen-separated metadata parts.

    Returns:
        Tuple containing the three metadata parts.

    Raises:
        TypeError: If ``run_id`` is not a string.
        ValueError: If the value is not one safe filename component or does not
            contain exactly three nonempty parts.
    """

    if not isinstance(run_id, str):
        raise TypeError("run_id must be a string.")
    if run_id in {"", ".", ".."} or any(
        separator in run_id for separator in ("/", "\\")
    ) or any(ord(character) < 32 or ord(character) == 127 for character in run_id):
        raise ValueError("run_id must be one safe filename component.")
    parts = tuple(run_id.split("-"))
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError(
            "run_id must contain exactly three nonempty hyphen-separated parts."
        )
    return parts


def config_digest(config: Mapping[str, Any] | Config) -> str:
    """Return a stable MD5 digest of a config.

    Keys are sorted before hashing so logically identical configs produce the
    same digest regardless of insertion order.

    Args:
        config: Config mapping or ``Config`` to hash.

    Returns:
        Hex MD5 digest string.
    """

    if isinstance(config, Config):
        config = config.to_dict()
    serialized = yaml.safe_dump(config, sort_keys=True).encode("utf-8")
    return hashlib.md5(serialized).hexdigest()


def merge_configs(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge ``override`` onto a copy of ``base``.

    Nested mappings are merged recursively; non-mapping values in ``override``
    replace those in ``base``.

    Args:
        base: Base config mapping (not mutated).
        override: Override config mapping whose entries take precedence.

    Returns:
        New merged ``dict``.
    """

    merged = deepcopy(dict(base))
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = merge_configs(existing, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def check_missing(config: Mapping[str, Any] | Config) -> None:
    """Raise if any config leaf is still the unfilled placeholder ``"?"``.

    Args:
        config: Config mapping or ``Config`` to validate.

    Returns:
        ``None``.

    Raises:
        ValueError: If any flattened leaf equals ``"?"``.
    """

    flat = Config(config).flatten()
    missing = sorted(key for key, value in flat.items() if value == _MISSING_PLACEHOLDER)
    if missing:
        raise ValueError(f"Config is missing required values for: {missing}.")
