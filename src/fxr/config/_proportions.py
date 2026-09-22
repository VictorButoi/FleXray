"""Numeric dataloader proportion helpers."""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from math import gcd
from typing import Any


def normalize_proportion_weight(raw_value: Any, name: str) -> Fraction:
    """Return one configured dataloader proportion as an exact numeric weight.

    Args:
        raw_value: Raw config value to normalize.
        name: Human-readable value name used in error messages.

    Returns:
        Exact non-negative sampling weight.

    Raises:
        ValueError: If ``raw_value`` is boolean, non-numeric, negative, or
            non-finite.
    """

    if isinstance(raw_value, bool):
        raise ValueError(
            f"{name} must be a finite non-negative sampling weight; "
            f"got {raw_value!r}."
        )
    if isinstance(raw_value, numbers.Integral):
        value = Fraction(int(raw_value), 1)
    elif isinstance(raw_value, numbers.Real):
        as_float = float(raw_value)
        if not math.isfinite(as_float):
            raise ValueError(
                f"{name} must be a finite non-negative sampling weight; "
                f"got {raw_value!r}."
            )
        value = _decimal_fraction(str(raw_value), name)
    elif isinstance(raw_value, str):
        value = _decimal_fraction(raw_value.strip(), name)
    else:
        raise ValueError(
            f"{name} must be a finite non-negative sampling weight; "
            f"got {raw_value!r}."
        )

    if value < 0:
        raise ValueError(f"{name} must be >= 0; got {raw_value!r}.")
    return value


def positive_proportion_weight(raw_value: Any, name: str) -> Fraction:
    """Return one configured dataloader proportion as a positive weight.

    Args:
        raw_value: Raw config value to normalize.
        name: Human-readable value name used in error messages.

    Returns:
        Exact positive sampling weight.

    Raises:
        ValueError: If ``raw_value`` is not a finite positive sampling weight.
    """

    value = normalize_proportion_weight(raw_value, name)
    if value <= 0:
        raise ValueError(
            f"{name} must be a positive sampling weight; got {raw_value!r}."
        )
    return value


def schedule_counts_from_proportions(
    source_names: Iterable[str],
    proportions: Mapping[str, Any],
    *,
    default: Any = 1,
    name: str = "dataloader.proportions",
) -> dict[str, int]:
    """Return integer schedule counts that exactly preserve relative weights.

    Args:
        source_names: Source names to include in the schedule.
        proportions: Configured sampling weights keyed by source name.
        default: Weight used for sources missing from ``proportions``.
        name: Human-readable base name used in error messages.

    Returns:
        Mapping from source name to a positive integer schedule count.

    Raises:
        ValueError: If any selected source weight is not positive.
    """

    weights = {
        source_name: positive_proportion_weight(
            proportions.get(source_name, default),
            f"{name}[{source_name!r}]",
        )
        for source_name in source_names
    }
    if not weights:
        return {}

    denominator_lcm = 1
    for weight in weights.values():
        denominator_lcm = (
            denominator_lcm
            * weight.denominator
            // gcd(denominator_lcm, weight.denominator)
        )

    return {
        source_name: int(weight * denominator_lcm)
        for source_name, weight in weights.items()
    }


def _decimal_fraction(raw_value: str, name: str) -> Fraction:
    """Return a finite decimal string as an exact fraction."""

    try:
        decimal = Decimal(raw_value)
    except InvalidOperation as exc:
        raise ValueError(
            f"{name} must be a finite non-negative sampling weight; "
            f"got {raw_value!r}."
        ) from exc
    if not decimal.is_finite():
        raise ValueError(
            f"{name} must be a finite non-negative sampling weight; "
            f"got {raw_value!r}."
        )
    return Fraction(decimal)
