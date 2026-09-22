"""Dynamic, config-driven object construction.

Experiment configs describe objects (models, losses, optimizers, callbacks) as
mappings carrying a ``_class`` or ``_fn`` dotted import string plus constructor
keyword arguments. ``eval_config`` walks such a config and instantiates those
objects, mirroring the pattern already used by ``fxr.callbacks.build_callback_runner``.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from importlib import import_module
from typing import Any

from .core import Config


def absolute_import(reference: str) -> Any:
    """Resolve a dotted import string to the referenced object.

    Args:
        reference: Dotted path such as ``"fxr.models.UNet"`` or
            ``"torch.optim.AdamW"``.

    Returns:
        The imported attribute (class or callable).

    Raises:
        TypeError: If ``reference`` is not a string.
        ValueError: If ``reference`` is not a dotted path.
        ImportError: If the module or attribute cannot be resolved.
    """

    if not isinstance(reference, str):
        raise TypeError(
            f"Import reference must be a string, got {type(reference).__name__}."
        )
    module_name, _, attr_name = reference.rpartition(".")
    if not module_name:
        raise ValueError(f"Import reference must be a dotted path, got {reference!r}.")
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise ImportError(f"Could not import module for {reference!r}.") from exc
    if not hasattr(module, attr_name):
        raise ImportError(f"Could not import {reference!r}.")
    return getattr(module, attr_name)


def eval_config(config: Any) -> Any:
    """Recursively instantiate ``_class``/``_fn`` objects within a config.

    Mappings are descended depth-first so nested objects are built before their
    parents. A mapping containing ``_class`` is instantiated by calling the
    resolved class with the remaining entries as keyword arguments; a mapping
    containing ``_fn`` becomes a ``functools.partial`` over the resolved callable.
    Lists are mapped element-wise; scalars pass through unchanged.

    Args:
        config: Config value to evaluate. ``Config``, ``dict``, ``list``, or
            scalar.

    Returns:
        The evaluated value with all ``_class``/``_fn`` mappings instantiated.

    Raises:
        ValueError: If a mapping declares both ``_class`` and ``_fn``.
    """

    if isinstance(config, Config):
        return eval_config(config.to_dict())

    if isinstance(config, list):
        return [eval_config(item) for item in config]

    if not isinstance(config, Mapping):
        return config

    evaluated = {key: eval_config(value) for key, value in config.items()}

    has_class = "_class" in evaluated
    has_fn = "_fn" in evaluated
    if has_class and has_fn:
        raise ValueError("Config may not declare both `_class` and `_fn`.")

    if has_class:
        target = absolute_import(evaluated.pop("_class"))
        return target(**evaluated)
    if has_fn:
        target = absolute_import(evaluated.pop("_fn"))
        return functools.partial(target, **evaluated)

    return evaluated
