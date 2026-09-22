from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from importlib import import_module
from typing import Any

from .runner import CallbackRunner


def build_callback_runner(specs: Sequence[Mapping[str, Any]]) -> CallbackRunner:
    """Instantiate callbacks from import-string specs and wrap them in a runner.

    Args:
        specs: Sequence of callback config mappings. Each mapping must define
            ``_class`` as a dotted import string; remaining entries are passed
            as constructor keyword arguments.

    Returns:
        ``CallbackRunner`` containing the constructed callbacks in spec order.

    Raises:
        TypeError: If ``specs`` or one of its entries has an invalid type.
        ValueError: If a spec omits ``_class`` or uses an invalid import string.
    """

    if isinstance(specs, (str, bytes)) or not isinstance(specs, Sequence):
        raise TypeError(
            f"Callback specs must be a sequence, got {type(specs).__name__}."
        )

    callbacks: list[object] = []
    for index, spec in enumerate(specs):
        if not isinstance(spec, Mapping):
            raise TypeError(
                "Callback specs must contain only mappings; "
                f"spec {index} has type {type(spec).__name__}."
            )
        spec_copy = deepcopy(dict(spec))
        class_value = spec_copy.pop("_class", None)
        if class_value is None:
            raise ValueError(f"Callback spec {index} must define `_class`.")
        callback_class = _resolve_callback_class(class_value)
        callbacks.append(callback_class(**spec_copy))

    return CallbackRunner(callbacks)


def _resolve_callback_class(value: Any) -> type:
    """Resolve a callback class object from a dotted import string.

    Args:
        value: Dotted import path to a callback class.

    Returns:
        Resolved callback class.

    Raises:
        TypeError: If ``value`` is not a string or the resolved object is not a
            class.
        ValueError: If a string is not a dotted import path.
    """

    if not isinstance(value, str):
        raise TypeError(
            "`_class` must be a dotted import string; " f"got {type(value).__name__}."
        )
    if "." not in value:
        raise ValueError(
            f"Callback `_class` must be a dotted import string, got {value!r}."
        )

    module_name, attr_name = value.rsplit(".", 1)
    module = import_module(module_name)
    callback_class = getattr(module, attr_name)
    if not isinstance(callback_class, type):
        raise TypeError(f"Resolved callback `_class` {value!r} is not a class.")
    return callback_class
