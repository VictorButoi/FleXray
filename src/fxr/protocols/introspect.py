"""Protocol and label-space introspection shared by the CLI and MCP server.

Every function is a plain JSON-in/JSON-out callable; `fxr.mcp.server` registers
them as MCP tools, so each docstring doubles as the tool description shown to
AI clients. Only `fxr.protocols` is used.
"""

from __future__ import annotations

from .compiler import compile_training_lut
from .registry import (
    list_dataset_spec_names,
    list_protocol_names,
    load_dataset_spec_by_name,
    load_protocol_by_name,
)
from .schemas import DatasetSpec, ProtocolSpec


def list_protocols() -> dict:
    """List the FleXray segmentation protocols packaged with this install.

    A protocol is a canonical channel label space: channel 0 is always
    `background` and foreground labels follow in fixed order. Use
    `describe_protocol` for the per-channel labels.

    Returns:
        Dictionary with `protocols`, the sorted list of protocol names.
    """

    return {"protocols": list(list_protocol_names())}


def describe_protocol(name: str) -> dict:
    """Describe one FleXray protocol's channel label space.

    Args:
        name: Protocol name from `list_protocols`, for example
            `all_structures_flexray_v4`.

    Returns:
        Dictionary with `protocol_name`, `num_labels`, and `labels`, a list of
        `{label, channel}` entries in channel order (`background` at 0).

    Raises:
        FileNotFoundError: If no packaged protocol has this name.
        ValueError: If the name is empty or path-like.
    """

    protocol = load_protocol_by_name(name)
    return {
        "protocol_name": protocol.protocol_name,
        "num_labels": len(protocol.labels),
        "labels": [
            {"label": label, "channel": channel}
            for channel, label in enumerate(protocol.labels)
        ],
    }


def list_datasets() -> dict:
    """List the dataset specs packaged with this install.

    A dataset spec declares a training dataset's native stored mask ids and how
    they map into FleXray protocols. Use `describe_dataset` for details.

    Returns:
        Dictionary with `datasets`, a list of `{name, aliases}` entries sorted
        by canonical dataset name; `aliases` are alternate names accepted by
        `describe_dataset` and `explain_mapping`.
    """

    datasets = []
    for name in list_dataset_spec_names():
        spec = load_dataset_spec_by_name(name)
        datasets.append({"name": name, "aliases": list(spec.dataset_aliases)})
    return {"datasets": datasets}


def describe_dataset(name: str) -> dict:
    """Describe one dataset spec's native labels and protocol mappings.

    Args:
        name: Dataset name or declared alias from `list_datasets`.

    Returns:
        Dictionary with `dataset_name`, `aliases`, `stored_labels` (a list of
        `{native_id, label}` entries sorted by native id),
        `protocol_label_aliases` (per-protocol mapping from native label to
        protocol label), and `protocol_drop_labels` (per-protocol native labels
        mapped to background).

    Raises:
        FileNotFoundError: If no packaged dataset spec matches the name.
        ValueError: If the name is empty or path-like.
    """

    spec = load_dataset_spec_by_name(name)
    return {
        "dataset_name": spec.dataset_name,
        "aliases": list(spec.dataset_aliases),
        "stored_labels": [
            {"native_id": native_id, "label": label}
            for native_id, label in sorted(spec.stored_labels.items())
        ],
        "protocol_label_aliases": {
            protocol: dict(aliases)
            for protocol, aliases in spec.protocol_label_aliases.items()
        },
        "protocol_drop_labels": {
            protocol: list(drops)
            for protocol, drops in spec.protocol_drop_labels.items()
        },
    }


def explain_mapping(protocol_name: str, dataset_name: str) -> dict:
    """Explain how a dataset's native mask ids map into a protocol.

    Compiles the training lookup table for the pair and reports each declared
    native id's destination channel. A `label_lut` value of `-1` marks a
    native-id gap that is absent from the dataset's stored labels.

    Args:
        protocol_name: Protocol name from `list_protocols`.
        dataset_name: Dataset name or alias from `list_datasets`.

    Returns:
        Dictionary with `protocol_name`, `dataset_name`, `protocol_labels`
        (channel-ordered), `label_lut` (dense list indexed by native id), and
        `rows`, a list of `{native_id, native_label, rule, protocol_channel,
        protocol_label}` entries for every declared native id, where `rule` is
        `background`, `identity`, `alias`, or `drop`.

    Raises:
        FileNotFoundError: If the protocol or dataset spec is missing.
        ValueError: If a name is empty or path-like, or a native label has no
            valid protocol destination.
    """

    return explain_lut(load_protocol_by_name(protocol_name), load_dataset_spec_by_name(dataset_name))


def explain_lut(protocol: ProtocolSpec, dataset_spec: DatasetSpec) -> dict:
    """Explain the compiled mapping of one loaded dataset spec into a protocol.

    Args:
        protocol: Target protocol.
        dataset_spec: Dataset spec (packaged or loaded from a user file).

    Returns:
        The same structure as `explain_mapping`.
    """

    lut = compile_training_lut(protocol, dataset_spec)
    aliases = dataset_spec.protocol_label_aliases.get(protocol.protocol_name, {})
    drops = dataset_spec.protocol_drop_labels.get(protocol.protocol_name, ())
    rows = []
    for native_id, protocol_id in sorted(lut.native_id_to_protocol_id.items()):
        native_label = dataset_spec.stored_labels[native_id]
        if native_id == 0:
            rule = "background"
        elif native_label in drops:
            rule = "drop"
        elif native_label in aliases:
            rule = "alias"
        else:
            rule = "identity"
        rows.append(
            {
                "native_id": native_id,
                "native_label": native_label,
                "rule": rule,
                "protocol_channel": protocol_id,
                "protocol_label": lut.native_id_to_protocol_label[native_id],
            }
        )
    return {
        "protocol_name": lut.protocol_name,
        "dataset_name": lut.dataset_name,
        "protocol_labels": list(lut.protocol_labels),
        "label_lut": list(lut.label_lut),
        "rows": rows,
    }
