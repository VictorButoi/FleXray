"""``fxr-protocol``: inspect protocols and compile dataset label mappings."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .introspect import describe_protocol, explain_lut, list_datasets, list_protocols
from .io import load_dataset_spec
from .registry import load_dataset_spec_by_name, load_protocol_by_name
from .schemas import DatasetSpec

_DEFAULT_PROTOCOL = "all_structures_flexray_v4"


def build_parser() -> argparse.ArgumentParser:
    """Build the ``fxr-protocol`` argument parser.

    Returns:
        Parser with ``list``, ``show``, and ``compile`` commands.
    """

    parser = argparse.ArgumentParser(
        prog="fxr-protocol",
        description="Inspect FleXray label protocols and compile dataset label mappings.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list packaged protocols and dataset specs")
    show = commands.add_parser("show", help="print a protocol's channels in order")
    show.add_argument("protocol", help="protocol name")
    compile_parser = commands.add_parser(
        "compile",
        help="compile and print the native-id -> protocol-channel mapping of a dataset",
    )
    compile_parser.add_argument(
        "--dataset",
        required=True,
        help="packaged dataset spec name/alias, or a path to a dataset spec YAML",
    )
    compile_parser.add_argument(
        "--protocol", default=_DEFAULT_PROTOCOL, help=f"protocol name (default {_DEFAULT_PROTOCOL})"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the protocol command-line interface.

    Args:
        argv: Optional arguments excluding the program name.

    Returns:
        Process exit status ``0`` on success.

    Raises:
        SystemExit: With status 2 and a concise message when a name or spec
            file is invalid.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            print(_format_list(list_protocols(), list_datasets()))
        elif args.command == "show":
            print(_format_protocol(describe_protocol(args.protocol)))
        else:
            protocol = load_protocol_by_name(args.protocol)
            print(_format_mapping(explain_lut(protocol, _dataset_argument(args.dataset))))
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


def _dataset_argument(value: str) -> DatasetSpec:
    """Load a dataset spec from an existing YAML path or a packaged name.

    Args:
        value: Path to a spec file, or a packaged dataset name/alias.

    Returns:
        The loaded dataset spec.
    """

    path = Path(value).expanduser()
    if path.suffix.lower() in {".yml", ".yaml"} or path.is_file():
        return load_dataset_spec(path)
    return load_dataset_spec_by_name(value)


def _format_list(protocols: dict, datasets: dict) -> str:
    """Render the ``list`` output.

    Args:
        protocols: Result of :func:`list_protocols`.
        datasets: Result of :func:`list_datasets`.

    Returns:
        Multi-line text listing protocols (with label counts) and dataset specs.
    """

    lines = [f"protocols ({len(protocols['protocols'])}):"]
    for name in protocols["protocols"]:
        lines.append(f"  {name}  {describe_protocol(name)['num_labels']} labels")
    lines.append(f"dataset specs ({len(datasets['datasets'])}):")
    for entry in datasets["datasets"]:
        suffix = f"  aliases: {', '.join(entry['aliases'])}" if entry["aliases"] else ""
        lines.append(f"  {entry['name']}{suffix}")
    return "\n".join(lines)


def _format_protocol(described: dict) -> str:
    """Render the ``show`` output.

    Args:
        described: Result of :func:`describe_protocol`.

    Returns:
        Header line, one ``channel  label`` line per channel, and a closing note.
    """

    lines = [f"{described['protocol_name']}: {described['num_labels']} labels"]
    lines.extend(f"  {entry['channel']:>3}  {entry['label']}" for entry in described["labels"])
    lines.append(
        "note: channels are protocol positions; a model bundle's output channels "
        "follow its label_schema.json"
    )
    return "\n".join(lines)


def _format_mapping(explained: dict) -> str:
    """Render the ``compile`` output as a fixed-width table.

    Args:
        explained: Result of :func:`explain_lut`.

    Returns:
        Header, one row per declared native id, and the dense ``label_lut``.
    """

    rows = explained["rows"]
    label_width = max(len("native_label"), *(len(r["native_label"]) for r in rows))
    lines = [
        f"{explained['dataset_name']} -> {explained['protocol_name']}",
        f"{'native_id':>9}  {'native_label':<{label_width}}  {'rule':<10}  {'channel':>7}  protocol_label",
    ]
    for row in rows:
        lines.append(
            f"{row['native_id']:>9}  {row['native_label']:<{label_width}}  "
            f"{row['rule']:<10}  {row['protocol_channel']:>7}  {row['protocol_label']}"
        )
    gaps = [i for i, value in enumerate(explained["label_lut"]) if value < 0]
    if gaps:
        lines.append(f"undeclared native ids (LUT -1): {gaps}")
    lines.append(f"label_lut: {explained['label_lut']}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
