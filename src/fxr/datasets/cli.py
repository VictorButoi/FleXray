"""Command-line dataset validation and ThunderDB packaging."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .inspect import format_inspection, inspect_thunderdb
from .packaging import (
    DatasetPackageReport,
    pack_dataset,
    validate_dataset_manifest,
    validate_packed_dataset,
)
from .scaffold import ScaffoldReport, scaffold_manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the ``fxr-dataset`` argument parser.

    Returns:
        Configured parser with ``scaffold``, ``validate``, ``pack``, ``check``,
        and ``inspect`` commands.
    """

    parser = argparse.ArgumentParser(
        prog="fxr-dataset",
        description="Validate and package FleXray training datasets.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    scaffold_parser = commands.add_parser(
        "scaffold",
        help="write a manifest from an image directory and a mask directory",
    )
    scaffold_parser.add_argument(
        "dataset_type", choices=("xray-seg", "ct-seg"), help="package type to scaffold"
    )
    scaffold_parser.add_argument("--images", type=Path, required=True, help="image directory")
    scaffold_parser.add_argument(
        "--masks", type=Path, required=True, help="mask directory (same file stems)"
    )
    scaffold_parser.add_argument("--name", required=True, help="dataset name")
    scaffold_parser.add_argument(
        "--output", type=Path, required=True, help="manifest path to write"
    )
    scaffold_parser.add_argument(
        "--split",
        type=int,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        help="subject-level split percentages (default 70 15 15 / 90 10 0 for CT)",
    )
    scaffold_parser.add_argument("--seed", type=int, default=1337, help="split seed")
    scaffold_parser.add_argument(
        "--subject-regex",
        help="regex whose first capture group on the file stem is the subject id",
    )

    validate_parser = commands.add_parser(
        "validate",
        help="validate a YAML packaging manifest without writing a database",
    )
    validate_parser.add_argument(
        "manifest",
        type=Path,
        metavar="MANIFEST",
        help="YAML manifest describing one ct-seg or xray-seg dataset.",
    )

    pack_parser = commands.add_parser(
        "pack",
        help="pack a YAML manifest into canonical ThunderDB storage",
    )
    pack_parser.add_argument(
        "manifest",
        type=Path,
        metavar="MANIFEST",
        help="Validated YAML packaging manifest.",
    )
    pack_parser.add_argument(
        "output",
        type=Path,
        metavar="OUTPUT",
        help="Destination directory for the canonical ThunderDB package.",
    )
    pack_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output only after the new package validates",
    )

    check_parser = commands.add_parser(
        "check",
        help="validate and smoke-load an existing packaged ThunderDB",
    )
    check_parser.add_argument(
        "dataset",
        type=Path,
        metavar="DATASET",
        help="Existing packaged dataset directory to validate and smoke-load.",
    )

    inspect_parser = commands.add_parser(
        "inspect",
        help="describe any split/sample ThunderDB (including non-canonical ones)",
    )
    inspect_parser.add_argument(
        "dataset", type=Path, metavar="DATASET", help="ThunderDB directory to describe."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dataset command-line interface.

    Args:
        argv: Optional arguments excluding the program name. ``None`` reads
            process arguments through ``argparse``.

    Returns:
        Process exit status ``0`` after successful validation or packaging.

    Raises:
        SystemExit: With status 2 and a concise parser error when expected user
            input, dependency, filesystem, or package validation fails.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "scaffold":
            report = scaffold_manifest(
                args.dataset_type,
                images=args.images,
                masks=args.masks,
                dataset_name=args.name,
                output=args.output,
                seed=args.seed,
                percentages=args.split,
                subject_regex=args.subject_regex,
            )
            print(_format_scaffold(report))
            return 0
        if args.command == "inspect":
            print(format_inspection(inspect_thunderdb(args.dataset)))
            return 0
        if args.command == "validate":
            report = validate_dataset_manifest(args.manifest)
        elif args.command == "pack":
            report = pack_dataset(
                args.manifest,
                args.output,
                overwrite=args.overwrite,
            )
        else:
            report = validate_packed_dataset(args.dataset)
    except (AssertionError, ImportError, KeyError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(_format_report(report))
    return 0


def _format_scaffold(report: ScaffoldReport) -> str:
    """Format a scaffold summary with the follow-up step.

    Args:
        report: Scaffold result.

    Returns:
        Two-line summary for terminal output.
    """

    splits = ", ".join(f"{name}={count}" for name, count in report.split_counts.items())
    ids = ", ".join(str(i) for i in report.observed_label_ids)
    return (
        f"{report.dataset_name}: {report.num_samples} samples ({splits}), mask ids [{ids}] "
        f"-> {report.manifest}\n"
        f"Rename the stored_labels stub in {report.manifest.name} to protocol label names, "
        "then run `fxr-dataset validate`."
    )


def _format_report(report: DatasetPackageReport) -> str:
    """Format a compact human-readable package summary.

    Args:
        report: Validated package report.

    Returns:
        One-line summary for terminal output.
    """

    splits = ", ".join(
        f"{name}={count}" for name, count in report.split_counts.items()
    )
    return (
        f"{report.dataset_name}: {report.dataset_type}, "
        f"{report.label_encoding} labels, {report.num_subjects} subjects, "
        f"{report.num_samples} samples ({splits}) -> {report.path}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
