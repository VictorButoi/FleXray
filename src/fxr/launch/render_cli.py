"""``fxr-render``: render DRR training samples from a packed CT dataset."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from fxr.launch.cli import _parse_overrides, load_training_base
from fxr.launch.sweep import apply_overrides


def _absolute_path(value: str) -> Path:
    """Resolve a command-line path so a relative dataset location is accepted.

    Args:
        value: Path text as typed on the command line.

    Returns:
        The path with ``~`` expanded and made absolute, as the dataset config
        requires.
    """

    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    """Build the ``fxr-render`` argument parser.

    Returns:
        Configured parser.
    """

    parser = argparse.ArgumentParser(
        prog="fxr-render",
        description=(
            "Render DRR image/mask samples from a packed ct-seg dataset with a "
            "training DRR profile, then pack them with `fxr-dataset pack`."
        ),
    )
    parser.add_argument(
        "dataset", type=_absolute_path, help="packed ct-seg ThunderDB directory"
    )
    parser.add_argument("--dataset-name", required=True, help="CT dataset name")
    parser.add_argument(
        "--base",
        default="base",
        help="packaged training config (fxr/configs/training/NAME.yml) or a YAML path",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="overrides",
        help="config override applied before profile resolution (repeatable)",
    )
    parser.add_argument(
        "--profile", default="MOOSE", help="drr_model.datasets profile to render with"
    )
    parser.add_argument("--output", type=Path, required=True, help="output directory")
    parser.add_argument("--split", default="train", help="CT package split to render")
    parser.add_argument("--renders", type=int, default=1, help="camera draws per volume")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--device", default="auto", help="'auto', 'cpu', or a CUDA device string"
    )
    parser.add_argument("--png", action="store_true", help="also write PNG previews")
    parser.add_argument(
        "--air-clamp-hu", type=float, default=-900.0, help="air threshold in HU"
    )
    parser.add_argument("--output-name", help="dataset name of the rendered manifest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline rendering command.

    Args:
        argv: Optional arguments excluding the program name.

    Returns:
        Process exit status ``0`` on success.

    Raises:
        SystemExit: With status 2 and a concise message on user input errors.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    # Rendering needs the training extras (kornia, nanodrr); keep --help usable without them.
    from fxr.launch.render import OfflineRenderRequest, render_dataset

    try:
        config = apply_overrides(load_training_base(args.base), _parse_overrides(args.overrides))
        report = render_dataset(
            OfflineRenderRequest(
                dataset_path=args.dataset,
                dataset_name=args.dataset_name,
                config=config,
                profile=args.profile,
                output=args.output,
                split=args.split,
                renders=args.renders,
                seed=args.seed,
                device=_resolve_device(args.device),
                png=args.png,
                air_clamp_hu=args.air_clamp_hu,
                output_name=args.output_name,
            )
        )
    except (AssertionError, FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"rendered {report.num_samples} samples from {report.num_volumes} volumes "
        f"({len(report.label_names)} mask channels) -> {report.manifest}"
    )
    print(f"Next: fxr-dataset pack {report.manifest} /path/to/output")
    return 0


def _resolve_device(name: str):
    """Resolve the ``--device`` option to a torch device."""
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
