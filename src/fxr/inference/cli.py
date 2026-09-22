"""Command line prediction for Hugging Face FleXray model bundles."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ._devices import resolve_device
from .artifacts import DEFAULT_MODEL_ID, ENSEMBLE_MEMBER_SUBFOLDERS
from .dicom import DICOM_SUFFIXES, is_dicom_path
from .preprocessing import DEFAULT_IMAGE_SIZE, ImagePreprocessing, _load_image_file
from .segmenter import FleXraySegmenter
from .tta import _validate_tta_samples

_PILLOW_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})
_IMAGE_SUFFIXES = _PILLOW_SUFFIXES | DICOM_SUFFIXES
# These exceptions are recoverable only at the image-decoding boundary.
_IMAGE_READ_FAILURES = (
    OSError, ValueError, RuntimeError, AttributeError, EOFError,
    Image.DecompressionBombError,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the ``flexify`` argument parser.

    Returns:
        Configured argument parser.
    """

    parser = argparse.ArgumentParser(
        prog="flexify",
        description="Run image-file inference with a Hugging Face FleXray model.",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        default=None,
        help=(
            "Hugging Face model id or local bundle directory. Defaults to "
            f"{DEFAULT_MODEL_ID}. Repeat the flag to average a custom set of "
            "them as an ensemble."
        ),
    )
    bundles = parser.add_mutually_exclusive_group()
    bundles.add_argument(
        "--ensemble",
        action="store_true",
        help=(
            "Average every member declared by the model repo's ensemble.json "
            f"({DEFAULT_MODEL_ID} lists {len(ENSEMBLE_MEMBER_SUBFOLDERS)}: "
            f"{', '.join(ENSEMBLE_MEMBER_SUBFOLDERS)})."
        ),
    )
    bundles.add_argument(
        "--subfolder",
        default=None,
        help=(
            "Load one bundle directory of the model repo, e.g. "
            f"{ENSEMBLE_MEMBER_SUBFOLDERS[1]}. Defaults to the repo's flagship."
        ),
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face branch, tag, or commit id.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        dest="input_path",
        help="Image file or directory of image files to predict.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where .npy prediction artifacts are written.",
    )
    parser.add_argument(
        "--binary",
        default=None,
        metavar="LABEL",
        dest="binary_label",
        help=(
            "Restrict output to one label, e.g. --binary femurs. "
            "By default every model label is written."
        ),
    )
    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
        help="Probability threshold for saved uint8 masks. Defaults to 0.5.",
    )
    parser.add_argument(
        "--tta-samples",
        default=1,
        type=int,
        help=(
            "Total number of forward passes per image, including the plain one. "
            "0 or 1 (the default) is a single normal forward pass with no TTA."
        ),
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Seed the test-time-augmentation draws so repeated runs match.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Torch device for inference, e.g. cuda, cuda:1, or cpu. "
            "Defaults to auto: CUDA when available, otherwise CPU."
        ),
    )
    return parser


def _is_supported_image(path: Path) -> bool:
    """Return whether a path is a Pillow-readable image or a DICOM file.

    Args:
        path: Candidate path.

    Returns:
        ``True`` for regular files with a supported suffix or DICOM header.
    """

    return path.is_file() and (
        path.suffix.lower() in _PILLOW_SUFFIXES or is_dicom_path(path)
    )


def collect_input_images(input_path: str | Path) -> tuple[Path, ...]:
    """Return sorted supported image files from a file or directory input.

    Supported inputs are BMP/JPEG/PNG/TIFF files plus DICOM files (``.dcm``,
    ``.dicom``, or extension-less files carrying the ``DICM`` magic).

    Args:
        input_path: One image file or a directory containing image files.

    Returns:
        Tuple of image paths in deterministic order.

    Raises:
        FileNotFoundError: If the input path is missing or a directory contains
            no supported image files.
        ValueError: If the input path is neither a supported image file nor a
            directory.
    """

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}.")
    if path.is_file():
        if not _is_supported_image(path):
            raise ValueError(
                f"Input file is not a supported image type: {path}. "
                f"Supported suffixes: {sorted(_IMAGE_SUFFIXES)} (extension-less "
                "DICOM files are detected by their DICM header)."
            )
        return (path,)
    if path.is_dir():
        images = tuple(
            sorted(child for child in path.iterdir() if _is_supported_image(child))
        )
        if not images:
            raise FileNotFoundError(
                f"No supported image files found in input directory: {path}."
            )
        return images
    raise ValueError(f"Input path must be an image file or directory: {path}.")


def preprocess_image_file(
    image_path: str | Path,
    *,
    size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> torch.Tensor:
    """Load and prepare one image for FleXray inference.

    The preprocessing contract is grayscale conversion, zero padding to square,
    bilinear resize to ``size``, and scaling to ``[0, 1]`` as ``float32``.

    Args:
        image_path: PNG/JPEG-style image path readable by Pillow, or a
            DICOM file (requires ``pydicom``).
        size: Output ``(height, width)``. Defaults to ``(256, 256)``.

    Returns:
        Tensor shaped ``1x1xHxW`` with dtype ``torch.float32``.

    Raises:
        ValueError: If ``size`` does not contain two positive integers.
    """

    return ImagePreprocessing(image_size=size).prepare(image_path)


def predict_from_paths(
    *,
    model_id: str | Sequence[str] = DEFAULT_MODEL_ID,
    input_path: str | Path,
    output_dir: str | Path,
    binary_label: str | None = None,
    threshold: float = 0.5,
    revision: str | None = None,
    subfolder: str | None = None,
    ensemble: bool = False,
    tta_samples: int = 1,
    seed: int | None = None,
    device: str | torch.device | None = "auto",
    on_error: Callable[[Path, Exception], None] | None = None,
) -> list[dict[str, Path]]:
    """Run prediction for one file or directory and write NumPy artifacts.

    Args:
        model_id: Hugging Face model id containing FleXray public artifacts, a
            local bundle directory, or a sequence of either loaded together as
            an ensemble.
        input_path: Image file or directory of image files.
        output_dir: Directory where per-image logits, probabilities, and masks
            are written.
        binary_label: Optional output label name. ``None`` writes every model
            label. A label name restricts artifacts to that single channel.
        threshold: Probability threshold used for saved uint8 masks.
        revision: Optional Hugging Face branch, tag, or commit id.
        subfolder: Optional bundle directory inside a single ``model_id``.
        ensemble: Load every member declared by a single ``model_id``'s
            ``ensemble.json``.
        tta_samples: Total number of test-time-augmentation forward passes per
            image. ``0`` or ``1`` disables TTA; negative values are rejected.
        seed: Optional seed for the test-time-augmentation draws.
        device: Inference device. ``"auto"`` (default) or ``None`` selects
            CUDA when available and CPU otherwise; any other value is passed
            to ``torch.device``.
        on_error: Optional callback receiving an unreadable image path and
            its decoding exception. Providing it skips those inputs after
            calling it; ``None`` preserves fail-fast behavior. Argument,
            preprocessing, model, and output errors always propagate.

    Returns:
        List of dictionaries keyed by ``logits``, ``probabilities``, and
        ``masks`` with paths to written artifacts. Named outputs also receive
        one ``label_names.json`` in channel order. Existing output metadata
        must agree; arrays without metadata cannot establish label order.

    Raises:
        FileNotFoundError: If required model artifacts or inputs are missing.
        NotADirectoryError: If ``output_dir`` exists and is not a directory.
        ValueError: If threshold, label, model metadata, device, or input
            paths are invalid.
        TypeError: If model metadata has an unsupported type.
    """

    _validate_threshold(threshold)
    tta_samples = _validate_tta_samples(tta_samples)
    if seed is None and tta_samples > 1:
        warnings.warn(
            "Test-time augmentation draws from the global torch RNG, so "
            "repeated runs differ; pass --seed (seed=) to reproduce them.",
            UserWarning,
            stacklevel=2,
        )
    target_device = resolve_device(device)
    image_paths = collect_input_images(input_path)
    output_path = Path(output_dir)
    if output_path.exists() and not output_path.is_dir():
        raise NotADirectoryError(
            f"Output directory exists and is not a directory: {output_path}."
        )
    segmenter = FleXraySegmenter.from_pretrained(
        repo_id=model_id,
        revision=revision,
        subfolder=subfolder,
        ensemble=ensemble,
        device=target_device,
    )

    if binary_label is not None:
        segmenter._label_channel(binary_label)
    label_names = (
        (binary_label,) if binary_label is not None else segmenter.label_names
    )
    _validate_output_labels(output_path, label_names)
    written: list[dict[str, Path]] = []
    used_stems: set[str] = set()
    for index, image_path in enumerate(image_paths):
        try:
            image = _load_image_file(image_path)
        except _IMAGE_READ_FAILURES as exc:
            if on_error is None:
                raise
            on_error(image_path, exc)
            continue
        result = segmenter.predict(
            image,
            label=binary_label,
            threshold=threshold,
            tta_samples=tta_samples,
            seed=seed,
        )
        if not written:
            output_path.mkdir(parents=True, exist_ok=True)
            labels_path = output_path / "label_names.json"
            if label_names is not None and not labels_path.exists():
                labels_path.write_text(
                    f"{json.dumps(list(label_names), indent=2)}\n", encoding="utf-8"
                )
        stem = _unique_stem(image_path, index=index, used=used_stems)
        written.append(
            _write_prediction_artifacts(
                output_dir=output_path,
                stem=stem,
                logits=result.logits,
                probabilities=result.probabilities,
                masks=result.masks,
            )
        )
    return written


def _validate_output_labels(output_dir: Path, names: tuple[str, ...] | None) -> None:
    """Check that an output directory can describe the requested channels.

    Args:
        output_dir: Destination directory, which may not yet exist.
        names: Ordered output labels, or ``None`` for unnamed outputs.

    Returns:
        None; no files are modified.

    Raises:
        ValueError: If existing metadata differs, is malformed, or named outputs
            would share a directory with arrays of unknown label order.
    """

    labels_path = output_dir / "label_names.json"
    if labels_path.exists():
        try:
            existing = json.loads(labels_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ValueError(
                f"Invalid label metadata in {labels_path}; use a new output directory."
            ) from exc
        if names is None or existing != list(names):
            raise ValueError(
                f"Output label metadata in {labels_path} conflicts with the requested "
                "channels; use a new output directory."
            )
    elif names is not None and any(
        any(output_dir.glob(f"*_{kind}.npy"))
        for kind in ("logits", "probabilities", "masks")
    ):
        raise ValueError(
            f"Existing prediction arrays in {output_dir} have no label metadata; "
            "their channel order cannot be verified. Use a new output directory."
        )


def _validate_threshold(threshold: float) -> None:
    """Validate a probability threshold.

    Args:
        threshold: Candidate threshold.

    Returns:
        None.

    Raises:
        ValueError: If the threshold is not finite or outside ``[0, 1]``.
    """

    value = float(threshold)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(
            f"threshold must be a finite value in [0, 1], got {threshold!r}."
        )


def _unique_stem(image_path: Path, *, index: int, used: set[str]) -> str:
    """Return a deterministic artifact stem that avoids collisions.

    Args:
        image_path: Source image path.
        index: Zero-based input position.
        used: Mutable set of stems already returned.

    Returns:
        Unique filename stem for prediction artifacts.
    """

    base = image_path.stem or f"image_{index:04d}"
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _write_prediction_artifacts(
    *,
    output_dir: Path,
    stem: str,
    logits: torch.Tensor,
    probabilities: torch.Tensor,
    masks: torch.Tensor,
) -> dict[str, Path]:
    """Write logits, probabilities, and masks for one sample.

    Args:
        output_dir: Destination directory.
        stem: Filename stem used for all artifacts.
        logits: Batched logit tensor shaped ``1xCxHxW``.
        probabilities: Batched probability tensor shaped ``1xCxHxW``.
        masks: Batched uint8 mask tensor shaped ``1xCxHxW``.

    Returns:
        Mapping from artifact kind to written path.
    """

    logits_path = output_dir / f"{stem}_logits.npy"
    probabilities_path = output_dir / f"{stem}_probabilities.npy"
    masks_path = output_dir / f"{stem}_masks.npy"

    np.save(logits_path, logits[0].detach().float().cpu().numpy())
    np.save(probabilities_path, probabilities[0].detach().float().cpu().numpy())
    np.save(masks_path, masks[0].detach().to(torch.uint8).cpu().numpy())
    return {
        "logits": logits_path,
        "probabilities": probabilities_path,
        "masks": masks_path,
    }


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run prediction, and write output artifacts.

    Args:
        argv: Optional argument vector. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code ``0`` on success, or ``1`` when an input was skipped.
    """

    parser = _build_parser()
    args = parser.parse_args(argv)
    model_ids = args.model_id or [DEFAULT_MODEL_ID]
    if (args.ensemble or args.subfolder) and len(model_ids) > 1:
        parser.error("--ensemble and --subfolder apply to a single --model-id.")
    if args.tta_samples < 0:
        parser.error("--tta-samples must not be negative; 0 or 1 disables TTA.")
    failures: list[Path] = []

    def report_unreadable(image_path: Path, error: Exception) -> None:
        """Record and report a skipped image.

        Args:
            image_path: Input file that could not be decoded.
            error: Decoder exception explaining the failure.

        Returns:
            None.
        """

        failures.append(image_path)
        print(f"{parser.prog}: skipped {image_path}: {error}", file=sys.stderr)

    try:
        written = predict_from_paths(
            model_id=model_ids[0] if len(model_ids) == 1 else model_ids,
            revision=args.revision,
            subfolder=args.subfolder,
            ensemble=args.ensemble,
            input_path=args.input_path,
            output_dir=args.output_dir,
            binary_label=args.binary_label,
            threshold=args.threshold,
            tta_samples=args.tta_samples,
            seed=args.seed,
            device=args.device,
            on_error=report_unreadable,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
    print(f"Wrote {len(written)} prediction(s) to {args.output_dir}.")
    if failures:
        print(f"Skipped {len(failures)} unreadable input(s).", file=sys.stderr)
        return 1
    return 0


__all__ = [
    "collect_input_images",
    "main",
    "predict_from_paths",
    "preprocess_image_file",
    "resolve_device",
]


if __name__ == "__main__":
    raise SystemExit(main())
