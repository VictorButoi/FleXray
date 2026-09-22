"""Generate a packaging manifest from image/mask directories.

``fxr-dataset scaffold`` pairs images and masks by file stem, assigns seeded
subject-level train/val/test splits, and writes a ``schema_version: 1``
manifest with a ``stored_labels`` stub listing every mask id it observed. The
user renames the stub labels to protocol (or dataset-spec) names and runs
``fxr-dataset validate``/``pack``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .nifti import is_nifti_path, load_nifti

_XRAY_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy")
_CT_SUFFIXES = (".nii", ".nii.gz", ".npy")
_SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class ScaffoldReport:
    """Summary of a scaffolded manifest.

    Attributes:
        manifest: Written manifest path.
        dataset_name: Dataset name recorded in the manifest.
        num_samples: Number of paired samples.
        split_counts: Samples per split.
        observed_label_ids: Sorted mask ids observed across all masks.
    """

    manifest: Path
    dataset_name: str
    num_samples: int
    split_counts: dict[str, int]
    observed_label_ids: tuple[int, ...]


def stable_subject_splits(
    subjects: Sequence[str],
    *,
    seed: int,
    percentages: Sequence[int],
) -> dict[str, str]:
    """Assign subjects to train/val/test deterministically.

    Subjects are sorted, permuted with a seeded generator, and divided by the
    requested percentages using largest-remainder rounding so counts sum
    exactly to the subject count.

    Args:
        subjects: Subject identifiers (duplicates collapse to one subject).
        seed: Random seed for the permutation.
        percentages: ``(train, val, test)`` integer percentages summing to 100.

    Returns:
        Mapping from subject id to split name.
    """

    shares = tuple(int(v) for v in percentages)
    assert len(shares) == 3 and sum(shares) == 100 and min(shares) >= 0, (
        f"split percentages must be three non-negative integers summing to 100; got {percentages!r}."
    )
    ordered = sorted(set(subjects))
    permuted = [ordered[i] for i in np.random.default_rng(seed).permutation(len(ordered))]
    exact = [len(ordered) * share / 100 for share in shares]
    counts = [int(np.floor(v)) for v in exact]
    for index in sorted(range(3), key=lambda i: exact[i] - counts[i], reverse=True):
        if sum(counts) == len(ordered):
            break
        counts[index] += 1
    assignment: dict[str, str] = {}
    start = 0
    for split, count in zip(_SPLIT_NAMES, counts):
        for subject in permuted[start : start + count]:
            assignment[subject] = split
        start += count
    return assignment


def scaffold_manifest(
    dataset_type: str,
    *,
    images: Path,
    masks: Path,
    dataset_name: str,
    output: Path,
    seed: int = 1337,
    percentages: Sequence[int] | None = None,
    subject_regex: str | None = None,
) -> ScaffoldReport:
    """Write a packaging manifest for an image directory and a mask directory.

    Args:
        dataset_type: ``"xray-seg"`` or ``"ct-seg"``.
        images: Directory of images (``.png/.jpg/.tif/.npy``) or CT volumes
            (``.nii/.nii.gz/.npy``).
        masks: Directory of masks with the same file stems.
        dataset_name: Dataset name written to the manifest.
        output: Manifest path to write.
        seed: Split permutation seed.
        percentages: ``(train, val, test)`` percentages; defaults to
            ``(70, 15, 15)`` for X-ray and ``(90, 10, 0)`` for CT.
        subject_regex: Optional regex whose first capture group on the file
            stem is the subject id (default: the stem itself).

    Returns:
        Summary of the written manifest.
    """

    assert dataset_type in {"xray-seg", "ct-seg"}, dataset_type
    suffixes = _XRAY_SUFFIXES if dataset_type == "xray-seg" else _CT_SUFFIXES
    pairs = _pair_by_stem(images, masks, suffixes)
    assert pairs, f"No image/mask pairs with matching stems found under {images} and {masks}."
    pattern = re.compile(subject_regex) if subject_regex else None
    subjects = {stem: _subject_id(stem, pattern) for stem in pairs}
    shares = percentages or ((70, 15, 15) if dataset_type == "xray-seg" else (90, 10, 0))
    splits = stable_subject_splits(list(subjects.values()), seed=seed, percentages=shares)
    observed: set[int] = set()
    samples = []
    for stem, (image_path, mask_path) in pairs.items():
        observed.update(_mask_ids(mask_path))
        samples.append(
            {
                "sample_id": stem,
                "subject_id": subjects[stem],
                "split": splits[subjects[stem]],
                "image": _relative(image_path, output.parent),
                "label": _relative(mask_path, output.parent),
            }
        )
    label_ids = tuple(sorted(observed))
    manifest = {
        "schema_version": 1,
        "dataset_name": dataset_name,
        "dataset_type": dataset_type,
        "stored_labels": {
            label_id: "background" if label_id == 0 else f"label_{label_id}"
            for label_id in label_ids
        },
        "samples": samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    counts = {split: sum(1 for s in samples if s["split"] == split) for split in _SPLIT_NAMES}
    return ScaffoldReport(
        manifest=output,
        dataset_name=dataset_name,
        num_samples=len(samples),
        split_counts={k: v for k, v in counts.items() if v},
        observed_label_ids=label_ids,
    )


def _pair_by_stem(
    images: Path, masks: Path, suffixes: tuple[str, ...]
) -> dict[str, tuple[Path, Path]]:
    """Pair image and mask files sharing a stem (``.nii.gz`` counts as one suffix)."""
    image_files = _files_by_stem(images, suffixes)
    mask_files = _files_by_stem(masks, suffixes)
    return {stem: (image_files[stem], mask_files[stem]) for stem in sorted(image_files) if stem in mask_files}


def _files_by_stem(directory: Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
    """Index a directory's supported files by stem."""
    assert directory.is_dir(), f"{directory} is not a directory."
    indexed: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        name = path.name.lower()
        matched = next((s for s in suffixes if name.endswith(s)), None)
        if path.is_file() and matched is not None:
            stem = path.name[: -len(matched)]
            assert stem not in indexed, f"Duplicate stem {stem!r} under {directory}."
            indexed[stem] = path
    return indexed


def _subject_id(stem: str, pattern: re.Pattern[str] | None) -> str:
    """Derive a subject id from a stem, via the regex capture group when given."""
    if pattern is None:
        return stem
    match = pattern.search(stem)
    assert match is not None and match.groups(), (
        f"--subject-regex must match stem {stem!r} with one capture group."
    )
    return match.group(1)


def _mask_ids(path: Path) -> set[int]:
    """Return the integer ids present in a mask file."""
    if is_nifti_path(path):
        array, _, _ = load_nifti(path, context=str(path))
    elif path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            array = np.array(image)
    return {int(v) for v in np.unique(array)}


def _relative(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` when possible, else absolute."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())
