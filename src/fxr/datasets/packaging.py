"""Validated packaging of user-owned segmentation data into ThunderDB."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from PIL import Image, UnidentifiedImageError

from ._thunderdb import open_thunderdb
from .ct_crops import (
    AIR_HU,
    CropPlan,
    CropSpec,
    crop_metadata,
    crop_sample_id,
    crop_z,
    offset_affine,
    plan_z_crops,
)
from .nifti import is_nifti_path, load_nifti
from .preprocessing import CTPreprocessing, XrayPreprocessing
from .runtime import CTTrainingDataset, XrayTrainingDataset
from .storage import SplitThunderDBStorageBackend

DatasetType = Literal["ct-seg", "xray-seg"]
LabelEncoding = Literal["dense", "channels"]

_SCHEMA_VERSION = 1
_SCHEMA_NAME = "flexray-training-dataset"
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "dataset_name",
    "dataset_type",
    "protocol_name",
    "label_names",
    "stored_labels",
    "samples",
    "preprocessing",
    "crops",
}
_SAMPLE_FIELDS = {
    "sample_id",
    "subject_id",
    "split",
    "image",
    "label",
    "affine",
    "spacing",
    "metadata",
}


@dataclass(frozen=True)
class DatasetPackageReport:
    """Summary of a validated manifest or packed ThunderDB dataset.

    Attributes:
        path: Manifest or ThunderDB path described by the report.
        dataset_name: Dataset identity stored in the package.
        dataset_type: Supported segmentation type, either ``"ct-seg"`` or
            ``"xray-seg"``.
        label_encoding: ``"dense"`` for integer maps or ``"channels"`` for
            named channel-first X-ray masks.
        num_subjects: Number of distinct subject identifiers.
        num_samples: Number of stored samples.
        split_counts: Sample counts keyed by explicit split name.
    """

    path: Path
    dataset_name: str
    dataset_type: DatasetType
    label_encoding: LabelEncoding
    num_subjects: int
    num_samples: int
    split_counts: dict[str, int]


@dataclass(frozen=True)
class _PayloadPlan:
    """On-disk location and validated structural contract for one payload.

    Attributes:
        path: Resolved source file that will be reopened during serialization.
        shape: Canonical array shape serialization must reproduce.
        dtype: Canonical NumPy dtype string serialization must reproduce.
    """

    path: Path
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class _SamplePlan:
    """Validated payload descriptors and metadata for one package sample.

    Attributes:
        sample_id: Stable key used to store the sample in ThunderDB.
        subject_id: Subject identifier used to enforce split disjointness.
        split: Explicit split containing the sample.
        image: Validated image or volume source descriptor.
        label: Validated dense or channel-first label source descriptor.
        affine: Optional CT voxel-to-world affine source descriptor.
        spacing: Optional CT voxel spacing source descriptor.
        label_encoding: Validated label representation for this sample.
        metadata: User metadata augmented with ``subject_id``.
        crop: Planned axial crop window for crop-backed CT packages, or ``None``.
    """

    sample_id: str
    subject_id: str
    split: str
    image: _PayloadPlan
    label: _PayloadPlan
    affine: _PayloadPlan | None
    spacing: _PayloadPlan | None
    label_encoding: LabelEncoding
    metadata: dict[str, Any]
    crop: CropSpec | None = None


@dataclass(frozen=True)
class _PackagePlan:
    """Fully validated dataset package ready for ThunderDB serialization.

    Attributes:
        source_path: YAML manifest that produced the plan.
        dataset_name: Dataset identity for every sample.
        dataset_type: Supported segmentation package type.
        protocol_name: Optional target protocol declared by the user.
        label_names: Ordered names for channel-first labels, if applicable.
        stored_labels: Dense native label names keyed by contiguous integer id,
            if applicable.
        label_encoding: Common label representation used by all samples.
        samples: Validated samples in manifest order.
        subjects: Distinct subject identifiers in first-seen order.
        splits: Sample identifiers grouped by explicit split.
        preprocessing: Optional packaging-time preprocessing applied to every sample.
        crops: Optional CT crop plan (``storage_layout: crops``).
        dropped_background_only_crops: Ids of planned crops without foreground.
    """

    source_path: Path
    dataset_name: str
    dataset_type: DatasetType
    protocol_name: str | None
    label_names: tuple[str, ...] | None
    stored_labels: dict[int, str] | None
    label_encoding: LabelEncoding
    samples: tuple[_SamplePlan, ...]
    subjects: tuple[str, ...]
    splits: dict[str, tuple[str, ...]]
    preprocessing: XrayPreprocessing | CTPreprocessing | None = None
    crops: CropPlan | None = None
    dropped_background_only_crops: tuple[str, ...] = ()


def validate_dataset_manifest(manifest: str | Path) -> DatasetPackageReport:
    """Load and validate a FleXray dataset-packaging manifest.

    Validation checks every referenced payload one sample at a time, retaining
    only its path and structural contract, and verifies that a subject never
    appears in more than one split. It does not write a database.

    Args:
        manifest: YAML manifest path.

    Returns:
        Report describing the validated input.

    Raises:
        FileNotFoundError: If the manifest or a referenced payload is missing.
        TypeError: If a manifest container or metadata value has the wrong type.
        ValueError: If the schema, payloads, or subject splits are invalid.
    """

    plan = _load_package_plan(manifest)
    return _report_from_plan(plan, path=plan.source_path)


def pack_dataset(
    manifest: str | Path,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> DatasetPackageReport:
    """Pack a validated segmentation manifest into canonical ThunderDB storage.

    The database is built in a sibling temporary directory and smoke-tested
    through FleXray's storage backend and runtime dataset before it replaces the
    requested destination. Existing output is preserved unless ``overwrite`` is
    true.

    Args:
        manifest: YAML manifest path.
        output: Destination ThunderDB directory.
        overwrite: Whether a successfully built package may replace an existing
            destination.

    Returns:
        Report describing the committed ThunderDB package.

    Raises:
        FileExistsError: If ``output`` exists and ``overwrite`` is false.
        ImportError: If the training dependency ``thunderpack`` is unavailable.
        OSError: If the database cannot be written or committed.
        TypeError: If manifest values have incompatible types.
        ValueError: If manifest or payload validation fails.
    """

    plan = _load_package_plan(manifest)
    requested_output = Path(output).expanduser()
    if requested_output.is_symlink():
        raise FileExistsError(
            f"Output ThunderDB cannot be a symbolic link: {requested_output}."
        )
    output_path = requested_output.resolve()
    broad_targets = {Path.cwd().resolve(), Path.home().resolve()}
    if output_path.parent == output_path or output_path in broad_targets:
        raise ValueError(
            f"Output must name a dedicated non-root dataset directory: {output_path}."
        )
    contained_sources = [
        path
        for path in _planned_source_paths(plan)
        if path.is_relative_to(output_path)
    ]
    if contained_sources:
        raise ValueError(
            "Output cannot contain the source manifest or payload files: "
            f"{output_path}. Contained sources: "
            f"{[str(path) for path in contained_sources]!r}."
        )
    if output_path.exists() and not output_path.is_dir():
        raise FileExistsError(
            f"Output ThunderDB must be a directory path: {output_path}."
        )
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output ThunderDB already exists: {output_path}. "
            "Pass overwrite=True or --overwrite to replace it."
        )
    if output_path.exists():
        _validate_overwrite_target(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.parent / (
        f".{output_path.name}.fxr-pack-{uuid.uuid4().hex}"
    )
    try:
        temporary_path.mkdir()
        _write_package(plan, temporary_path)
        validate_packed_dataset(temporary_path)
        _commit_package(
            temporary_path,
            output_path,
            overwrite=overwrite,
        )
    except BaseException:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        raise
    return _report_from_plan(plan, path=output_path)


def validate_packed_dataset(path: str | Path) -> DatasetPackageReport:
    """Validate and smoke-load a canonical FleXray ThunderDB package.

    Args:
        path: ThunderDB directory produced by :func:`pack_dataset`.

    Returns:
        Report describing the validated package.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ImportError: If the training dependency ``thunderpack`` is unavailable.
        KeyError: If a canonical metadata or payload key is absent.
        TypeError: If stored metadata has an incompatible container type.
        ValueError: If package identities, splits, payloads, or shapes are
            invalid, including a pickled stored value.
    """

    package_path = Path(path).expanduser().resolve()
    if not package_path.exists():
        raise FileNotFoundError(package_path)
    with open_thunderdb(package_path) as db:
        report = _validate_database_mapping(db, package_path)
        _smoke_load_runtime(db, report)
    return report


def _load_package_plan(manifest: str | Path) -> _PackagePlan:
    """Parse a YAML manifest and validate all referenced samples.

    Args:
        manifest: YAML manifest path.

    Returns:
        Validated in-memory package plan.
    """

    manifest_path = Path(manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    try:
        with open(manifest_path, encoding="utf-8") as stream:
            raw_manifest = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Could not parse YAML manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(raw_manifest, Mapping):
        raise TypeError("Dataset packaging manifest must be a YAML mapping.")
    data = dict(raw_manifest)
    _reject_unknown_fields(data, _TOP_LEVEL_FIELDS, context="manifest")
    schema_version = data.get("schema_version")
    if schema_version != _SCHEMA_VERSION:
        raise ValueError(
            f"manifest schema_version must be {_SCHEMA_VERSION}, "
            f"got {schema_version!r}."
        )
    dataset_name = _required_text(data.get("dataset_name"), "dataset_name")
    dataset_type = _dataset_type(data.get("dataset_type"))
    protocol_name = _optional_text(data.get("protocol_name"), "protocol_name")
    label_names = _optional_label_names(data.get("label_names"))
    stored_labels = _optional_stored_labels(data.get("stored_labels"))
    preprocessing = _preprocessing(data.get("preprocessing"), dataset_type=dataset_type)
    crops = _crop_plan(data.get("crops"), dataset_type=dataset_type)
    raw_samples = data.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("manifest samples must be a non-empty list.")

    samples: list[_SamplePlan] = []
    sample_ids: set[str] = set()
    subject_splits: dict[str, str] = {}
    subjects: list[str] = []
    splits: dict[str, list[str]] = {}
    encodings: set[LabelEncoding] = set()
    dropped_crops: list[str] = []
    planned: list[_SamplePlan] = []
    for index, raw_sample in enumerate(raw_samples):
        sample = _load_sample_plan(
            raw_sample,
            index=index,
            root=manifest_path.parent,
            dataset_type=dataset_type,
            label_names=label_names,
            stored_labels=stored_labels,
            preprocessing=preprocessing,
        )
        if crops is None:
            planned.append(sample)
            continue
        expanded, dropped = _expand_ct_crops(sample, crops, preprocessing=preprocessing)
        planned.extend(expanded)
        dropped_crops.extend(dropped)
    for sample in planned:
        if sample.sample_id in sample_ids:
            raise ValueError(f"Duplicate sample_id {sample.sample_id!r}.")
        sample_ids.add(sample.sample_id)
        previous_split = subject_splits.setdefault(sample.subject_id, sample.split)
        if previous_split != sample.split:
            raise ValueError(
                f"Subject {sample.subject_id!r} appears in both "
                f"{previous_split!r} and {sample.split!r}; splits must be "
                "subject-disjoint."
            )
        if sample.subject_id not in subjects:
            subjects.append(sample.subject_id)
        splits.setdefault(sample.split, []).append(sample.sample_id)
        encodings.add(sample.label_encoding)
        samples.append(sample)

    if dataset_type == "xray-seg":
        expected_shape = samples[0].image.shape[-2:]
        mismatched_shapes = {
            sample.sample_id: sample.image.shape[-2:]
            for sample in samples
            if sample.image.shape[-2:] != expected_shape
        }
        if mismatched_shapes:
            raise ValueError(
                "All xray-seg images must share one spatial shape; expected "
                f"{expected_shape}, found {mismatched_shapes!r}."
            )

    if len(encodings) != 1:
        raise ValueError(
            "All samples must use the same label encoding; found "
            f"{sorted(encodings)!r}."
        )
    label_encoding = next(iter(encodings))
    if dataset_type == "ct-seg" and label_names is not None:
        raise ValueError("ct-seg manifests do not accept label_names.")
    if label_encoding == "channels" and label_names is None:
        raise ValueError("Channel-first X-ray labels require ordered label_names.")
    if label_encoding == "channels" and stored_labels is not None:
        raise ValueError("Channel-first labels reject stored_labels.")
    if label_encoding == "dense" and label_names is not None:
        raise ValueError("Dense label maps must omit label_names.")
    if label_encoding == "dense" and stored_labels is None:
        raise ValueError("Dense label maps require stored_labels.")
    return _PackagePlan(
        source_path=manifest_path,
        dataset_name=dataset_name,
        dataset_type=dataset_type,
        protocol_name=protocol_name,
        stored_labels=stored_labels,
        label_names=label_names,
        label_encoding=label_encoding,
        samples=tuple(samples),
        subjects=tuple(subjects),
        splits={name: tuple(ids_) for name, ids_ in splits.items()},
        preprocessing=preprocessing,
        crops=crops,
        dropped_background_only_crops=tuple(dropped_crops),
    )


def _crop_plan(raw_value: Any, *, dataset_type: DatasetType) -> CropPlan | None:
    """Validate the optional manifest ``crops`` block (``ct-seg`` only).

    Args:
        raw_value: Raw YAML value, or ``None`` when absent.
        dataset_type: Package type; X-ray packages reject crops.

    Returns:
        Parsed crop plan, or ``None``.
    """

    if raw_value is None:
        return None
    if dataset_type != "ct-seg":
        raise ValueError("crops are only valid for ct-seg manifests.")
    return CropPlan.from_manifest(raw_value)


def _expand_ct_crops(
    sample: _SamplePlan,
    crops: CropPlan,
    *,
    preprocessing: XrayPreprocessing | CTPreprocessing | None,
) -> tuple[list[_SamplePlan], list[str]]:
    """Replace one validated CT sample by its planned axial crops.

    Args:
        sample: Validated whole-volume sample plan.
        crops: Manifest crop plan.
        preprocessing: Optional CT preprocessing (applied before cropping).

    Returns:
        ``(crop_samples, dropped_ids)``: one plan per crop with foreground and
        the ids of background-only crops that were dropped.
    """

    context = f"sample {sample.sample_id!r}"
    derived = sample.affine is not None and sample.affine.path == sample.image.path
    image, label, affine, spacing = _load_ct_sample(
        sample.image.path,
        sample.label.path,
        None if derived else sample.affine.path,
        None if derived else sample.spacing.path,
        preprocessing=preprocessing,
        context=context,
    )
    if tuple(image.shape[-3:-1]) != crops.size[:2]:
        raise ValueError(
            f"{context} volume xy extent {tuple(image.shape[-3:-1])} must equal crops.size "
            f"xy {crops.size[:2]}; crops are axial only."
        )
    expanded: list[_SamplePlan] = []
    dropped: list[str] = []
    for spec in plan_z_crops(int(image.shape[-1]), crops.size[2], max_overlap=crops.max_overlap):
        crop_id = crop_sample_id(sample.sample_id, spec.crop_index)
        label_crop = crop_z(label, spec, fill_value=0)
        if not np.any(label_crop != 0):
            dropped.append(crop_id)
            continue
        metadata = {
            **sample.metadata,
            **crop_metadata(
                spec,
                label_crop,
                subject_id=sample.subject_id,
                source_sample_id=sample.sample_id,
            ),
        }
        expanded.append(
            _SamplePlan(
                sample_id=crop_id,
                subject_id=sample.subject_id,
                split=sample.split,
                image=_payload_plan(sample.image.path, crop_z(image, spec, fill_value=AIR_HU)),
                label=_payload_plan(sample.label.path, label_crop),
                affine=_payload_plan(sample.affine.path, affine),
                spacing=_payload_plan(sample.spacing.path, spacing),
                label_encoding="dense",
                metadata=metadata,
                crop=spec,
            )
        )
    return expanded, dropped


def _preprocessing(
    raw_value: Any, *, dataset_type: DatasetType
) -> XrayPreprocessing | CTPreprocessing | None:
    """Validate the optional manifest ``preprocessing`` block.

    Args:
        raw_value: Raw YAML value, or ``None`` when absent.
        dataset_type: Package type selecting the accepted keys.

    Returns:
        Parsed preprocessing spec, or ``None``.
    """

    if raw_value is None:
        return None
    if dataset_type == "xray-seg":
        return XrayPreprocessing.from_manifest(raw_value)
    return CTPreprocessing.from_manifest(raw_value)


def _load_sample_plan(
    raw_sample: Any,
    *,
    index: int,
    root: Path,
    dataset_type: DatasetType,
    label_names: tuple[str, ...] | None,
    stored_labels: dict[int, str] | None,
    preprocessing: XrayPreprocessing | CTPreprocessing | None = None,
) -> _SamplePlan:
    """Validate one manifest sample and retain only payload descriptors.

    Args:
        raw_sample: Raw sample mapping from YAML.
        index: Zero-based manifest sample index used in errors.
        root: Manifest directory used to resolve relative payload paths.
        dataset_type: Package type controlling payload requirements.
        label_names: Optional ordered names for channel-first masks.
        stored_labels: Optional native names for dense integer ids.
        preprocessing: Optional packaging-time preprocessing applied to every
            sample (its geometry is recorded in ``metadata.preprocessing``).

    Returns:
        Validated sample plan.
    """

    context = f"samples[{index}]"
    if not isinstance(raw_sample, Mapping):
        raise TypeError(f"{context} must be a mapping.")
    data = dict(raw_sample)
    _reject_unknown_fields(data, _SAMPLE_FIELDS, context=context)
    sample_id = _required_text(data.get("sample_id"), f"{context}.sample_id")
    if sample_id.startswith("_"):
        raise ValueError(f"{context}.sample_id cannot start with '_'.")
    subject_id = _required_text(data.get("subject_id"), f"{context}.subject_id")
    split = _required_text(data.get("split"), f"{context}.split")
    image_path = _payload_path(data.get("image"), root, f"{context}.image")
    label_path = _payload_path(data.get("label"), root, f"{context}.label")
    metadata = _metadata(data.get("metadata"), context=f"{context}.metadata")
    stored_subject = metadata.get("subject_id")
    if stored_subject is not None and str(stored_subject) != subject_id:
        raise ValueError(
            f"{context}.metadata.subject_id conflicts with subject_id "
            f"{subject_id!r}."
        )
    metadata["subject_id"] = subject_id

    if dataset_type == "xray-seg":
        if data.get("affine") is not None or data.get("spacing") is not None:
            raise ValueError(f"{context} xray-seg samples reject affine and spacing.")
        image, label, geometry = _load_xray_sample(
            image_path, label_path, preprocessing=preprocessing, context=context
        )
        if geometry is not None:
            metadata["preprocessing"] = geometry
        _validate_xray_shapes(
            image,
            label,
            label_names=label_names,
            context=context,
        )
        label_encoding = _label_encoding(label, dataset_type=dataset_type)
        if label_encoding == "dense" and stored_labels is not None:
            _validate_declared_dense_ids(
                label,
                stored_labels,
                context=f"{context}.label",
            )
        return _SamplePlan(
            sample_id=sample_id,
            subject_id=subject_id,
            split=split,
            image=_payload_plan(image_path, image),
            label=_payload_plan(label_path, label),
            affine=None,
            spacing=None,
            label_encoding=label_encoding,
            metadata=metadata,
        )

    affine_path = _optional_ct_geometry_path(
        data.get("affine"), root, f"{context}.affine", image_path=image_path
    )
    spacing_path = _optional_ct_geometry_path(
        data.get("spacing"), root, f"{context}.spacing", image_path=image_path
    )
    image, label, affine, spacing = _load_ct_sample(
        image_path,
        label_path,
        affine_path,
        spacing_path,
        preprocessing=preprocessing,
        context=context,
    )
    if stored_labels is not None:
        _validate_declared_dense_ids(
            label,
            stored_labels,
            context=f"{context}.label",
        )
    return _SamplePlan(
        sample_id=sample_id,
        subject_id=subject_id,
        split=split,
        image=_payload_plan(image_path, image),
        label=_payload_plan(label_path, label),
        affine=_payload_plan(affine_path or image_path, affine),
        spacing=_payload_plan(spacing_path or image_path, spacing),
        label_encoding="dense",
        metadata=metadata,
    )


def _load_xray_sample(
    image_path: Path,
    label_path: Path,
    *,
    preprocessing: XrayPreprocessing | CTPreprocessing | None,
    context: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None]:
    """Load one X-ray image/label pair and apply optional preprocessing.

    Shared by manifest validation and package writing so both passes see the
    identical processed arrays.

    Args:
        image_path: Image payload path.
        label_path: Label payload path.
        preprocessing: Optional X-ray preprocessing spec.
        context: Manifest location used in errors.

    Returns:
        ``(image, label, geometry)``; ``geometry`` is ``None`` without
        preprocessing.
    """

    image = _load_xray_image(image_path, context=f"{context}.image")
    label = _load_xray_label(label_path, context=f"{context}.label")
    if not isinstance(preprocessing, XrayPreprocessing):
        return image, label, None
    if tuple(image.shape[-2:]) != tuple(label.shape[-2:]):
        raise ValueError(
            f"{context} image and label spatial shapes differ: "
            f"{image.shape[-2:]} versus {label.shape[-2:]}."
        )
    image, label, geometry = preprocessing.apply(image, label)
    return image, label, geometry


def _load_ct_sample(
    image_path: Path,
    label_path: Path,
    affine_path: Path | None,
    spacing_path: Path | None,
    *,
    preprocessing: XrayPreprocessing | CTPreprocessing | None,
    context: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate one CT volume, label, affine, and spacing group.

    NIfTI images supply their own affine and spacing; explicit ``.npy`` geometry
    payloads take precedence but must agree with the header. Shared by manifest
    validation and package writing.

    Args:
        image_path: ``.npy`` or NIfTI CT volume path.
        label_path: ``.npy`` or NIfTI dense label path.
        affine_path: Optional ``.npy`` affine path (required for ``.npy`` images).
        spacing_path: Optional ``.npy`` spacing path (required for ``.npy`` images).
        preprocessing: Optional CT preprocessing spec (HU window).
        context: Manifest location used in errors.

    Returns:
        ``(image, label, affine, spacing)`` arrays.
    """

    image, header_affine, header_spacing = _load_ct_volume(
        image_path, context=f"{context}.image"
    )
    label, _, _ = _load_ct_volume(label_path, context=f"{context}.label")
    affine = header_affine if affine_path is None else _load_npy(affine_path, context=f"{context}.affine")
    spacing = header_spacing if spacing_path is None else _load_npy(spacing_path, context=f"{context}.spacing")
    if affine is None or spacing is None:
        raise ValueError(f"{context} .npy CT images require affine and spacing payloads.")
    if header_affine is not None and affine_path is not None:
        if not np.allclose(affine, header_affine, atol=1e-4):
            raise ValueError(f"{context}.affine disagrees with the NIfTI header affine.")
    if isinstance(preprocessing, CTPreprocessing):
        image = preprocessing.apply(image)
    if label.dtype.kind == "f":
        label = np.rint(label).astype(np.int64)
    _validate_ct_payloads(image, label, affine, spacing, context=context)
    return image, label, affine, spacing


def _load_ct_volume(
    path: Path, *, context: str
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load a CT payload as ``.npy`` or NIfTI.

    Args:
        path: Payload path.
        context: Manifest location used in errors.

    Returns:
        ``(array, affine, spacing)``; affine and spacing are ``None`` for
        ``.npy`` payloads.
    """

    if is_nifti_path(path):
        return load_nifti(path, context=context)
    if path.suffix.lower() in {".dcm", ".dicom"}:
        raise ValueError(
            f"{context} DICOM is not supported; convert to NIfTI first (e.g. dcm2niix)."
        )
    return _load_npy(path, context=context), None, None


def _optional_ct_geometry_path(
    raw_value: Any, root: Path, name: str, *, image_path: Path
) -> Path | None:
    """Resolve an affine/spacing payload that NIfTI images may omit.

    Args:
        raw_value: Optional path string.
        root: Manifest directory for relative paths.
        name: Field name used in errors.
        image_path: Image payload path deciding whether the field is optional.

    Returns:
        Existing payload path, or ``None`` when derived from a NIfTI header.
    """

    if raw_value is None and is_nifti_path(image_path):
        return None
    return _payload_path(raw_value, root, name)


def _payload_plan(path: Path, array: np.ndarray) -> _PayloadPlan:
    """Describe a validated payload without retaining its array.

    Args:
        path: Resolved payload source path.
        array: Validated array loaded from the source.

    Returns:
        Immutable source and structural contract used during serialization.
    """

    return _PayloadPlan(
        path=path,
        shape=tuple(int(size) for size in array.shape),
        dtype=array.dtype.str,
    )


def _load_xray_image(path: Path, *, context: str) -> np.ndarray:
    """Load and canonicalize one single-channel X-ray image.

    Pillow inputs are decoded as grayscale. NumPy inputs must be two-dimensional
    or have exactly one leading channel. Integer intensities are scaled across
    the full numeric range of their dtype. Floating-point intensities must
    already be finite and lie in ``[0, 1]``.

    Args:
        path: Image payload path.
        context: Manifest location used in errors.

    Returns:
        Contiguous ``float32`` array shaped ``(1, H, W)`` in ``[0, 1]``.
    """

    source_dtype: np.dtype[Any] | None = None
    if path.suffix.lower() == ".npy":
        image = _load_npy(path, context=context)
        _validate_numeric_array(image, context=context)
        if image.ndim == 3 and image.shape[0] == 1:
            image = image[0]
        elif image.ndim != 2:
            raise ValueError(
                f"{context} NumPy image must be 2D or one-channel CHW, "
                f"got shape {image.shape}."
            )
    else:
        image, source_dtype = _load_pillow_xray_image(path, context=context)
    canonical = _normalize_xray_image(
        image,
        context=context,
        source_dtype=source_dtype,
    )
    return np.ascontiguousarray(canonical[None])


def _load_pillow_xray_image(
    path: Path,
    *,
    context: str,
) -> tuple[np.ndarray, np.dtype[Any]]:
    """Decode one Pillow-supported image as grayscale.

    Args:
        path: Image payload path.
        context: Manifest location used in errors.

    Returns:
        Two-dimensional grayscale array and the decoded source dtype whose
        numeric range controls integer scaling. Intensity normalization is
        deliberately deferred to :func:`_normalize_xray_image`.

    Raises:
        ValueError: If Pillow cannot identify, decode, or grayscale the image.
    """

    try:
        with Image.open(path) as image:
            image.load()
            decoded = np.array(image)
            source_dtype = decoded.dtype
            if decoded.ndim == 2 and image.mode != "P":
                return decoded, source_dtype
            grayscale = np.array(image.convert("F"), dtype=np.float32)
            return grayscale, source_dtype
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(
            f"{context} is neither .npy nor a Pillow-supported image: {path}."
        ) from exc


def _normalize_xray_image(
    image: np.ndarray,
    *,
    context: str,
    source_dtype: np.dtype[Any] | None = None,
) -> np.ndarray:
    """Convert X-ray intensities to the canonical unit interval.

    Integer arrays use the fixed numeric range of their dtype, so equal source
    values have equal packaged values across images. Floating-point arrays are
    not rescaled and must already lie in ``[0, 1]``.

    Args:
        image: Validated two-dimensional numeric image.
        context: Manifest location used in errors.
        source_dtype: Optional decoded integer dtype for a Pillow image that was
            converted to floating-point grayscale.

    Returns:
        Contiguous two-dimensional ``float32`` array in ``[0, 1]``.
    """

    _validate_numeric_array(image, context=context)
    if image.ndim != 2:
        raise ValueError(f"{context} must be two-dimensional, got {image.shape}.")
    input_dtype = np.dtype(
        source_dtype if source_dtype is not None else image.dtype
    )
    values = np.asarray(image, dtype=np.longdouble)
    if input_dtype.kind == "b":
        normalized = values
    elif input_dtype.kind in {"i", "u"}:
        limits = np.iinfo(input_dtype)
        normalized = (values - limits.min) / (limits.max - limits.min)
    else:
        minimum = np.min(values)
        maximum = np.max(values)
        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(
                f"{context} floating-point intensities must already be in "
                f"[0, 1], got range [{minimum}, {maximum}]."
            )
        normalized = values
    result = np.asarray(normalized, dtype=np.float32)
    return np.ascontiguousarray(result)


def _load_xray_label(path: Path, *, context: str) -> np.ndarray:
    """Load a dense or channel-first X-ray label payload.

    Args:
        path: Label payload path.
        context: Manifest location used in errors.

    Returns:
        Numeric label array. Pillow masks remain two-dimensional; multichannel
        masks must be supplied as channel-first ``.npy`` arrays.
    """

    if path.suffix.lower() == ".npy":
        label = _load_npy(path, context=context)
    else:
        label = _load_pillow(path, context=context)
        if label.ndim != 2:
            raise ValueError(
                f"{context} Pillow mask must be single-channel; use a "
                "channel-first .npy array for multichannel labels."
            )
    _validate_numeric_array(label, context=context)
    if label.ndim not in {2, 3}:
        raise ValueError(
            f"{context} must be a dense 2D map or channel-first 3D mask, "
            f"got shape {label.shape}."
        )
    if label.ndim == 2:
        _validate_dense_label(label, context=context)
    else:
        _validate_channel_label(label, context=context)
    return np.ascontiguousarray(label)


def _load_pillow(path: Path, *, context: str) -> np.ndarray:
    """Load one Pillow-supported image into a writable NumPy array.

    Args:
        path: Image or mask path.
        context: Manifest location used in errors.

    Returns:
        Copied NumPy representation of the first image frame.

    Raises:
        ValueError: If Pillow cannot identify or decode the file.
    """

    try:
        with Image.open(path) as image:
            image.load()
            return np.array(image)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(
            f"{context} is neither .npy nor a Pillow-supported image: {path}."
        ) from exc


def _load_npy(path: Path, *, context: str) -> np.ndarray:
    """Load a non-pickled NumPy payload.

    Args:
        path: Required ``.npy`` path.
        context: Manifest location used in errors.

    Returns:
        Loaded NumPy array.

    Raises:
        ValueError: If the path is not ``.npy`` or contains object data.
    """

    if path.suffix.lower() != ".npy":
        raise ValueError(f"{context} must reference a .npy payload, got {path}.")
    try:
        return np.load(path, allow_pickle=False)
    except ValueError as exc:
        raise ValueError(f"{context} could not be loaded safely from {path}.") from exc


def _validate_canonical_xray_image(
    image: np.ndarray,
    *,
    context: str,
) -> None:
    """Validate the storage contract for one packaged X-ray image.

    Args:
        image: Stored image array to inspect.
        context: Database location used in errors.

    Returns:
        ``None``.
    """

    _validate_numeric_array(image, context=context)
    if image.dtype != np.dtype(np.float32):
        raise ValueError(f"{context} must use float32, got {image.dtype}.")
    if image.ndim != 3 or image.shape[0] != 1:
        raise ValueError(
            f"{context} must have canonical shape (1, H, W), got {image.shape}."
        )
    if np.any(image < 0) or np.any(image > 1):
        raise ValueError(f"{context} must contain values in [0, 1].")


def _validate_xray_shapes(
    image: np.ndarray,
    label: np.ndarray,
    *,
    label_names: tuple[str, ...] | None,
    context: str,
) -> None:
    """Validate X-ray image/mask spatial agreement and channel metadata.

    Args:
        image: Canonical one-channel, channel-first image.
        label: Dense or channel-first mask.
        label_names: Ordered channel names declared by the manifest.
        context: Manifest sample location used in errors.

    Returns:
        ``None``.
    """

    if tuple(image.shape[-2:]) != tuple(label.shape[-2:]):
        raise ValueError(
            f"{context} image and label spatial shapes differ: "
            f"{image.shape[-2:]} versus {label.shape[-2:]}."
        )
    if label.ndim == 3:
        if label_names is None:
            raise ValueError(
                f"{context} channel-first label requires top-level label_names."
            )
        if label.shape[0] != len(label_names):
            raise ValueError(
                f"{context} has {label.shape[0]} label channels but "
                f"{len(label_names)} label_names."
            )


def _validate_ct_payloads(
    image: np.ndarray,
    label: np.ndarray,
    affine: np.ndarray,
    spacing: np.ndarray,
    *,
    context: str,
) -> None:
    """Validate one CT image, dense label, affine, and spacing group.

    Args:
        image: CT volume, optionally with a leading channel.
        label: Dense CT label map, optionally with one leading channel.
        affine: Voxel-to-world matrix.
        spacing: Positive voxel spacing vector.
        context: Manifest sample location used in errors.

    Returns:
        ``None``.
    """

    _validate_numeric_array(image, context=f"{context}.image")
    _validate_numeric_array(label, context=f"{context}.label")
    _validate_numeric_array(affine, context=f"{context}.affine")
    _validate_numeric_array(spacing, context=f"{context}.spacing")
    if image.ndim not in {3, 4} or (image.ndim == 4 and image.shape[0] != 1):
        raise ValueError(
            f"{context}.image must be a 3D volume or one-channel 4D volume, "
            f"got {image.shape}."
        )
    if label.ndim not in {3, 4} or (label.ndim == 4 and label.shape[0] != 1):
        raise ValueError(
            f"{context}.label must be a dense 3D map or one-channel 4D map, "
            f"got {label.shape}."
        )
    if tuple(image.shape[-3:]) != tuple(label.shape[-3:]):
        raise ValueError(
            f"{context} image and label spatial shapes differ: "
            f"{image.shape[-3:]} versus {label.shape[-3:]}."
        )
    _validate_dense_label(label, context=f"{context}.label")
    if not np.any(label != 0):
        raise ValueError(
            f"{context}.label must contain at least one foreground label id."
        )
    if affine.shape != (4, 4):
        raise ValueError(
            f"{context}.affine must have shape (4, 4), got {affine.shape}."
        )
    affine_sign, _ = np.linalg.slogdet(np.asarray(affine, dtype=np.float64))
    if not np.isfinite(affine_sign) or affine_sign == 0:
        raise ValueError(f"{context}.affine must be invertible.")
    if spacing.shape not in {(3,), (1, 3)}:
        raise ValueError(
            f"{context}.spacing must contain exactly three values, "
            f"got shape {spacing.shape}."
        )
    if np.any(np.asarray(spacing) <= 0):
        raise ValueError(f"{context}.spacing values must be positive.")


def _validate_numeric_array(array: np.ndarray, *, context: str) -> None:
    """Validate that an array is non-empty, numeric, and finite.

    Args:
        array: Array to inspect.
        context: Manifest location used in errors.

    Returns:
        ``None``.
    """

    if not isinstance(array, np.ndarray):
        raise TypeError(f"{context} must load as a NumPy array.")
    if array.size == 0 or any(size <= 0 for size in array.shape):
        raise ValueError(f"{context} cannot be empty.")
    if array.dtype.kind not in {"b", "i", "u", "f"}:
        raise ValueError(f"{context} must have a numeric dtype, got {array.dtype}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{context} must contain only finite values.")


def _validate_dense_label(label: np.ndarray, *, context: str) -> None:
    """Validate a non-negative integer-valued dense segmentation map.

    Args:
        label: Dense label array.
        context: Manifest location used in errors.

    Returns:
        ``None``.
    """

    if label.dtype.kind not in {"b", "i", "u"}:
        raise ValueError(f"{context} dense labels must use an integer dtype.")
    if label.dtype.kind == "i" and np.any(label < 0):
        raise ValueError(f"{context} dense labels cannot contain negative ids.")


def _validate_declared_dense_ids(
    label: np.ndarray,
    stored_labels: Mapping[int, str],
    *,
    context: str,
) -> None:
    """Verify that every observed dense id has a declared native label.

    Args:
        label: Already validated dense label map.
        stored_labels: Declared native label names keyed by integer id.
        context: Manifest or database location used in errors.

    Returns:
        ``None``.
    """

    observed_ids = {int(value) for value in np.unique(label)}
    unknown_ids = sorted(observed_ids.difference(stored_labels))
    if unknown_ids:
        raise ValueError(
            f"{context} contains ids absent from stored_labels: {unknown_ids!r}."
        )


def _validate_channel_label(label: np.ndarray, *, context: str) -> None:
    """Validate a channel-first mask with values in the unit interval.

    Args:
        label: Channel-first mask array.
        context: Manifest location used in errors.

    Returns:
        ``None``.
    """

    if np.any(label < 0) or np.any(label > 1):
        raise ValueError(f"{context} channel masks must contain values in [0, 1].")


def _write_package(plan: _PackagePlan, path: Path) -> None:
    """Stream a validated package plan into a ThunderDB directory.

    Each sample is reopened and revalidated immediately before serialization.
    Payload arrays leave scope before the next sample is loaded.

    Args:
        plan: Validated package plan.
        path: Empty destination directory.

    Returns:
        ``None``.
    """

    thunderpack = _import_thunderpack()
    attrs: dict[str, Any] = {
        "schema_name": _SCHEMA_NAME,
        "schema_version": _SCHEMA_VERSION,
        "dataset_name": plan.dataset_name,
        "dataset_type": plan.dataset_type,
        "modality": "ct" if plan.dataset_type == "ct-seg" else "xray",
        "clean_dataset": True,
        "payload_keys": {
            "image": "img",
            "label": "seg",
            **(
                {"affine": "affine", "spacing": "spacing"}
                if plan.dataset_type == "ct-seg"
                else {}
            ),
        },
        "label_encoding": plan.label_encoding,
        "num_subjects": len(plan.subjects),
        "num_samples": len(plan.samples),
    }
    if plan.protocol_name is not None:
        attrs["protocol_name"] = plan.protocol_name
    if plan.preprocessing is not None:
        attrs["preprocessing"] = plan.preprocessing.to_attrs()
    if plan.dataset_type == "ct-seg":
        attrs["storage_layout"] = "crops" if plan.crops is not None else "subjects"
    if plan.crops is not None:
        attrs["crops"] = {
            **plan.crops.to_attrs(),
            "num_crops": len(plan.samples),
            "num_source_samples": len({s.metadata["source_sample_id"] for s in plan.samples}),
            "dropped_background_only": list(plan.dropped_background_only_crops),
        }
    if plan.label_names is not None:
        attrs["label_names"] = list(plan.label_names)
    if plan.stored_labels is not None:
        attrs["stored_labels"] = {
            str(label_id): label_name
            for label_id, label_name in plan.stored_labels.items()
        }

    cache: dict[str, Any] = {}
    with thunderpack.ThunderDB.open(str(path), "c") as db:
        for sample in plan.samples:
            _write_sample(db, plan=plan, sample=sample, cache=cache)
        db["_subjects"] = list(plan.subjects)
        db["_samples"] = [sample.sample_id for sample in plan.samples]
        db["_splits"] = {
            split: list(sample_ids) for split, sample_ids in plan.splits.items()
        }
        db["_metadata"] = {
            sample.sample_id: sample.metadata for sample in plan.samples
        }
        db["_attrs"] = attrs


def _write_sample(
    db: Any,
    *,
    plan: _PackagePlan,
    sample: _SamplePlan,
    cache: dict[str, Any] | None = None,
) -> None:
    """Reload, revalidate, and serialize exactly one planned sample.

    Args:
        db: Open writable ThunderDB mapping.
        plan: Package-level label and dataset contract.
        sample: Descriptor for the sample being serialized.
        cache: Optional one-volume cache shared by consecutive crops of the
            same source volume.

    Returns:
        ``None``.
    """

    payload = _reload_sample_payload(plan=plan, sample=sample, cache=cache)
    db[sample.sample_id] = payload


def _reload_sample_payload(
    *,
    plan: _PackagePlan,
    sample: _SamplePlan,
    cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reload and revalidate one sample against its planned contract.

    Args:
        plan: Package-level label and dataset contract.
        sample: Validated payload descriptors for one sample.
        cache: Optional one-volume cache keyed by source image path, so the
            crops of one volume reload it once.

    Returns:
        Canonical ThunderDB payload mapping containing only this sample's
        arrays.
    """

    context = f"sample {sample.sample_id!r}"
    if plan.dataset_type == "xray-seg":
        image, label, _ = _load_xray_sample(
            sample.image.path,
            sample.label.path,
            preprocessing=plan.preprocessing,
            context=context,
        )
        _validate_payload_contract(
            image,
            sample.image,
            context=f"{context}.image",
        )
        _validate_payload_contract(
            label,
            sample.label,
            context=f"{context}.label",
        )
        _validate_xray_shapes(
            image,
            label,
            label_names=plan.label_names,
            context=context,
        )
        label_encoding = _label_encoding(label, dataset_type=plan.dataset_type)
        if (
            label_encoding != sample.label_encoding
            or label_encoding != plan.label_encoding
        ):
            raise ValueError(
                f"{context}.label encoding changed after manifest validation."
            )
        if label_encoding == "dense":
            if plan.stored_labels is None:
                raise ValueError("Dense labels require stored_labels.")
            _validate_declared_dense_ids(
                label,
                plan.stored_labels,
                context=f"{context}.label",
            )
        return {"img": image, "seg": label}

    if sample.affine is None or sample.spacing is None:
        raise ValueError(f"{context} is missing CT affine or spacing descriptors.")
    derived = sample.affine.path == sample.image.path
    key = str(sample.image.path)
    if cache is not None and key in cache:
        image, label, affine, spacing = cache[key]
    else:
        image, label, affine, spacing = _load_ct_sample(
            sample.image.path,
            sample.label.path,
            None if derived else sample.affine.path,
            None if derived else sample.spacing.path,
            preprocessing=plan.preprocessing,
            context=context,
        )
        if cache is not None:
            cache.clear()
            cache[key] = (image, label, affine, spacing)
    if sample.crop is not None:
        image = crop_z(image, sample.crop, fill_value=AIR_HU)
        label = crop_z(label, sample.crop, fill_value=0)
        affine = offset_affine(affine, sample.crop)
        _validate_ct_payloads(image, label, affine, spacing, context=context)
    _validate_payload_contract(image, sample.image, context=f"{context}.image")
    _validate_payload_contract(label, sample.label, context=f"{context}.label")
    _validate_payload_contract(
        affine,
        sample.affine,
        context=f"{context}.affine",
    )
    _validate_payload_contract(
        spacing,
        sample.spacing,
        context=f"{context}.spacing",
    )
    if plan.stored_labels is None:
        raise ValueError("Dense labels require stored_labels.")
    _validate_declared_dense_ids(
        label,
        plan.stored_labels,
        context=f"{context}.label",
    )
    return {
        "img": image,
        "seg": label,
        "affine": affine,
        "spacing": spacing,
    }


def _validate_payload_contract(
    array: np.ndarray,
    payload_plan: _PayloadPlan,
    *,
    context: str,
) -> None:
    """Reject a payload whose canonical structure changed after validation.

    Args:
        array: Reloaded and independently validated payload array.
        payload_plan: Shape and dtype observed during initial validation.
        context: Sample field used in errors.

    Returns:
        ``None``.
    """

    shape = tuple(int(size) for size in array.shape)
    if shape != payload_plan.shape:
        raise ValueError(
            f"{context} changed shape after manifest validation: expected "
            f"{payload_plan.shape}, got {shape}."
        )
    if array.dtype.str != payload_plan.dtype:
        raise ValueError(
            f"{context} changed dtype after manifest validation: expected "
            f"{payload_plan.dtype}, got {array.dtype.str}."
        )


def _validate_database_mapping(
    db: Mapping[str, Any],
    path: Path,
) -> DatasetPackageReport:
    """Validate canonical metadata, split invariants, and every stored payload.

    Args:
        db: Open ThunderDB mapping.
        path: Package path reported in validation errors and results.

    Returns:
        Validation report for the database.
    """

    attrs = _required_mapping(db, "_attrs")
    if attrs.get("schema_name") != _SCHEMA_NAME:
        raise ValueError(
            f"_attrs.schema_name must be {_SCHEMA_NAME!r}, "
            f"got {attrs.get('schema_name')!r}."
        )
    if attrs.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(
            f"_attrs.schema_version must be {_SCHEMA_VERSION}, "
            f"got {attrs.get('schema_version')!r}."
        )
    if attrs.get("clean_dataset") is not True:
        raise ValueError("_attrs.clean_dataset must be true.")
    dataset_name = _required_text(attrs.get("dataset_name"), "_attrs.dataset_name")
    dataset_type = _dataset_type(attrs.get("dataset_type"), context="_attrs")
    expected_modality = "ct" if dataset_type == "ct-seg" else "xray"
    if attrs.get("modality") != expected_modality:
        raise ValueError(
            f"_attrs.modality must be {expected_modality!r} for "
            f"{dataset_type!r}."
        )
    expected_payload_keys = {"image": "img", "label": "seg"}
    if dataset_type == "ct-seg":
        expected_payload_keys.update({"affine": "affine", "spacing": "spacing"})
    payload_keys = attrs.get("payload_keys")
    if (
        not isinstance(payload_keys, Mapping)
        or dict(payload_keys) != expected_payload_keys
    ):
        raise ValueError(
            f"_attrs.payload_keys must equal {expected_payload_keys!r}."
        )
    label_encoding = _stored_label_encoding(attrs.get("label_encoding"))
    label_names = _optional_label_names(attrs.get("label_names"))
    stored_labels = _optional_stored_labels(attrs.get("stored_labels"))
    if dataset_type == "ct-seg" and label_encoding != "dense":
        raise ValueError("_attrs.label_encoding must be \"dense\" for ct-seg.")
    if label_encoding == "channels" and label_names is None:
        raise ValueError("_attrs.label_names is required for channel labels.")
    if label_encoding == "channels" and stored_labels is not None:
        raise ValueError("_attrs.stored_labels must be absent for channel labels.")
    if label_encoding == "dense" and label_names is not None:
        raise ValueError("_attrs.label_names must be absent for dense labels.")
    if label_encoding == "dense" and stored_labels is None:
        raise ValueError("_attrs.stored_labels is required for dense labels.")

    subjects = _required_string_list(db, "_subjects")
    samples = _required_string_list(db, "_samples")
    if len(subjects) != len(set(subjects)):
        raise ValueError("_subjects cannot contain duplicate ids.")
    if len(samples) != len(set(samples)):
        raise ValueError("_samples cannot contain duplicate ids.")
    splits = _required_mapping(db, "_splits")
    split_samples: list[str] = []
    sample_split: dict[str, str] = {}
    for raw_split, raw_ids in splits.items():
        split = _required_text(raw_split, "_splits key")
        ids_ = _string_sequence(raw_ids, context=f"_splits[{split!r}]")
        for sample_id in ids_:
            if sample_id in sample_split:
                raise ValueError(
                    f"Sample {sample_id!r} appears in both "
                    f"{sample_split[sample_id]!r} and {split!r}."
                )
            sample_split[sample_id] = split
        split_samples.extend(ids_)
    if set(split_samples) != set(samples) or len(split_samples) != len(samples):
        raise ValueError("_splits must partition _samples exactly.")

    metadata = _required_mapping(db, "_metadata")
    subject_split: dict[str, str] = {}
    observed_subjects: list[str] = []
    xray_spatial_shape: tuple[int, int] | None = None
    for sample_id in samples:
        sample_metadata = _metadata(
            metadata.get(sample_id),
            context=f"_metadata[{sample_id!r}]",
        )
        subject_id = _required_text(
            sample_metadata.get("subject_id"),
            f"_metadata[{sample_id!r}].subject_id",
        )
        split = sample_split[sample_id]
        previous_split = subject_split.setdefault(subject_id, split)
        if previous_split != split:
            raise ValueError(
                f"Subject {subject_id!r} appears in both {previous_split!r} "
                f"and {split!r}."
            )
        if subject_id not in observed_subjects:
            observed_subjects.append(subject_id)
        raw_payload = db[sample_id]
        if not isinstance(raw_payload, Mapping):
            raise TypeError(f"ThunderDB sample {sample_id!r} must be a mapping.")
        _validate_stored_payload(
            raw_payload,
            dataset_type=dataset_type,
            label_encoding=label_encoding,
            label_names=label_names,
            stored_labels=stored_labels,
            context=f"ThunderDB sample {sample_id!r}",
        )
        if dataset_type == "xray-seg":
            current_shape = tuple(
                int(size)
                for size in np.asarray(raw_payload["img"]).shape[-2:]
            )
            if xray_spatial_shape is None:
                xray_spatial_shape = current_shape
            elif current_shape != xray_spatial_shape:
                raise ValueError(
                    "All xray-seg images must share one spatial shape; expected "
                    f"{xray_spatial_shape}, sample {sample_id!r} has {current_shape}."
                )
        if attrs.get("storage_layout") == "crops":
            _validate_crop_metadata(sample_metadata, context=f"_metadata[{sample_id!r}]")
    if set(observed_subjects) != set(subjects):
        raise ValueError("_subjects must equal subjects referenced by _metadata.")
    if attrs.get("num_subjects") != len(subjects):
        raise ValueError("_attrs.num_subjects does not match _subjects.")
    if attrs.get("num_samples") != len(samples):
        raise ValueError("_attrs.num_samples does not match _samples.")
    return DatasetPackageReport(
        path=path,
        dataset_name=dataset_name,
        dataset_type=dataset_type,
        label_encoding=label_encoding,
        num_subjects=len(subjects),
        num_samples=len(samples),
        split_counts={
            str(split): len(_string_sequence(ids_, context=f"_splits[{split!r}]"))
            for split, ids_ in splits.items()
        },
    )


def _validate_crop_metadata(metadata: Mapping[str, Any], *, context: str) -> None:
    """Validate the per-crop metadata of a crop-backed CT package.

    Args:
        metadata: Stored sample metadata.
        context: Database location used in errors.

    Returns:
        ``None``.
    """

    required = {
        "source_sample_id",
        "crop_index",
        "z_start",
        "z_stop",
        "z_pad_before",
        "z_pad_after",
        "crop_foreground_label_ids",
        "fg_centroids_ijk",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"{context} crop metadata is missing {missing}.")
    label_ids = list(metadata["crop_foreground_label_ids"])
    centroids = list(metadata["fg_centroids_ijk"])
    if len(label_ids) != len(centroids) or not label_ids:
        raise ValueError(
            f"{context} must store one fg_centroids_ijk row per crop_foreground_label_ids "
            "entry and at least one foreground label."
        )
    if any(len(row) != 3 for row in centroids):
        raise ValueError(f"{context}.fg_centroids_ijk rows must be (i, j, k) triples.")


def _validate_stored_payload(
    payload: Mapping[str, Any],
    *,
    dataset_type: DatasetType,
    label_encoding: LabelEncoding,
    label_names: tuple[str, ...] | None,
    stored_labels: dict[int, str] | None,
    context: str,
) -> None:
    """Validate one canonical per-sample payload mapping.

    Args:
        payload: Stored sample mapping.
        dataset_type: Declared package type.
        label_encoding: Declared common label representation.
        label_names: Optional ordered channel names.
        stored_labels: Optional native names for dense integer ids.
        context: Database location used in errors.

    Returns:
        ``None``.
    """

    if "img" not in payload or "seg" not in payload:
        raise KeyError(f"{context} must define 'img' and 'seg'.")
    image = np.asarray(payload["img"])
    label = np.asarray(payload["seg"])
    if dataset_type == "xray-seg":
        _validate_canonical_xray_image(image, context=f"{context}.img")
        _validate_numeric_array(label, context=f"{context}.seg")
        expected_ndim = 2 if label_encoding == "dense" else 3
        if label.ndim != expected_ndim:
            raise ValueError(
                f"{context}.seg conflicts with label_encoding "
                f"{label_encoding!r}."
            )
        if label_encoding == "dense":
            _validate_dense_label(label, context=f"{context}.seg")
            if stored_labels is None:
                raise ValueError("Dense labels require stored_labels.")
            _validate_declared_dense_ids(
                label,
                stored_labels,
                context=f"{context}.seg",
            )
        else:
            _validate_channel_label(label, context=f"{context}.seg")
        _validate_xray_shapes(
            image,
            label,
            label_names=label_names,
            context=context,
        )
        return
    for key in ("affine", "spacing"):
        if key not in payload:
            raise KeyError(f"{context} must define {key!r}.")
    _validate_ct_payloads(
        image,
        label,
        np.asarray(payload["affine"]),
        np.asarray(payload["spacing"]),
        context=context,
    )
    if stored_labels is None:
        raise ValueError("Dense labels require stored_labels.")
    _validate_declared_dense_ids(label, stored_labels, context=f"{context}.seg")


def _smoke_load_runtime(
    db: Mapping[str, Any],
    report: DatasetPackageReport,
) -> None:
    """Load one sample per non-empty split through public runtime classes.

    Args:
        db: Open canonical ThunderDB mapping.
        report: Validated package report.

    Returns:
        ``None``.
    """

    splits = _required_mapping(db, "_splits")
    modality = "ct" if report.dataset_type == "ct-seg" else "xray"
    dataset_class = (
        CTTrainingDataset if report.dataset_type == "ct-seg" else XrayTrainingDataset
    )
    for split, raw_ids in splits.items():
        sample_ids = _string_sequence(raw_ids, context=f"_splits[{split!r}]")
        if not sample_ids:
            continue
        backend = SplitThunderDBStorageBackend(
            db,
            dataset_name=report.dataset_name,
            modality=modality,
            split=str(split),
            subject_grouping="subject",
        )
        dataset = dataset_class(
            backend=backend,
            dataset_name=report.dataset_name,
            label_mode="native",
        )
        dataset[0]


def _planned_source_paths(plan: _PackagePlan) -> tuple[Path, ...]:
    """Collect the manifest and every payload path referenced by a plan.

    Args:
        plan: Validated package plan whose sources should be protected.

    Returns:
        Manifest and payload paths in deterministic manifest order.
    """

    paths = [plan.source_path]
    for sample in plan.samples:
        paths.extend((sample.image.path, sample.label.path))
        if sample.affine is not None:
            paths.append(sample.affine.path)
        if sample.spacing is not None:
            paths.append(sample.spacing.path)
    return tuple(paths)


def _validate_overwrite_target(output_path: Path) -> None:
    """Require an existing overwrite target to be a canonical package.

    Args:
        output_path: Existing destination proposed for replacement.

    Returns:
        ``None`` when the destination passes full package validation.

    Raises:
        ImportError: If the storage dependency is unavailable.
        ValueError: If any other validation failure makes replacement unsafe.
    """

    try:
        validate_packed_dataset(output_path)
    except ImportError:
        raise
    except Exception as exc:
        raise ValueError(
            "Refusing to overwrite an existing destination that is not a "
            f"canonical FleXray package: {output_path}."
        ) from exc


def _commit_package(
    temporary_path: Path,
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    """Commit a validated temporary directory while preserving failed targets.

    Args:
        temporary_path: Fully written sibling temporary directory.
        output_path: Requested final directory.
        overwrite: Whether an existing destination may be replaced.

    Returns:
        ``None``.
    """

    if not output_path.exists():
        os.replace(temporary_path, output_path)
        return
    if not overwrite:
        raise FileExistsError(output_path)
    backup_path = output_path.parent / (
        f".{output_path.name}.fxr-backup-{uuid.uuid4().hex}"
    )
    os.replace(output_path, backup_path)
    try:
        os.replace(temporary_path, output_path)
    except BaseException:
        os.replace(backup_path, output_path)
        raise
    shutil.rmtree(backup_path)


def _report_from_plan(plan: _PackagePlan, *, path: Path) -> DatasetPackageReport:
    """Build a public summary from a validated package plan.

    Args:
        plan: Validated plan.
        path: Manifest or output path represented by the report.

    Returns:
        Public package report.
    """

    return DatasetPackageReport(
        path=path,
        dataset_name=plan.dataset_name,
        dataset_type=plan.dataset_type,
        label_encoding=plan.label_encoding,
        num_subjects=len(plan.subjects),
        num_samples=len(plan.samples),
        split_counts={name: len(ids_) for name, ids_ in plan.splits.items()},
    )


def _dataset_type(raw_value: Any, *, context: str = "manifest") -> DatasetType:
    """Validate a supported dataset type.

    Args:
        raw_value: Candidate dataset type.
        context: Container name used in errors.

    Returns:
        Narrowed dataset type literal.
    """

    value = _required_text(raw_value, f"{context}.dataset_type")
    if value not in {"ct-seg", "xray-seg"}:
        raise ValueError(
            f"{context}.dataset_type must be 'ct-seg' or 'xray-seg', "
            f"got {value!r}."
        )
    return value  # type: ignore[return-value]


def _label_encoding(
    label: np.ndarray,
    *,
    dataset_type: DatasetType,
) -> LabelEncoding:
    """Infer the supported label encoding from an already validated array.

    Args:
        label: Validated label array.
        dataset_type: Dataset type controlling expected dimensionality.

    Returns:
        ``"dense"`` or ``"channels"``.
    """

    if dataset_type == "ct-seg" or label.ndim == 2:
        return "dense"
    return "channels"


def _stored_label_encoding(raw_value: Any) -> LabelEncoding:
    """Validate a label encoding read from package attributes.

    Args:
        raw_value: Stored label encoding.

    Returns:
        Narrowed label encoding literal.
    """

    value = _required_text(raw_value, "_attrs.label_encoding")
    if value not in {"dense", "channels"}:
        raise ValueError("_attrs.label_encoding must be 'dense' or 'channels'.")
    return value  # type: ignore[return-value]


def _payload_path(raw_value: Any, root: Path, name: str) -> Path:
    """Resolve and verify one manifest payload path.

    Args:
        raw_value: Required path string.
        root: Manifest directory for relative paths.
        name: Field name used in errors.

    Returns:
        Existing regular-file path.
    """

    value = _required_text(raw_value, name)
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} payload was not found: {path}")
    return path


def _required_text(raw_value: Any, name: str) -> str:
    """Validate and strip a required text value.

    Args:
        raw_value: Candidate value.
        name: Field name used in errors.

    Returns:
        Non-empty stripped text.
    """

    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return raw_value.strip()


def _optional_text(raw_value: Any, name: str) -> str | None:
    """Validate and strip optional text.

    Args:
        raw_value: Candidate value or ``None``.
        name: Field name used in errors.

    Returns:
        ``None`` or non-empty stripped text.
    """

    if raw_value is None:
        return None
    return _required_text(raw_value, name)


def _optional_label_names(raw_value: Any) -> tuple[str, ...] | None:
    """Validate optional ordered channel label names.

    Args:
        raw_value: Optional YAML list.

    Returns:
        Tuple of unique non-empty names, or ``None``.
    """

    if raw_value is None:
        return None
    if not isinstance(raw_value, list):
        raise TypeError("label_names must be a list of strings.")
    names = tuple(
        _required_text(name, f"label_names[{index}]")
        for index, name in enumerate(raw_value)
    )
    if not names:
        raise ValueError("label_names cannot be empty.")
    if len(set(names)) != len(names):
        raise ValueError("label_names cannot contain duplicates.")
    return names


def _optional_stored_labels(raw_value: Any) -> dict[int, str] | None:
    """Validate optional dense native label declarations.

    Args:
        raw_value: Optional mapping from integer-like ids to label names.

    Returns:
        Contiguous integer-keyed labels, or ``None`` when omitted.

    Raises:
        TypeError: If the declaration is not a mapping.
        ValueError: If ids or names violate the dense-label contract.
    """

    if raw_value is None:
        return None
    if not isinstance(raw_value, Mapping):
        raise TypeError("stored_labels must be a mapping from ids to names.")
    labels: dict[int, str] = {}
    for raw_id, raw_name in raw_value.items():
        if isinstance(raw_id, bool):
            raise ValueError("stored_labels ids must be non-negative integers.")
        if isinstance(raw_id, int):
            label_id = raw_id
        elif isinstance(raw_id, str) and raw_id.strip().isdigit():
            label_id = int(raw_id.strip())
        else:
            raise ValueError("stored_labels ids must be non-negative integers.")
        if label_id < 0 or label_id in labels:
            raise ValueError("stored_labels ids must be unique non-negative integers.")
        labels[label_id] = _required_text(
            raw_name,
            f"stored_labels[{label_id}]",
        )
    if not labels:
        raise ValueError("stored_labels cannot be empty.")
    expected_ids = list(range(len(labels)))
    if sorted(labels) != expected_ids:
        raise ValueError(
            "stored_labels ids must be contiguous starting at 0; "
            f"expected {expected_ids!r}, got {sorted(labels)!r}."
        )
    ordered = {label_id: labels[label_id] for label_id in expected_ids}
    if ordered[0] != "background":
        raise ValueError("stored_labels id 0 must be named 'background'.")
    if len(set(ordered.values())) != len(ordered):
        raise ValueError("stored_labels names must be unique.")
    return ordered


def _metadata(raw_value: Any, *, context: str) -> dict[str, Any]:
    """Validate user sample metadata as JSON-compatible data.

    Args:
        raw_value: Optional metadata mapping.
        context: Manifest location used in errors.

    Returns:
        Shallow metadata copy.
    """

    if raw_value is None:
        return {}
    if not isinstance(raw_value, Mapping):
        raise TypeError(f"{context} must be a mapping.")
    metadata = dict(raw_value)
    if any(not isinstance(key, str) or not key for key in metadata):
        raise ValueError(f"{context} keys must be non-empty strings.")
    try:
        encoded = json.dumps(metadata, allow_nan=False)
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context} must contain JSON-compatible values.") from exc
    if normalized != metadata:
        raise TypeError(
            f"{context} nested mappings must use string keys and lists."
        )
    return metadata


def _reject_unknown_fields(
    data: Mapping[str, Any],
    allowed: set[str],
    *,
    context: str,
) -> None:
    """Reject misspelled or unsupported schema fields.

    Args:
        data: Mapping whose keys should be inspected.
        allowed: Accepted field names.
        context: Manifest location used in errors.

    Returns:
        ``None``.
    """

    unknown = sorted(set(data).difference(allowed))
    if unknown:
        raise ValueError(f"{context} contains unsupported fields: {unknown!r}.")


def _required_mapping(db: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Read a required mapping from a database-like object.

    Args:
        db: Open database mapping.
        key: Required database key.

    Returns:
        Shallow dictionary copy.
    """

    try:
        raw_value = db[key]
    except (KeyError, LookupError):
        raise KeyError(f"ThunderDB package must define {key!r}.") from None
    if not isinstance(raw_value, Mapping):
        raise TypeError(f"ThunderDB {key} must be a mapping.")
    return dict(raw_value)


def _required_string_list(db: Mapping[str, Any], key: str) -> list[str]:
    """Read a required list of non-empty strings from a database.

    Args:
        db: Open database mapping.
        key: Required database key.

    Returns:
        Validated list of strings.
    """

    try:
        raw_value = db[key]
    except (KeyError, LookupError):
        raise KeyError(f"ThunderDB package must define {key!r}.") from None
    return _string_sequence(raw_value, context=key)


def _string_sequence(raw_value: Any, *, context: str) -> list[str]:
    """Validate a list or tuple of non-empty strings.

    Args:
        raw_value: Candidate sequence.
        context: Storage location used in errors.

    Returns:
        Validated string list.
    """

    if not isinstance(raw_value, (list, tuple)):
        raise TypeError(f"{context} must be a list of strings.")
    return [
        _required_text(value, f"{context}[{index}]")
        for index, value in enumerate(raw_value)
    ]


def _import_thunderpack() -> Any:
    """Import the optional training storage dependency with a useful error.

    Returns:
        Imported ``thunderpack`` module.

    Raises:
        ImportError: If the training dependency is unavailable.
    """

    try:
        import thunderpack
    except ImportError as exc:
        raise ImportError(
            "Dataset packaging requires thunderpack. Install FleXray with "
            'the train extra: python -m pip install "flexray[train]".'
        ) from exc
    return thunderpack
