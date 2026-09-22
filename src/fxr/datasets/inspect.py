"""Report the layout of any split/sample ThunderDB without blessing it.

``fxr-dataset check`` only accepts canonical FleXray packages; ``inspect``
describes what a database declares (including legacy-built ones) so users can
diagnose a rejected or foreign ThunderDB.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._thunderdb import open_thunderdb

_ATTR_KEYS = (
    "schema_name",
    "schema_version",
    "dataset_name",
    "dataset",
    "dataset_type",
    "modality",
    "storage_layout",
    "seg_storage",
    "label_encoding",
    "protocol_name",
    "version",
    "resolution",
)


@dataclass(frozen=True)
class ThunderDBInspection:
    """Declared layout of one ThunderDB.

    Attributes:
        path: Database path.
        attrs: Selected ``_attrs`` entries that were present.
        split_counts: Samples per split from ``_splits``.
        num_subjects: Distinct ``subject_id`` values in ``_metadata`` (or the
            sample count when metadata carries none).
        label_names: Declared ``label_names`` / ``mask_label_names``, if any.
        stored_labels: Declared ``stored_labels``, if any.
        payload_fields: Field name to ``(shape, dtype)`` of the first sample.
    """

    path: Path
    attrs: dict[str, Any]
    split_counts: dict[str, int]
    num_subjects: int
    label_names: tuple[str, ...] | None
    stored_labels: dict[str, str] | None
    payload_fields: dict[str, tuple[tuple[int, ...], str]]


def inspect_thunderdb(path: str | Path) -> ThunderDBInspection:
    """Describe a ThunderDB's declared layout and first sample.

    Args:
        path: Existing ThunderDB directory.

    Returns:
        Inspection summary.

    Raises:
        ValueError: If the database stores a pickled value, which FleXray
            refuses to decode.
    """

    root = Path(path).expanduser().resolve()
    assert root.is_dir(), f"{root} is not a ThunderDB directory."
    with open_thunderdb(root) as db:
        attrs = dict(db["_attrs"]) if "_attrs" in db else {}
        splits = dict(db["_splits"]) if "_splits" in db else {}
        metadata = dict(db["_metadata"]) if "_metadata" in db else {}
        split_counts = {str(k): len(list(v)) for k, v in splits.items()}
        sample_ids = [str(s) for ids in splits.values() for s in ids]
        subjects = {
            str(metadata.get(s, {}).get("subject_id", s))
            for s in sample_ids
            if isinstance(metadata.get(s, {}), Mapping)
        }
        payload_fields = _first_payload_fields(db, sample_ids)
    names = attrs.get("mask_label_names", attrs.get("label_names"))
    stored = attrs.get("stored_labels")
    return ThunderDBInspection(
        path=root,
        attrs={k: attrs[k] for k in _ATTR_KEYS if k in attrs},
        split_counts=split_counts,
        num_subjects=len(subjects),
        label_names=None if names is None else tuple(str(n) for n in names),
        stored_labels=None if stored is None else {str(k): str(v) for k, v in dict(stored).items()},
        payload_fields=payload_fields,
    )


def format_inspection(inspection: ThunderDBInspection) -> str:
    """Render an inspection as plain text.

    Args:
        inspection: Result of :func:`inspect_thunderdb`.

    Returns:
        Multi-line human-readable report.
    """

    canonical = inspection.attrs.get("schema_name") == "flexray-training-dataset"
    lines = [
        f"{inspection.path}",
        f"  package: {'canonical FleXray' if canonical else 'non-canonical (fxr-dataset check will reject it)'}",
    ]
    lines.extend(f"  {key}: {value}" for key, value in inspection.attrs.items())
    splits = ", ".join(f"{k}={v}" for k, v in inspection.split_counts.items()) or "none"
    lines.append(f"  splits: {splits}")
    lines.append(f"  subjects: {inspection.num_subjects}")
    if inspection.label_names is not None:
        lines.append(f"  label_names ({len(inspection.label_names)}): {', '.join(inspection.label_names)}")
    if inspection.stored_labels is not None:
        pairs = ", ".join(f"{k}={v}" for k, v in inspection.stored_labels.items())
        lines.append(f"  stored_labels: {pairs}")
    for field, (shape, dtype) in inspection.payload_fields.items():
        lines.append(f"  payload {field}: shape={shape} dtype={dtype}")
    return "\n".join(lines)


def _first_payload_fields(
    db: Any, sample_ids: list[str]
) -> dict[str, tuple[tuple[int, ...], str]]:
    """Return array shapes/dtypes of the first sample's payload fields."""
    if not sample_ids:
        return {}
    payload = db[sample_ids[0]]
    if not isinstance(payload, Mapping):
        return {}
    fields = {}
    for key, value in payload.items():
        array = np.asarray(value)
        fields[str(key)] = (tuple(int(v) for v in array.shape), str(array.dtype))
    return fields
