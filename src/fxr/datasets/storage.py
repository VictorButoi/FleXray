from __future__ import annotations

import json
import pickle
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import yaml

from .schemas import DatasetRecord


class StorageBackend(Protocol):
    """Storage backend that can load record payload fields.

    Attributes:
        records: Immutable sequence of records available from the backend.
    """

    records: tuple[DatasetRecord, ...]

    def load(self, record: DatasetRecord, field: str) -> Any:
        """Load one payload field for a dataset record.

        Args:
            record: Dataset record that owns the payload reference.
            field: Name of the ``DatasetRecord`` field containing the payload
                key or path.

        Returns:
            Loaded payload object for ``field``.
        """

        ...


class ManifestStorageBackend:
    """Manifest-backed storage for portable training datasets and fixtures.

    Attributes:
        root: Base directory used to resolve relative payload paths.
        payloads: Inline manifest payloads keyed by payload id.
        records: Immutable sequence of normalized dataset records.
    """

    def __init__(
        self,
        manifest: str | Path | Mapping[str, Any] | Iterable[Mapping[str, Any]],
        *,
        root: str | Path | None = None,
    ) -> None:
        """Initialize storage from a manifest path or in-memory manifest body.

        Args:
            manifest: Manifest path, manifest mapping, or iterable of record
                mappings.
            root: Optional base directory for relative payload paths. When
                omitted for file manifests, the manifest directory is used.

        Returns:
            ``None``.

        Raises:
            TypeError: If the manifest does not define a record list.
            ValueError: If record normalization rejects a record field.
        """

        data, manifest_root = _load_manifest(manifest)
        self.root = Path(root) if root is not None else manifest_root
        self.payloads = (
            dict(data.get("payloads", {})) if isinstance(data, Mapping) else {}
        )
        default_dataset_name = (
            data.get("dataset_name") if isinstance(data, Mapping) else None
        )
        default_modality = data.get("modality") if isinstance(data, Mapping) else None
        raw_records = data.get("records") if isinstance(data, Mapping) else data
        if not isinstance(raw_records, list):
            raise TypeError("Manifest storage must define a records list.")
        self.records = tuple(
            DatasetRecord.from_mapping(
                raw_record,
                default_dataset_name=default_dataset_name,
                default_modality=default_modality,
            )
            for raw_record in raw_records
        )

    def load(self, record: DatasetRecord, field: str) -> Any:
        """Load a payload from inline manifest data or a resolved file path.

        Args:
            record: Dataset record that owns the payload reference.
            field: Name of the record field containing the payload key or path.

        Returns:
            Loaded payload object.

        Raises:
            KeyError: If the record field is ``None``.
            FileNotFoundError: If a referenced payload path does not exist.
            ValueError: If the payload file extension is unsupported.
        """

        value = getattr(record, field)
        if value is None:
            raise KeyError(f"Record {record.data_id!r} has no payload field {field!r}.")
        if value in self.payloads:
            return self.payloads[value]
        path = Path(value)
        if not path.is_absolute():
            path = self.root / path
        return _load_path(path)

    @classmethod
    def from_file(cls, path: str | Path) -> "ManifestStorageBackend":
        """Build manifest storage from a manifest file path.

        Args:
            path: JSON or YAML manifest path.

        Returns:
            Manifest storage backend rooted at the manifest directory.
        """

        return cls(path)


class ThunderDBStorageBackend:
    """Small ThunderDB adapter using public key/value-style access.

    Attributes:
        db: Opened ThunderDB-like object or compatible key/value object.
        _owns_database: Whether this backend must close ``db``.
        _closed: Whether the owned reader has already been released.
        records: Immutable sequence of normalized dataset records read from the
            database or explicit ``records`` argument.
    """

    def __init__(
        self,
        db: Any,
        *,
        records: Iterable[Mapping[str, Any] | DatasetRecord] | None = None,
        record_key: str = "records",
        owns_database: bool | None = None,
    ) -> None:
        """Initialize storage around an opened database or database path.

        Args:
            db: Open ThunderDB-like object, compatible key/value object, or path
                that ``thunderpack`` can open.
            records: Optional explicit record mappings. When omitted, records
                are loaded from ``record_key`` in the database.
            record_key: Database key containing record mappings when ``records``
                is omitted.
            owns_database: Whether this backend closes ``db``. By default, paths
                opened by the backend are owned and passed objects are borrowed.

        Returns:
            ``None``.

        Raises:
            ValueError: If no records are provided or stored under
                ``record_key``.
        """

        self.db = _open_thunderdb(db)
        inferred_ownership = isinstance(db, (str, Path))
        self._owns_database = (
            inferred_ownership if owns_database is None else bool(owns_database)
        )
        self._closed = False
        try:
            raw_records = (
                records
                if records is not None
                else _read_db_value(self.db, record_key)
            )
            if raw_records is None:
                raise ValueError(
                    "ThunderDBStorageBackend requires records=... or a 'records' "
                    "entry in the database."
                )
            self.records = tuple(_coerce_record(record) for record in raw_records)
        except BaseException:
            self.close()
            raise

    def load(self, record: DatasetRecord, field: str) -> Any:
        """Load a payload from the database key named by a record field.

        Args:
            record: Dataset record that owns the payload reference.
            field: Name of the record field containing the database key.

        Returns:
            Loaded database payload.

        Raises:
            KeyError: If the record field is ``None`` or the payload key is not
                present in the database.
        """

        key = getattr(record, field)
        if key is None:
            raise KeyError(f"Record {record.data_id!r} has no payload field {field!r}.")
        return _read_db_value(self.db, key, required=True)

    def close(self) -> None:
        """Close the database reader when this backend owns it.

        Returns:
            ``None``. Repeated calls are safe.
        """

        if self._closed:
            return
        self._closed = True
        if self._owns_database:
            _close_thunderdb(self.db)


class SplitThunderDBStorageBackend:
    """Adapter for split/sample ThunderDB training layouts.

    Attributes:
        db: Opened ThunderDB-like object or compatible key/value object.
        _owns_database: Whether this backend must close ``db``.
        _closed: Whether the owned reader has already been released.
        dataset_name: Dataset name assigned to materialized records.
        modality: Modality assigned to materialized records.
        split: Split name read from the database ``_splits`` mapping.
        attrs: Database ``_attrs`` mapping copied at construction. Clean
            layouts declare ``dataset_name`` and ``clean_dataset=True``; legacy
            ThunderDBs may instead declare ``dataset``.
        metadata: Optional database ``_metadata`` mapping used for records.
        image_key: Per-sample payload field used for image arrays.
        label_key: Optional per-sample payload field used for segmentation arrays.
        affine_key: Optional per-sample payload field used for CT affine arrays.
        spacing_key: Optional per-sample payload field used for CT spacing arrays.
        label_names: Optional ordered native-id or channel names shared by label
            payloads, read from ``_attrs.label_names`` (or ``mask_label_names``
            for overlapping channel-mask storage).
        seg_storage: Declared label storage, ``"indexed_label_mask"`` (dense id
            maps) or ``"overlapping_channel_mask"`` (binary channel stacks whose
            channel index equals the native id); ``None`` when undeclared.
        subject_grouping: ``"sample"`` to use one subject per sample, or
            ``"subject"`` to read subject identifiers from sample metadata.
        record_metadata: Whether per-sample metadata is copied onto records.
        lazy_payload_refs: Whether to skip per-sample payload validation while
            materializing records and trust the configured payload keys. This is
            useful for CT ThunderDBs whose sample records contain large arrays.
        _payload_refs: Payload references indexed by sample id.
        records: Immutable sequence of records materialized from the requested split.
    """

    def __init__(
        self,
        db: Any,
        *,
        dataset_name: str,
        modality: str,
        split: str,
        image_key: str | None = "img",
        label_key: str | None = "seg",
        affine_key: str | None = "affine",
        spacing_key: str | None = "spacing",
        subject_grouping: str = "sample",
        record_metadata: bool = True,
        lazy_payload_refs: bool = False,
        owns_database: bool | None = None,
    ) -> None:
        """Initialize storage from a split/sample ThunderDB layout.

        Args:
            db: Open ThunderDB-like object, compatible key/value object, or path
                that ``thunderpack`` can open.
            dataset_name: Dataset name assigned to every materialized record.
            modality: Modality assigned to every materialized record.
            split: Split name to read from ``_splits``.
            image_key: Fallback image payload field inside each sample record.
            label_key: Fallback label payload field inside each sample record,
                or ``None`` to create records without labels.
            affine_key: Fallback CT affine payload field inside each sample
                record, or ``None`` to omit affine payload references.
            spacing_key: Fallback CT spacing payload field inside each sample
                record, or ``None`` to omit spacing payload references.
            subject_grouping: ``"sample"`` for one subject per sample, or
                ``"subject"`` to read subject ids from metadata.
            record_metadata: Whether to copy metadata into each record.
            lazy_payload_refs: Whether to build record payload references from
                split keys and configured payload names without loading each
                per-sample payload record up front.
            owns_database: Whether this backend closes ``db``. By default, paths
                opened by the backend are owned and passed objects are borrowed.

        Returns:
            ``None``.

        Raises:
            KeyError: If ``_attrs``, ``_splits``, the requested split, a sample
                record, or a required image payload field is missing.
            TypeError: If layout metadata is not mapping-like.
            ValueError: If constructor strings, dataset identity attributes, or
                split entries are invalid.
        """

        self.db = _open_thunderdb(db)
        inferred_ownership = isinstance(db, (str, Path))
        self._owns_database = (
            inferred_ownership if owns_database is None else bool(owns_database)
        )
        self._closed = False
        try:
            self.dataset_name = _required_text(dataset_name, "dataset_name")
            self.modality = _required_text(modality, "modality")
            self.split = _required_text(split, "split")
            self.subject_grouping = _validate_subject_grouping(subject_grouping)
            self.record_metadata = bool(record_metadata)
            self.attrs = _read_db_mapping(self.db, "_attrs", required=True)
            _validate_clean_dataset_attrs(self.attrs, dataset_name=self.dataset_name)
            _validate_canonical_package_modality(
                self.attrs,
                dataset_name=self.dataset_name,
                modality=self.modality,
            )
            self.metadata = _read_db_mapping(self.db, "_metadata", required=False)
            self.image_key = _resolve_payload_key(
                self.attrs,
                semantic_name="image",
                aliases=("image", "img"),
                fallback=image_key,
                required=True,
            )
            self.label_key = _resolve_payload_key(
                self.attrs,
                semantic_name="label",
                aliases=("label", "seg", "mask"),
                fallback=label_key,
                required=False,
            )
            self.affine_key = _resolve_payload_key(
                self.attrs,
                semantic_name="affine",
                aliases=("affine",),
                fallback=affine_key,
                required=False,
            )
            self.spacing_key = _resolve_payload_key(
                self.attrs,
                semantic_name="spacing",
                aliases=("spacing",),
                fallback=spacing_key,
                required=False,
            )
            self.label_names = (
                _label_names_from_attrs(self.attrs, context="ThunderDB _attrs")
                if "label_names" in self.attrs
                else None
            )
            self.seg_storage = _seg_storage_from_attrs(self.attrs, context="ThunderDB _attrs")
            if "mask_label_names" in self.attrs:
                self.label_names = _mask_label_names_from_attrs(
                    self.attrs, context="ThunderDB _attrs"
                )
            self.lazy_payload_refs = bool(lazy_payload_refs)
            self._payload_refs: dict[str, _SplitPayloadRef] = {}
            self.records = self._build_records(_split_entries(self.db, self.split))
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Close the database reader when this backend owns it.

        Returns:
            ``None``. Repeated calls are safe.
        """

        if self._closed:
            return
        self._closed = True
        if self._owns_database:
            _close_thunderdb(self.db)

    def load(self, record: DatasetRecord, field: str) -> Any:
        """Load one structured per-sample payload field.

        Args:
            record: Dataset record created by this backend.
            field: Name of the record field containing an opaque backend key.

        Returns:
            Payload stored under the referenced per-sample field.

        Raises:
            KeyError: If the record field is empty, unknown to this backend, or
                points to a missing payload field.
        """

        key = getattr(record, field)
        if key is None:
            raise KeyError(f"Record {record.data_id!r} has no payload field {field!r}.")
        ref = self._payload_refs.get(key)
        if ref is None:
            raise KeyError(
                f"Record {record.data_id!r} field {field!r} is not managed by this "
                "SplitThunderDBStorageBackend."
            )
        payload_record = _sample_payload_record(self.db, ref.sample_key)
        if ref.payload_key not in payload_record:
            raise KeyError(
                f"ThunderDB sample {ref.sample_key!r} does not define payload "
                f"field {ref.payload_key!r}."
            )
        return payload_record[ref.payload_key]

    def _build_records(self, split_entries: Iterable[Any]) -> tuple[DatasetRecord, ...]:
        """Materialize dataset records from split entries.

        Args:
            split_entries: Entries from the requested ``_splits`` value.

        Returns:
            Immutable tuple of normalized ``DatasetRecord`` values.
        """

        records: list[DatasetRecord] = []
        for entry in split_entries:
            sample_key = _split_entry_sample_key(entry)
            metadata = (
                _record_metadata(self.metadata, sample_key)
                if self.record_metadata
                else {}
            )
            payload_record = (
                None
                if self.lazy_payload_refs
                else _sample_payload_record(self.db, sample_key)
            )
            if payload_record is not None and self.image_key not in payload_record:
                raise KeyError(
                    f"ThunderDB sample {sample_key!r} does not define image "
                    f"payload field {self.image_key!r}."
                )
            record = DatasetRecord(
                dataset_name=self.dataset_name,
                modality=self.modality,
                data_id=sample_key,
                image=self._register_payload_ref(sample_key, self.image_key),
                label=self._payload_ref(payload_record, sample_key, self.label_key),
                split=self.split,
                subject_id=_subject_id(sample_key, metadata, self.subject_grouping),
                sample_id=sample_key,
                affine=self._payload_ref(
                    payload_record,
                    sample_key,
                    self.affine_key,
                ),
                spacing=self._payload_ref(
                    payload_record,
                    sample_key,
                    self.spacing_key,
                ),
                label_names=self.label_names,
                metadata=metadata,
            )
            records.append(record)
        return tuple(records)

    def _register_payload_ref(self, sample_key: str, payload_key: str) -> str:
        """Register a structured payload reference and return its opaque key.

        Args:
            sample_key: Database key for the sample record.
            payload_key: Field inside the sample record.

        Returns:
            Opaque record payload key understood only by this backend.
        """

        opaque_key = f"__fxr_payload_ref_{len(self._payload_refs)}"
        self._payload_refs[opaque_key] = _SplitPayloadRef(
            sample_key=sample_key,
            payload_key=payload_key,
        )
        return opaque_key

    def _payload_ref(
        self,
        payload_record: Mapping[str, Any] | None,
        sample_key: str,
        payload_key: str | None,
    ) -> str | None:
        """Return a payload reference when configured and, if checked, present.

        Args:
            payload_record: Per-sample payload mapping, or None when lazy
                payload references are enabled.
            sample_key: Database key for the sample record.
            payload_key: Optional field name inside the sample payload.

        Returns:
            Opaque payload key, or None when the configured field is absent
            or not present in an eagerly checked sample payload.
        """

        if payload_key is None:
            return None
        if payload_record is not None and payload_key not in payload_record:
            return None
        return self._register_payload_ref(sample_key, payload_key)


@dataclass(frozen=True)
class _SplitPayloadRef:
    """Structured reference to one field inside one ThunderDB sample record.

    Attributes:
        sample_key: Database key that resolves to the per-sample payload mapping.
        payload_key: Field inside the sample payload mapping.
    """

    sample_key: str
    payload_key: str


def _load_manifest(
    manifest: str | Path | Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> tuple[Mapping[str, Any] | list[Mapping[str, Any]], Path]:
    """Load a manifest body and determine its root directory.

    Args:
        manifest: Manifest path, manifest mapping, or iterable of record
            mappings.

    Returns:
        Pair of loaded manifest data and root path used for relative payloads.
    """

    if isinstance(manifest, (str, Path)):
        path = Path(manifest)
        with open(path, encoding="utf-8") as f:
            if path.suffix.lower() == ".json":
                data = json.load(f)
            else:
                data = yaml.safe_load(f) or {}
        return data, path.parent
    if isinstance(manifest, Mapping):
        return manifest, Path.cwd()
    return list(manifest), Path.cwd()


def _load_path(path: Path) -> Any:
    """Load a manifest payload file by extension.

    Args:
        path: Absolute or resolved payload file path.

    Returns:
        Loaded array, tensor, mapping, or scalar payload object.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file extension is unsupported.
    """

    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) == 1:
                return archive[archive.files[0]]
            return {name: archive[name] for name in archive.files}
    if suffix in {".pt", ".pth"}:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError as exc:
            raise ValueError(
                f"Unsafe or unsupported tensor payload in {path}; .pt/.pth "
                "manifest files must use the weights-only tensor contract."
            ) from exc
        return _validate_safe_torch_payload(payload, path=path)
    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    if suffix in {".yml", ".yaml"}:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    raise ValueError(f"Unsupported manifest payload file extension for {path}.")


def _validate_safe_torch_payload(value: Any, *, path: Path) -> Any:
    """Validate an object returned by PyTorch weights-only loading.

    Args:
        value: Deserialized tensor payload to inspect recursively.
        path: Source file used in validation errors.

    Returns:
        The original validated payload object.

    Raises:
        TypeError: If the payload contains an unsupported container, key, or
            leaf object.
    """

    scalar_types = (str, bytes, int, float, bool, type(None))
    if isinstance(value, (torch.Tensor, *scalar_types)):
        return value
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, scalar_types):
                raise TypeError(
                    f"Safe tensor payload mapping keys in {path} must be basic "
                    f"scalars, got {type(key).__name__}."
                )
            _validate_safe_torch_payload(item, path=path)
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_safe_torch_payload(item, path=path)
        return value
    raise TypeError(
        f"Safe tensor payload {path} contains unsupported "
        f"{type(value).__name__}; expected tensors, basic scalars, and nested "
        "mapping/list/tuple containers."
    )


def _open_thunderdb(db: Any) -> Any:
    """Open a ThunderDB path or pass through an opened database object.

    Paths are opened read-only through :func:`fxr.datasets._thunderdb.open_thunderdb`,
    which refuses pickled values; the other ``thunderpack`` openers decode them.

    Args:
        db: Open database-like object or filesystem path accepted by
            ``thunderpack``.

    Returns:
        Opened database-like object.

    Raises:
        ImportError: If the training dependency ``thunderpack`` is unavailable.
    """

    if not isinstance(db, (str, Path)):
        return db
    from ._thunderdb import open_thunderdb

    return open_thunderdb(db)


def _close_thunderdb(db: Any) -> None:
    """Close a ThunderDB-like object when it exposes ``close``.

    Args:
        db: Open database-like object.

    Returns:
        ``None``.
    """

    close = getattr(db, "close", None)
    if callable(close):
        close()


def _read_db_value(db: Any, key: str, *, required: bool = False) -> Any:
    """Read one value from a database-like object.

    Args:
        db: Mapping-like database object.
        key: Key to read.
        required: Whether absence should raise instead of returning ``None``.

    Returns:
        Stored value, or ``None`` when absent and not required.

    Raises:
        KeyError: If ``required`` is true and ``key`` is absent.
    """

    if hasattr(db, "get"):
        value = db.get(key)
        if value is not None or not required:
            return value
    try:
        return db[key]
    except (KeyError, LookupError, TypeError, AttributeError):
        if required:
            raise KeyError(f"ThunderDB payload key {key!r} was not found.") from None
        return None


def _read_db_mapping(db: Any, key: str, *, required: bool) -> dict[str, Any]:
    """Read and validate a mapping-valued database entry.

    Args:
        db: Mapping-like database object.
        key: Key whose value should be a mapping.
        required: Whether absence should raise instead of returning ``{}``.

    Returns:
        Shallow dictionary copied from the stored mapping, or an empty
        dictionary for optional missing entries.

    Raises:
        KeyError: If a required mapping is absent.
        TypeError: If the stored value is not mapping-like.
    """

    raw_value = _read_db_value(db, key, required=required)
    if raw_value is None:
        if required:
            raise KeyError(f"ThunderDB layout must define {key!r}.")
        return {}
    if not isinstance(raw_value, Mapping):
        raise TypeError(f"ThunderDB layout entry {key!r} must be a mapping.")
    return dict(raw_value)


def _validate_clean_dataset_attrs(
    attrs: Mapping[str, Any], *, dataset_name: str
) -> None:
    """Validate dataset identity metadata stored in ThunderDB ``_attrs``.

    Args:
        attrs: ThunderDB ``_attrs`` mapping.
        dataset_name: Requested dataset name that the database must declare.

    Returns:
        ``None``.

    Raises:
        ValueError: If the stored dataset name does not match ``dataset_name``
            or an explicit clean-dataset flag is false.
    """

    actual_name = attrs.get("dataset_name", attrs.get("dataset"))
    if actual_name != dataset_name:
        raise ValueError(
            "ThunderDB _attrs dataset identity must equal requested dataset "
            f"{dataset_name!r}; got {actual_name!r}."
        )
    if "clean_dataset" in attrs and attrs.get("clean_dataset") is not True:
        raise ValueError("ThunderDB _attrs.clean_dataset must be true when set.")


def _validate_canonical_package_modality(
    attrs: Mapping[str, Any],
    *,
    dataset_name: str,
    modality: str,
) -> None:
    """Validate modality metadata for a canonical FleXray package.

    Args:
        attrs: ThunderDB ``_attrs`` mapping.
        dataset_name: Requested dataset identity used in validation errors.
        modality: Runtime modality requested by the dataset builder.

    Returns:
        ``None`` for matching canonical packages and legacy layouts.

    Raises:
        ValueError: If canonical package modality or dataset type conflicts with
            the requested runtime modality.
    """

    if attrs.get("schema_name") != "flexray-training-dataset":
        return
    expected_types = {"ct": "ct-seg", "xray": "xray-seg"}
    expected_type = expected_types.get(modality)
    if expected_type is None:
        raise ValueError(
            f"Canonical FleXray package {dataset_name!r} cannot be opened as "
            f"modality {modality!r}; use 'ct' or 'xray'."
        )
    stored_modality = attrs.get("modality")
    stored_type = attrs.get("dataset_type")
    if stored_modality != modality or stored_type != expected_type:
        raise ValueError(
            f"Canonical FleXray package {dataset_name!r} declares modality "
            f"{stored_modality!r} and dataset_type {stored_type!r}, but was "
            f"configured as modality {modality!r} (expected dataset_type "
            f"{expected_type!r})."
        )


def _resolve_payload_key(
    attrs: Mapping[str, Any],
    *,
    semantic_name: str,
    aliases: tuple[str, ...],
    fallback: str | None,
    required: bool,
) -> str | None:
    """Resolve a per-sample payload key from attributes and fallback values.

    Args:
        attrs: ThunderDB ``_attrs`` mapping.
        semantic_name: Canonical semantic payload name such as ``"image"``.
        aliases: Alternative names accepted in ``_attrs``.
        fallback: Constructor-supplied fallback payload key.
        required: Whether the resolved key is mandatory.

    Returns:
        Resolved payload field name, or ``None`` for optional omitted fields.

    Raises:
        TypeError: If an ``_attrs`` payload-key container is not a mapping.
        ValueError: If a required key cannot be resolved to a non-empty string.
    """

    for container_key in ("payload_keys", "payload_fields", "keys"):
        if container_key not in attrs:
            continue
        container = attrs[container_key]
        if not isinstance(container, Mapping):
            raise TypeError(f"ThunderDB _attrs[{container_key!r}] must be a mapping.")
        payload_key = _payload_key_from_mapping(
            container,
            semantic_name=semantic_name,
            aliases=aliases,
            context=f"ThunderDB _attrs[{container_key!r}]",
            include_raw_aliases=True,
        )
        if payload_key is not None:
            return payload_key

    payload_key = _payload_key_from_mapping(
        attrs,
        semantic_name=semantic_name,
        aliases=aliases,
        context="ThunderDB _attrs",
        include_raw_aliases=False,
    )
    if payload_key is not None:
        return payload_key
    if fallback is None:
        if required:
            raise ValueError(f"{semantic_name}_key must be a non-empty string.")
        return None
    return _required_text(fallback, f"{semantic_name}_key")


def _payload_key_from_mapping(
    data: Mapping[str, Any],
    *,
    semantic_name: str,
    aliases: tuple[str, ...],
    context: str,
    include_raw_aliases: bool,
) -> str | None:
    """Resolve one payload key from a mapping of possible declarations.

    Args:
        data: Mapping containing candidate payload-key declarations.
        semantic_name: Canonical semantic payload name.
        aliases: Alternative semantic names to inspect.
        context: Human-readable location used in validation errors.
        include_raw_aliases: Whether bare semantic names are accepted as keys.

    Returns:
        Resolved non-empty payload field name, or ``None`` when no declaration
        is present.

    Raises:
        ValueError: If a present declaration is not a non-empty string.
    """

    candidate_keys: list[str] = [f"{semantic_name}_key", f"{semantic_name}_field"]
    for alias in aliases:
        candidate_keys.extend((f"{alias}_key", f"{alias}_field"))
    if include_raw_aliases:
        candidate_keys.extend((semantic_name, *aliases))

    seen: set[str] = set()
    for candidate_key in candidate_keys:
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        if candidate_key not in data or data[candidate_key] is None:
            continue
        return _required_text(data[candidate_key], f"{context}[{candidate_key!r}]")
    return None


def _split_entries(db: Any, split: str) -> tuple[Any, ...]:
    """Read split entries from a ThunderDB ``_splits`` mapping.

    Args:
        db: Mapping-like database object.
        split: Requested split name.

    Returns:
        Tuple of raw split entries.

    Raises:
        KeyError: If ``_splits`` or the requested split is absent.
        TypeError: If the split value is not an iterable of entries.
    """

    splits = _read_db_mapping(db, "_splits", required=True)
    if split not in splits:
        raise KeyError(f"ThunderDB _splits does not define split {split!r}.")
    entries = splits[split]
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Iterable):
        raise TypeError(
            f"ThunderDB _splits[{split!r}] must be an iterable of sample keys."
        )
    return tuple(entries)


def _split_entry_sample_key(entry: Any) -> str:
    """Extract a sample database key from one split entry.

    Args:
        entry: Split entry string or mapping with a key-like field.

    Returns:
        Sample key string.

    Raises:
        ValueError: If the entry does not contain a non-empty sample key.
    """

    if isinstance(entry, Mapping):
        for key in ("data_id", "sample_key", "key", "id"):
            if key in entry:
                return _required_text(entry[key], f"ThunderDB split entry {key!r}")
        raise ValueError(
            "ThunderDB split entry mappings must define 'data_id', 'sample_key', "
            "'key', or 'id'."
        )
    return _required_text(entry, "ThunderDB split entry")


def _sample_payload_record(db: Any, sample_key: str) -> Mapping[str, Any]:
    """Read and validate one per-sample payload mapping.

    Args:
        db: Mapping-like database object.
        sample_key: Database key for the sample.

    Returns:
        Stored payload mapping for the sample.

    Raises:
        KeyError: If the sample key is absent.
        TypeError: If the stored sample value is not a mapping.
    """

    raw_value = _read_db_value(db, sample_key, required=True)
    if not isinstance(raw_value, Mapping):
        raise TypeError(f"ThunderDB sample {sample_key!r} must be a payload mapping.")
    return raw_value


def _record_metadata(metadata: Mapping[str, Any], sample_key: str) -> dict[str, Any]:
    """Read per-sample metadata from the ThunderDB metadata mapping.

    Args:
        metadata: Database ``_metadata`` mapping.
        sample_key: Sample key whose metadata should be copied.

    Returns:
        Shallow metadata dictionary, or ``{}`` when no metadata exists.

    Raises:
        TypeError: If metadata for ``sample_key`` is not mapping-like.
    """

    raw_value = metadata.get(sample_key, {})
    if raw_value is None:
        return {}
    if not isinstance(raw_value, Mapping):
        raise TypeError(f"ThunderDB _metadata[{sample_key!r}] must be a mapping.")
    return dict(raw_value)


def _subject_id(
    sample_key: str, metadata: Mapping[str, Any], subject_grouping: str
) -> str:
    """Resolve a subject identifier for a materialized record.

    Args:
        sample_key: Sample key used as the default subject.
        metadata: Per-sample metadata mapping.
        subject_grouping: Grouping mode, either ``"sample"`` or ``"subject"``.

    Returns:
        Subject identifier string. In subject mode, known metadata keys are
        preferred and the sample key is used as a fallback.
    """

    if subject_grouping == "sample":
        return sample_key
    for key in ("subject_id", "subject", "subject_key"):
        raw_value = metadata.get(key)
        if raw_value is None:
            continue
        return _required_text(str(raw_value), f"metadata[{key!r}]")
    return sample_key


def _validate_subject_grouping(subject_grouping: str) -> str:
    """Validate the subject grouping mode.

    Args:
        subject_grouping: Requested grouping mode.

    Returns:
        Validated grouping mode.

    Raises:
        ValueError: If the value is empty or not ``"sample"`` or ``"subject"``.
    """

    value = _required_text(subject_grouping, "subject_grouping")
    if value not in {"sample", "subject"}:
        raise ValueError("subject_grouping must be 'sample' or 'subject'.")
    return value


def _seg_storage_from_attrs(attrs: Mapping[str, Any], *, context: str) -> str | None:
    """Read the optional declared label storage kind from ``_attrs``.

    Args:
        attrs: ThunderDB ``_attrs`` mapping.
        context: Human-readable location used in validation errors.

    Returns:
        ``"indexed_label_mask"``, ``"overlapping_channel_mask"``, or ``None``.

    Raises:
        ValueError: If ``seg_storage`` declares an unsupported value.
    """

    value = attrs.get("seg_storage")
    if value is None:
        return None
    if value not in _SEG_STORAGE_KINDS:
        raise ValueError(
            f"{context}['seg_storage'] must be one of {sorted(_SEG_STORAGE_KINDS)}; "
            f"got {value!r}."
        )
    return str(value)


def _mask_label_names_from_attrs(
    attrs: Mapping[str, Any], *, context: str
) -> tuple[str, ...]:
    """Read overlapping channel-mask names, enforcing channel index == native id.

    Args:
        attrs: ThunderDB ``_attrs`` mapping declaring ``mask_label_names``.
        context: Human-readable location used in validation errors.

    Returns:
        Ordered channel names.

    Raises:
        ValueError: If ``mask_label_ids`` is present and is not the identity
            range over the declared names.
    """

    names = _label_names_from_attrs(
        {"label_names": attrs["mask_label_names"]}, context=f"{context}.mask"
    )
    ids = attrs.get("mask_label_ids")
    if ids is not None and [int(i) for i in ids] != list(range(len(names))):
        raise ValueError(
            f"{context}['mask_label_ids'] must equal range(len(mask_label_names)) so "
            "that channel index equals native id."
        )
    return names


_SEG_STORAGE_KINDS = frozenset({"indexed_label_mask", "overlapping_channel_mask"})


def _label_names_from_attrs(
    attrs: Mapping[str, Any], *, context: str
) -> tuple[str, ...]:
    """Read and validate label names from an ``_attrs`` mapping.

    Args:
        attrs: ThunderDB ``_attrs`` mapping.
        context: Human-readable location used in validation errors.

    Returns:
        Tuple of non-empty unique label names.

    Raises:
        KeyError: If ``label_names`` is absent.
        TypeError: If ``label_names`` is not a list or tuple.
        ValueError: If names are empty, missing, or duplicated.
    """

    if "label_names" not in attrs:
        raise KeyError(f"{context} must define 'label_names'.")
    raw_value = attrs["label_names"]
    if not isinstance(raw_value, (list, tuple)):
        raise TypeError(f"{context}['label_names'] must be a list of strings.")
    label_names = tuple(
        _required_text(label_name, f"{context}['label_names'] entry")
        for label_name in raw_value
    )
    if not label_names:
        raise ValueError(f"{context}['label_names'] must not be empty.")
    if len(set(label_names)) != len(label_names):
        raise ValueError(f"{context}['label_names'] must not contain duplicates.")
    return label_names


def _required_text(raw_value: Any, name: str) -> str:
    """Validate and strip a required string value.

    Args:
        raw_value: Value to validate.
        name: Human-readable field name used in validation errors.

    Returns:
        Stripped string value.

    Raises:
        ValueError: If the value is not a non-empty string.
    """

    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return raw_value.strip()


def _coerce_record(raw_record: Mapping[str, Any] | DatasetRecord) -> DatasetRecord:
    """Normalize a record mapping or pass through an existing record.

    Args:
        raw_record: Dataset record object or mapping accepted by
            ``DatasetRecord.from_mapping``.

    Returns:
        Normalized dataset record.

    Raises:
        TypeError: If the mapping shape is invalid.
        ValueError: If required record fields are invalid.
    """

    if isinstance(raw_record, DatasetRecord):
        return raw_record
    return DatasetRecord.from_mapping(dict(raw_record))
