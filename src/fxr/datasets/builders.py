from __future__ import annotations

import os
import string
from collections.abc import Iterable, Mapping, Sized
from dataclasses import dataclass, field, replace
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

import yaml
from torch.utils.data import Dataset

from .runtime import CTTrainingDataset, XrayTrainingDataset
from .storage import SplitThunderDBStorageBackend, _open_thunderdb

BuilderModality = Literal["ct", "xray"]

_CONFIG_MODALITY_TO_RUNTIME = {
    "ct": "ct",
    "xray": "xray",
}
_RUNTIME_TO_CONFIG_MODALITY = {
    "ct": "CT",
    "xray": "Xray",
}
_CONFIG_MODALITY_KEYS = frozenset(_RUNTIME_TO_CONFIG_MODALITY.values()).union(
    _RUNTIME_TO_CONFIG_MODALITY
)
_ROOT_ENV_BY_MODALITY = {
    "ct": "CT_DATAPATH",
    "xray": "XRAY_DATAPATH",
}
_FORBIDDEN_CONFIG_KEYS = {"_class", "data_root", "root"}
_COMMON_RUNTIME_KEYS = {
    "exclude_subjects",
    "label_mode",
    "num_subjects",
    "require_seg",
    "return_data_id",
    "return_metadata",
    "return_native_label",
}
_CT_RUNTIME_KEYS = {
    "air_clamp_hu",
    "compute_fg_centroids",
    "crop_mode",
    "crop_size",
    "generator",
    "hu_min",
    "sample_weighting",
}
_XRAY_RUNTIME_KEYS: set[str] = set()


@dataclass(frozen=True)
class DatasetLayout:
    """Registered ThunderDB storage layout for one trainable dataset.

    Attributes:
        dataset_name: Public dataset key accepted by the builder.
        modality: Runtime modality served by this layout: ``"ct"`` or ``"xray"``.
        relative_path_template: Path template resolved under ``root_env_var``. The
            template may reference ``dataset_name`` and selector config fields
            such as ``version`` and ``resolution``.
        required_fields: Selector config keys that must be supplied before
            resolving the template.
        storage_kwargs: Keyword defaults passed to
            ``SplitThunderDBStorageBackend``.
        dataset_kwargs: Keyword defaults passed to the runtime dataset class.
        root_env_var: Environment variable naming the root directory that owns
            this layout. When omitted, it defaults from ``modality``.
    """

    dataset_name: str
    modality: BuilderModality
    relative_path_template: str
    required_fields: tuple[str, ...] = ()
    storage_kwargs: Mapping[str, Any] = field(default_factory=dict)
    dataset_kwargs: Mapping[str, Any] = field(default_factory=dict)
    root_env_var: str | None = None


@dataclass(frozen=True)
class TrainingDatasetBundle:
    """Training and optional validation datasets from a multimodal config.

    Attributes:
        train: Mapping from dataset name to the train-split dataset instance.
        val: Mapping from dataset name to validation-split datasets; empty when
            validation was not requested.
        modalities: Mapping from dataset name to runtime modality.
        _databases: Database readers owned by this bundle and shared by its
            train/validation dataset pairs.
        _closed: Whether owned readers have already been released.
        train_datasets: Alias for ``train`` for callers that prefer explicit
            field names.
        val_datasets: Alias for ``val`` for callers that prefer explicit field
            names.
        dataset_modalities: Alias for ``modalities`` for callers that prefer
            explicit field names.
        closed: Whether the bundle has released its owned readers.
    """

    train: dict[str, Dataset]
    val: dict[str, Dataset]
    modalities: dict[str, BuilderModality]
    _databases: tuple[Any, ...] = field(
        default_factory=tuple, repr=False, compare=False
    )
    _closed: bool = field(default=False, init=False, repr=False, compare=False)

    @property
    def closed(self) -> bool:
        """Return whether this bundle has released its owned database readers.

        Returns:
            ``True`` after the first call to :meth:`close`, otherwise ``False``.
        """

        return self._closed

    def close(self) -> None:
        """Close every owned database reader exactly once.

        Train and validation datasets for one source share a reader, so cleanup
        is performed from this ownership list rather than from both mappings.

        Returns:
            ``None``. Repeated calls are safe.
        """

        if self._closed:
            return
        object.__setattr__(self, "_closed", True)
        _close_databases(self._databases)

    def __enter__(self) -> "TrainingDatasetBundle":
        """Return this bundle for context-managed use.

        Returns:
            This open bundle.
        """

        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Close owned readers when leaving a context manager.

        Args:
            exc_type: Exception type raised in the context, if any.
            exc_value: Exception value raised in the context, if any.
            traceback: Traceback raised in the context, if any.

        Returns:
            ``None``.
        """

        del exc_type, exc_value, traceback
        self.close()

    @property
    def train_datasets(self) -> dict[str, Dataset]:
        """Return datasets built for the train split.

        Returns:
            Mapping from dataset name to train-split dataset instance.
        """

        return self.train

    @property
    def val_datasets(self) -> dict[str, Dataset]:
        """Return datasets built for the validation split.

        Returns:
            Mapping from dataset name to validation-split dataset instance.
        """

        return self.val

    @property
    def dataset_modalities(self) -> dict[str, BuilderModality]:
        """Return runtime modalities keyed by dataset name.

        Returns:
            Mapping from dataset name to runtime modality.
        """

        return self.modalities


_LAYOUTS: dict[str, DatasetLayout] = {}


def register_dataset_layout(layout: DatasetLayout, *, replace: bool = False) -> None:
    """Register a dataset layout for builder construction.

    Args:
        layout: Layout definition to add to the process-local registry.
        replace: Whether an existing layout with the same dataset name may be
            replaced.

    Returns:
        ``None``.

    Raises:
        TypeError: If ``layout`` is not a ``DatasetLayout``.
        ValueError: If the layout is invalid or a duplicate is registered
            without ``replace=True``.
    """

    if not isinstance(layout, DatasetLayout):
        raise TypeError("layout must be a DatasetLayout.")
    dataset_name = _required_text(layout.dataset_name, "layout.dataset_name")
    modality = _normalize_modality(layout.modality)
    root_env_var = _normalize_root_env_var(layout.root_env_var, modality)
    relative_path_template = _required_text(
        layout.relative_path_template,
        "layout.relative_path_template",
    )
    required_fields = tuple(
        _required_text(field_name, "layout.required_fields entry")
        for field_name in layout.required_fields
    )
    template_fields = _template_fields(relative_path_template)
    unknown_required = sorted(
        field_name
        for field_name in required_fields
        if field_name not in template_fields and field_name != "dataset_name"
    )
    if unknown_required:
        raise ValueError(
            f"Layout {dataset_name!r} requires fields not used by its path "
            f"template: {unknown_required}."
        )
    normalized = DatasetLayout(
        dataset_name=dataset_name,
        modality=modality,
        relative_path_template=relative_path_template,
        required_fields=required_fields,
        storage_kwargs=dict(layout.storage_kwargs),
        dataset_kwargs=dict(layout.dataset_kwargs),
        root_env_var=root_env_var,
    )
    if dataset_name in _LAYOUTS and not replace:
        raise ValueError(
            f"Dataset layout {dataset_name!r} is already registered; pass "
            "replace=True to replace it."
        )
    _LAYOUTS[dataset_name] = normalized


def load_dataset_layouts(
    source: str | Path | Mapping[str, Any] | Any,
    *,
    replace: bool = False,
) -> tuple[DatasetLayout, ...]:
    """Load and register dataset layouts from a YAML layout config.

    Args:
        source: YAML path, package resource with ``read_text()``, or in-memory
            mapping with a top-level ``layouts`` list.
        replace: Whether loaded layouts may replace existing registered names.

    Returns:
        Tuple of normalized layouts that were registered.

    Raises:
        TypeError: If the config shape or nested fields are invalid.
        ValueError: If unknown keys, duplicate names, or invalid layouts are
            present.
    """

    data = _load_layout_config(source)
    _reject_unknown_keys(data, {"layouts"}, context="dataset layout config")
    raw_layouts = data.get("layouts")
    if not isinstance(raw_layouts, list):
        raise TypeError("dataset layout config must define a layouts list.")
    layouts: list[DatasetLayout] = []
    seen: set[str] = set()
    for idx, raw_layout in enumerate(raw_layouts):
        layout = _dataset_layout_from_mapping(
            raw_layout,
            context=f"dataset layout config layouts[{idx}]",
        )
        if layout.dataset_name in seen:
            raise ValueError(
                "dataset layout config repeats dataset layout "
                f"{layout.dataset_name!r}."
            )
        seen.add(layout.dataset_name)
        layouts.append(layout)
    normalized_layouts: list[DatasetLayout] = []
    for layout in layouts:
        register_dataset_layout(layout, replace=replace)
        normalized_layouts.append(_LAYOUTS[layout.dataset_name])
    return tuple(normalized_layouts)


def build_training_dataset(
    dataset_name: str,
    split: str,
    modality: str,
    cfg: Mapping[str, Any] | None = None,
) -> Dataset:
    """Build one native-label dataset from a registered layout or package path.

    Args:
        dataset_name: Registered layout name or explicit package identity.
        split: Split name to read from the ThunderDB ``_splits`` mapping.
        modality: Config or runtime modality name: ``"CT"``, ``"Xray"``,
            ``"ct"``, or ``"xray"``.
        cfg: Dataset config mapping. ``path`` may name an absolute packaged
            ThunderDB directly; otherwise registered selector fields and modality
            root environment variables resolve the storage path.

    Returns:
        A ``CTTrainingDataset`` or ``XrayTrainingDataset`` with
        ``label_mode="native"``.

    Raises:
        KeyError: If the modality-specific root env var is not set, the layout is
            unknown, or the layout modality does not match ``modality``.
        TypeError: If ``cfg`` is not mapping-like.
        ValueError: If required fields are missing, an explicit package path is relative,
            unsupported config keys are supplied, or model label mode is
            requested.
    """

    layout, config, db_path = _resolve_dataset_build(
        dataset_name,
        modality,
        cfg,
    )
    return _build_dataset_from_layout(layout, split=split, cfg=config, db_path=db_path)


def _resolve_dataset_build(
    dataset_name: str,
    modality: str,
    cfg: Mapping[str, Any] | None,
) -> tuple[DatasetLayout, dict[str, Any], Path]:
    """Resolve one dataset config into a layout, runtime config, and path.

    Args:
        dataset_name: Registered layout name or explicit package identity.
        modality: Config or runtime modality requested by the caller.
        cfg: Optional per-dataset builder config.

    Returns:
        Tuple containing the resolved layout, path-free runtime config, and
        absolute or root-relative ThunderDB path.

    Raises:
        KeyError: If a registered layout has a different modality.
        TypeError: If the per-dataset config is not mapping-like.
        ValueError: If a direct package path or runtime config is invalid.
    """

    name = _required_text(dataset_name, "dataset_name")
    requested_modality = _normalize_modality(modality)
    config = _normalize_dataset_cfg(cfg, dataset_name=name)
    explicit_path = config.pop("path", None)
    spec_path = config.pop("dataset_spec", None)
    if spec_path is not None:
        assert explicit_path is not None, (
            f"data[{name!r}].dataset_spec requires an explicit package path."
        )
        assert Path(spec_path).expanduser().is_absolute(), (
            f"data[{name!r}].dataset_spec must be an absolute path; got {spec_path!r}."
        )
    if explicit_path is None:
        layout = _layout_for_name(name)
        db_path = _resolve_layout_path(layout, config)
    else:
        registered_layout = _LAYOUTS.get(name)
        if registered_layout is None:
            layout = DatasetLayout(
                dataset_name=name,
                modality=requested_modality,
                relative_path_template="{dataset_name}",
                storage_kwargs={"subject_grouping": "subject"},
            )
        else:
            layout = replace(
                registered_layout,
                storage_kwargs={
                    **registered_layout.storage_kwargs,
                    "subject_grouping": "subject",
                },
            )
        db_path = _absolute_dataset_path(explicit_path, dataset_name=name)
    if layout.modality != requested_modality:
        raise KeyError(
            f"Dataset layout {name!r} is registered for modality "
            f"{_RUNTIME_TO_CONFIG_MODALITY[layout.modality]!r}, not "
            f"{_RUNTIME_TO_CONFIG_MODALITY[requested_modality]!r}."
        )
    _dataset_kwargs(layout, config)
    return layout, config, db_path


def _absolute_dataset_path(raw_path: Any, *, dataset_name: str) -> Path:
    """Validate one explicit packaged-dataset path.

    Args:
        raw_path: Path value supplied by a per-dataset config.
        dataset_name: Dataset name used in validation messages.

    Returns:
        Expanded absolute filesystem path.

    Raises:
        TypeError: If the path is not a string or ``Path``.
        ValueError: If the path is relative.
    """

    if not isinstance(raw_path, (str, Path)):
        raise TypeError(f"data[{dataset_name!r}].path must be a path string.")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise ValueError(
            f"data[{dataset_name!r}].path must be absolute; got {str(path)!r}."
        )
    return path


def build_named_datasets(
    data_cfg: Mapping[str, Any],
    split: str,
    modality: str,
) -> dict[str, Dataset]:
    """Build all datasets for one modality from a FleXray data config.

    Args:
        data_cfg: Either the full config containing ``data:``, the ``data``
            mapping itself, or a mapping from dataset name to dataset config for
            the selected modality.
        split: Split name to read from each ThunderDB layout.
        modality: Config or runtime modality name to select.

    Returns:
        Mapping from dataset name to runtime dataset instance.

    Raises:
        TypeError: If the selected modality config is not a mapping.
        ValueError: If private root overrides or malformed dataset entries are present.
    """

    selected = _select_named_configs(data_cfg, modality)
    datasets: dict[str, Dataset] = {}
    try:
        for dataset_name, dataset_cfg in selected.items():
            name = _required_text(dataset_name, "dataset config key")
            datasets[name] = build_training_dataset(
                name,
                split=split,
                modality=modality,
                cfg=_require_mapping_or_none(
                    dataset_cfg, context=f"data[{name!r}]"
                ),
            )
    except BaseException:
        _close_datasets(datasets.values(), suppress_errors=True)
        raise
    return datasets


def build_multimodal_datasets(
    data_cfg: Mapping[str, Any],
    *,
    train_split: str = "train",
    val_split: str = "val",
    include_validation: bool = True,
) -> TrainingDatasetBundle:
    """Build training and optional validation datasets for each modality.

    Args:
        data_cfg: Full FleXray config containing ``data:`` or the ``data``
            mapping itself.
        train_split: Split name used for ``TrainingDatasetBundle.train``.
        val_split: Split name used for ``TrainingDatasetBundle.val`` when
            ``include_validation`` is true.
        include_validation: Whether to require and construct validation datasets.
            Set this false only for a run whose core validation is disabled.

    Returns:
        Bundle containing train datasets, validation datasets, and per-dataset
        runtime modalities.

    Raises:
        TypeError: If ``data_cfg`` or a modality section is not mapping-like.
        ValueError: If a dataset name is repeated across modalities.
    """

    if not isinstance(include_validation, bool):
        raise TypeError("include_validation must be a bool.")
    data = _extract_data_section(data_cfg)
    _reject_forbidden_config_keys(data, context="data")
    sections: list[tuple[BuilderModality, Mapping[str, Any]]] = []
    seen_modalities: dict[str, str] = {}
    for modality_key, raw_named_cfgs in data.items():
        modality = _normalize_modality(modality_key)
        named_cfgs = _require_mapping_or_none(
            raw_named_cfgs,
            context=f"data[{modality_key!r}]",
        )
        if named_cfgs is None:
            continue
        for dataset_name in named_cfgs:
            name = _required_text(dataset_name, "dataset config key")
            previous = seen_modalities.get(name)
            if previous is not None:
                raise ValueError(
                    f"Dataset {name!r} is configured under both {previous!r} and "
                    f"{_RUNTIME_TO_CONFIG_MODALITY[modality]!r}; dataset names "
                    "must be unique across modalities."
                )
            seen_modalities[name] = _RUNTIME_TO_CONFIG_MODALITY[modality]
        sections.append((modality, named_cfgs))

    train: dict[str, Dataset] = {}
    val: dict[str, Dataset] = {}
    modalities: dict[str, BuilderModality] = {}
    databases: list[Any] = []
    try:
        for modality, named_cfgs in sections:
            for raw_dataset_name, raw_dataset_cfg in named_cfgs.items():
                dataset_name = _required_text(
                    raw_dataset_name, "dataset config key"
                )
                dataset_cfg = _require_mapping_or_none(
                    raw_dataset_cfg,
                    context=f"data[{dataset_name!r}]",
                )
                layout, config, db_path = _resolve_dataset_build(
                    dataset_name,
                    modality,
                    dataset_cfg,
                )
                database = _open_database(db_path)
                databases.append(database)
                _require_training_splits(
                    database,
                    dataset_name=dataset_name,
                    train_split=train_split,
                    val_split=val_split if include_validation else None,
                )
                train_dataset = _build_dataset_from_layout(
                    layout,
                    split=train_split,
                    cfg=config,
                    db_path=database,
                )
                train[dataset_name] = train_dataset
                if include_validation:
                    val[dataset_name] = _build_dataset_from_layout(
                        layout,
                        split=val_split,
                        cfg=config,
                        db_path=database,
                    )
                modalities[dataset_name] = modality
    except BaseException:
        _close_databases(databases, suppress_errors=True)
        raise
    return TrainingDatasetBundle(
        train=train,
        val=val,
        modalities=modalities,
        _databases=tuple(databases),
    )


def _require_training_splits(
    database: Any,
    *,
    dataset_name: str,
    train_split: str,
    val_split: str | None,
) -> None:
    """Require the non-empty splits consumed by the experiment.

    Args:
        database: Open ThunderDB-like mapping shared by runtime datasets.
        dataset_name: Dataset identity used in validation errors.
        train_split: Required training split name.
        val_split: Required validation split name, or ``None`` when core
            validation is disabled.

    Returns:
        ``None`` when every requested split is present and non-empty.

    Raises:
        KeyError: If ``_splits`` or a requested split is absent.
        TypeError: If ``_splits`` is not mapping-like or a requested split is
            not a sized iterable.
        ValueError: If a requested split contains no samples.
    """

    try:
        splits = database["_splits"]
    except (KeyError, LookupError, TypeError, AttributeError) as exc:
        raise KeyError(
            f"Training dataset {dataset_name!r} must define a '_splits' mapping."
        ) from exc
    if not isinstance(splits, Mapping):
        raise TypeError(
            f"Training dataset {dataset_name!r} '_splits' entry must be a mapping."
        )
    requested = [_required_text(train_split, "train_split")]
    if val_split is not None:
        requested.append(_required_text(val_split, "val_split"))
    required_splits = tuple(dict.fromkeys(requested))
    missing = [
        split_name for split_name in required_splits if split_name not in splits
    ]
    if missing:
        raise KeyError(
            f"Training dataset {dataset_name!r} is missing required split(s) "
            f"{missing}."
        )
    for split_name in required_splits:
        entries = splits[split_name]
        if (
            isinstance(entries, (str, bytes))
            or not isinstance(entries, Iterable)
            or not isinstance(entries, Sized)
        ):
            raise TypeError(
                f"Training dataset {dataset_name!r} split {split_name!r} must "
                "be a sized iterable of sample entries."
            )
        if len(entries) == 0:
            raise ValueError(
                f"Training dataset {dataset_name!r} split {split_name!r} must "
                "contain at least one sample."
            )


def _build_dataset_from_layout(
    layout: DatasetLayout,
    *,
    split: str,
    cfg: dict[str, Any],
    db_path: Any,
) -> Dataset:
    """Build a runtime dataset once layout validation has completed.

    Args:
        layout: Registered layout to instantiate.
        split: Split name to read from storage.
        cfg: Normalized dataset config mapping.
        db_path: Resolved ThunderDB path or an already-open database object.

    Returns:
        Runtime training dataset instance for the layout modality.
    """

    split_name = _required_text(split, "split")
    dataset_kwargs = _dataset_kwargs(layout, cfg)
    storage_kwargs = dict(layout.storage_kwargs)
    owns_database = isinstance(db_path, (str, Path))
    db = _open_database(db_path)
    try:
        if layout.modality == "ct":
            backend = SplitThunderDBStorageBackend(
                db,
                dataset_name=layout.dataset_name,
                modality="ct",
                split=split_name,
                owns_database=False,
                **storage_kwargs,
            )
            backend._owns_database = owns_database
            return CTTrainingDataset(
                backend=backend,
                dataset_name=layout.dataset_name,
                split=split_name,
                **dataset_kwargs,
            )
        backend = SplitThunderDBStorageBackend(
            db,
            dataset_name=layout.dataset_name,
            modality="xray",
            split=split_name,
            owns_database=False,
            **storage_kwargs,
        )
        backend._owns_database = owns_database
        return XrayTrainingDataset(
            backend=backend,
            dataset_name=layout.dataset_name,
            split=split_name,
            **dataset_kwargs,
        )
    except BaseException:
        if owns_database:
            _close_database(db)
        raise


def _dataset_kwargs(layout: DatasetLayout, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Select runtime dataset constructor keywords from a builder config.

    Args:
        layout: Layout whose modality controls the allowed runtime keys.
        cfg: Normalized dataset config mapping.

    Returns:
        Keyword arguments for the runtime dataset class.

    Raises:
        ValueError: If unsupported config keys are supplied.
    """

    allowed_runtime_keys = set(_COMMON_RUNTIME_KEYS)
    if layout.modality == "ct":
        allowed_runtime_keys.update(_CT_RUNTIME_KEYS)
    else:
        allowed_runtime_keys.discard("require_seg")
        allowed_runtime_keys.update(_XRAY_RUNTIME_KEYS)
    layout_fields = _template_fields(layout.relative_path_template).union(
        layout.required_fields,
        {"dataset_name"},
    )
    allowed_keys = allowed_runtime_keys.union(layout_fields)
    unknown = sorted(key for key in cfg if key not in allowed_keys)
    if unknown:
        raise ValueError(
            f"Dataset {layout.dataset_name!r} builder config has unsupported keys: "
            f"{unknown}."
        )
    label_mode = cfg.get("label_mode", "native")
    if label_mode != "native":
        raise ValueError(
            "FleXray dataset builders currently support label_mode='native' "
            f"only; got {label_mode!r} for dataset {layout.dataset_name!r}."
        )
    dataset_kwargs = dict(layout.dataset_kwargs)
    dataset_kwargs["label_mode"] = "native"
    for key in sorted(allowed_runtime_keys):
        if key in cfg and key != "label_mode":
            dataset_kwargs[key] = cfg[key]
    return dataset_kwargs


def _resolve_layout_path(layout: DatasetLayout, cfg: Mapping[str, Any]) -> Path:
    """Resolve a layout path below its required modality root.

    Args:
        layout: Layout whose template should be resolved.
        cfg: Normalized dataset config mapping.

    Returns:
        Absolute ThunderDB path under the root named by ``layout.root_env_var``.

    Raises:
        KeyError: If the required root environment variable is not set.
        ValueError: If required fields are missing or empty.
    """

    root_env_var = _required_text(layout.root_env_var, "layout.root_env_var")
    root_value = os.environ.get(root_env_var)
    if root_value is None or not root_value.strip():
        raise KeyError(
            f"{root_env_var} must be set to the root directory for "
            f"{layout.modality} FleXray training datasets."
        )
    if os.pathsep in root_value:
        raise ValueError(
            f"{root_env_var} must name one dataset root directory, not a path-list."
        )
    root = Path(root_value)
    values = {"dataset_name": layout.dataset_name, **cfg}
    missing = [
        field_name
        for field_name in layout.required_fields
        if _missing_config_value(values.get(field_name))
    ]
    if missing:
        raise ValueError(
            f"Dataset {layout.dataset_name!r} requires builder config field(s): "
            f"{missing}."
        )
    relative_path = layout.relative_path_template.format(**values)
    if Path(relative_path).is_absolute():
        raise ValueError(
            f"Dataset layout {layout.dataset_name!r} resolved an absolute path."
        )
    return _path_below_root(
        root,
        relative_path,
        context=f"Dataset layout {layout.dataset_name!r}",
    )


def _path_below_root(root: Path, relative_path: str, *, context: str) -> Path:
    """Resolve one relative path while confining it below ``root``.

    Args:
        root: Directory that must contain the resolved path.
        relative_path: Relative path produced by a registered layout.
        context: Description used in validation errors.

    Returns:
        Absolute, normalized path contained by ``root``.

    Raises:
        ValueError: If the path is absolute or escapes through ``..`` or a
            symbolic-link prefix.
    """

    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"{context} resolved an absolute path.")
    resolved_root = root.expanduser().resolve()
    resolved_path = (resolved_root / relative).resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"{context} resolved outside its dataset root.")
    return resolved_path


def _select_named_configs(
    data_cfg: Mapping[str, Any],
    modality: str,
) -> Mapping[str, Any]:
    """Select one modality section from any supported data-config shape.

    Args:
        data_cfg: Full config, data section, or selected modality mapping.
        modality: Requested modality name.

    Returns:
        Mapping from dataset name to dataset config.
    """

    data = _extract_data_section(data_cfg)
    _reject_forbidden_config_keys(data, context="data")
    runtime_modality = _normalize_modality(modality)
    preferred_key = _RUNTIME_TO_CONFIG_MODALITY[runtime_modality]
    lower_key = runtime_modality
    if preferred_key in data:
        selected = data[preferred_key]
    elif lower_key in data:
        selected = data[lower_key]
    elif any(str(key) in _CONFIG_MODALITY_KEYS for key in data):
        return {}
    else:
        selected = data
    selected_mapping = _require_mapping_or_none(
        selected,
        context=f"data[{preferred_key!r}]",
    )
    if selected_mapping is None:
        return {}
    _reject_forbidden_config_keys(selected_mapping, context=f"data[{preferred_key!r}]")
    return selected_mapping


def _extract_data_section(data_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """Normalize a full config or data section into a data mapping.

    Args:
        data_cfg: Full config containing ``data`` or the data section itself.

    Returns:
        Data mapping.

    Raises:
        TypeError: If ``data_cfg`` or ``data`` is not mapping-like.
    """

    if not isinstance(data_cfg, Mapping):
        raise TypeError("data_cfg must be a mapping.")
    if "data" not in data_cfg:
        return data_cfg
    data = data_cfg["data"]
    if not isinstance(data, Mapping):
        raise TypeError("data_cfg['data'] must be a mapping.")
    return data


def _normalize_dataset_cfg(
    cfg: Mapping[str, Any] | None,
    *,
    dataset_name: str,
) -> dict[str, Any]:
    """Validate and copy a per-dataset builder config.

    Args:
        cfg: Raw per-dataset config or ``None``.
        dataset_name: Dataset name used in validation errors.

    Returns:
        Shallow config copy.
    """

    if cfg is None:
        config: dict[str, Any] = {}
    elif isinstance(cfg, Mapping):
        config = dict(cfg)
    else:
        raise TypeError(f"data[{dataset_name!r}] must be a mapping or null.")
    _reject_forbidden_config_keys(config, context=f"data[{dataset_name!r}]")
    return config


def _load_layout_config(source: str | Path | Mapping[str, Any] | Any) -> dict[str, Any]:
    """Load a YAML layout config from a path, resource, or mapping.

    Args:
        source: YAML path, package resource, or mapping.

    Returns:
        Shallow config mapping.

    Raises:
        TypeError: If the loaded body is not a mapping.
    """

    if isinstance(source, Mapping):
        data = dict(source)
    elif isinstance(source, (str, Path)):
        with open(source, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    elif hasattr(source, "read_text"):
        data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    else:
        raise TypeError("layout source must be a path, package resource, or mapping.")
    if not isinstance(data, dict):
        raise TypeError("dataset layout config must be a mapping.")
    return data


def _dataset_layout_from_mapping(raw_layout: Any, *, context: str) -> DatasetLayout:
    """Build one ``DatasetLayout`` from YAML mapping data.

    Args:
        raw_layout: Raw layout mapping.
        context: Human-readable config location used in errors.

    Returns:
        Normalized dataset layout.

    Raises:
        TypeError: If nested fields have incompatible types.
        ValueError: If required fields are missing or unknown keys are present.
    """

    if not isinstance(raw_layout, Mapping):
        raise TypeError(f"{context} must be a mapping.")
    data = dict(raw_layout)
    _reject_unknown_keys(
        data,
        {
            "dataset_kwargs",
            "dataset_name",
            "modality",
            "relative_path_template",
            "required_fields",
            "root_env_var",
            "storage_kwargs",
        },
        context=context,
    )
    return DatasetLayout(
        dataset_name=_required_text(
            data.get("dataset_name"), f"{context}.dataset_name"
        ),
        modality=_normalize_modality(data.get("modality")),
        relative_path_template=_required_text(
            data.get("relative_path_template"),
            f"{context}.relative_path_template",
        ),
        required_fields=_string_tuple(
            data.get("required_fields", ()),
            context=f"{context}.required_fields",
        ),
        storage_kwargs=_optional_mapping(
            data.get("storage_kwargs", {}),
            context=f"{context}.storage_kwargs",
        ),
        dataset_kwargs=_optional_mapping(
            data.get("dataset_kwargs", {}),
            context=f"{context}.dataset_kwargs",
        ),
        root_env_var=_optional_text(
            data.get("root_env_var"),
            context=f"{context}.root_env_var",
        ),
    )


def _optional_text(raw_value: Any, *, context: str) -> str | None:
    """Load an optional non-empty string field.

    Args:
        raw_value: Value to validate.
        context: Human-readable config location used in errors.

    Returns:
        Stripped string, or ``None`` when omitted.
    """

    if raw_value is None:
        return None
    return _required_text(raw_value, context)


def _string_tuple(raw_value: Any, *, context: str) -> tuple[str, ...]:
    """Load a YAML list of non-empty strings as a tuple.

    Args:
        raw_value: Value to validate.
        context: Human-readable config location used in errors.

    Returns:
        Tuple of stripped strings.

    Raises:
        TypeError: If ``raw_value`` is not a sequence of strings.
        ValueError: If any item is empty or duplicated.
    """

    if not isinstance(raw_value, (list, tuple)):
        raise TypeError(f"{context} must be a list of strings.")
    values = tuple(_required_text(value, f"{context} entry") for value in raw_value)
    if len(set(values)) != len(values):
        raise ValueError(f"{context} cannot contain duplicates.")
    return values


def _optional_mapping(raw_value: Any, *, context: str) -> dict[str, Any]:
    """Load an optional YAML mapping as a shallow dictionary.

    Args:
        raw_value: Value to validate.
        context: Human-readable config location used in errors.

    Returns:
        Shallow dictionary.

    Raises:
        TypeError: If ``raw_value`` is not mapping-like.
    """

    if not isinstance(raw_value, Mapping):
        raise TypeError(f"{context} must be a mapping.")
    return dict(raw_value)


def _reject_unknown_keys(
    data: Mapping[str, Any],
    allowed_keys: set[str],
    *,
    context: str,
) -> None:
    """Reject unknown YAML keys in a parsed layout config mapping.

    Args:
        data: Mapping to validate.
        allowed_keys: Accepted key set.
        context: Human-readable config location used in errors.

    Returns:
        ``None``.

    Raises:
        ValueError: If any key is unsupported.
    """

    unknown = sorted(str(key) for key in data if str(key) not in allowed_keys)
    if unknown:
        raise ValueError(f"{context} has unknown keys: {unknown}.")


def _reject_forbidden_config_keys(cfg: Mapping[str, Any], *, context: str) -> None:
    """Reject legacy class and implicit root override keys in a config mapping.

    Args:
        cfg: Config mapping to validate.
        context: Human-readable location used in the error message.

    Returns:
        ``None``.

    Raises:
        ValueError: If a forbidden key is present.
    """

    forbidden = sorted(str(key) for key in cfg if str(key) in _FORBIDDEN_CONFIG_KEYS)
    if forbidden:
        raise ValueError(
            f"{context} cannot define {forbidden}; use a registered dataset root "
            "or a per-dataset absolute path."
        )


def _require_mapping_or_none(
    raw_value: Any, *, context: str
) -> Mapping[str, Any] | None:
    """Validate an optional mapping value from a data config.

    Args:
        raw_value: Value to validate.
        context: Human-readable config location used in errors.

    Returns:
        Mapping value or ``None``.

    Raises:
        TypeError: If the value is neither ``None`` nor mapping-like.
    """

    if raw_value is None:
        return None
    if not isinstance(raw_value, Mapping):
        raise TypeError(f"{context} must be a mapping or null.")
    return raw_value


def _layout_for_name(dataset_name: str) -> DatasetLayout:
    """Return a registered layout or raise with registration guidance.

    Args:
        dataset_name: Dataset name to look up.

    Returns:
        Registered layout.

    Raises:
        KeyError: If no layout is registered for ``dataset_name``.
    """

    layout = _LAYOUTS.get(dataset_name)
    if layout is None:
        known = ", ".join(sorted(_LAYOUTS))
        raise KeyError(
            f"Unknown FleXray dataset layout {dataset_name!r}. Known layouts: "
            f"{known}. Register custom layouts at process startup with "
            "register_dataset_layout(...)."
        )
    return layout


def _normalize_root_env_var(
    root_env_var: str | None, modality: BuilderModality
) -> str:
    """Return the layout root environment variable for a modality.

    Args:
        root_env_var: Explicit root environment variable from a layout, or
            ``None`` to use the modality default.
        modality: Normalized runtime modality.

    Returns:
        Non-empty environment variable name.
    """

    if root_env_var is None:
        return _ROOT_ENV_BY_MODALITY[modality]
    return _required_text(root_env_var, "layout.root_env_var")


def _normalize_modality(modality: str) -> BuilderModality:
    """Normalize a config or runtime modality string.

    Args:
        modality: Modality value to normalize.

    Returns:
        Runtime modality string.

    Raises:
        ValueError: If the modality is unsupported.
    """

    key = _required_text(modality, "modality").lower()
    if key not in _CONFIG_MODALITY_TO_RUNTIME:
        raise ValueError(
            "modality must be one of 'CT', 'Xray', 'ct', or 'xray'."
        )
    return _CONFIG_MODALITY_TO_RUNTIME[key]  # type: ignore[return-value]


def _required_text(raw_value: Any, context: str) -> str:
    """Validate a required non-empty string value.

    Args:
        raw_value: Value to validate.
        context: Human-readable field name used in validation errors.

    Returns:
        Stripped string.

    Raises:
        ValueError: If the value is missing or empty.
    """

    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    return raw_value.strip()


def _missing_config_value(raw_value: Any) -> bool:
    """Return whether a required config value is missing or empty.

    Args:
        raw_value: Value to inspect.

    Returns:
        ``True`` when the value should fail required-field validation.
    """

    return raw_value is None or (isinstance(raw_value, str) and not raw_value.strip())


def _template_fields(template: str) -> set[str]:
    """Return named replacement fields used by a format-string template.

    Args:
        template: Format string to inspect.

    Returns:
        Set of top-level replacement field names.
    """

    fields: set[str] = set()
    for _, field_name, _, _ in string.Formatter().parse(template):
        if field_name:
            fields.add(field_name.split(".", 1)[0].split("[", 1)[0])
    return fields


def _open_database(path: Any) -> Any:
    """Open a database path or pass through an opened database object.

    Args:
        path: Resolved ThunderDB path or an already-open database object.

    Returns:
        Opened ThunderDB-like object.
    """

    return _open_thunderdb(path)


def _close_database(database: Any) -> None:
    """Close a database object when it exposes an ownership hook.

    Args:
        database: Open database-like object.

    Returns:
        ``None``.
    """

    close = getattr(database, "close", None)
    if callable(close):
        close()


def _close_datasets(
    datasets: Iterable[Dataset],
    *,
    suppress_errors: bool = False,
) -> None:
    """Close built datasets while attempting every cleanup.

    Args:
        datasets: Runtime datasets whose storage backends may own readers.
        suppress_errors: Whether close failures should be ignored after all
            datasets have been attempted.

    Returns:
        ``None``.

    Raises:
        BaseException: The first close failure when ``suppress_errors=False``.
    """

    first_error: BaseException | None = None
    for dataset in reversed(tuple(datasets)):
        close = getattr(dataset, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None and not suppress_errors:
        raise first_error


def _close_databases(
    databases: Iterable[Any],
    *,
    suppress_errors: bool = False,
) -> None:
    """Close unique database objects while attempting every cleanup.

    Args:
        databases: Database objects to close. Repeated object identities are
            closed once.
        suppress_errors: Whether close failures should be ignored after all
            database objects have been attempted.

    Returns:
        ``None``.

    Raises:
        BaseException: The first close failure when ``suppress_errors=False``.
    """

    first_error: BaseException | None = None
    seen: set[int] = set()
    for database in reversed(tuple(databases)):
        identity = id(database)
        if identity in seen:
            continue
        seen.add(identity)
        try:
            _close_database(database)
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None and not suppress_errors:
        raise first_error


def _register_builtin_layouts() -> None:
    """Populate the process-local registry from packaged YAML layouts.

    Returns:
        ``None``.
    """

    layout_resource = files("fxr.configs").joinpath("dataset_layouts", "training.yml")
    load_dataset_layouts(
        layout_resource,
        replace=True,
    )


_register_builtin_layouts()
