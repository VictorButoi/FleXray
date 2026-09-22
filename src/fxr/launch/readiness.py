"""Training launch readiness validation.

The checks here run on resolved training config dictionaries before a local run
or cluster dry run. They intentionally avoid constructing a full experiment.
Direct package metadata is opened to validate its label contract; sample arrays
are loaded only when ``smoke_data=True``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch

from fxr.config import check_missing, prune_zero_proportion_datasets
from fxr.config._proportions import normalize_proportion_weight
from fxr.config.training import requires_training_source_validation
from fxr.datasets import (
    build_training_dataset,
    compile_training_label_remap_by_name,
    compile_package_label_remap,
)
from fxr.datasets.builders import (
    _absolute_dataset_path,
    _layout_for_name,
    _path_below_root,
    _require_training_splits,
)
from fxr.datasets.remap import normalize_training_channel_label_names
from fxr.experiment._loader_config import loader_options
from fxr.models.camera import resolve_ct_profiles

_DRR_RANDOM_LABEL = "random_label"
_MISSING_PLACEHOLDER = "?"
_MODALITY_KEYS = (("CT", "ct"), ("Xray", "xray"))
_ROOT_ENV_VARS = ("CT_DATAPATH", "XRAY_DATAPATH", "GENERATED_DATAPATH")


def validate_training_config(
    config: Mapping[str, Any],
    *,
    smoke_data: bool = False,
    ct_data_root: str | Path | None = None,
    xray_data_root: str | Path | None = None,
    generated_data_root: str | Path | None = None,
) -> None:
    """Validate that a resolved training config is ready to launch.

    Args:
        config: Resolved experiment config mapping.
        smoke_data: Whether to open one sample from each active dataset.
        ct_data_root: Optional CT dataset root to use instead of ``CT_DATAPATH``.
        xray_data_root: Optional X-ray dataset root to use instead of
            ``XRAY_DATAPATH``.
        generated_data_root: Optional generated dataset root to use instead of
            ``GENERATED_DATAPATH``.

    Returns:
        ``None`` when the config passes readiness checks.

    Raises:
        ValueError: If one or more readiness checks fail.
    """

    cfg = prune_zero_proportion_datasets(_plain(config))
    issues: list[str] = []
    _append_check_missing_issue(cfg, issues)
    _validate_runtime_scalars(cfg, issues)
    active = _active_dataset_configs(cfg)
    callback_eval_sets = _callback_eval_dataset_configs(cfg, issues)
    if not active and not _requires_training_data(cfg):
        if issues:
            joined = "\n- ".join(issues)
            raise ValueError(f"Training readiness validation failed:\n- {joined}")
        return
    data_roots = _data_root_overrides(
        ct_data_root=ct_data_root,
        xray_data_root=xray_data_root,
        generated_data_root=generated_data_root,
    )
    _validate_dataloader(cfg, active, issues)
    include_validation = requires_training_source_validation(cfg)
    core_splits = ("train", "val") if include_validation else ("train",)
    _validate_protocol_remaps(cfg, active, issues, required_splits=core_splits)
    profiles = _validate_drr_profiles(cfg, active, issues)
    _validate_routed_loss(cfg, active, issues)
    _validate_dataset_paths(active, issues, data_roots=data_roots)
    for callback_active, callback_split in callback_eval_sets:
        _validate_protocol_remaps(
            cfg,
            callback_active,
            issues,
            required_splits=(callback_split,),
        )
        _validate_dataset_paths(callback_active, issues, data_roots=data_roots)
        if smoke_data:
            _validate_smoke_data(
                callback_active,
                issues,
                profiles=None,
                data_roots=data_roots,
                split=callback_split,
            )
    if smoke_data:
        _validate_smoke_data(
            active,
            issues,
            profiles=profiles,
            data_roots=data_roots,
            split="train",
        )
    if issues:
        joined = "\n- ".join(issues)
        raise ValueError(f"Training readiness validation failed:\n- {joined}")


def _plain(value: Any) -> Any:
    """Recursively convert config-like values into plain Python containers."""
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_plain(child) for child in value]
    return value


def _append_check_missing_issue(config: Mapping[str, Any], issues: list[str]) -> None:
    """Append any unresolved placeholder issue to ``issues``."""
    try:
        check_missing(config)
    except ValueError as exc:
        issues.append(str(exc))


def _validate_runtime_scalars(
    config: Mapping[str, Any], issues: list[str]
) -> None:
    """Validate scalar values consumed only when the training loop starts.

    Args:
        config: Resolved training config.
        issues: Aggregate readiness issue list.

    Returns:
        ``None``.
    """

    if not _requires_training_data(config):
        return
    train_cfg = config.get("train")
    if not isinstance(train_cfg, Mapping):
        issues.append("train must be a mapping for FleXrayTrainExperiment.")
    else:
        if "epochs" not in train_cfg:
            issues.append("train.epochs is required for FleXrayTrainExperiment.")
        else:
            try:
                _positive_integer(train_cfg["epochs"], "train.epochs")
            except ValueError as exc:
                issues.append(str(exc))
        if "eval_freq" in train_cfg:
            try:
                _integer_value(train_cfg["eval_freq"], "train.eval_freq")
            except ValueError as exc:
                issues.append(str(exc))

    log_cfg = config.get("log")
    if not isinstance(log_cfg, Mapping):
        return
    model_weights = log_cfg.get("model_weights")
    if isinstance(model_weights, Mapping) and "save_freq" in model_weights:
        try:
            _integer_value(
                model_weights["save_freq"], "log.model_weights.save_freq"
            )
        except ValueError as exc:
            issues.append(str(exc))


def _active_dataset_configs(
    config: Mapping[str, Any],
) -> dict[str, tuple[str, dict[str, Any]]]:
    """Return active dataset configs keyed by dataset name.

    Args:
        config: Pruned training config mapping.

    Returns:
        Mapping from dataset name to ``(runtime_modality, dataset_config)``.
    """

    data = config.get("data")
    active: dict[str, tuple[str, dict[str, Any]]] = {}
    if not isinstance(data, Mapping):
        return active
    for config_key, runtime_modality in _MODALITY_KEYS:
        section = data.get(config_key)
        if section is None:
            section = data.get(runtime_modality)
        if section is None:
            continue
        if not isinstance(section, Mapping):
            active[f"<invalid {config_key}>"] = (runtime_modality, {})
            continue
        for dataset_name, dataset_cfg in section.items():
            if isinstance(dataset_cfg, Mapping):
                cfg = dict(dataset_cfg)
            elif dataset_cfg is None:
                cfg = {}
            else:
                cfg = {"<invalid>": dataset_cfg}
            active[str(dataset_name)] = (runtime_modality, cfg)
    return active


def _callback_eval_dataset_configs(
    config: Mapping[str, Any],
    issues: list[str],
) -> list[tuple[dict[str, tuple[str, dict[str, Any]]], str]]:
    """Return callback-local eval datasets and splits that launch must validate.

    Args:
        config: Resolved training config.
        issues: Aggregate readiness issue list.

    Returns:
        Pairs of active X-ray dataset configs and requested split, one per
        configured ``EvalSetMetricLogger``.
    """

    callbacks = config.get("callbacks")
    if callbacks is None:
        return []
    if not isinstance(callbacks, Mapping):
        issues.append("callbacks must be a mapping or null.")
        return []

    result: list[tuple[dict[str, tuple[str, dict[str, Any]]], str]] = []
    for group_name, raw_group in callbacks.items():
        if raw_group is None:
            continue
        if not isinstance(raw_group, Mapping):
            issues.append(f"callbacks.{group_name} must be a mapping or null.")
            continue
        for callback_name, raw_spec in raw_group.items():
            if raw_spec is None:
                continue
            if not isinstance(raw_spec, Mapping):
                issues.append(
                    f"callback {callback_name!r} must be a mapping with _class."
                )
                continue
            class_name = str(raw_spec.get("_class", ""))
            if class_name.split(".")[-1] != "EvalSetMetricLogger":
                continue
            raw_data = raw_spec.get("data")
            if not isinstance(raw_data, Mapping):
                issues.append(
                    f"EvalSetMetricLogger {callback_name!r} data must be a mapping."
                )
                continue
            unsupported = sorted(
                str(key) for key in raw_data if str(key).lower() != "xray"
            )
            if unsupported:
                issues.append(
                    f"EvalSetMetricLogger {callback_name!r} only supports Xray; "
                    f"got sections {unsupported}."
                )
                continue
            section = raw_data.get("Xray", raw_data.get("xray"))
            if section is None:
                section = {}
            if not isinstance(section, Mapping):
                issues.append(
                    f"EvalSetMetricLogger {callback_name!r} data.Xray must be a "
                    "mapping or null."
                )
                continue
            split = raw_spec.get("split", "val")
            if not isinstance(split, str) or not split.strip():
                issues.append(
                    f"EvalSetMetricLogger {callback_name!r} split must be a "
                    "non-empty string."
                )
                continue
            active = _active_dataset_configs({"data": {"Xray": section}})
            result.append((active, split.strip()))
    return result


def _requires_training_data(config: Mapping[str, Any]) -> bool:
    """Return whether the configured experiment is the FleXray trainer."""

    experiment_cfg = config.get("experiment")
    if not isinstance(experiment_cfg, Mapping):
        return False
    class_name = str(experiment_cfg.get("_class", ""))
    return class_name.split(".")[-1] == "FleXrayTrainExperiment"


def _validate_dataloader(
    config: Mapping[str, Any],
    active: Mapping[str, tuple[str, dict[str, Any]]],
    issues: list[str],
) -> None:
    """Validate dataloader proportions and fixed epoch length."""

    if not active and _requires_training_data(config):
        issues.append(
            "data must configure at least one active CT or Xray dataset."
        )
    dl_cfg = config.get("dataloader")
    if not isinstance(dl_cfg, Mapping):
        issues.append("dataloader must be a mapping.")
        return

    if "batch_size" in dl_cfg:
        try:
            _positive_integer(dl_cfg["batch_size"], "dataloader.batch_size")
        except ValueError as exc:
            issues.append(str(exc))
    if "num_workers" in dl_cfg:
        try:
            _integer_value(dl_cfg["num_workers"], "dataloader.num_workers")
        except ValueError as exc:
            issues.append(str(exc))

    for modality in sorted({value[0] for value in active.values()}):
        try:
            loader_options(dl_cfg, modality=modality)
        except (TypeError, ValueError) as exc:
            message = str(exc)
            if message not in issues:
                issues.append(message)

    proportions = dl_cfg.get("proportions")
    if proportions is not None:
        if not isinstance(proportions, Mapping):
            issues.append("dataloader.proportions must be a mapping.")
        else:
            for dataset_name, raw_weight in proportions.items():
                try:
                    weight = normalize_proportion_weight(
                        raw_weight,
                        f"dataloader.proportions[{dataset_name!r}]",
                    )
                except ValueError as exc:
                    issues.append(str(exc))
                    continue
                if weight > 0 and str(dataset_name) not in active:
                    issues.append(
                        "dataloader.proportions references active dataset "
                        f"{dataset_name!r}, but it is absent from data."
                    )

    if "iters_per_epoch" in dl_cfg:
        try:
            _positive_integer(dl_cfg["iters_per_epoch"], "dataloader.iters_per_epoch")
        except ValueError as exc:
            issues.append(str(exc))

    ct_cfg = dl_cfg.get("CT")
    if isinstance(ct_cfg, Mapping) and "batch_size" in ct_cfg:
        try:
            ct_batch_size = _positive_integer(
                ct_cfg["batch_size"], "dataloader.CT.batch_size"
            )
        except ValueError as exc:
            issues.append(str(exc))
        else:
            if ct_batch_size != 1:
                issues.append(
                    "dataloader.CT.batch_size must be 1 for CT volume loaders."
                )


def _validate_protocol_remaps(
    config: Mapping[str, Any],
    active: Mapping[str, tuple[str, dict[str, Any]]],
    issues: list[str],
    *,
    required_splits: tuple[str, ...],
) -> None:
    """Validate label remaps and required splits for active datasets.

    Args:
        config: Resolved training config.
        active: Active dataset configs keyed by dataset name.
        issues: Aggregate readiness issue list.
        required_splits: Non-empty package splits consumed by this caller.

    Returns:
        ``None``. Failures are appended to ``issues``.
    """

    protocol_cfg = config.get("protocol")
    protocol_name = (
        protocol_cfg.get("name") if isinstance(protocol_cfg, Mapping) else None
    )
    if not protocol_name:
        if active:
            issues.append("protocol.name is required for training label projection.")
        return
    model_names = _model_label_names(config)
    config_root = protocol_cfg.get("config_root")
    for dataset_name in sorted(active):
        modality, dataset_cfg = active[dataset_name]
        try:
            if dataset_cfg.get("path") is not None:
                _validate_packaged_protocol_remap(
                    str(protocol_name),
                    dataset_name,
                    modality,
                    dataset_cfg,
                    model_label_names=model_names,
                    config_root=config_root,
                    required_splits=required_splits,
                )
            else:
                compile_training_label_remap_by_name(
                    str(protocol_name),
                    dataset_name,
                    model_label_names=model_names,
                    config_root=config_root,
                )
        except Exception as exc:  # noqa: BLE001
            # Readiness reports independent configuration failures together.
            issues.append(f"protocol remap failed for {dataset_name!r}: {exc}")


def _validate_packaged_protocol_remap(
    protocol_name: str,
    dataset_name: str,
    modality: str,
    dataset_cfg: Mapping[str, Any],
    *,
    model_label_names: tuple[str, ...] | None,
    config_root: str | Path | None,
    required_splits: tuple[str, ...],
) -> None:
    """Validate label metadata stored by a directly configured package.

    Args:
        protocol_name: Target training protocol name.
        dataset_name: Package dataset identity.
        modality: Runtime modality used to open the package.
        dataset_cfg: Per-dataset config containing the explicit path.
        model_label_names: Optional model-output subset.
        config_root: Optional protocol config root.
        required_splits: Non-empty package splits consumed by this caller.

    Returns:
        ``None`` when dense or named-channel labels are compatible.

    Raises:
        ValueError: If package label metadata is missing or incompatible.
    """

    if not required_splits or len(required_splits) > 2:
        raise ValueError("required_splits must contain one or two split names.")
    primary_split = required_splits[0]
    secondary_split = required_splits[1] if len(required_splits) == 2 else None
    dataset = build_training_dataset(
        dataset_name, split=primary_split, modality=modality, cfg=dataset_cfg
    )
    try:
        backend = getattr(dataset, "backend", None)
        attrs = getattr(backend, "attrs", None)
        if not isinstance(attrs, Mapping):
            raise ValueError("packaged dataset backend has no _attrs mapping.")
        _require_training_splits(
            backend.db,
            dataset_name=dataset_name,
            train_split=primary_split,
            val_split=secondary_split,
        )

        declared_protocol = attrs.get("protocol_name")
        if declared_protocol is not None and str(declared_protocol) != protocol_name:
            raise ValueError(
                f"package protocol_name {declared_protocol!r} does not match "
                f"training protocol {protocol_name!r}."
            )
        source_names = getattr(backend, "label_names", None)
        dataset_spec = dataset_cfg.get("dataset_spec")
        if source_names is not None:
            assert dataset_spec is None, (
                f"data[{dataset_name!r}].dataset_spec applies to dense packages only."
            )
            normalize_training_channel_label_names(
                protocol_name,
                dataset_name,
                source_names,
                config_root=config_root,
            )
            return

        stored_labels = attrs.get("stored_labels")
        if stored_labels is None:
            raise ValueError(
                "dense package _attrs must define stored_labels; channel packages "
                "must define label_names."
            )
        compile_package_label_remap(
            protocol_name,
            dataset_name,
            stored_labels,
            dataset_spec=dataset_spec,
            model_label_names=model_label_names,
            config_root=config_root,
        )
    finally:
        _close_dataset_database(dataset)


def _close_dataset_database(dataset: Any) -> None:
    """Close a temporary dataset's database reader when it owns one.

    Args:
        dataset: Runtime dataset whose backend may expose a closable database.

    Returns:
        ``None``.
    """

    close = getattr(dataset, "close", None)
    if callable(close):
        close()
        return
    backend = getattr(dataset, "backend", None)
    database = getattr(backend, "db", None)
    close = getattr(database, "close", None)
    if callable(close):
        close()


def _validate_drr_profiles(
    config: Mapping[str, Any],
    active: Mapping[str, tuple[str, dict[str, Any]]],
    issues: list[str],
) -> dict[str, dict[str, Any]] | None:
    """Validate configured CT datasets have matching DRR profiles."""

    ct_names = {name for name, (modality, _) in active.items() if modality == "ct"}
    if not ct_names:
        return None
    if config.get("drr_model") is None:
        issues.append("drr_model is required when CT datasets are active.")
        return None
    try:
        return resolve_ct_profiles(config)
    except Exception as exc:  # noqa: BLE001 - report all readiness failures together.
        issues.append(f"CT DRR profile validation failed: {exc}")
        return None


def _validate_routed_loss(
    config: Mapping[str, Any],
    active: Mapping[str, tuple[str, dict[str, Any]]],
    issues: list[str],
) -> None:
    """Validate DatasetRoutedLoss coverage when a routed loss is configured."""

    loss_cfg = config.get("loss_func")
    if not isinstance(loss_cfg, Mapping):
        return
    class_name = str(loss_cfg.get("_class", ""))
    if class_name.split(".")[-1] != "DatasetRoutedLoss":
        return
    losses = loss_cfg.get("losses")
    routes = loss_cfg.get("dataset_losses")
    if not isinstance(losses, Mapping) or not isinstance(routes, Mapping):
        issues.append("DatasetRoutedLoss requires losses and dataset_losses mappings.")
        return
    active_names = set(active)
    routed_names = {str(name) for name in routes}
    missing = sorted(active_names - routed_names)
    stray = sorted(
        name for name in routed_names - active_names if _route_weight(config, name) != 0
    )
    if missing:
        issues.append(
            f"DatasetRoutedLoss.dataset_losses is missing active dataset(s): {missing}."
        )
    if stray:
        issues.append(
            f"DatasetRoutedLoss.dataset_losses references inactive dataset(s): {stray}."
        )
    unknown_profiles = sorted(
        str(profile) for profile in routes.values() if str(profile) not in losses
    )
    if unknown_profiles:
        issues.append(
            f"DatasetRoutedLoss routes reference unknown profile(s): {unknown_profiles}."
        )


def _validate_dataset_paths(
    active: Mapping[str, tuple[str, dict[str, Any]]],
    issues: list[str],
    *,
    data_roots: Mapping[str, str | Path | None],
) -> None:
    """Validate that active dataset layout paths exist under dataset roots."""

    if not active:
        return
    for dataset_name, (modality, dataset_cfg) in sorted(active.items()):
        try:
            if dataset_cfg.get("path") is not None:
                path = _absolute_dataset_path(
                    dataset_cfg["path"], dataset_name=dataset_name
                )
            else:
                layout = _layout_for_name(dataset_name)
                if layout.modality != modality:
                    raise KeyError(
                        f"layout is registered for modality {layout.modality!r}, "
                        f"not {modality!r}"
                    )
                root = _dataset_root(layout.root_env_var, data_roots)
                path = _resolve_dataset_path(
                    dataset_name, modality, dataset_cfg, root
                )
        except (
            Exception
        ) as exc:  # noqa: BLE001 - report all readiness failures together.
            issues.append(f"dataset path resolution failed for {dataset_name!r}: {exc}")
            continue
        if not path.exists():
            issues.append(f"dataset path for {dataset_name!r} does not exist: {path}")


def _validate_smoke_data(
    active: Mapping[str, tuple[str, dict[str, Any]]],
    issues: list[str],
    *,
    profiles: Mapping[str, Mapping[str, Any]] | None,
    data_roots: Mapping[str, str | Path | None],
    split: str,
) -> None:
    """Open one sample per active dataset split and validate CT fields."""

    with _temporary_datapaths(data_roots):
        for dataset_name, (modality, dataset_cfg) in sorted(active.items()):
            dataset = None
            try:
                dataset = build_training_dataset(
                    dataset_name, split=split, modality=modality, cfg=dataset_cfg
                )
                if len(dataset) <= 0:
                    issues.append(
                        f"dataset {dataset_name!r} has no {split!r} samples."
                    )
                    continue
                sample = dataset[0]
                if modality == "ct":
                    _validate_ct_sample(
                        dataset_name,
                        sample,
                        issues,
                        profiles=profiles,
                    )
            except Exception as exc:  # noqa: BLE001
                issues.append(f"smoke-data open failed for {dataset_name!r}: {exc}")
            finally:
                if dataset is not None:
                    _close_dataset_database(dataset)


def _validate_ct_sample(
    dataset_name: str,
    sample: Mapping[str, Any],
    issues: list[str],
    *,
    profiles: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    """Validate one CT sample has the fields required by CT rendering."""

    label = sample.get("label")
    if label is None:
        issues.append(f"CT dataset {dataset_name!r} sample is missing label.")
    else:
        label_t = torch.as_tensor(label)
        if label_t.ndim not in (3, 4):
            issues.append(
                f"CT dataset {dataset_name!r} label must be dense 3D or Cx3D; "
                f"got shape {tuple(label_t.shape)}."
            )
        if label_t.ndim == 4 and int(label_t.shape[0]) != 1:
            issues.append(
                f"CT dataset {dataset_name!r} label must be a dense single-channel "
                f"mask; got shape {tuple(label_t.shape)}."
            )
        if label_t.is_floating_point() and not torch.equal(label_t, label_t.round()):
            issues.append(
                f"CT dataset {dataset_name!r} label contains non-integer values."
            )

    metadata = sample.get("metadata")
    if not isinstance(metadata, Mapping):
        issues.append(f"CT dataset {dataset_name!r} sample is missing metadata.")
        return
    if "affine" not in metadata:
        issues.append(f"CT dataset {dataset_name!r} sample metadata is missing affine.")
    if (
        _requires_random_label_centroids(dataset_name, profiles)
        and "fg_centroids_ijk" not in metadata
    ):
        issues.append(
            f"CT dataset {dataset_name!r} uses random_label isocenter but sample "
            "metadata is missing fg_centroids_ijk."
        )


def _resolve_dataset_path(
    dataset_name: str,
    modality: str,
    dataset_cfg: Mapping[str, Any],
    root: Path,
) -> Path:
    """Resolve a registered training dataset path without opening the dataset."""

    layout = _layout_for_name(dataset_name)
    if layout.modality != modality:
        raise KeyError(
            f"layout is registered for modality {layout.modality!r}, not {modality!r}"
        )
    values = {"dataset_name": layout.dataset_name, **dict(dataset_cfg)}
    missing = [
        field
        for field in layout.required_fields
        if values.get(field) in (None, "", _MISSING_PLACEHOLDER)
    ]
    if missing:
        raise ValueError(f"missing required field(s): {missing}")
    relative_path = layout.relative_path_template.format(**values)
    return _path_below_root(
        root,
        relative_path,
        context=f"Dataset layout {dataset_name!r}",
    )


def _data_root_overrides(
    *,
    ct_data_root: str | Path | None,
    xray_data_root: str | Path | None,
    generated_data_root: str | Path | None,
) -> dict[str, str | Path | None]:
    """Return explicit root overrides keyed by dataset root env var."""

    return {
        "CT_DATAPATH": ct_data_root,
        "XRAY_DATAPATH": xray_data_root,
        "GENERATED_DATAPATH": generated_data_root,
    }


def _dataset_root(
    root_env_var: str | None, data_roots: Mapping[str, str | Path | None]
) -> Path:
    """Return a dataset root from an override or the environment."""

    env_var = _required_root_env_var(root_env_var)
    raw_value = data_roots.get(env_var)
    if raw_value is None or not str(raw_value).strip():
        raw_value = os.environ.get(env_var)
    if raw_value is None or not str(raw_value).strip():
        raise KeyError(
            f"{env_var} must be set or the matching cluster root must be provided."
        )
    value = str(raw_value)
    if os.pathsep in value:
        raise ValueError(
            f"{env_var} must name one dataset root directory, not a path-list."
        )
    return Path(value)


def _required_root_env_var(root_env_var: str | None) -> str:
    """Validate and return a layout root environment variable name."""

    if root_env_var is None or not str(root_env_var).strip():
        raise ValueError("dataset layout must define root_env_var.")
    return str(root_env_var).strip()


@contextmanager
def _temporary_datapaths(
    data_roots: Mapping[str, str | Path | None]
) -> Iterator[None]:
    """Temporarily expose root overrides as clean data-path env vars."""

    old_values = {env_var: os.environ.get(env_var) for env_var in _ROOT_ENV_VARS}
    for env_var, raw_value in data_roots.items():
        if raw_value is not None and str(raw_value).strip():
            os.environ[env_var] = str(raw_value)
    try:
        yield
    finally:
        for env_var, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(env_var, None)
            else:
                os.environ[env_var] = old_value


def _model_label_names(config: Mapping[str, Any]) -> tuple[str, ...] | None:
    """Return explicit protocol model labels when configured."""

    protocol_cfg = config.get("protocol")
    if not isinstance(protocol_cfg, Mapping):
        return None
    model_labels = protocol_cfg.get("model_labels")
    if not isinstance(model_labels, Mapping):
        return None
    names = model_labels.get("names")
    if names is None:
        return None
    return tuple(str(name) for name in names)


def _requires_random_label_centroids(
    dataset_name: str,
    profiles: Mapping[str, Mapping[str, Any]] | None,
) -> bool:
    """Return whether a CT profile uses random-label isocenter sampling."""

    if profiles is None or dataset_name not in profiles:
        return False
    isocenter_cfg = profiles[dataset_name].get("isocenter_cfg")
    return (
        isinstance(isocenter_cfg, Mapping)
        and isocenter_cfg.get("sample_scheme") == _DRR_RANDOM_LABEL
    )


def _route_weight(config: Mapping[str, Any], dataset_name: str) -> Any | None:
    """Return a configured dataloader proportion weight, if present."""

    dl_cfg = config.get("dataloader")
    if not isinstance(dl_cfg, Mapping):
        return None
    proportions = dl_cfg.get("proportions")
    if not isinstance(proportions, Mapping) or dataset_name not in proportions:
        return None
    try:
        return normalize_proportion_weight(
            proportions[dataset_name], f"dataloader.proportions[{dataset_name!r}]"
        )
    except ValueError:
        return None


def _positive_integer(raw_value: Any, name: str) -> int:
    """Return a positive integer config value."""

    value = _integer_value(raw_value, name)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {raw_value!r}.")
    return value


def _integer_value(raw_value: Any, name: str) -> int:
    """Return a non-negative integer config value."""

    if isinstance(raw_value, bool):
        raise ValueError(f"{name} must be an integer; got {raw_value!r}.")
    if isinstance(raw_value, int):
        value = raw_value
    elif isinstance(raw_value, float) and raw_value.is_integer():
        value = int(raw_value)
    elif isinstance(raw_value, str):
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer; got {raw_value!r}.") from exc
    else:
        raise ValueError(f"{name} must be an integer; got {raw_value!r}.")
    if value < 0:
        raise ValueError(f"{name} must be >= 0; got {value}.")
    return value
