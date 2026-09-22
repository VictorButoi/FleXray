from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schemas import DatasetSpec, EvalContract, ModelLabelSpace, ProtocolSpec

_BACKGROUND_LABEL = "background"


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys before schema parsing.

    Attributes:
        No additional public attributes beyond ``yaml.SafeLoader`` state.
    """


def _construct_unique_mapping(loader: yaml.SafeLoader, node: yaml.Node, deep=False):
    """Construct a YAML mapping while failing on duplicate keys.

    Args:
        loader: PyYAML loader instance constructing the node.
        node: YAML mapping node to construct.
        deep: Whether PyYAML should construct nested nodes eagerly.

    Returns:
        A Python mapping for the YAML node.
    """

    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_protocol(path: str | Path) -> ProtocolSpec:
    """Load and validate a protocol YAML file.

    Args:
        path: Path to a protocol YAML file with `protocol_name` and ordered
            `labels`.

    Returns:
        A `ProtocolSpec` whose label order is validated and preserved.
    """

    data = _load_yaml_mapping(path)
    _reject_unknown_keys(data, {"protocol_name", "labels"}, context=f"{path}")

    protocol_name = _required_name(data, "protocol_name", context=f"{path}")
    labels = _load_label_list(data.get("labels"), context=f"{path}: labels")
    _validate_ordered_label_space(labels, context=f"Protocol {protocol_name!r}")
    return ProtocolSpec(protocol_name=protocol_name, labels=labels)


def load_model_label_space(path: str | Path) -> ModelLabelSpace:
    """Load and validate a model label-space YAML file.

    Args:
        path: Path to a model YAML file with `model_name` and ordered `labels`.

    Returns:
        A `ModelLabelSpace` whose output channel order is validated and preserved.
    """

    data = _load_yaml_mapping(path)
    _reject_unknown_keys(data, {"model_name", "labels"}, context=f"{path}")

    model_name = _required_name(data, "model_name", context=f"{path}")
    labels = _load_label_list(data.get("labels"), context=f"{path}: labels")
    _validate_ordered_label_space(labels, context=f"Model {model_name!r}")
    return ModelLabelSpace(model_name=model_name, labels=labels)


def load_dataset_spec(path: str | Path) -> DatasetSpec:
    """Load and validate a segmentation dataset YAML file.

    Args:
        path: Path to a dataset YAML file declaring stored segmentation labels,
            optional aliases/drops, required `skip_subjects`, and aliases.

    Returns:
        A `DatasetSpec` for a segmentation dataset. Visibility-only and eval-only
        datasets are intentionally unsupported in this protocol slice.
    """

    data = _load_yaml_mapping(path)
    _reject_unknown_keys(
        data,
        {
            "dataset_name",
            "stored_labels",
            "skip_subjects",
            "protocol_label_aliases",
            "protocol_drop_labels",
            "dataset_aliases",
            "supervise_empty_labels",
        },
        context=f"{path}",
    )
    dataset_name = _required_name(data, "dataset_name", context=f"{path}")
    if "skip_subjects" not in data:
        raise ValueError(f'{path} must define "skip_subjects".')

    dataset_aliases = _load_dataset_aliases(
        data.get("dataset_aliases", []),
        dataset_name=dataset_name,
        context=f"Dataset {dataset_name!r} dataset_aliases",
    )
    stored_labels = _load_stored_labels(
        data.get("stored_labels"),
        context=f"Dataset {dataset_name!r} stored_labels",
    )
    protocol_label_aliases = _load_protocol_label_mappings(
        data.get("protocol_label_aliases", {}),
        context=f"Dataset {dataset_name!r} protocol_label_aliases",
    )
    protocol_drop_labels = _load_protocol_label_lists(
        data.get("protocol_drop_labels", {}),
        context=f"Dataset {dataset_name!r} protocol_drop_labels",
    )
    skip_subjects = _load_skip_subjects(
        data.get("skip_subjects"),
        context=f"Dataset {dataset_name!r} skip_subjects",
    )
    supervise_empty_labels = _load_label_list(
        data.get("supervise_empty_labels", []),
        context=f"Dataset {dataset_name!r} supervise_empty_labels",
    )
    stored_names = set(stored_labels.values())
    for protocol_aliases in protocol_label_aliases.values():
        stored_names.update(
            protocol_aliases[name] for name in stored_labels.values()
            if name in protocol_aliases
        )
    overlap = sorted(stored_names.intersection(supervise_empty_labels))
    if overlap:
        raise ValueError(
            f"Dataset {dataset_name!r} supervise_empty_labels must name protocol "
            f"labels the dataset never stores; got stored or aliased labels {overlap}."
        )
    _validate_dataset_spec(
        dataset_name=dataset_name,
        stored_labels=stored_labels,
        protocol_label_aliases=protocol_label_aliases,
        protocol_drop_labels=protocol_drop_labels,
    )
    return DatasetSpec(
        dataset_name=dataset_name,
        stored_labels=stored_labels,
        protocol_label_aliases=protocol_label_aliases,
        protocol_drop_labels=protocol_drop_labels,
        skip_subjects=skip_subjects,
        dataset_aliases=dataset_aliases,
        supervise_empty_labels=supervise_empty_labels,
    )


def load_eval_contract(path: str | Path) -> EvalContract:
    """Load and validate an eval-contract YAML file.

    Args:
        path: Path to an eval-contract YAML file with model name, eval sets, and
            optional identity defaults.

    Returns:
        An `EvalContract` with candidate model-label groups normalized to tuples.
    """

    data = _load_yaml_mapping(path)
    _reject_unknown_keys(
        data,
        {"model_name", "eval_sets", "identity_defaults"},
        context=f"{path}",
    )
    model_name = _required_name(data, "model_name", context=f"{path}")
    identity_defaults = _load_eval_identity_defaults(
        data.get("identity_defaults"),
        context=f"{path}: identity_defaults",
    )
    raw_eval_sets = data.get("eval_sets")
    if not isinstance(raw_eval_sets, dict):
        raise TypeError(f"Eval contract {model_name!r} eval_sets must be a mapping.")

    eval_sets: dict[str, dict[str, tuple[tuple[str, ...], ...]]] = {}
    for raw_eval_set_name, raw_eval_label_map in raw_eval_sets.items():
        eval_set_name = _trim_name(
            raw_eval_set_name,
            context=f"Eval contract {model_name!r} eval set name",
        )
        if eval_set_name in eval_sets:
            raise ValueError(
                f"Eval contract {model_name!r} repeats eval set {eval_set_name!r}."
            )
        if not isinstance(raw_eval_label_map, dict):
            raise TypeError(
                f"Eval contract {model_name!r} eval set {eval_set_name!r} "
                "must be a mapping."
            )
        eval_sets[eval_set_name] = _load_eval_label_map(
            model_name=model_name,
            eval_set_name=eval_set_name,
            raw_eval_label_map=raw_eval_label_map,
        )

    return EvalContract(
        model_name=model_name,
        eval_sets=eval_sets,
        identity_defaults=identity_defaults,
    )


def _load_eval_label_map(
    *,
    model_name: str,
    eval_set_name: str,
    raw_eval_label_map: dict[object, object],
) -> dict[str, tuple[tuple[str, ...], ...]]:
    """Load all eval labels for one eval set.

    Args:
        model_name: Model name used in validation errors.
        eval_set_name: Eval-set name used in validation errors.
        raw_eval_label_map: Raw YAML mapping from eval label to candidate config.

    Returns:
        Mapping from eval label to ordered candidate model-label tuples.
    """

    eval_label_map: dict[str, tuple[tuple[str, ...], ...]] = {}
    for raw_eval_label, raw_cfg in raw_eval_label_map.items():
        eval_label = _trim_name(
            raw_eval_label,
            context=(
                f"Eval contract {model_name!r} eval set {eval_set_name!r} "
                "label name"
            ),
        )
        if eval_label in eval_label_map:
            raise ValueError(
                f"Eval contract {model_name!r} eval set {eval_set_name!r} "
                f"repeats eval label {eval_label!r}."
            )
        if not isinstance(raw_cfg, dict):
            raise TypeError(
                f"Eval contract {model_name!r} eval set {eval_set_name!r} "
                f"label {eval_label!r} must be a mapping."
            )
        _reject_unknown_keys(
            raw_cfg,
            {"model_labels", "candidates"},
            context=(
                f"Eval contract {model_name!r} eval set {eval_set_name!r} "
                f"label {eval_label!r}"
            ),
        )
        eval_label_map[eval_label] = _load_eval_label_candidates(
            raw_cfg,
            context=(
                f"Eval contract {model_name!r} eval set {eval_set_name!r} "
                f"label {eval_label!r}"
            ),
        )
    return eval_label_map


def _load_eval_label_candidates(
    raw_cfg: dict[object, object],
    *,
    context: str,
) -> tuple[tuple[str, ...], ...]:
    """Load shorthand or explicit candidates for one eval label.

    Args:
        raw_cfg: Raw YAML candidate config using exactly one of `model_labels` or
            `candidates`.
        context: Human-readable location for validation errors.

    Returns:
        Ordered candidate groups as tuples of model label names.
    """

    has_shorthand = "model_labels" in raw_cfg
    has_candidates = "candidates" in raw_cfg
    if has_shorthand == has_candidates:
        raise ValueError(
            f"{context} must define exactly one of 'model_labels' or 'candidates'."
        )
    if has_shorthand:
        model_labels = _load_label_list(
            raw_cfg.get("model_labels"),
            context=f"{context} model_labels",
        )
        if not model_labels:
            raise ValueError(f"{context} model_labels must not be empty.")
        return (model_labels,)

    raw_candidates = raw_cfg.get("candidates")
    if not isinstance(raw_candidates, list):
        raise TypeError(f"{context} candidates must be a list.")
    if not raw_candidates:
        raise ValueError(f"{context} candidates must list at least one candidate.")
    candidates: list[tuple[str, ...]] = []
    for idx, raw_candidate in enumerate(raw_candidates):
        candidate_context = f"{context} candidates[{idx}]"
        if not isinstance(raw_candidate, dict):
            raise TypeError(f"{candidate_context} must be a mapping.")
        _reject_unknown_keys(raw_candidate, {"model_labels"}, context=candidate_context)
        model_labels = _load_label_list(
            raw_candidate.get("model_labels"),
            context=f"{candidate_context} model_labels",
        )
        if not model_labels:
            raise ValueError(f"{candidate_context} model_labels must not be empty.")
        candidates.append(model_labels)
    return tuple(candidates)


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Read a YAML file as a duplicate-key-checked mapping.

    Args:
        path: YAML file path.

    Returns:
        Top-level YAML mapping, or an empty dict for an empty file.
    """

    yaml_path = Path(path)
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.load(f, Loader=_UniqueKeySafeLoader) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{yaml_path} must contain a top-level mapping.")
    return data


def _reject_unknown_keys(
    data: dict[object, object],
    allowed_keys: set[str],
    *,
    context: str,
) -> None:
    """Reject mapping keys not declared by the local schema.

    Args:
        data: Mapping to validate.
        allowed_keys: String keys accepted at this schema location.
        context: Human-readable location for validation errors.

    Returns:
        None. Raises `ValueError` when unknown keys are present.
    """

    unknown_keys = sorted(str(key) for key in data if str(key) not in allowed_keys)
    if unknown_keys:
        raise ValueError(f"{context} uses unknown keys: {unknown_keys}.")


def _required_name(data: dict[str, Any], key: str, *, context: str) -> str:
    """Read a required string name field.

    Args:
        data: YAML mapping containing the field.
        key: Required field name.
        context: Human-readable location for validation errors.

    Returns:
        Trimmed non-empty string value for `key`.
    """

    if key not in data:
        raise ValueError(f"{context} must define {key!r}.")
    return _trim_name(data[key], context=f"{context}: {key}")


def _trim_name(raw_name: object, *, context: str) -> str:
    """Validate and trim a YAML string value.

    Args:
        raw_name: Raw YAML value expected to be a string.
        context: Human-readable location for validation errors.

    Returns:
        Trimmed non-empty string.
    """

    if not isinstance(raw_name, str):
        raise TypeError(f"{context} must be a string, got {raw_name!r}.")
    name = raw_name.strip()
    if not name:
        raise ValueError(f"{context} cannot be empty.")
    return name


def _load_eval_identity_defaults(raw_value: object, *, context: str) -> str | None:
    """Load the optional eval identity-default mode.

    Args:
        raw_value: Raw YAML value for `identity_defaults`.
        context: Human-readable location for validation errors.

    Returns:
        `None` or `"dataset_eval_labels"`.
    """

    if raw_value is None:
        return None
    value = _trim_name(raw_value, context=context)
    if value != "dataset_eval_labels":
        raise ValueError(f"{context} must be 'dataset_eval_labels', got {raw_value!r}.")
    return value


def _load_label_list(raw_labels: object, *, context: str) -> tuple[str, ...]:
    """Load a YAML list of unique label names.

    Args:
        raw_labels: Raw YAML value expected to be a list of strings.
        context: Human-readable location for validation errors.

    Returns:
        Tuple of trimmed label names in input order.
    """

    if not isinstance(raw_labels, list):
        raise TypeError(f"{context} must be a list of label names.")
    labels = tuple(
        _trim_name(raw_label, context=f"{context}[{idx}]")
        for idx, raw_label in enumerate(raw_labels)
    )
    _reject_duplicate_names(labels, context=context)
    return labels


def _load_skip_subjects(
    raw_subjects: object,
    *,
    context: str,
) -> tuple[str, ...] | None:
    """Load optional skipped subject identifiers.

    Args:
        raw_subjects: Raw YAML value, either null or a non-empty list of strings.
        context: Human-readable location for validation errors.

    Returns:
        `None` when no subjects are skipped, otherwise a tuple of subject keys.
    """

    if raw_subjects is None:
        return None
    if not isinstance(raw_subjects, list):
        raise TypeError(f"{context} must be null or a list of subject keys.")
    subjects = tuple(
        _trim_name(raw_subject, context=f"{context}[{idx}]")
        for idx, raw_subject in enumerate(raw_subjects)
    )
    if not subjects:
        raise ValueError(f"{context} must be null or list at least one subject key.")
    _reject_duplicate_names(subjects, context=context)
    return subjects


def _load_dataset_aliases(
    raw_aliases: object,
    *,
    dataset_name: str,
    context: str,
) -> tuple[str, ...]:
    """Load alternate names for a dataset config.

    Args:
        raw_aliases: Raw YAML value expected to be a list of strings.
        dataset_name: Canonical dataset name; aliases cannot equal it.
        context: Human-readable location for validation errors.

    Returns:
        Tuple of unique dataset aliases.
    """

    if not isinstance(raw_aliases, list):
        raise TypeError(f"{context} must be a list of dataset aliases.")
    aliases = tuple(
        _trim_name(raw_alias, context=f"{context}[{idx}]")
        for idx, raw_alias in enumerate(raw_aliases)
    )
    seen: set[str] = set()
    for alias in aliases:
        if alias in seen:
            raise ValueError(f"{context} repeats alias {alias!r}.")
        if alias == dataset_name:
            raise ValueError(
                f"{context} must not include owning dataset name {dataset_name!r}."
            )
        seen.add(alias)
    return aliases


def _load_label_mapping(
    raw_mapping: object,
    *,
    context: str,
) -> dict[str, str]:
    """Load a string-to-string label mapping.

    Args:
        raw_mapping: Raw YAML mapping from source labels to target labels.
        context: Human-readable location for validation errors.

    Returns:
        Dict mapping source labels to target labels.
    """

    if raw_mapping is None:
        raw_mapping = {}
    if not isinstance(raw_mapping, dict):
        raise TypeError(f"{context} must be a mapping.")
    mapping: dict[str, str] = {}
    for raw_key, raw_value in raw_mapping.items():
        key = _trim_name(raw_key, context=f"{context} key")
        value = _trim_name(raw_value, context=f"{context}[{key!r}]")
        if key in mapping:
            raise ValueError(f"{context} repeats label {key!r}.")
        mapping[key] = value
    return mapping


def _load_protocol_label_mappings(
    raw_mapping: object,
    *,
    context: str,
) -> dict[str, dict[str, str]]:
    """Load per-protocol label alias mappings.

    Args:
        raw_mapping: Raw YAML mapping from protocol names to label mappings.
        context: Human-readable location for validation errors.

    Returns:
        Nested mapping keyed by protocol name, then dataset-native label name.
    """

    if raw_mapping is None:
        raw_mapping = {}
    if not isinstance(raw_mapping, dict):
        raise TypeError(f"{context} must be a mapping.")
    protocol_mappings: dict[str, dict[str, str]] = {}
    for raw_protocol_name, raw_label_mapping in raw_mapping.items():
        protocol_name = _trim_name(
            raw_protocol_name,
            context=f"{context} protocol name",
        )
        if protocol_name in protocol_mappings:
            raise ValueError(f"{context} repeats protocol {protocol_name!r}.")
        protocol_mappings[protocol_name] = _load_label_mapping(
            raw_label_mapping,
            context=f"{context}[{protocol_name!r}]",
        )
    return protocol_mappings


def _load_protocol_label_lists(
    raw_mapping: object,
    *,
    context: str,
) -> dict[str, tuple[str, ...]]:
    """Load per-protocol lists of dataset-native labels.

    Args:
        raw_mapping: Raw YAML mapping from protocol names to label lists.
        context: Human-readable location for validation errors.

    Returns:
        Mapping from protocol name to tuple of label names.
    """

    if raw_mapping is None:
        raw_mapping = {}
    if not isinstance(raw_mapping, dict):
        raise TypeError(f"{context} must be a mapping.")
    protocol_lists: dict[str, tuple[str, ...]] = {}
    for raw_protocol_name, raw_label_list in raw_mapping.items():
        protocol_name = _trim_name(
            raw_protocol_name,
            context=f"{context} protocol name",
        )
        if protocol_name in protocol_lists:
            raise ValueError(f"{context} repeats protocol {protocol_name!r}.")
        protocol_lists[protocol_name] = _load_label_list(
            raw_label_list,
            context=f"{context}[{protocol_name!r}]",
        )
    return protocol_lists


def _load_stored_labels(raw_mapping: object, *, context: str) -> dict[int, str]:
    """Load native stored-id to label-name mappings.

    Args:
        raw_mapping: Raw YAML mapping from non-negative integer ids to labels.
        context: Human-readable location for validation errors.

    Returns:
        Dict sorted by native id.
    """

    if not isinstance(raw_mapping, dict):
        raise TypeError(f"{context} must be a mapping from native id to label name.")
    stored_labels: dict[int, str] = {}
    for raw_native_id, raw_label_name in raw_mapping.items():
        native_id = _parse_native_id(raw_native_id, context=context)
        if native_id in stored_labels:
            raise ValueError(f"{context} repeats native id {native_id}.")
        stored_labels[native_id] = _trim_name(
            raw_label_name,
            context=f"{context}[{native_id}]",
        )
    _reject_duplicate_names(stored_labels.values(), context=context)
    return dict(sorted(stored_labels.items()))


def _parse_native_id(raw_native_id: object, *, context: str) -> int:
    """Parse a non-negative integer native stored id.

    Args:
        raw_native_id: YAML mapping key, either an integer or decimal string.
        context: Human-readable location for validation errors.

    Returns:
        Parsed non-negative integer id.
    """

    if type(raw_native_id) is int:
        native_id = raw_native_id
    elif isinstance(raw_native_id, str) and raw_native_id.isdecimal():
        native_id = int(raw_native_id)
    else:
        raise TypeError(
            f"{context} keys must be non-negative integer ids, got {raw_native_id!r}."
        )
    if native_id < 0:
        raise ValueError(f"{context} keys must be non-negative, got {native_id}.")
    return native_id


def _reject_duplicate_names(names: object, *, context: str) -> None:
    """Reject duplicate names while preserving the caller's context.

    Args:
        names: Iterable of string names to check.
        context: Human-readable location for validation errors.

    Returns:
        None. Raises `ValueError` on duplicates.
    """

    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f"{context} repeats label {name!r}.")
        seen.add(name)


def _validate_ordered_label_space(labels: tuple[str, ...], *, context: str) -> None:
    """Validate a background-first ordered label space.

    Args:
        labels: Ordered label names.
        context: Human-readable location for validation errors.

    Returns:
        None. Raises `ValueError` when the label space is invalid.
    """

    if not labels:
        raise ValueError(f"{context} must define at least the background label.")
    if labels[0] != _BACKGROUND_LABEL:
        raise ValueError(f"{context} must put 'background' at channel 0.")
    if _BACKGROUND_LABEL not in labels:
        raise ValueError(f"{context} must include 'background'.")


def _validate_dataset_spec(
    *,
    dataset_name: str,
    stored_labels: dict[int, str],
    protocol_label_aliases: dict[str, dict[str, str]],
    protocol_drop_labels: dict[str, tuple[str, ...]],
) -> None:
    """Validate a segmentation dataset spec after field loading.

    Args:
        dataset_name: Dataset name used for validation errors.
        stored_labels: Native id to label-name mapping.
        protocol_label_aliases: Per-protocol alias mappings.
        protocol_drop_labels: Per-protocol dropped native labels.

    Returns:
        None. Raises `ValueError` when any dataset rule is invalid.
    """

    if stored_labels.get(0) != _BACKGROUND_LABEL:
        raise ValueError(
            f"Dataset {dataset_name!r} must put 'background' at stored id 0."
        )

    stored_label_names = set(stored_labels.values())
    _validate_alias_drop_rules(
        owner=f"Dataset {dataset_name!r}",
        source_label_names=stored_label_names,
        source_label_description="stored labels",
        alias_groups=tuple(
            (f"protocol_label_aliases[{protocol_name!r}]", protocol_aliases)
            for protocol_name, protocol_aliases in protocol_label_aliases.items()
        ),
        drop_groups=tuple(
            (f"protocol_drop_labels[{protocol_name!r}]", protocol_drops)
            for protocol_name, protocol_drops in protocol_drop_labels.items()
        ),
        conflict_groups=tuple(
            (
                f"for protocol {protocol_name!r}",
                protocol_label_aliases.get(protocol_name, {}),
                protocol_drop_labels.get(protocol_name, ()),
            )
            for protocol_name in sorted(
                set(protocol_label_aliases).union(protocol_drop_labels)
            )
        ),
    )


def _validate_alias_drop_rules(
    *,
    owner: str,
    source_label_names: set[str],
    source_label_description: str,
    alias_groups: tuple[tuple[str, dict[str, str]], ...],
    drop_groups: tuple[tuple[str, tuple[str, ...]], ...],
    conflict_groups: tuple[tuple[str, dict[str, str], tuple[str, ...]], ...],
) -> None:
    """Validate alias/drop source labels and conflicts.

    Args:
        owner: Object name used for validation errors.
        source_label_names: Labels allowed as alias/drop sources.
        source_label_description: Description for source labels in errors.
        alias_groups: Named groups of source-to-target aliases.
        drop_groups: Named groups of labels dropped to background.
        conflict_groups: Named alias/drop pairs checked for overlap.

    Returns:
        None. Raises `ValueError` when a rule is invalid.
    """

    for context, aliases in alias_groups:
        unknown_sources = sorted(
            source_label
            for source_label in aliases
            if source_label not in source_label_names
        )
        if unknown_sources:
            raise ValueError(
                f"{owner} {context} references unknown "
                f"{source_label_description}: {unknown_sources}."
            )
        if _BACKGROUND_LABEL in aliases:
            raise ValueError(f"{owner} cannot alias background.")
        background_targets = sorted(
            source_label
            for source_label, target_label in aliases.items()
            if target_label == _BACKGROUND_LABEL
        )
        if background_targets:
            raise ValueError(
                f"{owner} {context} cannot target background for labels: "
                f"{background_targets}."
            )
        identity_aliases = sorted(
            source_label
            for source_label, target_label in aliases.items()
            if source_label == target_label
        )
        if identity_aliases:
            raise ValueError(
                f"{owner} {context} must omit identity mappings: "
                f"{identity_aliases}."
            )

    for context, drop_labels in drop_groups:
        unknown_drops = sorted(
            label_name
            for label_name in drop_labels
            if label_name not in source_label_names
        )
        if unknown_drops:
            raise ValueError(
                f"{owner} {context} references unknown "
                f"{source_label_description}: {unknown_drops}."
            )
        if _BACKGROUND_LABEL in drop_labels:
            raise ValueError(f"{owner} cannot drop background by label.")

    for context, aliases, drop_labels in conflict_groups:
        alias_and_drop_labels = sorted(set(aliases).intersection(drop_labels))
        if alias_and_drop_labels:
            raise ValueError(
                f"{owner} labels cannot be both aliased and dropped {context}: "
                f"{alias_and_drop_labels}."
            )
