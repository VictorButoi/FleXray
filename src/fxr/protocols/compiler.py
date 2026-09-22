from __future__ import annotations

from collections.abc import Sequence

from .schemas import (
    CompiledEvalMapping,
    CompiledTrainingLut,
    DatasetSpec,
    EvalContract,
    ModelLabelSpace,
    ProtocolSpec,
)

_BACKGROUND_LABEL = "background"


def compile_training_lut(
    protocol: ProtocolSpec,
    dataset_spec: DatasetSpec,
) -> CompiledTrainingLut:
    """Compile dataset-native mask ids into protocol channel ids.

    Args:
        protocol: Target protocol label space. It must contain `background` at
            channel 0.
        dataset_spec: Segmentation dataset spec that declares native stored ids
            and any aliases/drops for `protocol.protocol_name`.

    Returns:
        A `CompiledTrainingLut` containing a dense LUT and sparse native-id
        mappings into the protocol label space.
    """

    _validate_protocol_background(protocol)
    _validate_dataset_background(dataset_spec)

    protocol_label_to_id = protocol.label_to_id
    native_id_to_protocol_id: dict[int, int] = {}
    native_id_to_protocol_label: dict[int, str] = {}
    for native_id, source_label in dataset_spec.stored_labels.items():
        protocol_label = _normalized_stored_label(
            dataset_spec,
            protocol.protocol_name,
            source_label,
            known_label_names=set(protocol_label_to_id),
        )
        if protocol_label is None:
            protocol_label = _BACKGROUND_LABEL
        native_id_to_protocol_id[native_id] = protocol_label_to_id[protocol_label]
        native_id_to_protocol_label[native_id] = protocol_label

    max_native_id = max(dataset_spec.stored_labels)
    label_lut = tuple(
        native_id_to_protocol_id.get(native_id, -1)
        for native_id in range(max_native_id + 1)
    )
    return CompiledTrainingLut(
        dataset_name=dataset_spec.dataset_name,
        protocol_name=protocol.protocol_name,
        label_lut=label_lut,
        native_id_to_protocol_id=native_id_to_protocol_id,
        native_id_to_protocol_label=native_id_to_protocol_label,
        protocol_labels=protocol.labels,
    )


def compile_eval_contract(
    model_label_space: ModelLabelSpace,
    eval_contract: EvalContract,
    dataset_spec: DatasetSpec,
    *,
    eval_set_name: str | None = None,
    unsupported_label_policy: str = "error",
    supported_model_label_names: Sequence[str] | None = None,
    strict_identity_defaults: bool = False,
    strict_identity_label_names: Sequence[str] | None = None,
) -> CompiledEvalMapping:
    """Compile one eval-set contract into model output channel groups.

    Args:
        model_label_space: Ordered model output labels that eval labels map onto.
        eval_contract: Eval contract for `model_label_space.model_name`.
        dataset_spec: Dataset spec used to derive identity-default eval labels.
        eval_set_name: Optional eval-set name inside the contract. When omitted,
            `dataset_spec.dataset_name` is used.
        unsupported_label_policy: Behavior when no candidate is fully supported:
            `"error"` raises, `"drop"` omits the eval label, and `"zero"` retains
            a supported model-label candidate for callers to score as zero.
        supported_model_label_names: Optional subset of model labels considered
            supported by the current evaluation run.
        strict_identity_defaults: When true, identity-default labels missing from
            the current model label space are emitted so unsupported handling can
            report them explicitly.
        strict_identity_label_names: Optional full label-space names used to
            validate identity-default labels when `model_label_space` is a subset.

    Returns:
        A `CompiledEvalMapping` with eval labels and selected model channel groups.
    """

    if unsupported_label_policy not in {"error", "drop", "zero"}:
        raise ValueError(
            "unsupported_label_policy must be 'error', 'drop', or 'zero', "
            f"got {unsupported_label_policy!r}."
        )
    if model_label_space.model_name != eval_contract.model_name:
        raise ValueError(
            f"Eval contract model {eval_contract.model_name!r} does not match "
            f"model label space {model_label_space.model_name!r}."
        )

    resolved_eval_set_name = (
        dataset_spec.dataset_name if eval_set_name is None else eval_set_name.strip()
    )
    if not resolved_eval_set_name:
        raise ValueError("eval_set_name cannot be empty.")
    if resolved_eval_set_name not in eval_contract.eval_sets:
        raise ValueError(
            f"Missing FleXray eval contract for model "
            f"{eval_contract.model_name!r} and eval set "
            f"{resolved_eval_set_name!r}."
        )

    model_label_to_id = model_label_space.label_to_id
    if supported_model_label_names is None:
        supported_model_label_set = set(model_label_to_id)
    else:
        supported_model_label_set = {str(name) for name in supported_model_label_names}
        unknown_supported_labels = sorted(
            label_name
            for label_name in supported_model_label_set
            if label_name not in model_label_to_id
        )
        if unknown_supported_labels:
            raise ValueError(
                "supported_model_label_names contains labels outside the resolved "
                f"model label space: {unknown_supported_labels}."
            )

    raw_eval_label_map = eval_contract.eval_sets[resolved_eval_set_name]
    if eval_contract.identity_defaults == "dataset_eval_labels":
        known_identity_label_names = (
            set(model_label_to_id)
            if strict_identity_label_names is None
            else {str(label_name) for label_name in strict_identity_label_names}
        )
        dataset_eval_labels = _dataset_eval_labels(
            dataset_spec,
            eval_contract.model_name,
            known_label_names=known_identity_label_names,
        )
        selector_labels = set(dataset_eval_labels)
        active_raw_eval_label_map = {
            eval_label: candidate_model_label_groups
            for eval_label, candidate_model_label_groups in raw_eval_label_map.items()
            if eval_label in selector_labels
            or any(
                model_label in selector_labels
                for candidate in candidate_model_label_groups
                for model_label in candidate
            )
        }
        eval_label_map = _identity_default_eval_label_map(
            dataset_eval_labels,
            model_label_to_id,
            active_raw_eval_label_map,
            strict_missing_model_labels=strict_identity_defaults,
            strict_identity_label_names=strict_identity_label_names,
        )
    elif eval_contract.identity_defaults is None:
        eval_label_map = raw_eval_label_map
    else:
        raise ValueError(
            f"Eval contract for model {eval_contract.model_name!r} has unsupported "
            f"identity_defaults {eval_contract.identity_defaults!r}."
        )
    if not eval_label_map:
        raise ValueError(
            f"Eval contract for model {eval_contract.model_name!r} and eval set "
            f"{resolved_eval_set_name!r} must define at least one eval label."
        )

    eval_label_names: list[str] = []
    model_channel_groups: list[tuple[int, ...]] = []
    eval_label_to_model_ids: dict[str, tuple[int, ...]] = {}
    eval_label_to_model_labels: dict[str, tuple[str, ...]] = {}
    dropped_eval_label_names: list[str] = []
    dropped_eval_label_to_model_labels: dict[str, tuple[tuple[str, ...], ...]] = {}
    unsupported_zero_eval_label_names: list[str] = []
    unsupported_zero_eval_label_to_model_labels: dict[str, tuple[str, ...]] = {}

    for eval_label, candidate_model_label_groups in eval_label_map.items():
        selected_model_labels, unsupported_zero = _select_supported_eval_candidate(
            eval_contract,
            resolved_eval_set_name,
            eval_label,
            candidate_model_label_groups,
            model_label_to_id,
            supported_model_label_set=supported_model_label_set,
            unsupported_label_policy=unsupported_label_policy,
        )
        if selected_model_labels is None:
            dropped_eval_label_names.append(str(eval_label))
            dropped_eval_label_to_model_labels[str(eval_label)] = tuple(
                tuple(str(label) for label in labels)
                for labels in candidate_model_label_groups
            )
            continue

        model_ids = tuple(
            model_label_to_id[model_label] for model_label in selected_model_labels
        )
        eval_label_names.append(eval_label)
        model_channel_groups.append(model_ids)
        eval_label_to_model_ids[eval_label] = model_ids
        eval_label_to_model_labels[eval_label] = selected_model_labels
        if unsupported_zero:
            unsupported_zero_eval_label_names.append(str(eval_label))
            unsupported_zero_eval_label_to_model_labels[str(eval_label)] = (
                selected_model_labels
            )

    return CompiledEvalMapping(
        model_name=model_label_space.model_name,
        eval_set_name=resolved_eval_set_name,
        eval_label_names=tuple(eval_label_names),
        model_channel_groups=tuple(model_channel_groups),
        eval_label_to_model_ids=eval_label_to_model_ids,
        eval_label_to_model_labels=eval_label_to_model_labels,
        model_labels=model_label_space.labels,
        dropped_eval_label_names=tuple(dropped_eval_label_names),
        dropped_eval_label_to_model_labels=dropped_eval_label_to_model_labels,
        unsupported_zero_eval_label_names=tuple(unsupported_zero_eval_label_names),
        unsupported_zero_eval_label_to_model_labels=(
            unsupported_zero_eval_label_to_model_labels
        ),
    )


def _identity_default_eval_label_map(
    dataset_eval_labels: tuple[str, ...],
    model_label_to_id: dict[str, int],
    active_raw_eval_label_map: dict[str, tuple[tuple[str, ...], ...]],
    *,
    strict_missing_model_labels: bool = False,
    strict_identity_label_names: Sequence[str] | None = None,
) -> dict[str, tuple[tuple[str, ...], ...]]:
    """Build an eval-label map from normalized dataset labels and explicit rules.

    Args:
        dataset_eval_labels: Normalized foreground labels derived from stored ids.
        model_label_to_id: Resolved model label ids keyed by label name.
        active_raw_eval_label_map: Explicit eval-set entries relevant to the
            dataset labels.
        strict_missing_model_labels: Whether to emit identity labels missing from
            the current model label space.
        strict_identity_label_names: Optional complete identity label universe.

    Returns:
        Eval-label candidate mapping ready for candidate selection.
    """

    eval_label_map: dict[str, tuple[tuple[str, ...], ...]] = {}
    emitted_explicit_labels: set[str] = set()
    explicit_label_by_reserved_model_label = _explicit_label_by_reserved_model_label(
        active_raw_eval_label_map,
        model_label_to_id,
    )
    strict_identity_label_set = (
        None
        if strict_identity_label_names is None
        else {str(label_name) for label_name in strict_identity_label_names}
    )

    for eval_label in dataset_eval_labels:
        if eval_label in active_raw_eval_label_map:
            eval_label_map[eval_label] = active_raw_eval_label_map[eval_label]
            emitted_explicit_labels.add(eval_label)
            continue
        explicit_eval_label = explicit_label_by_reserved_model_label.get(eval_label)
        if explicit_eval_label is not None:
            if explicit_eval_label not in emitted_explicit_labels:
                eval_label_map[explicit_eval_label] = active_raw_eval_label_map[
                    explicit_eval_label
                ]
                emitted_explicit_labels.add(explicit_eval_label)
            continue
        if eval_label not in model_label_to_id:
            if strict_missing_model_labels and (
                strict_identity_label_set is None
                or eval_label in strict_identity_label_set
            ):
                eval_label_map[eval_label] = ((eval_label,),)
            continue
        eval_label_map[eval_label] = ((eval_label,),)

    return eval_label_map


def _explicit_label_by_reserved_model_label(
    raw_eval_label_map: dict[str, tuple[tuple[str, ...], ...]],
    model_label_to_id: dict[str, int],
) -> dict[str, str]:
    """Index explicit aggregate eval labels by their covered model labels.

    Args:
        raw_eval_label_map: Explicit eval-set entries from the contract.
        model_label_to_id: Model label ids keyed by label name.

    Returns:
        Mapping from model label name to the first explicit eval label that uses
        it in a fully supported candidate.
    """

    explicit_label_by_model_label: dict[str, str] = {}
    for eval_label, candidate_model_label_groups in raw_eval_label_map.items():
        for model_labels in candidate_model_label_groups:
            if all(model_label in model_label_to_id for model_label in model_labels):
                for model_label in model_labels:
                    explicit_label_by_model_label.setdefault(model_label, eval_label)
                break
    return explicit_label_by_model_label


def _dataset_eval_labels(
    dataset_spec: DatasetSpec,
    protocol_name: str,
    *,
    known_label_names: set[str],
) -> tuple[str, ...]:
    """Derive eval-default labels from normalized dataset stored labels.

    Args:
        dataset_spec: Dataset with native stored labels and normalization rules.
        protocol_name: Protocol/model name selecting the dataset alias/drop rules.
        known_label_names: Valid label names in the target model/protocol label
            space.

    Returns:
        Ordered foreground labels after aliases and drops, with duplicates removed.
    """

    _validate_dataset_background(dataset_spec)
    labels: list[str] = []
    seen: set[str] = set()
    for _, source_label in sorted(dataset_spec.stored_labels.items()):
        normalized_label = _normalized_stored_label(
            dataset_spec,
            protocol_name,
            source_label,
            known_label_names=known_label_names,
        )
        if normalized_label is None or normalized_label == _BACKGROUND_LABEL:
            continue
        if normalized_label not in seen:
            labels.append(normalized_label)
            seen.add(normalized_label)
    return tuple(labels)


def _normalized_stored_label(
    dataset_spec: DatasetSpec,
    protocol_name: str,
    source_label: str,
    *,
    known_label_names: set[str],
) -> str | None:
    """Normalize one dataset-native label for a target label space.

    Args:
        dataset_spec: Dataset containing alias/drop rules.
        protocol_name: Protocol/model name selecting the rules.
        source_label: Dataset-native label to normalize.
        known_label_names: Valid labels in the target label space.

    Returns:
        The normalized target label, or `None` when the label is explicitly
        dropped to background.
    """

    label_aliases = _effective_label_aliases(dataset_spec, protocol_name)
    dropped_label_names = set(_effective_drop_labels(dataset_spec, protocol_name))
    if source_label == _BACKGROUND_LABEL:
        return _BACKGROUND_LABEL
    if source_label in dropped_label_names:
        return None

    normalized_label = label_aliases.get(source_label, source_label)
    if normalized_label in known_label_names:
        return normalized_label
    if normalized_label == source_label:
        raise ValueError(
            f"Dataset {dataset_spec.dataset_name!r} stored label {source_label!r} "
            f"is not in label space {protocol_name!r}; add it to "
            "protocol_label_aliases or protocol_drop_labels."
        )
    raise ValueError(
        f"Dataset {dataset_spec.dataset_name!r} label {source_label!r} maps to "
        f"unknown label space {protocol_name!r} label {normalized_label!r}."
    )


def _select_supported_eval_candidate(
    eval_contract: EvalContract,
    eval_set_name: str,
    eval_label: str,
    candidate_model_label_groups: tuple[tuple[str, ...], ...],
    model_label_to_id: dict[str, int],
    *,
    supported_model_label_set: set[str],
    unsupported_label_policy: str,
) -> tuple[tuple[str, ...] | None, bool]:
    """Choose the first candidate supported by the current model labels.

    Args:
        eval_contract: Contract used for error context.
        eval_set_name: Eval-set name used for error context.
        eval_label: Eval label whose candidates are being selected.
        candidate_model_label_groups: Ordered candidate groups from the contract.
        model_label_to_id: Model label ids keyed by label name.
        supported_model_label_set: Labels enabled for this evaluation run.
        unsupported_label_policy: `"error"`, `"drop"`, or `"zero"` fallback.

    Returns:
        A pair of selected model labels and whether the label is unsupported-zero.
        The selected labels are `None` only when the policy drops the eval label.
    """

    for model_labels in candidate_model_label_groups:
        if all(
            model_label in model_label_to_id
            and model_label in supported_model_label_set
            for model_label in model_labels
        ):
            return model_labels, False

    if unsupported_label_policy == "drop":
        return None, False
    if unsupported_label_policy == "zero":
        for model_labels in candidate_model_label_groups:
            if all(model_label in model_label_to_id for model_label in model_labels):
                return model_labels, True
        return None, False

    missing_by_candidate = [
        sorted(
            model_label
            for model_label in model_labels
            if model_label not in model_label_to_id
        )
        for model_labels in candidate_model_label_groups
    ]
    unsupported_by_candidate = [
        sorted(
            model_label
            for model_label in model_labels
            if model_label in model_label_to_id
            and model_label not in supported_model_label_set
        )
        for model_labels in candidate_model_label_groups
    ]
    raise ValueError(
        f"Eval contract for model {eval_contract.model_name!r} and eval set "
        f"{eval_set_name!r} label {eval_label!r} has no supported candidate "
        "for the resolved model label space. unknown model labels by candidate: "
        f"{missing_by_candidate}; unsupported model labels by candidate: "
        f"{unsupported_by_candidate}."
    )


def _validate_protocol_background(protocol: ProtocolSpec) -> None:
    """Validate that a protocol puts background at channel 0.

    Args:
        protocol: Protocol label space to validate.

    Returns:
        None. Raises `ValueError` when the protocol is invalid.
    """

    if not protocol.labels or protocol.labels[0] != _BACKGROUND_LABEL:
        raise ValueError(
            f"Protocol {protocol.protocol_name!r} must put background at channel 0."
        )


def _validate_dataset_background(dataset_spec: DatasetSpec) -> None:
    """Validate that a dataset stores background at native id 0.

    Args:
        dataset_spec: Dataset spec to validate.

    Returns:
        None. Raises `ValueError` when the dataset is invalid.
    """

    if dataset_spec.stored_labels.get(0) != _BACKGROUND_LABEL:
        raise ValueError(
            f"Dataset {dataset_spec.dataset_name!r} must put 'background' at "
            "stored id 0."
        )


def _effective_label_aliases(
    dataset_spec: DatasetSpec,
    protocol_name: str,
) -> dict[str, str]:
    """Return aliases for a dataset/protocol pair.

    Args:
        dataset_spec: Dataset spec containing per-protocol aliases.
        protocol_name: Protocol name selecting the alias mapping.

    Returns:
        A copy of the native-label to protocol-label alias mapping.
    """

    return dict(dataset_spec.protocol_label_aliases.get(protocol_name, {}))


def _effective_drop_labels(
    dataset_spec: DatasetSpec,
    protocol_name: str,
) -> tuple[str, ...]:
    """Return dropped native labels for a dataset/protocol pair.

    Args:
        dataset_spec: Dataset spec containing per-protocol drops.
        protocol_name: Protocol name selecting the drop list.

    Returns:
        Native dataset labels that should compile to background.
    """

    return dataset_spec.protocol_drop_labels.get(protocol_name, ())
