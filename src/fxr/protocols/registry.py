from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path, PurePath

from .compiler import compile_eval_contract, compile_training_lut
from .io import (
    load_dataset_spec,
    load_eval_contract,
    load_model_label_space,
    load_protocol,
)
from .schemas import (
    CompiledEvalMapping,
    CompiledTrainingLut,
    DatasetSpec,
    EvalContract,
    ModelLabelSpace,
    ProtocolSpec,
)

_DEFAULT_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def load_protocol_by_name(
    name: str,
    config_root: str | Path | None = None,
) -> ProtocolSpec:
    """Load a named protocol config.

    Args:
        name: Protocol config name without file extension.
        config_root: Optional root containing `protocols/`, `datasets/`,
            `eval_contracts/`, and optional `models/`. Defaults to packaged
            `fxr.configs`.

    Returns:
        The requested `ProtocolSpec` after filename and declared name validation.
    """

    path, requested_name = _resolve_named_yaml(
        name,
        "protocols",
        config_root=config_root,
        kind="protocol",
    )
    protocol = load_protocol(path)
    _require_matching_name(
        loaded_name=protocol.protocol_name,
        requested_name=requested_name,
        kind="Protocol",
    )
    return protocol


def resolve_dataset_spec_name(
    name: str,
    config_root: str | Path | None = None,
) -> str:
    """Resolve a dataset spec name through YAML-declared aliases.

    Args:
        name: Dataset name or alias without file extension.
        config_root: Optional config root. Defaults to packaged `fxr.configs`.

    Returns:
        Canonical dataset config name.
    """

    requested_name = _trim_requested_name(name, context="dataset spec name")
    _reject_path_like_name(requested_name, context="dataset spec name")
    dataset_paths = _list_named_yaml_paths(
        "datasets",
        config_root=config_root,
        kind="dataset spec",
    )
    aliases = _load_dataset_spec_alias_index(dataset_paths)
    return aliases.get(requested_name, requested_name)


def load_dataset_spec_by_name(
    name: str,
    config_root: str | Path | None = None,
) -> DatasetSpec:
    """Load a named segmentation dataset config.

    Args:
        name: Dataset config name or declared alias without file extension.
        config_root: Optional config root. Defaults to packaged `fxr.configs`.

    Returns:
        The resolved `DatasetSpec` after alias and declared-name validation.
    """

    canonical_name = resolve_dataset_spec_name(name, config_root=config_root)
    path, requested_name = _resolve_named_yaml(
        canonical_name,
        "datasets",
        config_root=config_root,
        kind="dataset spec",
    )
    dataset_spec = load_dataset_spec(path)
    _require_matching_name(
        loaded_name=dataset_spec.dataset_name,
        requested_name=requested_name,
        kind="Dataset spec",
    )
    return dataset_spec


def load_model_label_space_by_name(
    name: str,
    config_root: str | Path | None = None,
) -> ModelLabelSpace:
    """Load a named model label space, falling back to protocol labels.

    Args:
        name: Model label-space config name without file extension.
        config_root: Optional config root. Defaults to packaged `fxr.configs`.

    Returns:
        A `ModelLabelSpace`. If `models/{name}.yml` is absent, the protocol with
        the same name is loaded and used as the model label space.
    """

    try:
        path, requested_name = _resolve_named_yaml(
            name,
            "models",
            config_root=config_root,
            kind="model label space",
        )
    except FileNotFoundError:
        protocol_path, requested_name = _resolve_named_yaml(
            name,
            "protocols",
            config_root=config_root,
            kind="protocol",
        )
        protocol = load_protocol(protocol_path)
        _require_matching_name(
            loaded_name=protocol.protocol_name,
            requested_name=requested_name,
            kind="Protocol",
        )
        return ModelLabelSpace(model_name=protocol.protocol_name, labels=protocol.labels)

    model_label_space = load_model_label_space(path)
    _require_matching_name(
        loaded_name=model_label_space.model_name,
        requested_name=requested_name,
        kind="Model label space",
    )
    return model_label_space


def load_eval_contract_by_model_name(
    model_name: str,
    config_root: str | Path | None = None,
) -> EvalContract:
    """Load a named eval contract.

    Args:
        model_name: Model name whose eval contract should be loaded.
        config_root: Optional config root. Defaults to packaged `fxr.configs`.

    Returns:
        The requested `EvalContract` after declared-name validation.
    """

    path, requested_name = _resolve_named_yaml(
        model_name,
        "eval_contracts",
        config_root=config_root,
        kind="eval contract",
    )
    eval_contract = load_eval_contract(path)
    _require_matching_name(
        loaded_name=eval_contract.model_name,
        requested_name=requested_name,
        kind="Eval contract",
    )
    return eval_contract


def compile_training_lut_by_name(
    protocol_name: str,
    dataset_name: str,
    config_root: str | Path | None = None,
) -> CompiledTrainingLut:
    """Load named configs and compile a training label LUT.

    Args:
        protocol_name: Protocol config name used as the target channel space.
        dataset_name: Dataset config name or alias used as the native source.
        config_root: Optional config root. Defaults to packaged `fxr.configs`.

    Returns:
        A `CompiledTrainingLut` from dataset-native ids to protocol channel ids.
    """

    protocol = load_protocol_by_name(protocol_name, config_root=config_root)
    dataset_spec = load_dataset_spec_by_name(dataset_name, config_root=config_root)
    return compile_training_lut(protocol, dataset_spec)


def compile_eval_contract_by_name(
    model_name: str,
    dataset_name: str,
    eval_set_name: str | None = None,
    model_label_names: tuple[str, ...] | list[str] | None = None,
    config_root: str | Path | None = None,
    unsupported_label_policy: str = "error",
    supported_model_label_names: Sequence[str] | None = None,
) -> CompiledEvalMapping:
    """Load named configs and compile an eval mapping.

    Args:
        model_name: Model label-space and eval-contract name.
        dataset_name: Dataset config name or alias used for identity defaults.
        eval_set_name: Optional eval-set name. Defaults to the resolved dataset
            name.
        model_label_names: Optional ad hoc model-label subset to compile against.
        config_root: Optional config root. Defaults to packaged `fxr.configs`.
        unsupported_label_policy: `"error"`, `"drop"`, or `"zero"` behavior for
            eval labels with no supported candidate.
        supported_model_label_names: Optional enabled subset of the resolved model
            label space.

    Returns:
        A `CompiledEvalMapping` from eval labels to model output channel groups.
    """

    try:
        identity_label_names = load_protocol_by_name(
            model_name,
            config_root=config_root,
        ).labels
    except FileNotFoundError:
        identity_label_names = None

    if model_label_names is None:
        model_label_space = load_model_label_space_by_name(
            model_name,
            config_root=config_root,
        )
    else:
        full_model_label_space = load_model_label_space_by_name(
            model_name,
            config_root=config_root,
        )
        if identity_label_names is None:
            identity_label_names = full_model_label_space.labels
        model_label_space = ModelLabelSpace(
            model_name=str(model_name),
            labels=tuple(str(label_name) for label_name in model_label_names),
        )

    eval_contract = load_eval_contract_by_model_name(
        model_name,
        config_root=config_root,
    )
    dataset_spec = load_dataset_spec_by_name(dataset_name, config_root=config_root)
    return compile_eval_contract(
        model_label_space,
        eval_contract,
        dataset_spec,
        eval_set_name=eval_set_name,
        unsupported_label_policy=unsupported_label_policy,
        supported_model_label_names=supported_model_label_names,
        strict_identity_defaults=model_label_names is not None,
        strict_identity_label_names=identity_label_names,
    )


def list_protocol_names(
    config_root: str | Path | None = None,
) -> tuple[str, ...]:
    """List the protocol config names loadable by `load_protocol_by_name`.

    Args:
        config_root: Optional root containing `protocols/`. Defaults to
            packaged `fxr.configs`.

    Returns:
        Sorted tuple of protocol config names without file extensions.
    """

    return tuple(
        sorted(
            _list_named_yaml_paths(
                "protocols",
                config_root=config_root,
                kind="protocol",
            )
        )
    )


def list_dataset_spec_names(
    config_root: str | Path | None = None,
) -> tuple[str, ...]:
    """List the dataset spec names loadable by `load_dataset_spec_by_name`.

    Only canonical config names are returned; declared aliases resolve through
    `resolve_dataset_spec_name`.

    Args:
        config_root: Optional root containing `datasets/`. Defaults to
            packaged `fxr.configs`.

    Returns:
        Sorted tuple of dataset spec config names without file extensions.
    """

    return tuple(
        sorted(
            _list_named_yaml_paths(
                "datasets",
                config_root=config_root,
                kind="dataset spec",
            )
        )
    )


def _list_named_yaml_paths(
    subdir: str,
    *,
    config_root: str | Path | None,
    kind: str,
) -> dict[str, Path]:
    """List unique YAML configs in a named config subdirectory.

    Args:
        subdir: Config subdirectory under the root.
        config_root: Optional root. Defaults to packaged `fxr.configs`.
        kind: Human-readable config kind for errors.

    Returns:
        Mapping from config stem to YAML path.
    """

    root = _DEFAULT_CONFIG_ROOT if config_root is None else Path(config_root)
    directory = root / subdir
    paths_by_name: dict[str, Path] = {}
    for path in sorted((*directory.glob("*.yml"), *directory.glob("*.yaml"))):
        name = path.stem
        existing = paths_by_name.get(name)
        if existing is not None:
            raise ValueError(
                f"Ambiguous FleXray {kind} {name!r}; "
                f"multiple files exist: {existing}, {path}."
            )
        paths_by_name[name] = path
    return paths_by_name


def _load_dataset_spec_alias_index(dataset_paths: dict[str, Path]) -> dict[str, str]:
    """Build dataset alias to canonical name index.

    Args:
        dataset_paths: Dataset config paths keyed by real config name.

    Returns:
        Mapping from alias to canonical dataset config name.
    """

    real_names = set(dataset_paths)
    alias_targets: dict[str, str] = {}
    for requested_name, path in sorted(dataset_paths.items()):
        dataset_spec = load_dataset_spec(path)
        _require_matching_name(
            loaded_name=dataset_spec.dataset_name,
            requested_name=requested_name,
            kind="Dataset spec",
        )
        for alias in dataset_spec.dataset_aliases:
            if alias in real_names:
                raise ValueError(
                    f"Dataset spec {dataset_spec.dataset_name!r} alias {alias!r} "
                    "conflicts with a real dataset spec name."
                )
            existing_target = alias_targets.get(alias)
            if existing_target is not None:
                raise ValueError(
                    f"Dataset spec alias {alias!r} is declared by both "
                    f"{existing_target!r} and {dataset_spec.dataset_name!r}."
                )
            alias_targets[alias] = dataset_spec.dataset_name
    return alias_targets


def _resolve_named_yaml(
    name: str,
    subdir: str,
    *,
    config_root: str | Path | None,
    kind: str,
) -> tuple[Path, str]:
    """Resolve a single named YAML config file.

    Args:
        name: Config name without file extension.
        subdir: Config subdirectory under the root.
        config_root: Optional root. Defaults to packaged `fxr.configs`.
        kind: Human-readable config kind for errors.

    Returns:
        Pair of resolved path and trimmed requested name.
    """

    requested_name = _trim_requested_name(name, context=f"{kind} name")
    _reject_path_like_name(requested_name, context=f"{kind} name")
    root = _DEFAULT_CONFIG_ROOT if config_root is None else Path(config_root)
    directory = root / subdir
    candidates = [
        directory / f"{requested_name}.yml",
        directory / f"{requested_name}.yaml",
    ]
    existing = [candidate for candidate in candidates if candidate.exists()]
    if not existing:
        expected = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f"Could not find FleXray {kind} {requested_name!r}; "
            f"expected one of: {expected}."
        )
    if len(existing) > 1:
        matches = ", ".join(str(path) for path in existing)
        raise ValueError(
            f"Ambiguous FleXray {kind} {requested_name!r}; "
            f"multiple files exist: {matches}."
        )
    return existing[0], requested_name


def _trim_requested_name(raw_name: object, *, context: str) -> str:
    """Validate and trim a registry lookup name.

    Args:
        raw_name: Requested config name.
        context: Human-readable location for validation errors.

    Returns:
        Trimmed non-empty config name.
    """

    if not isinstance(raw_name, str):
        raise TypeError(f"{context} must be a string, got {raw_name!r}.")
    name = raw_name.strip()
    if not name:
        raise ValueError(f"{context} cannot be empty.")
    return name


def _reject_path_like_name(name: str, *, context: str) -> None:
    """Reject path-like config lookup names.

    Args:
        name: Trimmed config name.
        context: Human-readable location for validation errors.

    Returns:
        None. Raises `ValueError` for path separators or dot names.
    """

    path = PurePath(name)
    if path.name != name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"{context} must be a config name, got {name!r}.")


def _require_matching_name(
    *,
    loaded_name: str,
    requested_name: str,
    kind: str,
) -> None:
    """Validate that a loaded config declares the requested name.

    Args:
        loaded_name: Name declared inside the YAML config.
        requested_name: Name inferred from the requested filename.
        kind: Human-readable config kind for errors.

    Returns:
        None. Raises `ValueError` on mismatch.
    """

    if loaded_name != requested_name:
        raise ValueError(
            f"{kind} name {loaded_name!r} does not match requested name "
            f"{requested_name!r}."
        )
