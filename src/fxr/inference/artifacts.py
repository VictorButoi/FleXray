"""Shared helpers for Hugging Face FleXray inference artifacts."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml
from huggingface_hub import hf_hub_download

from fxr.models import UNet
from fxr.protocols import resolve_run_output_label_names

DEFAULT_MODEL_ID = "VictorButoi/flexray"
# The published repo keeps every bundle in a subfolder and describes them in
# ``ensemble.json`` (``flagship`` + ``members``). The tuple below mirrors that
# file for offline consumers (CLI help, the MCP registry); the loaders read the
# file itself. The flagship (FluXray proportion 0.375) comes first.
ENSEMBLE_MANIFEST_FILENAME = "ensemble.json"
FLAGSHIP_SUBFOLDER = "members/flux0375"
ENSEMBLE_MEMBER_SUBFOLDERS = (
    FLAGSHIP_SUBFOLDER,
    "members/flux000",
    "members/flux025",
    "members/flux050",
    "members/flux075",
)
WEIGHTS_FILENAME = "model.safetensors"
YAML_CONFIG_FILENAMES = ("config.yml", "config.yaml")
JSON_CONFIG_FILENAME = "config.json"
LABEL_SCHEMA_FILENAME = "label_schema.json"
PREPROCESSING_FILENAME = "preprocessing.json"
MODEL_CARD_FILENAME = "README.md"
CHECKSUMS_FILENAME = "checksums.json"

_PORTABLE_MODEL_BUILDERS: dict[str, type[torch.nn.Module]] = {
    "fxr.models.UNet": UNet,
}
_UNET_CONFIG_KEYS = frozenset(
    {
        "_class",
        "activation",
        "bottleneck_convs",
        "convs_per_block",
        "dropout",
        "filters",
        "in_channels",
        "norm",
        "norm_after_activation",
        "out_channels",
        "residual",
        "residual_projection",
        "residual_shortcut_norm",
        "skip_connections",
        "up_filters",
        "upsample_align_corners",
    }
)
_UNET_ACTIVATIONS = frozenset(
    {"elu", "gelu", "leakyrelu", "mish", "relu", "silu"}
)
_UNET_NORMS = frozenset({"none", "batch", "instance", "group", "layer"})


def load_model_config_file(path: str | Path) -> dict[str, Any]:
    """Load a model config from YAML or JSON.

    Args:
        path: Config file path ending in ``.yml``, ``.yaml``, or ``.json``.

    Returns:
        Parsed config mapping.

    Raises:
        TypeError: If the parsed document is not a mapping.
        ValueError: If the suffix is unsupported.
    """

    config_path = Path(path)
    suffix = config_path.suffix.lower()
    with config_path.open("r", encoding="utf-8") as handle:
        if suffix in {".yml", ".yaml"}:
            data = yaml.safe_load(handle)
        elif suffix == ".json":
            data = json.load(handle)
        else:
            raise ValueError(
                "Model config must be YAML or JSON; "
                f"got suffix {config_path.suffix!r}."
            )
    if not isinstance(data, Mapping):
        raise TypeError(f"Model config must parse as a mapping: {config_path}.")
    return dict(data)


def load_json_file(path: str | Path) -> dict[str, Any]:
    """Load a JSON artifact mapping.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON mapping.

    Raises:
        TypeError: If the parsed document is not a mapping.
    """

    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, Mapping):
        raise TypeError(f"JSON artifact must parse as a mapping: {json_path}.")
    return dict(data)


def download_bundle_file(
    repo_id: str,
    filename: str,
    *,
    subfolder: str | None = None,
    revision: str | None = None,
    required: bool = True,
) -> Path | None:
    """Resolve one bundle artifact from a model repository or local directory.

    A ``repo_id`` naming an existing directory is read in place, not downloaded.

    Args:
        repo_id: Hugging Face model repository id, or a local bundle directory.
        filename: Artifact filename relative to ``subfolder``.
        subfolder: Optional bundle directory inside the repository.
        revision: Optional branch, tag, or commit id; unused for directories.
        required: Whether a missing artifact is an error.

    Returns:
        Local cached path, or ``None`` when the artifact is optional and absent.

    Raises:
        FileNotFoundError: If a required artifact is missing.
    """

    local_bundle = Path(repo_id).expanduser()
    if local_bundle.is_dir():
        path = local_bundle / (subfolder or "") / filename
        if path.is_file():
            return path
        if not required:
            return None
        raise FileNotFoundError(
            f"Required artifact {filename!r} was not found in local model "
            f"bundle {bundle_location(repo_id, subfolder)!r}."
        )
    try:
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=subfolder or None,
                revision=revision,
            )
        )
    except Exception as exc:
        if not required:
            return None
        raise FileNotFoundError(
            f"Required Hugging Face artifact {filename!r} was not found in "
            f"model repo {bundle_location(repo_id, subfolder)!r}."
        ) from exc


def download_bundle_config(
    repo_id: str,
    *,
    subfolder: str | None = None,
    revision: str | None = None,
) -> Path:
    """Download the first supported model config file of one bundle.

    Args:
        repo_id: Hugging Face model repository id, or a local bundle directory.
        subfolder: Optional bundle directory inside the repository.
        revision: Optional branch, tag, or commit id.

    Returns:
        Local cached path of ``config.yml``, ``config.yaml``, or ``config.json``.

    Raises:
        FileNotFoundError: If the bundle has no supported config file.
    """

    for filename in (*YAML_CONFIG_FILENAMES, JSON_CONFIG_FILENAME):
        path = download_bundle_file(
            repo_id, filename, subfolder=subfolder, revision=revision, required=False
        )
        if path is not None:
            return path
    raise FileNotFoundError(
        f"Model repo {bundle_location(repo_id, subfolder)!r} must contain "
        "config.yml or config.json."
    )


def resolve_bundle_subfolders(
    repo_id: str,
    *,
    subfolder: str | None = None,
    ensemble: bool = False,
    revision: str | None = None,
) -> tuple[str | None, ...]:
    """Resolve which bundle directories of a repository to load.

    A repository that ships ``ensemble.json`` keeps its bundles in subfolders:
    the file names the ``flagship`` loaded by default and the ``members``
    loaded with ``ensemble=True``. A repository without that file is one bundle
    at its root. Resolving the flagship of an ensemble repository reports the
    chosen member on stderr, since the repository id alone does not name it.

    Args:
        repo_id: Hugging Face model repository id, or a local bundle directory.
        subfolder: Explicit bundle directory; skips ``ensemble.json``.
        ensemble: Whether to load every member declared by ``ensemble.json``.
        revision: Optional branch, tag, or commit id.

    Returns:
        Bundle subfolders to load in order; ``None`` means the repository root.

    Raises:
        FileNotFoundError: If ``ensemble`` is requested from a repository
            without ``ensemble.json``.
        ValueError: If ``subfolder`` and ``ensemble`` are combined, or the
            manifest is malformed.
    """

    if subfolder is not None and ensemble:
        raise ValueError("subfolder and ensemble=True are mutually exclusive.")
    if subfolder is not None:
        return (subfolder,)
    manifest_path = download_bundle_file(
        repo_id, ENSEMBLE_MANIFEST_FILENAME, revision=revision, required=False
    )
    if manifest_path is None:
        if ensemble:
            raise FileNotFoundError(
                f"Model repo {repo_id!r} has no {ENSEMBLE_MANIFEST_FILENAME}; "
                "ensemble=True needs a repository that declares its members."
            )
        return (None,)
    manifest = _validated_ensemble_manifest(load_json_file(manifest_path), repo_id)
    if ensemble:
        return tuple(member["subfolder"] for member in manifest["members"])
    flagship = manifest["flagship"]
    print(
        f"{repo_id} declares {len(manifest['members'])} ensemble members; "
        f"loading the flagship bundle {flagship}.",
        file=sys.stderr,
    )
    return (flagship,)


def bundle_location(repo_id: str, subfolder: str | None) -> str:
    """Return the human-readable ``repo_id[/subfolder]`` of one bundle."""

    return f"{repo_id}/{subfolder}" if subfolder else repo_id


def _validated_ensemble_manifest(
    manifest: Mapping[str, Any], repo_id: str
) -> dict[str, Any]:
    """Validate the ``flagship`` and ``members`` entries of ``ensemble.json``.

    Args:
        manifest: Parsed manifest mapping.
        repo_id: Repository id used in error messages.

    Returns:
        The manifest as a plain dict.

    Raises:
        ValueError: If ``flagship`` is not a non-empty string or ``members`` is
            not a non-empty list of unique ``{"subfolder": str}`` mappings.
    """

    flagship = manifest.get("flagship")
    members = manifest.get("members")
    where = f"{ENSEMBLE_MANIFEST_FILENAME} in {repo_id!r}"
    if not isinstance(flagship, str) or not flagship:
        raise ValueError(f"{where} must name a non-empty flagship subfolder.")
    if not isinstance(members, Sequence) or isinstance(members, str) or not members:
        raise ValueError(f"{where} must list at least one member.")
    subfolders = [
        member.get("subfolder") if isinstance(member, Mapping) else None
        for member in members
    ]
    if not all(isinstance(name, str) and name for name in subfolders):
        raise ValueError(f"{where} members must each declare a subfolder string.")
    if len(set(subfolders)) != len(subfolders):
        raise ValueError(f"{where} must not repeat a member subfolder.")
    return dict(manifest)


def resolve_bundle_label_names(
    config: Mapping[str, Any],
    label_schema: Mapping[str, Any] | None = None,
) -> tuple[str, ...] | None:
    """Resolve ordered output label names for an inference bundle.

    Args:
        config: Model or run config mapping.
        label_schema: Optional parsed ``label_schema.json`` mapping.

    Returns:
        Ordered label names, or ``None`` when neither schema nor protocol
        metadata declares them.

    Raises:
        ValueError: If the schema declares an empty or malformed label list.
    """

    if label_schema is not None:
        schema_labels = _labels_from_schema(label_schema)
        if schema_labels is not None:
            return schema_labels
    return resolve_run_output_label_names(config)


def normalized_model_config(
    config: Mapping[str, Any],
    *,
    label_names: Sequence[str] | None,
) -> dict[str, Any]:
    """Return the model constructor config with output channels injected.

    Args:
        config: A full config containing ``model`` or a direct model config.
        label_names: Ordered output label names used to set ``out_channels``.

    Returns:
        Model config mapping with runtime-only compile options removed.

    Raises:
        ValueError: If no model mapping can be found.
    """

    if "model" in config:
        raw_model = config["model"]
        if not isinstance(raw_model, Mapping):
            raise ValueError("Model config must define a mapping at key model.")
    else:
        raw_model = config
    model_cfg = dict(raw_model)
    _reject_executable_model_directives(model_cfg)
    model_cfg.pop("compile_cfg", None)
    if label_names is not None:
        model_cfg["out_channels"] = len(tuple(label_names))
    return model_cfg


def normalized_bundle_config(
    config: Mapping[str, Any],
    *,
    label_names: Sequence[str] | None,
) -> dict[str, Any]:
    """Return the portable config stored in a public model bundle.

    Args:
        config: Saved run or model config mapping.
        label_names: Optional ordered labels used to make the model head
            self-describing.

    Returns:
        Config containing the normalized ``model`` mapping and any public
        protocol declaration from the source config.
    """

    model_config = normalized_model_config(config, label_names=label_names)
    bundle: dict[str, Any] = {
        "model": _validated_portable_model_config(model_config)
    }
    protocol = config.get("protocol")
    if isinstance(protocol, Mapping):
        bundle["protocol"] = dict(protocol)
    return bundle


def build_model_from_config(
    config: Mapping[str, Any],
    *,
    label_names: Sequence[str] | None,
) -> torch.nn.Module:
    """Instantiate the PyTorch model described by an inference config.

    Args:
        config: Model or run config mapping.
        label_names: Optional output label names used to set ``out_channels``.

    Returns:
        Instantiated ``torch.nn.Module``.

    Raises:
        TypeError: If the config does not instantiate a PyTorch module.
        ValueError: If the model mapping is missing.
    """

    model_config = _validated_portable_model_config(
        normalized_model_config(config, label_names=label_names)
    )
    constructor_name = model_config.pop("_class")
    constructor = _PORTABLE_MODEL_BUILDERS[constructor_name]
    model = constructor(**model_config)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "Portable model constructor must return a torch.nn.Module, got "
            f"{type(model).__name__}."
        )
    return model


def _reject_executable_model_directives(config: Mapping[str, Any]) -> None:
    """Reject constructor directives below the portable model root.

    Args:
        config: Direct model configuration mapping to inspect recursively.

    Returns:
        ``None``.

    Raises:
        TypeError: If a model config key is not a string.
        ValueError: If ``_fn`` is present or ``_class`` is nested.
    """

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for raw_key, item in value.items():
                if not isinstance(raw_key, str):
                    raise TypeError(
                        "Portable model config keys must be strings; "
                        f"got {type(raw_key).__name__} at {'.'.join(path)}."
                    )
                item_path = (*path, raw_key)
                if raw_key == "_fn" or (
                    raw_key == "_class" and item_path != ("model", "_class")
                ):
                    raise ValueError(
                        "Portable model config rejects executable directive "
                        f"{'.'.join(item_path)!r}."
                    )
                visit(item, item_path)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, (*path, str(index)))

    visit(config, ("model",))


def _validated_portable_model_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and copy one allowlisted portable constructor config.

    Args:
        config: Direct model configuration with runtime options removed.

    Returns:
        Plain model config safe to pass to a known FleXray constructor.

    Raises:
        TypeError: If fields do not use their strict portable types.
        ValueError: If the constructor or any field is unsupported.
    """

    _reject_executable_model_directives(config)
    constructor_name = config.get("_class")
    if not isinstance(constructor_name, str) or not constructor_name:
        raise ValueError("Portable model config requires a non-empty _class.")
    if constructor_name not in _PORTABLE_MODEL_BUILDERS:
        supported = ", ".join(sorted(_PORTABLE_MODEL_BUILDERS))
        raise ValueError(
            f"Unsupported portable model constructor {constructor_name!r}; "
            f"supported constructors: {supported}."
        )
    model_config = dict(config)
    if constructor_name == "fxr.models.UNet":
        _validate_unet_model_config(model_config)
    return model_config


def _validate_unet_model_config(config: Mapping[str, Any]) -> None:
    """Validate the portable scalar and sequence schema for ``UNet``.

    Args:
        config: Direct ``fxr.models.UNet`` constructor config.

    Returns:
        ``None``.

    Raises:
        TypeError: If a field has the wrong portable type.
        ValueError: If fields are missing, unknown, or outside safe values.
    """

    unknown = sorted(set(config).difference(_UNET_CONFIG_KEYS))
    if unknown:
        raise ValueError(f"Unsupported fxr.models.UNet config keys: {unknown}.")
    required = ("in_channels", "out_channels", "filters")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"fxr.models.UNet config is missing keys: {missing}.")

    _positive_int(config["in_channels"], name="model.in_channels")
    _positive_int(config["out_channels"], name="model.out_channels")
    _positive_int_sequence(config["filters"], name="model.filters")
    if "up_filters" in config and config["up_filters"] is not None:
        _positive_int_sequence(config["up_filters"], name="model.up_filters")
    for key in ("convs_per_block", "bottleneck_convs"):
        if key in config and config[key] is not None:
            _positive_int(config[key], name=f"model.{key}")
    for key in (
        "norm_after_activation",
        "residual",
        "residual_shortcut_norm",
        "skip_connections",
        "upsample_align_corners",
    ):
        if key in config and not isinstance(config[key], bool):
            raise TypeError(f"model.{key} must be a bool.")
    if "dropout" in config:
        dropout = config["dropout"]
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise TypeError("model.dropout must be a finite number in [0, 1].")
        if not math.isfinite(float(dropout)) or not 0 <= float(dropout) <= 1:
            raise ValueError("model.dropout must be a finite number in [0, 1].")
    if "activation" in config:
        activation = config["activation"]
        if (
            not isinstance(activation, str)
            or activation.lower() not in _UNET_ACTIVATIONS
        ):
            raise ValueError(
                "model.activation must name an allowlisted activation: "
                f"{sorted(_UNET_ACTIVATIONS)}."
            )
    if "norm" in config and config["norm"] is not None:
        norm = config["norm"]
        if not isinstance(norm, str) or norm.lower() not in _UNET_NORMS:
            raise ValueError(
                "model.norm must be null or one of "
                f"{sorted(_UNET_NORMS)}."
            )
    if "residual_projection" in config and config["residual_projection"] not in {
        "auto",
        "always",
    }:
        raise ValueError("model.residual_projection must be 'auto' or 'always'.")


def _positive_int(value: Any, *, name: str) -> int:
    """Validate one strictly positive, non-boolean integer.

    Args:
        value: Candidate integer value.
        name: Config field name used in errors.

    Returns:
        Validated integer.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If ``value`` is not positive.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _positive_int_sequence(value: Any, *, name: str) -> tuple[int, ...]:
    """Validate a non-empty sequence of positive integers.

    Args:
        value: Candidate list or tuple.
        name: Config field name used in errors.

    Returns:
        Validated integer tuple.

    Raises:
        TypeError: If ``value`` is not a list or tuple.
        ValueError: If ``value`` is empty or contains invalid integers.
    """

    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list of positive integers.")
    if not value:
        raise ValueError(f"{name} must contain at least one integer.")
    return tuple(
        _positive_int(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )


def checkpoint_model_state(
    state: object,
    *,
    checkpoint_path: str | Path,
) -> Mapping[str, torch.Tensor]:
    """Extract a model state dict from a FleXray training checkpoint.

    Args:
        state: Object returned by ``torch.load``.
        checkpoint_path: Path used only for clear error messages.

    Returns:
        Mapping from state-dict keys to tensors.

    Raises:
        ValueError: If the checkpoint does not contain key ``model``.
        TypeError: If the model state is not a tensor mapping.
    """

    if not isinstance(state, Mapping) or "model" not in state:
        raise ValueError(
            "Checkpoint must contain a model state at key model: "
            f"{Path(checkpoint_path)}."
        )
    model_state = state["model"]
    if not isinstance(model_state, Mapping):
        raise TypeError(
            "Checkpoint key model must be a state-dict mapping, got "
            f"{type(model_state).__name__}."
        )
    for key, value in model_state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError("Checkpoint model state must map string keys to tensors.")
    return model_state


def _labels_from_schema(schema: Mapping[str, Any]) -> tuple[str, ...] | None:
    """Return labels from a schema mapping when present."""

    raw_labels = schema.get("label_names", schema.get("labels"))
    if raw_labels is None:
        return None
    if not isinstance(raw_labels, Sequence) or isinstance(raw_labels, (str, bytes)):
        raise ValueError("label_schema.json must define label_names as a list.")
    labels: list[str] = []
    for item in raw_labels:
        if isinstance(item, Mapping):
            name = item.get("name")
        else:
            name = item
        if not isinstance(name, str) or not name:
            raise ValueError("label_schema.json labels must be non-empty strings.")
        labels.append(name)
    if not labels:
        raise ValueError("label_schema.json must define at least one label.")
    return tuple(labels)


__all__ = [
    "CHECKSUMS_FILENAME",
    "DEFAULT_MODEL_ID",
    "ENSEMBLE_MANIFEST_FILENAME",
    "ENSEMBLE_MEMBER_SUBFOLDERS",
    "FLAGSHIP_SUBFOLDER",
    "JSON_CONFIG_FILENAME",
    "LABEL_SCHEMA_FILENAME",
    "MODEL_CARD_FILENAME",
    "PREPROCESSING_FILENAME",
    "WEIGHTS_FILENAME",
    "YAML_CONFIG_FILENAMES",
    "build_model_from_config",
    "bundle_location",
    "checkpoint_model_state",
    "download_bundle_config",
    "download_bundle_file",
    "load_json_file",
    "load_model_config_file",
    "normalized_bundle_config",
    "normalized_model_config",
    "resolve_bundle_label_names",
    "resolve_bundle_subfolders",
]
