"""Model-only initialization for new FleXray training runs.

Initialization is intentionally separate from checkpoint resume. A pretrained
source supplies only model parameters to a newly constructed experiment, so the
optimizer, scheduler, scaler, epoch, and global-step state all start fresh.
Trusted FleXray run checkpoints are supported explicitly because PyTorch
checkpoints use pickle; public or otherwise untrusted weights must use
``safetensors``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from fxr.inference.artifacts import (
    JSON_CONFIG_FILENAME,
    LABEL_SCHEMA_FILENAME,
    WEIGHTS_FILENAME,
    YAML_CONFIG_FILENAMES,
    checkpoint_model_state,
    download_bundle_config,
    download_bundle_file,
    load_json_file,
    load_model_config_file,
    normalized_model_config,
    resolve_bundle_label_names,
    resolve_bundle_subfolders,
)

_INITIALIZATION_KEYS = frozenset(
    {
        "kind",
        "source",
        "revision",
        "checkpoint",
        "allow_unverified_label_order",
        "replace_head",
        "freeze_backbone",
    }
)
_HEAD_PREFIX = "output_conv."


def initialize_model(
    model: torch.nn.Module,
    initialization: Mapping[str, Any] | None,
    *,
    expected_label_names: Sequence[str] | None,
    expected_model_config: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Strictly initialize ``model`` from a configured model-only source.

    With ``initialization.replace_head`` the source's ``output_conv`` head is
    skipped (so a model can be fine-tuned onto a different label protocol) and
    ``freeze_backbone`` leaves only that head trainable.

    Args:
        model: Newly constructed model whose parameters should be initialized.
        initialization: Top-level ``initialization`` config section, or
            ``None`` to keep the model's from-scratch initialization.
        expected_label_names: Ordered labels produced by the target model.
        expected_model_config: Target model constructor config. Runtime-only
            ``compile_cfg`` is ignored during metadata compatibility checks.

    Returns:
        JSON-serializable source provenance, or ``None`` when initialization is
        not configured.

    Raises:
        FileNotFoundError: If a configured local or remote artifact is missing.
        KeyError: If required initialization fields are absent.
        TypeError: If config or checkpoint structures are malformed.
        ValueError: If config keys, labels, model metadata, or state keys do not
            match the target model.
    """

    if initialization is None:
        return None
    config = _validate_initialization_config(initialization)
    kind = config["kind"]
    if kind == "pretrained":
        state, metadata, provenance = _load_pretrained(
            config["source"],
            revision=config.get("revision"),
            allow_unverified_label_order=config.get(
                "allow_unverified_label_order", False
            ),
        )
    else:
        state, metadata, provenance = _load_trusted_run(
            config["source"], checkpoint=config.get("checkpoint", "last")
        )

    replace_head = config.get("replace_head", False)
    _validate_metadata(
        metadata,
        expected_label_names=None if replace_head else expected_label_names,
        expected_model_config=expected_model_config,
        allow_unverified_label_order=(
            provenance.get("label_order_verified") is False
        ),
        ignore_out_channels=replace_head,
    )
    if replace_head:
        provenance = {
            **provenance,
            "replace_head": True,
            "source_out_channels": int(state[f"{_HEAD_PREFIX}weight"].shape[0]),
        }
        _load_state_except_head(model, state)
    else:
        model.load_state_dict(state, strict=True)
    if config.get("freeze_backbone", False):
        provenance = {**provenance, "freeze_backbone": True}
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(_HEAD_PREFIX))
    return provenance


def _load_state_except_head(model: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """Load every source tensor except the ``output_conv`` head, strictly.

    Args:
        model: Target model whose head keeps its fresh initialization.
        state: Source model state.

    Returns:
        ``None``.
    """

    target_state = model.state_dict()
    head_keys = {key for key in target_state if key.startswith(_HEAD_PREFIX)}
    assert head_keys, "initialization.replace_head requires a model with an output_conv head."
    backbone = {key: value for key, value in state.items() if not key.startswith(_HEAD_PREFIX)}
    missing = sorted(set(target_state) - head_keys - set(backbone))
    assert not missing, f"Initialization source lacks backbone tensor(s): {missing}."
    mismatched = sorted(
        key
        for key, value in backbone.items()
        if key in target_state and tuple(value.shape) != tuple(target_state[key].shape)
    )
    assert not mismatched, f"Initialization backbone shape mismatch for: {mismatched}."
    result = model.load_state_dict(backbone, strict=False)
    assert set(result.missing_keys) == head_keys and not result.unexpected_keys, (
        f"Unexpected state keys while replacing the head: {result.unexpected_keys}."
    )


def _validate_initialization_config(
    initialization: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize one initialization config section.

    Args:
        initialization: Raw initialization mapping.

    Returns:
        Normalized initialization mapping with strict scalar values.
    """

    if not isinstance(initialization, Mapping):
        raise TypeError("initialization must be a mapping.")
    unknown = sorted(set(initialization).difference(_INITIALIZATION_KEYS))
    if unknown:
        raise ValueError(f"initialization has unsupported keys: {unknown}.")

    kind = _required_text(initialization.get("kind"), "initialization.kind")
    if kind not in {"pretrained", "run"}:
        raise ValueError("initialization.kind must be 'pretrained' or 'run'.")
    source = _required_text(initialization.get("source"), "initialization.source")

    normalized: dict[str, Any] = {"kind": kind, "source": source}
    replace_head = initialization.get("replace_head", False)
    freeze_backbone = initialization.get("freeze_backbone", False)
    if not isinstance(replace_head, bool) or not isinstance(freeze_backbone, bool):
        raise TypeError("initialization.replace_head and freeze_backbone must be bools.")
    if freeze_backbone and not replace_head:
        raise ValueError("initialization.freeze_backbone requires replace_head=true.")
    if replace_head:
        normalized["replace_head"] = True
    if freeze_backbone:
        normalized["freeze_backbone"] = True
    allow_unverified = initialization.get("allow_unverified_label_order", False)
    if not isinstance(allow_unverified, bool):
        raise TypeError(
            "initialization.allow_unverified_label_order must be a bool."
        )
    if allow_unverified and kind != "pretrained":
        raise ValueError(
            "initialization.allow_unverified_label_order is valid only when "
            "kind='pretrained'."
        )
    if allow_unverified:
        normalized["allow_unverified_label_order"] = True
    revision = initialization.get("revision")
    checkpoint = initialization.get("checkpoint")
    if kind == "pretrained":
        if checkpoint is not None:
            raise ValueError(
                "initialization.checkpoint is only valid when kind='run'."
            )
        if revision is not None:
            normalized["revision"] = _required_text(
                revision, "initialization.revision"
            )
    else:
        if revision is not None:
            raise ValueError(
                "initialization.revision is only valid when kind='pretrained'."
            )
        if checkpoint is not None:
            normalized["checkpoint"] = _required_text(
                checkpoint, "initialization.checkpoint"
            )
    return normalized


def _load_pretrained(
    source: str,
    *,
    revision: str | None,
    allow_unverified_label_order: bool,
) -> tuple[
    Mapping[str, torch.Tensor],
    dict[str, Mapping[str, Any] | None],
    dict[str, Any],
]:
    """Load safetensors and optional compatibility metadata.

    Args:
        source: Local standalone file, local pretrained bundle, or Hugging Face
            repository id.
        revision: Optional Hugging Face revision.
        allow_unverified_label_order: Whether a standalone safetensors file may
            initialize weights despite having no ordered-label metadata.

    Returns:
        Tuple of model state, metadata mappings, and persisted provenance.
    """

    local = Path(source).expanduser()
    if local.is_file():
        if local.suffix.lower() != ".safetensors":
            raise ValueError(
                "A standalone pretrained file must have a .safetensors suffix."
            )
        if not allow_unverified_label_order:
            raise ValueError(
                "Standalone safetensors cannot verify model architecture or label "
                "order. Use a complete pretrained bundle, or explicitly set "
                "initialization.allow_unverified_label_order=true to accept the "
                "risk."
            )
        weights_path = local.resolve()
        metadata = {"config": None, "label_schema": None}
        provenance = {
            "kind": "pretrained",
            "source": str(weights_path),
            "artifact": "standalone_safetensors",
            "label_order_verified": False,
        }
    elif local.is_dir():
        if allow_unverified_label_order:
            raise ValueError(
                "initialization.allow_unverified_label_order applies only to a "
                "standalone .safetensors file; bundles must provide metadata."
            )
        bundle_dir = local.resolve()
        weights_path = bundle_dir / WEIGHTS_FILENAME
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Pretrained bundle is missing {WEIGHTS_FILENAME}: {bundle_dir}."
            )
        metadata = _load_local_bundle_metadata(bundle_dir)
        provenance = {
            "kind": "pretrained",
            "source": str(bundle_dir),
            "artifact": "local_bundle",
            "label_order_verified": True,
        }
    elif _looks_like_local_artifact(source):
        raise FileNotFoundError(f"Pretrained source does not exist: {local}.")
    else:
        if allow_unverified_label_order:
            raise ValueError(
                "initialization.allow_unverified_label_order applies only to a "
                "standalone .safetensors file; Hugging Face bundles must provide "
                "metadata."
            )
        subfolder = resolve_bundle_subfolders(source, revision=revision)[0]
        weights_path = download_bundle_file(
            source, WEIGHTS_FILENAME, subfolder=subfolder, revision=revision
        )
        metadata = _download_bundle_metadata(
            source, subfolder=subfolder, revision=revision
        )
        provenance = {
            "kind": "pretrained",
            "source": source,
            "artifact": "huggingface_bundle",
            "label_order_verified": True,
        }
        if subfolder is not None:
            provenance["subfolder"] = subfolder
        if revision is not None:
            provenance["revision"] = revision

    state = load_file(str(weights_path), device="cpu")
    return state, metadata, provenance


def _load_trusted_run(
    source: str,
    *,
    checkpoint: str,
) -> tuple[
    Mapping[str, torch.Tensor],
    dict[str, Mapping[str, Any] | None],
    dict[str, Any],
]:
    """Load only the model field from a trusted local run checkpoint.

    Args:
        source: FleXray run directory.
        checkpoint: Checkpoint stem or filename under ``checkpoints``.

    Returns:
        Tuple of model state, optional run metadata, and persisted provenance.
    """

    run_dir = Path(source).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Initialization run directory does not exist: {run_dir}."
        )
    checkpoint_name = _checkpoint_filename(checkpoint)
    checkpoint_path = run_dir / "checkpoints" / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Initialization checkpoint does not exist: {checkpoint_path}."
        )

    checkpoint_state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model_state = checkpoint_model_state(
        checkpoint_state, checkpoint_path=checkpoint_path
    )
    config_path = _first_existing_config(run_dir)
    metadata: dict[str, Mapping[str, Any] | None] = {
        "config": (
            load_model_config_file(config_path) if config_path is not None else None
        ),
        "label_schema": None,
    }
    provenance = {
        "kind": "run",
        "source": str(run_dir),
        "checkpoint": checkpoint_name,
        "artifact": "trusted_torch_checkpoint",
    }
    return model_state, metadata, provenance


def _validate_metadata(
    metadata: Mapping[str, Mapping[str, Any] | None],
    *,
    expected_label_names: Sequence[str] | None,
    expected_model_config: Mapping[str, Any],
    allow_unverified_label_order: bool,
    ignore_out_channels: bool = False,
) -> None:
    """Require exact target compatibility for all available metadata.

    Args:
        metadata: Optional source config and label-schema mappings.
        expected_label_names: Ordered labels expected by the target run.
        expected_model_config: Target model constructor config.
        allow_unverified_label_order: Whether missing source labels were
            explicitly accepted for standalone weights.
        ignore_out_channels: Whether ``out_channels`` may differ (head
            replacement).

    Returns:
        ``None``.
    """

    source_config = metadata.get("config")
    label_schema = metadata.get("label_schema")
    if (
        label_schema is not None
        and label_schema.get("label_order_verified") is False
    ):
        raise ValueError(
            "Initialization bundle marks its label order as unverified."
        )
    source_labels: tuple[str, ...] | None = None
    if source_config is not None or label_schema is not None:
        source_labels = resolve_bundle_label_names(source_config or {}, label_schema)
    target_labels = (
        None
        if expected_label_names is None
        else tuple(str(name) for name in expected_label_names)
    )
    if target_labels is not None:
        if source_labels is None and not allow_unverified_label_order:
            raise ValueError(
                "Initialization source does not provide ordered-label metadata "
                "for the target run."
            )
        if source_labels is not None and source_labels != target_labels:
            raise ValueError(
                "Initialization label order does not match the target run: "
                f"source={list(source_labels)!r}, target={list(target_labels)!r}."
            )

    if source_config is None:
        return
    source_model = _canonical_model_config(
        normalized_model_config(source_config, label_names=source_labels)
    )
    target_model = _canonical_model_config(expected_model_config)
    if ignore_out_channels:
        source_model.pop("out_channels", None)
        target_model.pop("out_channels", None)
    if source_model != target_model:
        raise ValueError(
            "Initialization model config does not exactly match the target model: "
            f"source={source_model!r}, target={target_model!r}."
        )


def _canonical_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep-copied constructor config without runtime compile options.

    Args:
        config: Full or direct model configuration mapping.

    Returns:
        Canonical direct model constructor config.
    """

    model_config = normalized_model_config(config, label_names=None)
    canonical = deepcopy(model_config)
    canonical.pop("compile_cfg", None)
    return canonical


def _load_local_bundle_metadata(
    bundle_dir: Path,
) -> dict[str, Mapping[str, Any] | None]:
    """Load required metadata files from a local pretrained bundle.

    Args:
        bundle_dir: Directory containing bundle artifacts.

    Returns:
        Parsed model config and ordered-label schema mappings.
    """

    config_path = _first_existing_config(bundle_dir)
    if config_path is None:
        expected = ", ".join((*YAML_CONFIG_FILENAMES, JSON_CONFIG_FILENAME))
        raise FileNotFoundError(
            f"Pretrained bundle is missing model config ({expected}): {bundle_dir}."
        )
    label_path = bundle_dir / LABEL_SCHEMA_FILENAME
    if not label_path.is_file():
        raise FileNotFoundError(
            f"Pretrained bundle is missing {LABEL_SCHEMA_FILENAME}: {bundle_dir}."
        )
    return {
        "config": load_model_config_file(config_path),
        "label_schema": load_json_file(label_path),
    }


def _download_bundle_metadata(
    repo_id: str,
    *,
    subfolder: str | None,
    revision: str | None,
) -> dict[str, Mapping[str, Any] | None]:
    """Download required compatibility metadata from a model repository.

    Args:
        repo_id: Hugging Face model repository id.
        subfolder: Bundle directory inside the repository, or ``None``.
        revision: Optional branch, tag, or commit.

    Returns:
        Parsed model config and ordered-label schema mappings.
    """

    config_path = download_bundle_config(
        repo_id, subfolder=subfolder, revision=revision
    )
    label_path = download_bundle_file(
        repo_id, LABEL_SCHEMA_FILENAME, subfolder=subfolder, revision=revision
    )
    return {
        "config": load_model_config_file(config_path),
        "label_schema": load_json_file(label_path),
    }


def _first_existing_config(root: Path) -> Path | None:
    """Return the first supported model config file under ``root``.

    Args:
        root: Bundle or run directory.

    Returns:
        Existing YAML/JSON config path, or ``None``.
    """

    for filename in (*YAML_CONFIG_FILENAMES, JSON_CONFIG_FILENAME):
        candidate = root / filename
        if candidate.is_file():
            return candidate
    return None


def _checkpoint_filename(checkpoint: str) -> str:
    """Normalize a trusted-run checkpoint stem into one safe filename.

    Args:
        checkpoint: Checkpoint stem or filename.

    Returns:
        Filename ending in ``.pt``.
    """

    name = _required_text(checkpoint, "initialization.checkpoint")
    path = Path(name)
    if path.name != name or name in {".", ".."}:
        raise ValueError(
            "initialization.checkpoint must be a filename under checkpoints/."
        )
    return name if path.suffix == ".pt" else f"{name}.pt"


def _looks_like_local_artifact(source: str) -> bool:
    """Return whether a missing source clearly names a local artifact path."""

    path = Path(source).expanduser()
    return (
        path.is_absolute()
        or source.startswith(("./", "../", "~"))
        or path.suffix.lower() == ".safetensors"
    )


def _required_text(value: Any, path: str) -> str:
    """Return a stripped required string value.

    Args:
        value: Value to validate.
        path: Config path used in validation errors.

    Returns:
        Non-empty stripped string.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string.")
    return value.strip()


__all__ = ["initialize_model"]
