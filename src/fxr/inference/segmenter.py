"""High-level Hugging Face backed FleXray segmenter."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file
from torch import Tensor

from ._devices import resolve_device
from .artifacts import (
    DEFAULT_MODEL_ID,
    LABEL_SCHEMA_FILENAME,
    PREPROCESSING_FILENAME,
    WEIGHTS_FILENAME,
    build_model_from_config,
    bundle_location,
    download_bundle_config,
    download_bundle_file,
    load_json_file,
    load_model_config_file,
    resolve_bundle_label_names,
    resolve_bundle_subfolders,
)
from .preprocessing import ImagePreprocessing
from .probabilities import ProbabilityMode
from .runner import InferenceRunner


@dataclass(frozen=True)
class FleXrayPrediction:
    """Prediction tensors returned by :meth:`FleXraySegmenter.predict`.

    Attributes:
        logits: Raw model output with shape ``BxCxHxW`` on the segmenter device.
        probabilities: Probability tensor with shape ``BxCxHxW`` on the
            segmenter device.
        masks: Thresholded ``uint8`` tensor with shape ``BxCxHxW`` on the
            segmenter device.
    """

    logits: Tensor
    probabilities: Tensor
    masks: Tensor


class FleXraySegmenter:
    """High-level public inference API for pretrained FleXray models.

    Attributes:
        models: Loaded PyTorch segmentation models, one per ensemble member (a
            single-model segmenter has exactly one). Ensembles average member
            probabilities.
        model: The sole member for single-model segmenters; raises for ensembles.
        label_names: Ordered output channel names loaded from
            ``label_schema.json`` when available.
        preprocessing: Image preprocessing contract loaded from
            ``preprocessing.json``.
        probability_mode: Default logits-to-probability mode for prediction.
        repo_id: Hugging Face model repository id used for loading, or a tuple
            of member repository ids for ensembles.
        subfolders: Bundle directory loaded for each member (``None`` for a
            repository-root bundle), aligned with ``models``; ``None`` when the
            segmenter was not loaded from Hugging Face.
        revision: Optional Hugging Face revision used for loading.
        device: Device that owns the models and returned prediction tensors.
        _runner: Low-level inference runner sharing the loaded models.
    """

    def __init__(
        self,
        model: torch.nn.Module | Sequence[torch.nn.Module],
        *,
        label_names: tuple[str, ...] | None,
        preprocessing: ImagePreprocessing,
        probability_mode: ProbabilityMode | str = "multilabel",
        repo_id: str | Sequence[str] | None = None,
        subfolders: Sequence[str | None] | None = None,
        revision: str | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Create a segmenter around already-loaded model(s).

        Args:
            model: PyTorch segmentation model returning ``BxCxHxW`` logits, or a
                sequence of such models sharing one label space that are
                averaged as an ensemble.
            label_names: Optional ordered output channel names.
            preprocessing: Input preprocessing contract.
            probability_mode: Default mode passed to ``probabilities_from_logits``.
            repo_id: Optional model repository id(s) for provenance.
            subfolders: Optional per-member bundle directories for provenance.
            revision: Optional model revision for provenance.
            device: Target device. ``None`` keeps the models on their current
                device.

        Returns:
            None.
        """

        self.probability_mode = str(probability_mode)
        self._runner = InferenceRunner(
            model, probability_mode=self.probability_mode
        )
        self._runner._configure_tta_normalizer(preprocessing._normalize_tensor)
        target_device = None if device is None else resolve_device(device)
        for member in self._runner.models:
            if target_device is not None:
                member.to(target_device)
            member.eval()
        self.models = self._runner.models
        self.label_names = label_names
        self.preprocessing = preprocessing
        self.repo_id = (
            repo_id
            if repo_id is None or isinstance(repo_id, str)
            else tuple(repo_id)
        )
        self.subfolders = None if subfolders is None else tuple(subfolders)
        self.revision = revision
        self.device = _infer_segmenter_device(self.models[0])

    @property
    def model(self) -> torch.nn.Module:
        """Return the single loaded model.

        Returns:
            The sole member module.

        Raises:
            ValueError: If the segmenter is an ensemble; use ``models`` instead.
        """

        return self._runner.model

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str | Sequence[str] = DEFAULT_MODEL_ID,
        revision: str | None = None,
        device: torch.device | str | None = None,
        *,
        subfolder: str | None = None,
        ensemble: bool = False,
    ) -> "FleXraySegmenter":
        """Load a pretrained FleXray segmenter from Hugging Face artifacts.

        A repository that ships ``ensemble.json`` keeps its bundles in
        subfolders; by default its ``flagship`` is loaded, ``ensemble=True``
        loads every declared member, and ``subfolder`` picks one bundle
        explicitly. A repository without that file is a single bundle at its
        root. A ``repo_id`` naming an existing portable bundle directory is
        read from disk instead of downloaded.

        Args:
            repo_id: Hugging Face model repository id or local bundle
                directory, or a sequence of either loaded as an ensemble (every
                member must declare the same label schema, preprocessing, and
                probability mode). Defaults to ``"VictorButoi/flexray"``.
            revision: Optional Hugging Face branch, tag, or commit id applied to
                every member.
            device: Target device. ``None`` (default) and ``"auto"`` select
                CUDA when it is available and CPU otherwise, like
                ``flexify --device auto``.
            subfolder: Bundle directory inside a single ``repo_id``.
            ensemble: Load every member declared by ``ensemble.json`` of a
                single ``repo_id``.

        Returns:
            Loaded ``FleXraySegmenter``.

        Raises:
            FileNotFoundError: If a required model artifact is missing, or
                ``ensemble=True`` names a repository without ``ensemble.json``.
            TypeError: If JSON/config artifacts are malformed.
            ValueError: If model config, label schema, or preprocessing metadata
                is invalid, if ``repo_id`` is empty or repeats an id, if
                ``subfolder``/``ensemble`` are combined with each other or with
                several repository ids, or if ensemble members disagree on
                labels, preprocessing, or probability mode.
        """

        device = resolve_device(device)
        repo_ids = _normalize_repo_ids(repo_id)
        if (subfolder is not None or ensemble) and len(repo_ids) != 1:
            raise ValueError(
                "subfolder and ensemble=True apply to a single repo_id; got "
                f"{repo_ids!r}."
            )
        members = [
            (member_repo, member_subfolder)
            for member_repo in repo_ids
            for member_subfolder in resolve_bundle_subfolders(
                member_repo,
                subfolder=subfolder,
                ensemble=ensemble,
                revision=revision,
            )
        ]
        bundles = [
            _load_member_bundle(member_repo, subfolder=member_subfolder, revision=revision)
            for member_repo, member_subfolder in members
        ]
        locations = [bundle_location(*member) for member in members]
        _require_matching_bundles(locations, bundles)
        first = bundles[0]
        return cls(
            [bundle.model for bundle in bundles],
            label_names=first.label_names,
            preprocessing=first.preprocessing,
            probability_mode=first.probability_mode,
            repo_id=repo_id if isinstance(repo_id, str) else repo_ids,
            subfolders=[member_subfolder for _, member_subfolder in members],
            revision=revision,
            device=device,
        )

    def preprocess(self, image: str | Path | Any) -> Tensor:
        """Prepare an image-like input for the loaded model.

        Args:
            image: Image path, Pillow image, NumPy array, or tensor accepted by
                ``ImagePreprocessing.prepare``.

        Returns:
            ``float32`` tensor shaped ``BxCxHxW`` on the segmenter device.
        """

        return self.preprocessing.prepare(image).to(self.device)

    def predict(
        self,
        image: str | Path | Any,
        *,
        label: str | None = None,
        threshold: float = 0.5,
        tta_samples: int = 1,
        seed: int | None = None,
    ) -> FleXrayPrediction:
        """Run segmentation on one image-like input.

        Args:
            image: Image path, Pillow image, NumPy array, or tensor.
            label: Optional output label name. ``None`` returns every model
                channel. A label name restricts logits, probabilities, and
                masks to that single channel (``Bx1xHxW``).
            threshold: Probability threshold used to produce ``masks``.
            tta_samples: Total number of test-time-augmentation forward passes
                per model. Values ``<= 1`` disable TTA. Larger values average
                probabilities over one plain pass plus ``tta_samples - 1``
                augmented passes (see ``fxr.inference.tta``). Ensembles run
                every pass through each member and average all of them.
            seed: Optional seed for the test-time-augmentation draws. ``None``
                draws from the global torch RNG, so repeated runs differ.

        Returns:
            ``FleXrayPrediction`` with logits, probabilities, and masks.

        Raises:
            ValueError: If threshold is outside ``[0, 1]``, ``tta_samples`` is
                negative, or ``label`` is unknown or unavailable.
        """

        _validate_threshold(threshold)
        channel = self._label_channel(label) if label is not None else None
        images = self.preprocess(image)
        result = self._runner.predict(images, tta_samples=tta_samples, seed=seed)
        logits = result.logits
        probabilities = result.probabilities
        if channel is not None:
            logits = logits[:, channel : channel + 1]
            probabilities = probabilities[:, channel : channel + 1]
        masks = (probabilities >= float(threshold)).to(torch.uint8)
        return FleXrayPrediction(
            logits=logits,
            probabilities=probabilities,
            masks=masks,
        )

    def _label_channel(self, label: str) -> int:
        """Resolve one output label name to its channel index.

        Args:
            label: Output label name from the bundle label schema.

        Returns:
            Zero-based channel index of ``label``.

        Raises:
            ValueError: If the bundle has no label schema or ``label`` is not in
                it.
        """

        if self.label_names is None:
            raise ValueError(
                "label selection requires label_schema.json with ordered "
                "label_names."
            )
        if label not in self.label_names:
            raise ValueError(
                f"Unknown label {label!r}; expected one of {list(self.label_names)}."
            )
        return self.label_names.index(label)


@dataclass(frozen=True)
class _MemberBundle:
    """One loaded Hugging Face bundle awaiting ensemble assembly.

    Attributes:
        model: Model with bundle weights loaded on CPU.
        label_names: Ordered output channel names, or ``None``.
        preprocessing: Bundle preprocessing contract.
        probability_mode: Bundle logits-to-probability mode.
    """

    model: torch.nn.Module
    label_names: tuple[str, ...] | None
    preprocessing: ImagePreprocessing
    probability_mode: str


def _load_member_bundle(
    repo_id: str, *, subfolder: str | None, revision: str | None
) -> _MemberBundle:
    """Download one bundle's artifacts and instantiate its model on CPU.

    Args:
        repo_id: Hugging Face model repository id.
        subfolder: Bundle directory inside the repository, or ``None`` for the
            repository root.
        revision: Optional Hugging Face revision.

    Returns:
        Loaded ``_MemberBundle``.

    Raises:
        FileNotFoundError: If a required model artifact is missing.
        RuntimeError: If the weights file cannot be read, which a partial
            download or a damaged cache entry causes.
        TypeError: If JSON/config artifacts are malformed.
        ValueError: If model config, label schema, or preprocessing metadata
            is invalid.
    """

    def download(filename: str) -> Path:
        return download_bundle_file(
            repo_id, filename, subfolder=subfolder, revision=revision
        )

    config_path = download_bundle_config(
        repo_id, subfolder=subfolder, revision=revision
    )
    weights_path = download(WEIGHTS_FILENAME)
    label_schema_path = download(LABEL_SCHEMA_FILENAME)
    preprocessing_path = download(PREPROCESSING_FILENAME)

    config = load_model_config_file(config_path)
    label_schema = load_json_file(label_schema_path)
    preprocessing_metadata = load_json_file(preprocessing_path)
    label_names = resolve_bundle_label_names(config, label_schema)
    model = build_model_from_config(config, label_names=label_names)
    try:
        weights = load_file(str(weights_path), device="cpu")
    except (OSError, SafetensorError) as exc:
        recovery = (
            "Restore the local model.safetensors file from the original bundle."
            if Path(repo_id).expanduser().is_dir()
            else "Re-download this model into a fresh Hugging Face cache."
        )
        raise RuntimeError(
            f"Failed to read model weights {weights_path}: {exc}. {recovery}"
        ) from exc
    model.load_state_dict(weights)
    return _MemberBundle(
        model=model,
        label_names=label_names,
        preprocessing=ImagePreprocessing.from_metadata(preprocessing_metadata),
        probability_mode=str(preprocessing_metadata.get("probability_mode", "multilabel")),
    )


def _normalize_repo_ids(repo_id: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize one repo id or a sequence of ids to a non-empty unique tuple.

    Args:
        repo_id: Single repository id or sequence of member ids.

    Returns:
        Tuple of repository ids in the given order.

    Raises:
        ValueError: If the sequence is empty, contains a non-string, or repeats
            an id (which would silently double-weight that member).
    """

    repo_ids = (repo_id,) if isinstance(repo_id, str) else tuple(repo_id)
    if not repo_ids:
        raise ValueError("repo_id must name at least one Hugging Face model repository.")
    if not all(isinstance(member_id, str) for member_id in repo_ids):
        raise ValueError(f"repo_id entries must be strings; got {repo_ids!r}.")
    if len(set(repo_ids)) != len(repo_ids):
        raise ValueError(f"repo_id must not repeat a repository; got {repo_ids!r}.")
    return repo_ids


def _require_matching_bundles(
    locations: Sequence[str],
    bundles: Sequence[_MemberBundle],
) -> None:
    """Require every member bundle to share the first member's contract.

    Args:
        locations: Member ``repo_id[/subfolder]`` names aligned with ``bundles``.
        bundles: Loaded member bundles.

    Returns:
        None.

    Raises:
        ValueError: If a member's label names, preprocessing, or probability
            mode differ from the first member's.
    """

    reference = bundles[0]
    for location, bundle in zip(locations[1:], bundles[1:]):
        mismatches = {
            "label_names": bundle.label_names != reference.label_names,
            "preprocessing": bundle.preprocessing.to_metadata()
            != reference.preprocessing.to_metadata(),
            "probability_mode": bundle.probability_mode != reference.probability_mode,
        }
        differing = [name for name, differs in mismatches.items() if differs]
        if differing:
            raise ValueError(
                f"Ensemble member {location!r} differs from {locations[0]!r} in "
                f"{differing}; members must share one label space and preprocessing."
            )


def _infer_segmenter_device(model: torch.nn.Module) -> torch.device:
    """Infer the current model device, defaulting to CPU for stateless modules."""

    for parameter in model.parameters():
        return parameter.device
    for buffer in model.buffers():
        return buffer.device
    return torch.device("cpu")


def _validate_threshold(threshold: float) -> None:
    """Validate a probability threshold."""

    value = float(threshold)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(
            f"threshold must be a finite value in [0, 1], got {threshold!r}."
        )


__all__ = [
    "FleXrayPrediction",
    "FleXraySegmenter",
]
