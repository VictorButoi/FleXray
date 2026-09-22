from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage
from torch import Tensor

_LOGIT_EPS = 1e-6
_SHOULDERMONCH_MIN_RELATIVE_CLAVICLE_COMPONENT_SIZE = 0.05


@dataclass(frozen=True)
class InferencePostprocessResult:
    """Result of optional in-memory inference postprocessing.

    Attributes:
        logits: Logit tensor after postprocessing. It has the same shape as the
            input logits; only channels listed in metadata ``changed_label_names``
            may differ.
        probabilities: Probability tensor after postprocessing. It has the same
            shape as the input probabilities.
        metadata: Plain metadata describing the selected postprocessor, whether
            it applied, and any skip status or per-sample status details.
    """

    logits: Tensor
    probabilities: Tensor
    metadata: dict[str, object]


@dataclass(frozen=True)
class _PostprocessorSpec:
    """Static registration data for one dataset postprocessor.

    Attributes:
        name: Dataset name this postprocessor handles.
        required_label_names: Label names that must be present in caller-provided
            output label names before the postprocessor can run.
        changed_label_names: Label names whose logits and probabilities may be
            changed by the postprocessor.
        fn: Callable that applies the dataset-specific postprocessor.
    """

    name: str
    required_label_names: tuple[str, ...]
    changed_label_names: tuple[str, ...]
    fn: Callable[..., InferencePostprocessResult]


def _label_names_tuple(label_names: Sequence[str] | None) -> tuple[str, ...]:
    """Return label names as a stable tuple of strings."""

    if label_names is None:
        return ()
    return tuple(str(name) for name in label_names)


def _label_index_by_name(label_names: Sequence[str]) -> dict[str, int]:
    """Return a channel index lookup keyed by label name."""

    return {str(name): idx for idx, name in enumerate(label_names)}


def _base_metadata(
    *,
    dataset_name: str,
    spec: _PostprocessorSpec | None,
    status: str,
    applied: bool,
    detail: str | None = None,
) -> dict[str, object]:
    """Build the standard postprocessing metadata dictionary."""

    if spec is None:
        return {
            "dataset_name": str(dataset_name),
            "name": "identity",
            "registered": False,
            "applied": False,
            "status": "identity",
            "required_label_names": [],
            "changed_label_names": [],
        }

    metadata: dict[str, object] = {
        "dataset_name": str(dataset_name),
        "name": spec.name,
        "registered": True,
        "applied": bool(applied),
        "status": str(status),
        "required_label_names": list(spec.required_label_names),
        "changed_label_names": list(spec.changed_label_names),
    }
    if detail is not None:
        metadata["detail"] = str(detail)
    return metadata


def _identity_result(
    *,
    logits: Tensor,
    probabilities: Tensor,
    metadata: dict[str, object],
) -> InferencePostprocessResult:
    """Return unmodified tensors with the supplied metadata."""

    return InferencePostprocessResult(
        logits=logits,
        probabilities=probabilities,
        metadata=metadata,
    )


def _missing_required_labels(
    *,
    required_label_names: Sequence[str],
    label_names: Sequence[str],
) -> tuple[str, ...]:
    """Return required labels that are absent from the output label names."""

    present = set(label_names)
    return tuple(name for name in required_label_names if name not in present)


def _require_prediction_shape(*, logits: Tensor, probabilities: Tensor) -> None:
    """Validate batched 2D prediction tensor shape before postprocessing."""

    if logits.shape != probabilities.shape:
        raise ValueError(
            "Inference postprocessing requires logits and probabilities to share "
            f"the same shape; got logits={tuple(logits.shape)} and "
            f"probabilities={tuple(probabilities.shape)}."
        )
    if logits.ndim != 4:
        raise ValueError(
            "Inference postprocessing currently expects 2D batched tensors with "
            f"shape BxCxHxW, got {tuple(logits.shape)}."
        )


def _require_label_shape(
    *,
    label: Tensor,
    probabilities: Tensor,
    label_idx: int,
    dataset_name: str,
) -> None:
    """Validate a prepared channel-first label tensor for postprocessing."""

    if label.ndim != 4:
        raise ValueError(
            f"{dataset_name} postprocessing expects a prepared 2D label tensor "
            f"with shape BxCxHxW, got {tuple(label.shape)}."
        )
    if int(label.shape[0]) != int(probabilities.shape[0]) or tuple(
        int(dim) for dim in label.shape[2:]
    ) != tuple(int(dim) for dim in probabilities.shape[2:]):
        raise ValueError(
            f"{dataset_name} postprocessing label shape {tuple(label.shape)} does "
            f"not match prediction shape {tuple(probabilities.shape)}."
        )
    if int(label.shape[1]) <= int(label_idx):
        raise ValueError(
            f"{dataset_name} postprocessing expected GT channel {label_idx}, "
            f"but label shape is {tuple(label.shape)}."
        )


def _binary_channel_logits(probability_channel: Tensor) -> Tensor:
    """Convert edited binary-channel probabilities back to logits."""

    return torch.logit(probability_channel.clamp(min=_LOGIT_EPS, max=1.0 - _LOGIT_EPS))


def _distance_to_positive_mask(mask: np.ndarray) -> np.ndarray | None:
    """Return each pixel distance to the nearest positive mask pixel."""

    if not bool(mask.any()):
        return None
    return ndimage.distance_transform_edt(~mask.astype(bool))


def _hipray_postprocess(
    *,
    logits: Tensor,
    probabilities: Tensor,
    label: Tensor | None,
    label_names: tuple[str, ...],
    dataset_name: str,
    mode: str,
    threshold: float,
) -> InferencePostprocessResult:
    """Remove HipRay hip probabilities near thresholded femur predictions."""

    del label, mode
    spec = _POSTPROCESSORS["HipRay"]
    _require_prediction_shape(logits=logits, probabilities=probabilities)
    label_to_idx = _label_index_by_name(label_names)
    femur_idx = label_to_idx["femurs"]
    hip_idx = label_to_idx["hips"]

    cleaned_probs = probabilities.clone()
    hip_binary = (probabilities[:, hip_idx] > float(threshold)).detach().cpu().numpy()
    femur_binary = (
        (probabilities[:, femur_idx] > float(threshold)).detach().cpu().numpy()
    )
    cleaned_hips = np.zeros_like(hip_binary, dtype=np.float32)

    for batch_idx in range(int(probabilities.shape[0])):
        hip_mask = hip_binary[batch_idx].astype(bool)
        femur_mask = femur_binary[batch_idx].astype(bool)
        distance = _distance_to_positive_mask(femur_mask)
        if distance is None:
            cleaned_hips[batch_idx] = hip_mask.astype(np.float32)
            continue
        cleaned_hips[batch_idx] = (hip_mask & (distance > 4.0)).astype(np.float32)

    cleaned_probs[:, hip_idx] = torch.as_tensor(
        cleaned_hips,
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    cleaned_logits = logits.clone()
    cleaned_logits[:, hip_idx] = _binary_channel_logits(cleaned_probs[:, hip_idx])
    return InferencePostprocessResult(
        logits=cleaned_logits,
        probabilities=cleaned_probs,
        metadata=_base_metadata(
            dataset_name=dataset_name,
            spec=spec,
            status="applied",
            applied=True,
        ),
    )


def _mendeleycxr_postprocess(
    *,
    logits: Tensor,
    probabilities: Tensor,
    label: Tensor | None,
    label_names: tuple[str, ...],
    dataset_name: str,
    mode: str,
    threshold: float,
) -> InferencePostprocessResult:
    """Zero MendeleyCXR lung probabilities where liver is predicted."""

    del label, mode
    spec = _POSTPROCESSORS["MendeleyCXR"]
    _require_prediction_shape(logits=logits, probabilities=probabilities)
    label_to_idx = _label_index_by_name(label_names)
    liver_idx = label_to_idx["liver"]
    lungs_idx = label_to_idx["lungs"]

    cleaned_probs = probabilities.clone()
    liver_mask = probabilities[:, liver_idx] > float(threshold)
    cleaned_probs[:, lungs_idx] = torch.where(
        liver_mask,
        torch.zeros((), device=probabilities.device, dtype=probabilities.dtype),
        probabilities[:, lungs_idx],
    )
    cleaned_logits = logits.clone()
    cleaned_logits[:, lungs_idx] = _binary_channel_logits(cleaned_probs[:, lungs_idx])
    return InferencePostprocessResult(
        logits=cleaned_logits,
        probabilities=cleaned_probs,
        metadata=_base_metadata(
            dataset_name=dataset_name,
            spec=spec,
            status="applied",
            applied=True,
        ),
    )


def _largest_component_ids(
    components: np.ndarray,
    *,
    num_components: int,
    limit: int,
    min_relative_size: float = 0.0,
) -> list[int]:
    """Return largest connected-component ids after optional relative filtering."""

    if num_components <= 0:
        return []
    sizes = np.bincount(components.reshape(-1), minlength=num_components + 1)
    sizes[0] = 0
    component_ids = [idx for idx in range(1, num_components + 1) if sizes[idx] > 0]
    component_ids.sort(key=lambda idx: (-int(sizes[idx]), int(idx)))
    if component_ids and min_relative_size > 0.0:
        largest_size = int(sizes[component_ids[0]])
        min_size = max(1, int(np.ceil(float(largest_size) * float(min_relative_size))))
        component_ids = [idx for idx in component_ids if int(sizes[idx]) >= min_size]
    return component_ids[: int(limit)]


def _shouldermonch_postprocess(
    *,
    logits: Tensor,
    probabilities: Tensor,
    label: Tensor | None,
    label_names: tuple[str, ...],
    dataset_name: str,
    mode: str,
    threshold: float,
) -> InferencePostprocessResult:
    """Keep the ShoulderMonch clavicle component closest to GT humeri."""

    del mode
    spec = _POSTPROCESSORS["ShoulderMonch"]
    _require_prediction_shape(logits=logits, probabilities=probabilities)
    if label is None:
        return _identity_result(
            logits=logits,
            probabilities=probabilities,
            metadata=_base_metadata(
                dataset_name=dataset_name,
                spec=spec,
                status="skipped_missing_ground_truth",
                applied=False,
            ),
        )

    label_to_idx = _label_index_by_name(label_names)
    clavicle_idx = label_to_idx["clavicles"]
    humerus_idx = label_to_idx["humeri"]
    _require_label_shape(
        label=label,
        probabilities=probabilities,
        label_idx=humerus_idx,
        dataset_name=dataset_name,
    )

    clavicle_binary = (
        (probabilities[:, clavicle_idx] > float(threshold)).detach().cpu().numpy()
    )
    humerus_gt = (label[:, humerus_idx] > 0.5).detach().cpu().numpy()
    cleaned_probs = probabilities.clone()
    cleaned_logits = logits.clone()
    sample_statuses: list[str] = []
    skipped_batch_indices: list[int] = []
    structure = np.ones((3, 3), dtype=bool)

    for batch_idx in range(int(probabilities.shape[0])):
        humerus_mask = humerus_gt[batch_idx].astype(bool)
        if not bool(humerus_mask.any()):
            sample_statuses.append("skipped_missing_humeri_ground_truth")
            skipped_batch_indices.append(int(batch_idx))
            continue

        sample_statuses.append("applied")
        clavicle_mask = clavicle_binary[batch_idx].astype(bool)
        components, num_components = ndimage.label(clavicle_mask, structure=structure)
        candidate_ids = _largest_component_ids(
            components,
            num_components=int(num_components),
            limit=2,
            min_relative_size=_SHOULDERMONCH_MIN_RELATIVE_CLAVICLE_COMPONENT_SIZE,
        )
        cleaned_clavicle = np.zeros_like(clavicle_binary[batch_idx], dtype=np.float32)
        if not candidate_ids:
            cleaned_probs[batch_idx, clavicle_idx] = torch.as_tensor(
                cleaned_clavicle,
                device=probabilities.device,
                dtype=probabilities.dtype,
            )
            cleaned_logits[batch_idx, clavicle_idx] = _binary_channel_logits(
                cleaned_probs[batch_idx, clavicle_idx]
            )
            continue

        distance = _distance_to_positive_mask(humerus_mask)
        if distance is None:
            raise ValueError(
                "ShoulderMonch postprocessing requires positive GT humeri pixels."
            )
        ranked_candidates: list[tuple[float, int, int]] = []
        for rank, component_id in enumerate(candidate_ids):
            component_mask = components == int(component_id)
            min_distance = float(distance[component_mask].min())
            ranked_candidates.append((min_distance, rank, int(component_id)))
        _min_distance, _rank, selected_id = min(ranked_candidates)
        cleaned_clavicle = (components == selected_id).astype(np.float32)
        cleaned_probs[batch_idx, clavicle_idx] = torch.as_tensor(
            cleaned_clavicle,
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
        cleaned_logits[batch_idx, clavicle_idx] = _binary_channel_logits(
            cleaned_probs[batch_idx, clavicle_idx]
        )

    batch_size = int(probabilities.shape[0])
    skipped_count = len(skipped_batch_indices)
    applied_count = batch_size - skipped_count
    if applied_count == 0:
        status = "skipped_missing_humeri_ground_truth"
        applied = False
    elif skipped_count == 0:
        status = "applied"
        applied = True
    else:
        status = "partially_applied"
        applied = True

    metadata = _base_metadata(
        dataset_name=dataset_name,
        spec=spec,
        status=status,
        applied=applied,
        detail=(
            f"postprocessed {applied_count}/{batch_size} samples; "
            f"skipped {skipped_count} without GT humeri"
        ),
    )
    metadata["sample_statuses"] = sample_statuses
    metadata["skipped_batch_indices"] = skipped_batch_indices
    return InferencePostprocessResult(
        logits=cleaned_logits,
        probabilities=cleaned_probs,
        metadata=metadata,
    )


_POSTPROCESSORS: dict[str, _PostprocessorSpec] = {
    "HipRay": _PostprocessorSpec(
        name="HipRay",
        required_label_names=("femurs", "hips"),
        changed_label_names=("hips",),
        fn=_hipray_postprocess,
    ),
    "MendeleyCXR": _PostprocessorSpec(
        name="MendeleyCXR",
        required_label_names=("liver", "lungs"),
        changed_label_names=("lungs",),
        fn=_mendeleycxr_postprocess,
    ),
    "ShoulderMonch": _PostprocessorSpec(
        name="ShoulderMonch",
        required_label_names=("clavicles", "humeri"),
        changed_label_names=("clavicles",),
        fn=_shouldermonch_postprocess,
    ),
}


def get_inference_postprocessing_metadata(
    *,
    dataset_name: str,
    label_names: Sequence[str] | None,
) -> dict[str, object]:
    """Return static metadata for the dataset postprocessor selection.

    Args:
        dataset_name: Dataset name used to select a registered postprocessor.
        label_names: Ordered output label names for the logits/probabilities that
            will be postprocessed.

    Returns:
        Metadata describing whether a postprocessor is registered, whether the
        required labels are present, and which labels may be changed.
    """

    resolved_dataset_name = str(dataset_name)
    spec = _POSTPROCESSORS.get(resolved_dataset_name)
    if spec is None:
        return _base_metadata(
            dataset_name=resolved_dataset_name,
            spec=None,
            status="identity",
            applied=False,
        )

    resolved_label_names = _label_names_tuple(label_names)
    if not resolved_label_names:
        return _base_metadata(
            dataset_name=resolved_dataset_name,
            spec=spec,
            status="skipped_missing_label_names",
            applied=False,
        )
    missing = _missing_required_labels(
        required_label_names=spec.required_label_names,
        label_names=resolved_label_names,
    )
    if missing:
        return _base_metadata(
            dataset_name=resolved_dataset_name,
            spec=spec,
            status="skipped_missing_required_labels",
            applied=False,
            detail=f"missing labels: {', '.join(missing)}",
        )
    return _base_metadata(
        dataset_name=resolved_dataset_name,
        spec=spec,
        status="enabled",
        applied=True,
    )


def apply_inference_postprocessing(
    *,
    logits: Tensor,
    probabilities: Tensor,
    label: Tensor | None = None,
    label_names: Sequence[str] | None = None,
    dataset_name: str,
    mode: str = "binary",
    threshold: float = 0.5,
) -> InferencePostprocessResult:
    """Apply registered dataset postprocessing to in-memory predictions.

    Args:
        logits: Raw logits with shape ``BxCxHxW``.
        probabilities: Probabilities with the same shape as ``logits``.
        label: Optional prepared ground-truth label tensor with shape
            ``BxCxHxW``. It is required only by postprocessors that need ground
            truth, currently ``ShoulderMonch``.
        label_names: Ordered output label names corresponding to channel
            dimension ``C``.
        dataset_name: Dataset name used to select a registered postprocessor.
        mode: Inference probability mode supplied by callers for metadata
            compatibility. Current postprocessors operate on binary channels.
        threshold: Probability threshold used to form temporary binary masks.

    Returns:
        ``InferencePostprocessResult`` containing postprocessed tensors and
        metadata. Unknown datasets or missing required labels return identity
        tensors with skip metadata.
    """

    resolved_dataset_name = str(dataset_name)
    spec = _POSTPROCESSORS.get(resolved_dataset_name)
    if spec is None:
        return _identity_result(
            logits=logits,
            probabilities=probabilities,
            metadata=_base_metadata(
                dataset_name=resolved_dataset_name,
                spec=None,
                status="identity",
                applied=False,
            ),
        )

    resolved_label_names = _label_names_tuple(label_names)
    if not resolved_label_names:
        return _identity_result(
            logits=logits,
            probabilities=probabilities,
            metadata=_base_metadata(
                dataset_name=resolved_dataset_name,
                spec=spec,
                status="skipped_missing_label_names",
                applied=False,
            ),
        )

    missing = _missing_required_labels(
        required_label_names=spec.required_label_names,
        label_names=resolved_label_names,
    )
    if missing:
        return _identity_result(
            logits=logits,
            probabilities=probabilities,
            metadata=_base_metadata(
                dataset_name=resolved_dataset_name,
                spec=spec,
                status="skipped_missing_required_labels",
                applied=False,
                detail=f"missing labels: {', '.join(missing)}",
            ),
        )

    return spec.fn(
        logits=logits,
        probabilities=probabilities,
        label=label,
        label_names=resolved_label_names,
        dataset_name=resolved_dataset_name,
        mode=str(mode),
        threshold=float(threshold),
    )


__all__ = [
    "InferencePostprocessResult",
    "apply_inference_postprocessing",
    "get_inference_postprocessing_metadata",
]
