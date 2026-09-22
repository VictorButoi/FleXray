"""Per-crop sampling weights for crop-backed CT training datasets.

Crop-backed CT ThunderDBs store, for every crop, the native ids of the
foreground labels it contains (``crop_foreground_label_ids``). Tempered inverse
label-frequency weighting oversamples crops that contain rare anatomy so the
DRR renderer sees every structure often enough, without diluting rare-label
crops by the common structures they co-occur with.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

_SCHEMES = ("uniform", "inverse_label_frequency")
_AGGREGATIONS = ("mean", "max")


@dataclass(frozen=True)
class SampleWeightingSpec:
    """Resolved ``sample_weighting`` configuration for one CT dataset.

    Attributes:
        scheme: ``"uniform"`` (plain shuffling) or ``"inverse_label_frequency"``.
        tau: Tempering exponent applied to inverse label frequencies; ``1.0`` is
            full inverse frequency and ``0.5`` is square-root sampling.
        class_aggregation: How a crop combines its per-class contributions:
            ``"mean"`` or ``"max"`` (``"max"`` keeps rare-class crops from being
            diluted by co-occurring common classes).
    """

    scheme: str
    tau: float = 1.0
    class_aggregation: str = "mean"

    @classmethod
    def parse(cls, value: str | Mapping[str, Any], *, dataset_name: str) -> "SampleWeightingSpec":
        """Parse a config value into a validated spec.

        Args:
            value: A scheme name, or a mapping with ``scheme`` plus optional
                ``tau`` and ``class_aggregation``.
            dataset_name: Dataset name used in assertion messages.

        Returns:
            The validated weighting spec.
        """

        cfg = {"scheme": value} if isinstance(value, str) else dict(value)
        unknown = sorted(set(cfg) - {"scheme", "tau", "class_aggregation"})
        assert not unknown, f"{dataset_name} sample_weighting has unknown key(s): {unknown}."
        scheme = cfg.get("scheme")
        assert scheme in _SCHEMES, (
            f"{dataset_name} sample_weighting scheme must be one of {_SCHEMES}; got {scheme!r}."
        )
        if scheme == "uniform":
            assert set(cfg) == {"scheme"}, (
                f"{dataset_name} sample_weighting tau/class_aggregation require "
                "scheme='inverse_label_frequency'."
            )
        tau = cfg.get("tau", 1.0)
        assert (
            isinstance(tau, (int, float))
            and not isinstance(tau, bool)
            and math.isfinite(tau)
            and tau > 0
        ), f"{dataset_name} sample_weighting tau must be a finite number > 0; got {tau!r}."
        aggregation = cfg.get("class_aggregation", "mean")
        assert aggregation in _AGGREGATIONS, (
            f"{dataset_name} sample_weighting class_aggregation must be one of "
            f"{_AGGREGATIONS}; got {aggregation!r}."
        )
        return cls(scheme=str(scheme), tau=float(tau), class_aggregation=str(aggregation))


def inverse_label_frequency_weights(
    fg_label_ids: Sequence[Sequence[int]],
    label_lut: Sequence[int],
    *,
    tau: float = 1.0,
    class_aggregation: str = "mean",
) -> Tensor:
    """Compute per-crop sampling weights that oversample rare model classes.

    Each foreground model class ``c`` contributes ``(N / n_c) ** tau`` where
    ``N`` is the crop count and ``n_c`` the number of crops containing ``c``.
    A crop's weight aggregates its classes' contributions; crops without
    supervised foreground get weight ``0``. Weights are normalized to mean 1.

    Args:
        fg_label_ids: Per-crop dataset-native foreground label ids.
        label_lut: Native-id to model-channel lookup table; entries ``<= 0``
            are background or dropped labels.
        tau: Tempering exponent in ``(0, inf)``.
        class_aggregation: ``"mean"`` or ``"max"`` over a crop's classes.

    Returns:
        Float tensor of shape ``(len(fg_label_ids),)`` with mean ``1.0``.
    """

    assert len(fg_label_ids) > 0, "Sample weighting requires at least one crop."
    assert class_aggregation in _AGGREGATIONS, class_aggregation
    lut = [int(v) for v in label_lut]
    classes_per_crop = []
    for native_ids in fg_label_ids:
        ids = [int(i) for i in native_ids]
        assert all(0 <= i < len(lut) for i in ids), (
            f"Crop native label ids {ids} fall outside the label LUT of size {len(lut)}."
        )
        classes_per_crop.append({lut[i] for i in ids if lut[i] > 0})
    crops_per_class = Counter(c for classes in classes_per_crop for c in classes)
    assert crops_per_class, "No crop contains supervised foreground under the active protocol."
    aggregate = max if class_aggregation == "max" else (lambda values: sum(values) / len(values))
    num_crops = float(len(fg_label_ids))
    weights = torch.zeros(len(fg_label_ids), dtype=torch.float64)
    for index, classes in enumerate(classes_per_crop):
        if classes:
            weights[index] = aggregate([(num_crops / crops_per_class[c]) ** tau for c in classes])
    return weights / weights.mean()
