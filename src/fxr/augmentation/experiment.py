"""Configurable per-sample input normalization for training experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import torch
from torch import Tensor

DEFAULT_NORMALIZATION_EPS = 1.0e-8
DEFAULT_NORMALIZATION_PERCENTILES = (0.5, 99.5)
NORMALIZATION_SCHEMES = frozenset({"minmax", "percentile_minmax", "standardize"})


class Standardize(torch.nn.Module):
    """Standardize each sample to zero mean and unit variance.

    Attributes:
        eps: Small constant added to the standard deviation for stability.
    """

    def __init__(self, eps: float = DEFAULT_NORMALIZATION_EPS) -> None:
        """Store the numerical-stability epsilon.

        Args:
            eps: Positive constant added to the per-sample standard deviation.

        Returns:
            ``None``.
        """

        super().__init__()
        self.eps = _validate_eps(eps)

    def forward(self, image: Tensor) -> Tensor:
        """Return ``image`` standardized independently per sample.

        Args:
            image: Tensor shaped ``(N, C, *spatial)``.

        Returns:
            Zero-mean, unit-variance tensor with the same shape as ``image``.
        """

        dims = tuple(range(1, image.ndim))
        mean = image.mean(dim=dims, keepdim=True)
        std = image.std(dim=dims, keepdim=True)
        return (image - mean) / (std + self.eps)


class MinMaxNormalize(torch.nn.Module):
    """Rescale each sample from its finite intensity range into ``[0, 1]``.

    Attributes:
        eps: Minimum dynamic range used to avoid division by zero.
    """

    def __init__(self, eps: float = DEFAULT_NORMALIZATION_EPS) -> None:
        """Store the minimum stable per-sample intensity range.

        Args:
            eps: Positive denominator floor used for constant-valued samples.

        Returns:
            ``None``.
        """

        super().__init__()
        self.eps = _validate_eps(eps)

    def forward(self, image: Tensor) -> Tensor:
        """Return a per-sample min-max rescaling of ``image``.

        Args:
            image: Tensor shaped ``(N, C, *spatial)``.

        Returns:
            Tensor with the same shape whose non-constant samples span
            ``[0, 1]``.
        """

        batch_size = image.shape[0]
        flat_image = image.reshape(batch_size, -1)
        image_min, image_max = flat_image.aminmax(dim=1)
        view_shape = [batch_size] + [1] * (image.ndim - 1)
        image_min = image_min.view(*view_shape)
        image_max = image_max.view(*view_shape)
        return (image - image_min) / (image_max - image_min).clamp(min=self.eps)


class PercentileMinMaxNormalize(torch.nn.Module):
    """Clip each sample to configured percentiles and rescale it to ``[0, 1]``.

    Attributes:
        percentiles: Ordered lower and upper percentiles in ``[0, 100]``.
        eps: Minimum clipped dynamic range used to avoid division by zero.
        quantiles: Non-persistent tensor form of ``percentiles`` used by Torch.
    """

    def __init__(
        self,
        percentiles: Sequence[float] = DEFAULT_NORMALIZATION_PERCENTILES,
        eps: float = DEFAULT_NORMALIZATION_EPS,
    ) -> None:
        """Store validated percentile and numerical-stability settings.

        Args:
            percentiles: Two ordered values ``(lower, upper)`` satisfying
                ``0 <= lower < upper <= 100``.
            eps: Positive denominator floor used for constant clipped samples.

        Returns:
            ``None``.
        """

        super().__init__()
        lower, upper = _validate_percentiles(percentiles)
        self.percentiles = (lower, upper)
        self.eps = _validate_eps(eps)
        self.register_buffer(
            "quantiles",
            torch.tensor([lower / 100.0, upper / 100.0], dtype=torch.float32),
            persistent=False,
        )

    def forward(self, image: Tensor) -> Tensor:
        """Return the per-sample percentile-clipped intensity rescaling.

        Args:
            image: Tensor shaped ``(N, C, *spatial)``.

        Returns:
            Tensor with the same shape and values clipped and scaled to
            ``[0, 1]``.
        """

        batch_size = image.shape[0]
        flat_image = image.reshape(batch_size, -1)
        quantile_input = (
            flat_image
            if flat_image.dtype in {torch.float32, torch.float64}
            else flat_image.float()
        )
        bounds = torch.quantile(
            quantile_input,
            self.quantiles.to(
                device=quantile_input.device,
                dtype=quantile_input.dtype,
            ),
            dim=1,
            interpolation="linear",
        ).to(dtype=image.dtype)

        view_shape = [batch_size] + [1] * (image.ndim - 1)
        lower = bounds[0].view(*view_shape)
        upper = bounds[1].view(*view_shape)
        clipped = image.clamp(min=lower, max=upper)
        return (clipped - lower) / (upper - lower).clamp(min=self.eps)


def resolve_input_normalization_config(config: Any) -> dict[str, Any]:
    """Resolve and validate the ``train.normalization`` config contract.

    Configs without the section retain legacy zero-mean/unit-variance standardization.
    When the section is present, its default scheme is percentile min-max with
    ``(0.5, 99.5)`` percentiles.

    Args:
        config: Full experiment config mapping, Config-like object, or ``None``.

    Returns:
        Plain mapping with canonical ``scheme``, ``percentiles``, and ``eps``.

    Raises:
        TypeError: If a config section or value has an invalid type.
        ValueError: If a scheme, percentile range, or epsilon is invalid.
    """

    raw_normalization: Any = None
    if config is not None:
        config_mapping = _plain_mapping(config, name="config")
        raw_train = config_mapping.get("train")
        if raw_train is not None:
            train_cfg = _plain_mapping(raw_train, name="train")
            raw_normalization = train_cfg.get("normalization")

    if raw_normalization is None:
        return {
            "scheme": "standardize",
            "percentiles": None,
            "eps": DEFAULT_NORMALIZATION_EPS,
        }

    normalization_cfg = _plain_mapping(
        raw_normalization, name="train.normalization"
    )
    unknown = sorted(set(normalization_cfg) - {"scheme", "percentiles", "eps"})
    if unknown:
        raise ValueError(f"Unexpected train.normalization keys: {unknown}.")

    scheme = normalization_cfg.get("scheme", "percentile_minmax")
    if not isinstance(scheme, str) or scheme not in NORMALIZATION_SCHEMES:
        expected = ", ".join(sorted(NORMALIZATION_SCHEMES))
        raise ValueError(
            f"train.normalization.scheme must be one of {expected}; got {scheme!r}."
        )

    eps = _validate_eps(normalization_cfg.get("eps", DEFAULT_NORMALIZATION_EPS))
    if scheme == "percentile_minmax":
        percentiles = _validate_percentiles(
            normalization_cfg.get(
                "percentiles", DEFAULT_NORMALIZATION_PERCENTILES
            )
        )
    else:
        raw_percentiles = normalization_cfg.get("percentiles")
        if raw_percentiles is not None:
            _validate_percentiles(raw_percentiles)
        percentiles = None

    return {"scheme": scheme, "percentiles": percentiles, "eps": eps}


def build_input_normalizer(config: Any) -> torch.nn.Module:
    """Build the normalizer selected by ``train.normalization``.

    Args:
        config: Full experiment config mapping or Config-like object.

    Returns:
        A :class:`Standardize` or :class:`PercentileMinMaxNormalize` module.
    """

    resolved = resolve_input_normalization_config(config)
    if resolved["scheme"] == "standardize":
        return Standardize(eps=resolved["eps"])
    if resolved["scheme"] == "minmax":
        return MinMaxNormalize(eps=resolved["eps"])
    return PercentileMinMaxNormalize(
        percentiles=resolved["percentiles"],
        eps=resolved["eps"],
    )


def _plain_mapping(value: Any, *, name: str) -> dict[str, Any]:
    """Return ``value`` as a plain mapping for config validation.

    Args:
        value: Mapping or Config-like object exposing ``to_dict``.
        name: Config path used in validation errors.

    Returns:
        Shallow plain-dictionary copy.

    Raises:
        TypeError: If ``value`` is not mapping-like.
    """

    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}.")
    return dict(value)


def _validate_percentiles(value: Any) -> tuple[float, float]:
    """Validate a lower/upper percentile pair.

    Args:
        value: Candidate two-value numeric sequence.

    Returns:
        Validated ``(lower, upper)`` floating-point pair.

    Raises:
        TypeError: If the value is not a numeric sequence.
        ValueError: If it is not a finite ordered percentile pair.
    """

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("train.normalization.percentiles must contain two numbers.")
    if len(value) != 2:
        raise ValueError("train.normalization.percentiles must contain two values.")
    if any(isinstance(item, bool) or not isinstance(item, Real) for item in value):
        raise TypeError("train.normalization.percentiles must contain two numbers.")

    lower, upper = (float(item) for item in value)
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("train.normalization.percentiles must be finite.")
    if not 0.0 <= lower < upper <= 100.0:
        raise ValueError(
            "train.normalization.percentiles must be ordered as "
            "0 <= lower < upper <= 100."
        )
    return lower, upper


def _validate_eps(value: Any) -> float:
    """Validate the normalization denominator floor.

    Args:
        value: Candidate positive finite numeric value.

    Returns:
        Validated floating-point epsilon.

    Raises:
        TypeError: If ``value`` is not numeric.
        ValueError: If ``value`` is non-finite or non-positive.
    """

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("train.normalization.eps must be a positive number.")
    eps = float(value)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("train.normalization.eps must be a positive finite number.")
    return eps
