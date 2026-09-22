from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from importlib import import_module
from typing import Any

import kornia.augmentation as KA
import torch
import torch.nn as nn


class SegmentationAugmentationPipeline(nn.Module):
    """Centralized train-time 2D segmentation augmentation pipeline.

    Attributes:
        label_aware_ops: Ordered modules that inspect both image and label
            tensors while sampling augmentation parameters.
        kornia_pipeline: Optional Kornia sequence containing remaining
            image-first transforms that are applied to image and mask data keys.
    """

    def __init__(
        self,
        label_aware_ops: Iterable[nn.Module] = (),
        kornia_pipeline: KA.AugmentationSequential | None = None,
    ) -> None:
        """Store label-aware and Kornia augmentation stages.

        Args:
            label_aware_ops: Modules called as ``op(image, label)`` before the
                Kornia sequence.
            kornia_pipeline: Optional ``AugmentationSequential`` for standard
                image and mask transforms.

        Returns:
            ``None``.
        """

        super().__init__()
        self.label_aware_ops = nn.ModuleList(label_aware_ops)
        self.kornia_pipeline = kornia_pipeline

    def forward(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the augmentation pipeline to a 4D image and label batch.

        Args:
            image: Image tensor shaped ``(N, C, H, W)``.
            label: Label tensor shaped ``(N, C_label, H, W)``.

        Returns:
            Augmented image and label tensors.
        """

        return apply_segmentation_augmentation(self, image, label)


def build_segmentation_augmentation_pipeline(
    transform_map: Mapping[str, Mapping[str, Any]] | None,
) -> SegmentationAugmentationPipeline:
    """Build the centralized train-time segmentation augmentation pipeline.

    Args:
        transform_map: Ordered mapping from transform name to constructor
            configuration. Each config may set ``module`` to
            ``fxr.augmentation`` or ``kornia.augmentation``.

    Returns:
        Pipeline with label-aware transforms split ahead of the Kornia sequence.

    Raises:
        TypeError: If ``transform_map`` or any transform config has the wrong
            type.
        ValueError: If a transform name or module declaration is invalid.
        AttributeError: If a declared transform class does not exist.
    """

    if transform_map is None:
        transform_map = {}
    if not isinstance(transform_map, Mapping):
        raise TypeError(
            "Segmentation augmentation transform_map must be a mapping or None, "
            f"got {type(transform_map).__name__}."
        )

    ops: list[nn.Module] = []
    for raw_name, raw_config in transform_map.items():
        name = _normalize_transform_name(raw_name)
        config = _normalize_transform_config(raw_config, transform_name=name)
        transform_class, kwargs = _resolve_transform(name, config)
        op = transform_class(**kwargs)
        if not isinstance(op, nn.Module):
            raise TypeError(f"Transform {name!r} did not create a torch.nn.Module.")
        ops.append(op)

    label_aware_ops = [op for op in ops if getattr(op, "label_aware", False)]
    kornia_ops = [op for op in ops if not getattr(op, "label_aware", False)]
    kornia_pipeline = None
    if kornia_ops:
        kornia_pipeline = KA.AugmentationSequential(
            *kornia_ops,
            data_keys=["input", "mask"],
            keepdim=True,
            same_on_batch=False,
        )
    return SegmentationAugmentationPipeline(
        label_aware_ops=label_aware_ops,
        kornia_pipeline=kornia_pipeline,
    )


def apply_segmentation_augmentation(
    pipeline: SegmentationAugmentationPipeline,
    image: torch.Tensor,
    label: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a segmentation augmentation pipeline to one 4D batch.

    Args:
        pipeline: Pipeline returned by
            ``build_segmentation_augmentation_pipeline``.
        image: Image tensor shaped ``(N, C, H, W)``.
        label: Label tensor shaped ``(N, C_label, H, W)``.

    Returns:
        Pair of augmented image and label tensors.

    Raises:
        TypeError: If ``pipeline`` is not a
            ``SegmentationAugmentationPipeline``.
        ValueError: If image or label tensors are not 4D or their batch/spatial
            dimensions do not align.
    """

    if not isinstance(pipeline, SegmentationAugmentationPipeline):
        raise TypeError(
            "pipeline must be a SegmentationAugmentationPipeline, "
            f"got {type(pipeline).__name__}."
        )
    _validate_image_label_batch(image, label)

    augmented_image = image
    augmented_label = label
    for op in pipeline.label_aware_ops:
        augmented_image, augmented_label = op(augmented_image, augmented_label)

    if pipeline.kornia_pipeline is not None:
        augmented_image, augmented_label = pipeline.kornia_pipeline(
            augmented_image,
            augmented_label,
        )
    return augmented_image, restore_hard_background(augmented_label)


def restore_hard_background(label: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Re-harden a multichannel label and rebuild its background channel.

    Interpolating spatial transforms leave fractional mask values and a
    background channel that no longer complements the foreground; this restores
    the hard binary contract that FleXray labels carry into the loss.

    Args:
        label: Label batch shaped ``(N, C, H, W)``; channel 0 is background.
        threshold: Foreground values above this become ``1``.

    Returns:
        Binary label batch with channel 0 equal to the foreground complement.
        Single-channel labels are returned unchanged.
    """

    if label.shape[1] <= 1:
        return label
    foreground = (label[:, 1:] > threshold).to(label.dtype)
    background = (foreground.amax(dim=1, keepdim=True) <= 0).to(label.dtype)
    return torch.cat((background, foreground), dim=1)


def _validate_image_label_batch(image: torch.Tensor, label: torch.Tensor) -> None:
    """Validate 4D image and label tensors for segmentation augmentation.

    Args:
        image: Candidate image tensor.
        label: Candidate label tensor.

    Returns:
        ``None``.

    Raises:
        ValueError: If either tensor is not 4D or their batch/spatial axes do
            not match.
    """

    if image.ndim != 4:
        raise ValueError(
            f"image must be a 4D (N, C, H, W) tensor, got shape {tuple(image.shape)}."
        )
    if label.ndim != 4:
        raise ValueError(
            f"label must be a 4D (N, C, H, W) tensor, got shape {tuple(label.shape)}."
        )
    if image.shape[0] != label.shape[0] or image.shape[-2:] != label.shape[-2:]:
        raise ValueError(
            "image and label must have aligned batch and spatial dimensions, "
            f"got image={tuple(image.shape)} and label={tuple(label.shape)}."
        )


def _normalize_transform_name(raw_name: object) -> str:
    """Validate a transform name from a transform map.

    Args:
        raw_name: Mapping key expected to be a non-empty string.

    Returns:
        Trimmed transform name.

    Raises:
        TypeError: If ``raw_name`` is not a string.
        ValueError: If the normalized name is empty.
    """

    if not isinstance(raw_name, str):
        raise TypeError(f"Transform names must be strings, got {raw_name!r}.")
    name = raw_name.strip()
    if not name:
        raise ValueError("Transform names cannot be empty.")
    return name


def _normalize_transform_config(
    raw_config: object,
    *,
    transform_name: str,
) -> dict[str, Any]:
    """Validate and copy one transform constructor config.

    Args:
        raw_config: Raw transform config expected to be a mapping.
        transform_name: Transform name used in validation errors.

    Returns:
        Shallow config copy with YAML lists converted to tuples.

    Raises:
        TypeError: If ``raw_config`` is not a mapping.
    """

    if not isinstance(raw_config, Mapping):
        raise TypeError(
            f"Transform {transform_name!r} config must be a mapping, "
            f"got {type(raw_config).__name__}."
        )
    config = deepcopy(dict(raw_config))
    for key, value in tuple(config.items()):
        if isinstance(value, list):
            config[key] = tuple(value)
    return config


def _resolve_transform(
    name: str,
    config: Mapping[str, Any],
) -> tuple[type[nn.Module], dict[str, Any]]:
    """Resolve one configured transform class and keyword arguments.

    Args:
        name: Transform class name.
        config: Normalized transform config.

    Returns:
        Pair of module class and constructor kwargs.

    Raises:
        TypeError: If the ``module`` field is not a string.
        ValueError: If the module field points outside supported public
            augmentation modules.
        AttributeError: If the requested transform class is absent.
    """

    kwargs = dict(config)
    module_name = kwargs.pop("module", None)
    if module_name is None:
        module_name = (
            "fxr.augmentation"
            if name in _LOCAL_TRANSFORM_NAMES
            else "kornia.augmentation"
        )
    if not isinstance(module_name, str) or not module_name.strip():
        raise TypeError(f"Transform {name!r} module must be a non-empty string.")
    module_name = module_name.strip()
    if module_name not in {"fxr.augmentation", "kornia.augmentation"}:
        raise ValueError(
            f"Unsupported augmentation module {module_name!r} for transform {name!r}."
        )

    module = import_module(module_name)
    try:
        transform_class = getattr(module, name)
    except AttributeError as exc:
        raise AttributeError(
            f"Module {module_name!r} has no augmentation transform {name!r}."
        ) from exc
    if not isinstance(transform_class, type) or not issubclass(
        transform_class, nn.Module
    ):
        raise TypeError(
            f"Resolved augmentation transform {module_name}.{name} is not a "
            "torch.nn.Module class."
        )
    return transform_class, kwargs


_LOCAL_TRANSFORM_NAMES = {
    "RandomAspectCrop",
    "RandomClaheOrGamma",
    "RandomIntensityScale",
    "RandomLabelZoom",
    "RandomLetterDrop",
}
