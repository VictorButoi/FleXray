"""Train-time 2D segmentation augmentation helpers for FleXray."""

from .experiment import (
    DEFAULT_NORMALIZATION_EPS,
    DEFAULT_NORMALIZATION_PERCENTILES,
    MinMaxNormalize,
    PercentileMinMaxNormalize,
    Standardize,
    build_input_normalizer,
    resolve_input_normalization_config,
)
from .pipeline import (
    SegmentationAugmentationPipeline,
    apply_segmentation_augmentation,
    build_segmentation_augmentation_pipeline,
    restore_hard_background,
)
from .presets import (
    load_named_augmentation_preset,
    resolve_augmentation_presets,
    snapshot_augmentation_presets,
)
from .transforms import (
    RandomAspectCrop,
    RandomClaheOrGamma,
    RandomIntensityScale,
    RandomLabelZoom,
    RandomLetterDrop,
)

__all__ = [
    "DEFAULT_NORMALIZATION_EPS",
    "DEFAULT_NORMALIZATION_PERCENTILES",
    "MinMaxNormalize",
    "PercentileMinMaxNormalize",
    "RandomAspectCrop",
    "RandomClaheOrGamma",
    "RandomIntensityScale",
    "RandomLabelZoom",
    "RandomLetterDrop",
    "SegmentationAugmentationPipeline",
    "Standardize",
    "apply_segmentation_augmentation",
    "build_input_normalizer",
    "build_segmentation_augmentation_pipeline",
    "load_named_augmentation_preset",
    "resolve_augmentation_presets",
    "resolve_input_normalization_config",
    "restore_hard_background",
    "snapshot_augmentation_presets",
]
