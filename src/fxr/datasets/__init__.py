"""Training dataset structures for FleXray."""

from .builders import (
    DatasetLayout,
    TrainingDatasetBundle,
    build_multimodal_datasets,
    build_named_datasets,
    build_training_dataset,
    load_dataset_layouts,
    register_dataset_layout,
)
from .composition import (
    CompositeSourceDataset,
    HomogeneousSourceBatchSampler,
    MixedDataLoader,
    SequentialDataLoader,
    SourceSpec,
)
from .packaging import (
    DatasetPackageReport,
    pack_dataset,
    validate_dataset_manifest,
    validate_packed_dataset,
)
from .remap import (
    apply_label_lut,
    channelize_integer_mask,
    compile_package_label_remap,
    compile_training_label_remap_by_name,
    compile_training_label_remap_from_stored_labels,
    project_channel_mask,
)
from .inspect import ThunderDBInspection, inspect_thunderdb
from .runtime import CTTrainingDataset, XrayTrainingDataset
from .sampling import SampleWeightingSpec, inverse_label_frequency_weights
from .scaffold import ScaffoldReport, scaffold_manifest, stable_subject_splits
from .schemas import (
    DatasetRecord,
    NamedBatch,
    TrainingLabelRemap,
    TrainingSample,
)
from .storage import (
    ManifestStorageBackend,
    SplitThunderDBStorageBackend,
    ThunderDBStorageBackend,
)

__all__ = [
    "register_dataset_layout",
    "load_dataset_layouts",
    "build_training_dataset",
    "build_named_datasets",
    "build_multimodal_datasets",
    "TrainingDatasetBundle",
    "DatasetLayout",
    "DatasetPackageReport",
    "CTTrainingDataset",
    "CompositeSourceDataset",
    "DatasetRecord",
    "HomogeneousSourceBatchSampler",
    "ManifestStorageBackend",
    "SplitThunderDBStorageBackend",
    "MixedDataLoader",
    "NamedBatch",
    "SampleWeightingSpec",
    "ScaffoldReport",
    "SequentialDataLoader",
    "SourceSpec",
    "ThunderDBInspection",
    "ThunderDBStorageBackend",
    "TrainingLabelRemap",
    "TrainingSample",
    "XrayTrainingDataset",
    "apply_label_lut",
    "channelize_integer_mask",
    "compile_package_label_remap",
    "compile_training_label_remap_by_name",
    "compile_training_label_remap_from_stored_labels",
    "inspect_thunderdb",
    "inverse_label_frequency_weights",
    "project_channel_mask",
    "pack_dataset",
    "scaffold_manifest",
    "stable_subject_splits",
    "validate_dataset_manifest",
    "validate_packed_dataset",
]
