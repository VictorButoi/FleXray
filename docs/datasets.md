# FleXray Training Datasets

`fxr.datasets` is the training-only dataset structure surface. It does not
contain training loops, model code, inference, rendering, visibility-only data,
or eval-only data.

## Package Your Own Dataset

`fxr-dataset` turns a small YAML manifest into the canonical ThunderDB layout
used by FleXray training. The v1 format supports exactly `xray-seg` and
`ct-seg`. Generated 2D images are ordinary `xray-seg` samples; there is no
separate generated-data package type.

Payload paths may be absolute or relative to the manifest directory. Every
sample must declare both `subject_id` and `split`. Packaging fails if one
subject appears in more than one split. `fxr-dataset pack` never changes the
splits a manifest declares; `fxr-dataset scaffold` is the one command that
generates them.

### The data-engine commands

```bash
fxr-dataset scaffold xray-seg --images ./images --masks ./masks --name MyXrays \
    --output dataset.yml [--split 70 15 15] [--seed 1337] [--subject-regex '^(P\d+)']
fxr-dataset scaffold ct-seg --images ./ct --masks ./seg --name MyCT --output dataset.yml
fxr-dataset validate dataset.yml          # schema + every payload, no writes
fxr-dataset pack dataset.yml /data/MyXrays
fxr-dataset check /data/MyXrays           # canonical packages only
fxr-dataset inspect /data/AnyThunderDB    # describes any split/sample ThunderDB
fxr-render /data/MyCT --dataset-name MyCT --profile MOOSE --output ./drr  # see camera.md
```

When FleXray opens a ThunderDB path for `inspect`, `check`, or training, each
value is decoded with restricted loaders as it is read. `.pkl` and `.pickle`
values are refused, including compressed variants. MessagePack values retain
ordinary NumPy arrays and scalars, but object arrays and object-containing
structured dtypes are rejected before array construction or unpickling. Tensor
(`.pt`) values, including compressed ones, load on CPU with explicit
`weights_only=True`; unsupported Python objects and TorchScript archives are
refused without retrying an unrestricted loader. Existing numeric FleXray
packages keep the same storage format.

These guards apply to paths opened by FleXray. If you supply an already-open
database or custom mapping, its decoding behavior remains your responsibility;
FleXray uses that object as supplied.

`scaffold` pairs images and masks by file stem (`.png/.jpg/.jpeg/.tif/.tiff/.bmp/.npy`;
`.nii/.nii.gz/.npy` for CT), derives subject ids from the stem (or the first
capture group of `--subject-regex`), assigns seeded subject-level splits with
largest-remainder rounding (defaults 70/15/15 for X-ray, 90/10/0 for CT), and
writes a `stored_labels` stub naming every observed mask id `label_<id>`.
Rename those to protocol names (or the names of a dataset spec, see
[protocols.md](protocols.md)) before validating. COCO/LabelMe/YOLO annotations
and DICOM series are out of scope: convert them to mask images or NIfTI first.

### Reproducing the released preprocessing

The released FleXray weights were trained on X-rays that were min-max
scaled per image, zero-padded to a square, and area-resized to `256x256`
(masks with nearest-neighbour sampling), and on CT volumes clipped to
`[-1000, 2000]` HU and stored as `512x512x256` axial crops. Two optional
manifest blocks reproduce this inside `pack`:

```yaml
# xray-seg
preprocessing:
  intensity: per_image_minmax   # or dtype_range (the default without this block)
  pad_to_square: true
  output_size: [256, 256]

# ct-seg
preprocessing:
  hu_window: [-1000, 2000]      # clipped volumes are stored as float16
crops:
  size: [512, 512, 256]         # x and y must equal the volume extent; crops are axial
  max_overlap: 0.0              # minimal cover; raise it for denser crops
```

Without a `preprocessing` block nothing changes: integer X-ray input keeps the
dtype-range scaling described below. Training applies per-sample percentile
normalization, which makes the two intensity rules equivalent for unpadded
images; they differ only once padding is added (the pad value must equal the
tissue minimum), which is why `per_image_minmax` runs before `pad_to_square`.
Each processed X-ray records its geometry in `_metadata[*].preprocessing`
(`original_shape`, `pad_before`, `pad_after`, `padded_shape`,
`processed_shape`, `resize_scale`) so predictions can be mapped back, and the
package records the block in `_attrs.preprocessing`. Image resizing uses
Pillow's box filter (area averaging; identical to OpenCV `INTER_AREA` for
integer ratios) and masks use the exact `INTER_NEAREST` index rule.

`ct-seg` manifests accept `.nii`/`.nii.gz` for `image` and `label`; the affine
and spacing then come from the header and the `affine`/`spacing` fields may be
omitted (explicit `.npy` values must agree with the header). With a `crops`
block every volume becomes its planned axial crops (`<sample_id>__crop000`,
...): the top and bottom crops are always kept, intermediate crops are spread
evenly with at most `ceil(size[2] * (1 - max_overlap))` slices between their
starts. The default `0.0` selects the fewest crops that cover the volume;
larger values request denser crops. Despite its name, `max_overlap` is an
overlap target, not an upper bound: covering both ends can force extra overlap,
and rounding the step up to whole slices can give slightly less than the
target. Short volumes are padded symmetrically with air, background-only crops
are dropped (listed in `_attrs.crops.dropped_background_only`), and every crop
stores the metadata the training runtime reads (`_attrs.storage_layout: crops`, per-crop
`subject_id`, `source_sample_id`, `crop_index`, `z_start`, `z_stop`, `z_pad_*`,
`affine_offset_xyz`, `crop_foreground_label_ids`, `fg_centroids_ijk`). See
"Crop-backed layouts and sample weighting" below for how training uses them.

Before the overlap fix, every valid `max_overlap` value produced the minimal
cover. The default changed from `0.5` to `0.0` to preserve those crop windows
when the field is omitted. Repacking a manifest with an explicit nonzero value
can now increase sample counts and epoch length. For example, a 700-slice volume
with 256-slice crops starts at `[0, 222, 444]` by default and at
`[0, 111, 222, 333, 444]` with `max_overlap: 0.5`.

Every training run requires a non-empty `train` split. A non-empty,
subject-disjoint `val` split is also required when core validation is enabled
(`train.eval_freq > 0`) or an enabled `WandbSamplePredictionLogger` consumes the
training-source validation datasets. Set `train.eval_freq: 0` and disable that
callback for a train-only run. Callback-local `EvalSetMetricLogger` data remains
independent and must contain the split configured on that callback. Other split
names are valid, and FleXray never invents one.

Dense X-ray example:

```yaml
schema_version: 1
dataset_name: MyChestXrays
dataset_type: xray-seg
stored_labels:
  0: background
  1: lungs
  2: heart
samples:
  - sample_id: patient-001-pa
    subject_id: patient-001
    split: train
    image: images/patient-001-pa.png
    label: masks/patient-001-pa.png
    metadata:
      view: PA
  - sample_id: patient-002-pa
    subject_id: patient-002
    split: val
    image: images/patient-002-pa.npy
    label: masks/patient-002-pa.npy
```

The v1 packager writes each X-ray image as `(1, H, W)` channel-first
`float32` with finite values in `[0, 1]`. Pillow-readable inputs, including RGB
images, are converted to grayscale. NumPy inputs must be 2D or one-channel
`(1, H, W)` arrays; multichannel NumPy images are rejected. Integer input is
scaled using its dtype range: `(value - dtype_min) / (dtype_max - dtype_min)`.
This fixed mapping preserves intensity comparability between images instead of
applying per-image min-max normalization. Floating-point input is preserved and
must already be finite and in `[0, 1]`.

All X-ray samples in one package must share the same spatial shape. There is no
arbitrary minimum height or width; the packager accepts any positive common
shape and never resizes images silently.

Dense masks are 2D non-negative integer maps. Dense packages require
`stored_labels`: a contiguous native-id mapping
starting at `0: background`, with unique non-empty names. Every observed mask id
must be declared. To train the package, each name must resolve in the selected
FleXray protocol; launch readiness validates that mapping. This mapping lets
training project a user dataset without assuming
its integer ids already match the model label ids.

Partially labeled or generated 2D data can instead provide a channel-first
`.npy` mask and ordered names. The names declare the package-level subset of
structures available for supervision:

```yaml
schema_version: 1
dataset_name: MyPartialXrays
dataset_type: xray-seg
protocol_name: all_structures_flexray_v4
label_names: [lungs, heart]
samples:
  - sample_id: generated-001
    subject_id: source-volume-001
    split: train
    image: generated/images/001.npy
    label: generated/masks/001.npy
  - sample_id: generated-002
    subject_id: source-volume-002
    split: val
    image: generated/images/002.npy
    label: generated/masks/002.npy
```

Channel masks must have shape `(C, H, W)`, contain finite values in `[0, 1]`,
and match `label_names`. `label_names` declares the package-level subset of
available structures; it does not distinguish an unannotated pixel from an
annotated negative. Empty declared channels are ignored by the partial-label
loss. A package cannot mix dense and channel encodings.

The v1 CT packager accepts `.npy` payloads. Images must be single-channel:
either a 3D volume or a one-channel `(1, X, Y, Z)` array. Labels contain exactly
one dense integer label id per voxel and must be a 3D map or one-channel 4D map;
CT does not support named multichannel masks. Image and label spatial shapes
must match. CT spatial shapes may vary between samples because DRR rendering
produces the configured 2D output and CT training uses batch size 1.
Every sample must contain at least one non-background label id because the CT
package contract validates foreground-centroid data used by DRR sampling. It
also supplies a finite, invertible `(4, 4)` affine and three positive spacing
values:

```yaml
schema_version: 1
dataset_name: MyCT
dataset_type: ct-seg
stored_labels:
  0: background
  1: femurs
samples:
  - sample_id: volume-001
    subject_id: patient-001
    split: train
    image: volumes/001.npy
    label: labels/001.npy
    affine: affines/001.npy
    spacing: spacing/001.npy
  - sample_id: volume-002
    subject_id: patient-002
    split: val
    image: volumes/002.npy
    label: labels/002.npy
    affine: affines/002.npy
    spacing: spacing/002.npy
```

Validate inputs, build the database, and independently check the result:

```bash
fxr-dataset validate dataset.yml
fxr-dataset pack dataset.yml /data/MyDataset
fxr-dataset check /data/MyDataset
```

Validation and packing process one sample at a time. Memory use is therefore
bounded by the largest individual sample instead of the full dataset size.

`--overwrite` accepts only an existing canonical FleXray package and replaces it
only after the new temporary package passes structural validation and a runtime
load from every non-empty split. Filesystem roots, the working/home directory,
and destinations containing the source manifest or payloads are rejected. Canonical output contains `_subjects`, `_samples`,
`_splits`, `_metadata`, and `_attrs`. Each sample key stores `img` and `seg`, plus
`affine` and `spacing` for CT.

## Records

`DatasetRecord` describes one sample independently of storage. The required
fields are `dataset_name`, `modality`, `data_id`, and `image`. Segmentation
records usually define `label`; CT records also define `affine` and `spacing`.
X-ray records can declare `label_names` on the record or in metadata. For
channel-first masks, the names correspond to channel positions. Existing
ThunderDBs can also attach names to a dense map, where the order corresponds to
native integer ids. Canonical packages use the less ambiguous `stored_labels`
field for dense maps.

## Storage

`ManifestStorageBackend` reads a YAML/JSON manifest with a `records:` list. Each
payload field is a path relative to the manifest file. `.npy`, `.npz`, `.pt`,
`.pth`, `.json`, `.yml`, and `.yaml` payloads are supported. PyTorch `.pt` and
`.pth` files are loaded with `weights_only=True`; they may contain tensors,
basic scalar metadata, and nested mapping/list/tuple containers only. Pickled
custom classes and executable reducers are rejected.

`ThunderDBStorageBackend` accepts an opened ThunderDB-like object or a path that
can be opened by `thunderpack`. Record payload fields are keys into the database.

`SplitThunderDBStorageBackend` adapts split/sample ThunderDB layouts whose
records live under `_splits[split]`, per-sample metadata under `_metadata`, and
payload arrays inside each sample record. Clean databases declare
`_attrs.dataset_name` and `_attrs.clean_dataset: true`; legacy
ThunderDBs may declare `_attrs.dataset` instead. Configure payload field names
with constructor keywords such as `image_key`, `label_key`, `affine_key`, and
`spacing_key`; `_attrs.payload_keys` and `_attrs.payload_fields` declarations
are honored when present. Channel-mask packages declare their channel order in
`_attrs["label_names"]`.

`_attrs.seg_storage`, when declared, fixes how label payloads are read:
`indexed_label_mask` (dense 2D id maps) or `overlapping_channel_mask` (binary
`(C, H, W)` stacks whose channel index equals the native id, as written by
the legacy X-ray packages for datasets with overlapping structures). A payload whose rank
contradicts the declaration is rejected. Overlapping packages that also declare
`mask_label_names` (with identity `mask_label_ids`) use those names as the
channel names; FluXray packages keep `label_names`.

## Construction

Instantiate `XrayTrainingDataset` or `CTTrainingDataset`
directly with a storage backend and explicit keyword arguments such as `backend`,
`protocol_name`, `dataset_name`, `label_mode`, `split`, `require_seg`, and the
return flags. `from_manifest(...)` and `from_thunderdb(...)` are convenience
constructors that create the matching storage backend before applying the same
runtime options.

## Real-Run Builders

`build_training_dataset(...)` builds one training dataset from a registered
ThunderDB layout. `build_named_datasets(...)` builds all datasets for one
modality, and `build_multimodal_datasets(...)` returns a
`TrainingDatasetBundle` with training datasets, optional validation datasets, and
dataset modalities. Pass `include_validation=False` only when no configured
feature consumes training-source validation data. Each source ThunderDB is
opened once and shared by its train/val datasets when both are present; the
bundle owns those readers. Call `bundle.close()` when finished or
use it as a context manager. A training experiment performs this cleanup automatically on success
and failure. Datasets returned by the single/named builders also expose
idempotent `close()` methods.

Built-in layouts use `CT_DATAPATH` and `XRAY_DATAPATH`. The historical
FluXray layout reads `GENERATED_DATAPATH` for storage-root compatibility; its runtime modality is X-ray. Each root names one directory,
not a colon-separated search path. Built-in X-ray layouts have no selector
fields, so `XRAY_DATAPATH` points directly to the directory containing the X-ray
ThunderDB dataset folders (the base recipe trains on 256 px packages). Built-in CT layouts group records by the stored
`subject_id` metadata so `skip_subjects` also applies to crop-backed databases.

A user package does not need a registered layout or a root environment variable.
Set an absolute `path` in that dataset config under `Xray` or `CT`:

```yaml
data:
  Xray:
    MyPartialXrays:
      path: /data/flexray/MyPartialXrays
  CT:
    MyCT:
      path: /data/flexray/MyCT
```

Generated 2D data is configured under `Xray`; it is not a third package modality.
Dense-package `stored_labels` must use canonical names from the selected FleXray
protocol unless the entry also sets `dataset_spec: /abs/Spec.yml`, in which
case that spec's aliases and drops harmonize the package's native names (see
[protocols.md](protocols.md)). Channel-mask `label_names` may list only the
annotated subset. Builder configs reject `root`, `data_root`, and legacy
`_class`.

Built-in layouts are declared in packaged YAML at
`fxr.configs/dataset_layouts/training.yml` and registered at import time.
Downstream code can call `register_dataset_layout(...)` or
`load_dataset_layouts(...)` during process startup to add layouts. Registered
layout selector values and resolved symbolic-link targets must stay below the
configured dataset root; traversal and root escape are rejected. An explicit
absolute user-package `path` is intentionally direct and is not resolved below a
registered root.

Supported built-ins:

- CT: `MOOSE`, `ElbowCT`, `RSNAFrac`, `PedsCT`, `HANSeg`, `ShoulderCT`
- Xray: `HandBones`, `FootBones`, `DeepFluoro`, `HipRay`, `LowerLimbs`,
  `MendeleyCXR`, `RAM-W600`, `ThoracoaMonch`, `VinDr-Rib`, `ShoulderMonch`, `JRST`,
  `MURA_FOREARM`, `MURA_HUMERUS`, `DarwinCVD19`, `ElbowLat`, `KidneyStone`,
  `FluXray`

Config shape:

```yaml
data:
  CT:
    MOOSE:
      version: "2.0"
      crop_mode: random_slices
      crop_size: [512, 512, 256]
  Xray:
    HipRay: {}
    FluXray:
      version: "1.0"
```

Builder-created datasets always use `label_mode="native"`. X-ray builders also
rely on the `XrayTrainingDataset` default `require_seg=True` and reject
`require_seg` in builder configs. Supplying
`label_mode: native` is allowed for explicitness; `label_mode: model` is
rejected so label projection stays in the training runtime boundary.

Default built-in paths:

- `MOOSE`: `CT_DATAPATH/MOOSE/thunder_moose/3D/all_categories/{version}`
- `ElbowCT`, `RSNAFrac`, `PedsCT`: `CT_DATAPATH/{dataset_name}/thunder_dbs/{version}`
- Real X-ray datasets: `XRAY_DATAPATH/{dataset_name}`
- `FluXray`: `GENERATED_DATAPATH/FluXray/thunder_dbs/{version}`

The public data release,
[`VictorButoi/flexray-data`](https://huggingface.co/datasets/VictorButoi/flexray-data),
ships the redistributable real X-ray sources as `fxr-dataset` packages (pack
each `<Dataset>/dataset.yml` below `XRAY_DATAPATH`) and the FluXray database
as `FluXray/thunder_dbs/1.0/`, so `GENERATED_DATAPATH` is simply the download
root. Its card lists the download pointer for every source we cannot
redistribute.

## Builder Boundary

Real-run builder configs do not accept legacy `_class`, arbitrary `root` or
`data_root` overrides, or manifest-backed storage. Dataloader and training-loop
construction, eval-only and visibility-only datasets, inference, rendering,
and model code remain outside `fxr.datasets`.

## Runtime Classes

- `XrayTrainingDataset`: native labels are either dense integer masks or named
  channel-first masks. Model labels are channel-first and named masks are
  projected by label name.
- `CTTrainingDataset`: requires image, affine, and spacing payloads; segmentation
  records also define labels. Images and dense native integer labels are returned
  channel-first as `(C, X, Y, Z)`, with 3D payloads promoted to one channel.
  If constructed in model mode directly, model labels remain dense integer masks
  rather than one-hot masks; builder-created training datasets stay native.

All classes return a `TrainingSample` dictionary with `image`, optional `label`,
optional `native_label`, `dataset_name`, `modality`, `data_id`, and `metadata`.
`return_data_id`, `return_metadata`, and `return_native_label` control optional
fields.

## CT Cropping

`CTTrainingDataset` accepts constructor-only crop options:
`crop_mode="none"`, `"random"`, or `"random_slices"`; optional
`crop_size=(x, y, z)`; and an optional `torch.Generator` for deterministic random
starts. `crop_mode="none"` ignores `crop_size`. Random modes require exactly
three positive crop sizes. When the crop is larger than the source volume, images
are padded with air HU (`-1000`) and labels with background (`0`).

CT cropping updates `metadata["affine"]` to the cropped origin and keeps original
spacing in `metadata["spacing"]`. Labels are cropped before any model-label
remapping. If `require_seg=False` and a CT record has no label, only the image and
affine are cropped and no centroid metadata is produced.

HU bounds are applied after crop or pad. `hu_min` clamps the lower bound, and
`air_clamp_hu` optionally marks air voxels before the clamp so they can be
restored to `-1000`. With `compute_fg_centroids=True`, foreground centroids from
the returned dense label are stored in `metadata["fg_centroids_ijk"]`; background-only
labels raise an error.

### Crop-backed layouts and sample weighting

The base recipe's CT sources are crop-backed ThunderDBs
(`_attrs.storage_layout: crops`): every record is one `512x512x256` crop whose
metadata stores `subject_id`, `crop_foreground_label_ids`, and
`fg_centroids_ijk` (one row per foreground id, or a `{native_id: (i, j, k)}`
mapping). With `compute_fg_centroids=False` those stored centroids are emitted
keyed by native id, so the experiment can drop labels the protocol maps to
background before offering them as `random_label` DRR isocenters. Runtime
cropping cannot be combined with a crop-backed layout.

`sample_weighting` (`"uniform"`, or
`{scheme: inverse_label_frequency, tau: 0.5, class_aggregation: max}`) selects
`CTTrainingDataset.sample_weights(label_lut)`: each model class contributes
`(N / n_c) ** tau` and a crop takes the mean or max over its classes, crops
without supervised foreground weigh `0`, and the training loader draws crops
with replacement from those weights. Inverse label-frequency weighting requires
a crop-backed layout. `fxr.datasets.inverse_label_frequency_weights` exposes the
formula directly.

## Composition

`CompositeSourceDataset` concatenates named datasets and injects source metadata.
`HomogeneousSourceBatchSampler` yields same-source batches. `MixedDataLoader` and
`SequentialDataLoader` wrap named loaders and emit `NamedBatch` values with the
source name, modality, and batch. `MixedDataLoader` uses an exact-count,
seed-controlled source schedule for fixed-length epochs; `set_epoch(epoch)`
restores that schedule when a checkpoint resumes.
