# FleXray Protocols

A FleXray protocol is a strict ground-truth/evaluation label-space contract for
segmentation. Together with its model label space, it defines model/output
channel order, dataset-native label normalization, and eval-set label mappings
without importing training, inference, renderer, or storage code.

## v4 Label Space

`all_structures_flexray_v4` is the bundled FleXray protocol.
`background` is channel 0. Foreground labels start at channel 1 and preserve the
YAML order exactly.

The bundled protocol has 63 labels. Its packaged model label space has 61 and is
the default training/inference head: the aggregate evaluation labels
`lumbar_spine` and `thoracolumbar_spine` are omitted because they are derived
from the individual vertebra outputs during evaluation. An explicit
`protocol.model_labels.names` list can still select a different validated
subset for a run.

Callers should treat both orders as part of the public contract. Changing a
label name, inserting a label, or moving a label changes ground-truth or model
compatibility.

## Dataset Specs

Dataset specs define stored segmentation mask ids:

```yaml
dataset_name: Example
skip_subjects: null
stored_labels:
  0: background
  1: hip_left
  2: hip_right
protocol_label_aliases:
  all_structures_flexray_v4:
    hip_left: hips
    hip_right: hips
```

Identity mappings are implicit. A stored label that is not aliased or dropped
must exist in the active protocol. `protocol_drop_labels` maps named stored
labels to background. Native-id gaps that are absent from `stored_labels` compile
to `-1` in the training LUT. The bundled `MOOSE` spec declares the full native
`0..117` label table; native labels outside `all_structures_flexray_v4`, such as
`vertebra_l6` and vessel labels, are explicit protocol drops.

`skip_subjects` is required for dataset specs. Use `null` when no subjects are
excluded.

`supervise_empty_labels` optionally lists protocol labels that are anatomically
absent from every image of the dataset (for example `femurs` in an elbow
dataset). They must not be stored labels, nor the protocol labels stored
labels are aliased to. Training supervises those channels as known negatives
instead of ignoring them as unannotated; see [losses.md](losses.md).

## Dataset Aliases

Dataset specs can declare public aliases. The bundled `MOOSE` spec declares
`FluXray`, so:

```python
load_dataset_spec_by_name("FluXray").dataset_name
```

returns `"MOOSE"`.

Aliases cannot collide with real dataset spec names or with aliases declared by
another spec.

## Listing Packaged Names

`list_protocol_names()` and `list_dataset_spec_names()` return the sorted
config names accepted by `load_protocol_by_name` and
`load_dataset_spec_by_name`:

```python
from fxr.protocols import list_dataset_spec_names, list_protocol_names

list_protocol_names()       # ("all_structures_flexray_v4",)
list_dataset_spec_names()   # ("DarwinCVD19", ..., "VinDr-Rib")
```

Both accept the same optional `config_root` as the named loaders and raise
`ValueError` when a name is declared by both a `.yml` and a `.yaml` file.
Dataset aliases are not listed; they resolve through
`load_dataset_spec_by_name`.

## Eval Contracts

Eval contracts map eval-set labels to model output labels. A named model YAML
defines the default output space for its matching protocol. When none is
present, the model label space falls back to the protocol labels.

`identity_defaults: dataset_eval_labels` means the compiler creates identity
entries from the dataset's normalized stored labels for the active protocol,
excluding background and explicitly dropped labels. Explicit contract entries
remain available for aggregate labels and candidate model-label groups.

When an eval label can map through multiple candidates, the compiler picks the
first candidate that exists in the resolved model label space.

## Harmonizing a Custom Dataset

A user dataset keeps its own native mask ids; a dataset spec next to the data
declares how they map into the protocol. `examples/custom_dataset/` is the
runnable walkthrough:

```yaml
dataset_name: CustomHips
skip_subjects: null
stored_labels:
  0: background
  1: hip_left
  2: hip_right
  3: femurs
  4: implant
protocol_label_aliases:
  all_structures_flexray_v4:
    hip_left: hips
    hip_right: hips
protocol_drop_labels:
  all_structures_flexray_v4:
    - implant
```

`fxr-protocol compile --dataset CustomHips.yml` prints the compiled table:

```text
CustomHips -> all_structures_flexray_v4
native_id  native_label  rule        channel  protocol_label
        0  background    background        0  background
        1  hip_left      alias            54  hips
        2  hip_right     alias            54  hips
        3  femurs        identity         11  femurs
        4  implant       drop              0  background
label_lut: [0, 54, 54, 11, 0]
```

Wire the spec into a training config through the package's data entry:

```yaml
data:
  Xray:
    CustomHips:
      path: /abs/packed/CustomHips
      dataset_spec: /abs/CustomHips.yml
```

The spec's `dataset_name` must equal the config key and its `stored_labels`
must equal the package's `_attrs.stored_labels`; readiness validation
(`fxr-train --dry-run`) fails otherwise. Without `dataset_spec`, a package's
`stored_labels` must already be protocol names. `skip_subjects` from a
path-configured spec is not applied (packaged specs apply it through their
registered layouts). Channel-mask packages (`label_names`) map by name and do
not take a `dataset_spec`.

## `fxr-protocol`

```bash
fxr-protocol list                                  # packaged protocols and dataset specs
fxr-protocol show all_structures_flexray_v4        # channels in order
fxr-protocol compile --dataset HipRay              # packaged spec by name or alias
fxr-protocol compile --dataset ./CustomHips.yml    # user spec by path (also validates it)
```

`compile` defaults to `--protocol all_structures_flexray_v4`. Invalid names or
spec files exit with status 2 and a one-line message.

## Introspection API

`fxr.protocols.introspect` provides the JSON-style helpers behind the CLI and
the MCP server: `list_protocols()`, `describe_protocol(name)`,
`list_datasets()`, `describe_dataset(name)`, `explain_mapping(protocol_name,
dataset_name)`, and `explain_lut(protocol, dataset_spec)` for loaded specs.
Mapping rows carry `native_id`, `native_label`, `rule` (`background`,
`identity`, `alias`, `drop`), `protocol_channel`, and `protocol_label`.
`load_protocol(path)`, `load_dataset_spec(path)`, and
`compile_training_lut(protocol, spec)` are exported for path-based use.
