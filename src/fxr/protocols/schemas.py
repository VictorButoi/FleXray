from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProtocolSpec:
    """Shared protocol label space with explicit channel order.

    Attributes:
        protocol_name: Stable name used to resolve the protocol from config files
            and to select dataset normalization rules.
        labels: Ordered channel names for the ground-truth/evaluation protocol.
            `labels[0]` must be `"background"`; every later entry is a
            foreground protocol channel. A model label space may omit derived
            evaluation labels from this sequence.
        label_to_id: Mapping from each protocol label name to its integer channel
            id in `labels`.
    """

    protocol_name: str
    labels: tuple[str, ...]

    @property
    def label_to_id(self) -> dict[str, int]:
        """Return protocol labels keyed by name with channel ids as values.

        Returns:
            Mapping from protocol label name to channel id.
        """

        return {label_name: idx for idx, label_name in enumerate(self.labels)}


@dataclass(frozen=True)
class ModelLabelSpace:
    """Model output label space with explicit channel order.

    Attributes:
        model_name: Stable name for the model label space. It must match the
            corresponding `EvalContract.model_name` when compiling eval mappings.
        labels: Ordered output channel names emitted by the model. When no model
            config exists, registry helpers build this from a protocol's labels.
        label_to_id: Mapping from each model output label name to its integer
            channel id in `labels`.
    """

    model_name: str
    labels: tuple[str, ...]

    @property
    def label_to_id(self) -> dict[str, int]:
        """Return model labels keyed by name with channel ids as values.

        Returns:
            Mapping from model label name to channel id.
        """

        return {label_name: idx for idx, label_name in enumerate(self.labels)}


@dataclass(frozen=True)
class DatasetSpec:
    """Segmentation dataset labels and protocol normalization rules.

    Attributes:
        dataset_name: Canonical dataset config name. Registry helpers validate
            that this matches the YAML filename after alias resolution.
        stored_labels: Mapping from native stored mask ids to dataset-native label
            names. Native id 0 must be `"background"`.
        protocol_label_aliases: Per-protocol mappings from dataset-native labels
            to protocol labels. Use this when a stored label has the same meaning
            as a protocol label but a different name.
        protocol_drop_labels: Per-protocol lists of dataset-native labels that
            should compile to background and be excluded from identity eval
            defaults.
        skip_subjects: Optional subject keys to exclude for this dataset. `None`
            means no exclusions are declared.
        supervise_empty_labels: Protocol labels that are anatomically absent
            from every image of this dataset and may therefore be supervised as
            known negatives even though the dataset never annotates them.
        dataset_aliases: Alternative public names that resolve to this dataset
            spec, such as `"FluXray"` resolving to `"MOOSE"`.
        label_to_native_id: Mapping from each dataset-native label name to its
            stored integer id.
    """

    dataset_name: str
    stored_labels: dict[int, str]
    protocol_label_aliases: dict[str, dict[str, str]] = field(default_factory=dict)
    protocol_drop_labels: dict[str, tuple[str, ...]] = field(default_factory=dict)
    skip_subjects: tuple[str, ...] | None = None
    dataset_aliases: tuple[str, ...] = ()
    supervise_empty_labels: tuple[str, ...] = ()

    @property
    def label_to_native_id(self) -> dict[str, int]:
        """Return dataset-native labels keyed by name with stored ids as values.

        Returns:
            Mapping from dataset-native label name to stored integer id.
        """

        return {
            label_name: native_id
            for native_id, label_name in self.stored_labels.items()
        }


@dataclass(frozen=True)
class EvalContract:
    """Rules for mapping eval-set labels onto model output labels.

    Attributes:
        model_name: Model label-space name this contract applies to.
        eval_sets: Mapping from eval-set name to eval-label definitions. Each
            eval label maps to one or more candidate tuples of model label names;
            the compiler chooses the first supported candidate.
        identity_defaults: Optional defaulting mode. `"dataset_eval_labels"`
            means eval labels are derived from normalized dataset stored labels
            before explicit eval-set entries are applied.
    """

    model_name: str
    eval_sets: dict[str, dict[str, tuple[tuple[str, ...], ...]]]
    identity_defaults: str | None = None


@dataclass(frozen=True)
class CompiledTrainingLut:
    """Compiled lookup table from dataset-native ids to protocol channel ids.

    Attributes:
        dataset_name: Canonical dataset name used to compile the LUT.
        protocol_name: Protocol name used as the target label space.
        label_lut: Dense tuple indexed by native stored id. Values are protocol
            channel ids, with `-1` for native id gaps that do not appear in
            `stored_labels`.
        native_id_to_protocol_id: Sparse mapping from every declared native id to
            its compiled protocol channel id.
        native_id_to_protocol_label: Sparse mapping from every declared native id
            to its normalized protocol label name.
        protocol_labels: Ordered protocol labels used when compiling the LUT.
    """

    dataset_name: str
    protocol_name: str
    label_lut: tuple[int, ...]
    native_id_to_protocol_id: dict[int, int]
    native_id_to_protocol_label: dict[int, str]
    protocol_labels: tuple[str, ...]


@dataclass(frozen=True)
class CompiledEvalMapping:
    """Compiled mapping from eval labels to model output channel groups.

    Attributes:
        model_name: Model label-space name used to compile the mapping.
        eval_set_name: Eval-set name selected from the eval contract.
        eval_label_names: Ordered eval labels that should be evaluated.
        model_channel_groups: Ordered channel-id groups parallel to
            `eval_label_names`; each group should be unioned for that eval label.
        eval_label_to_model_ids: Mapping from eval label name to selected model
            channel ids.
        eval_label_to_model_labels: Mapping from eval label name to selected model
            label names.
        model_labels: Ordered model labels used when compiling the mapping.
        dropped_eval_label_names: Eval labels dropped because no supported
            candidate existed and `unsupported_label_policy="drop"` was used.
        dropped_eval_label_to_model_labels: Original candidate model-label groups
            for dropped eval labels.
        unsupported_zero_eval_label_names: Eval labels retained as zero-valued
            unsupported labels because `unsupported_label_policy="zero"` was used.
        unsupported_zero_eval_label_to_model_labels: Selected model-label groups
            for zero-valued unsupported eval labels.
    """

    model_name: str
    eval_set_name: str
    eval_label_names: tuple[str, ...]
    model_channel_groups: tuple[tuple[int, ...], ...]
    eval_label_to_model_ids: dict[str, tuple[int, ...]]
    eval_label_to_model_labels: dict[str, tuple[str, ...]]
    model_labels: tuple[str, ...]
    dropped_eval_label_names: tuple[str, ...] = ()
    dropped_eval_label_to_model_labels: dict[
        str,
        tuple[tuple[str, ...], ...],
    ] = field(default_factory=dict)
    unsupported_zero_eval_label_names: tuple[str, ...] = ()
    unsupported_zero_eval_label_to_model_labels: dict[
        str,
        tuple[str, ...],
    ] = field(default_factory=dict)
