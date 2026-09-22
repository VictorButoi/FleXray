from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class TrainContext:
    """Context passed to train-level callback hooks.

    Attributes:
        metadata: Read-only framework-owned metadata for the training run.
    """

    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze metadata after dataclass initialization.

        Args:
            None.

        Returns:
            ``None``.
        """

        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))


@dataclass(frozen=True)
class EpochContext:
    """Context passed to epoch-level callback hooks.

    Attributes:
        phase: Training phase name, such as ``"train"`` or ``"val"``.
        epoch: Epoch index supplied by the caller.
        num_epochs: Optional total number of epochs in the run.
        metadata: Read-only framework-owned metadata for the epoch.
    """

    phase: str
    epoch: int
    num_epochs: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze metadata after dataclass initialization.

        Args:
            None.

        Returns:
            ``None``.
        """

        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))


@dataclass(frozen=True)
class BatchContext:
    """Context passed to batch-level callback hooks.

    Attributes:
        phase: Training phase name, such as ``"train"`` or ``"val"``.
        epoch: Epoch index supplied by the caller.
        batch_idx: Zero-based batch index within the current phase and epoch.
        num_batches: Total number of batches in the current phase and epoch.
        metadata: Read-only framework-owned metadata for the batch.
    """

    phase: str
    epoch: int
    batch_idx: int
    num_batches: int
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze metadata after dataclass initialization.

        Args:
            None.

        Returns:
            ``None``.
        """

        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))


@dataclass(frozen=True)
class MetricsContext:
    """Context passed to metrics callback hooks.

    Attributes:
        records: Read-only sequence of metric records. Each record must contain
            ``epoch`` and ``phase`` fields, may contain ``dataset``, and may
            contain any number of metric-value columns.
        metadata: Read-only framework-owned metadata for the metrics event.
    """

    records: Sequence[Mapping[str, object]]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze records and metadata after dataclass initialization.

        Args:
            None.

        Returns:
            ``None``.
        """

        object.__setattr__(self, "records", _freeze_records(self.records))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))


def _freeze_mapping(
    value: Mapping[str, object], field_name: str
) -> Mapping[str, object]:
    """Copy a mapping into a read-only mapping proxy.

    Args:
        value: Mapping to freeze.
        field_name: Field name used in validation errors.

    Returns:
        Read-only mapping proxy containing the copied items.

    Raises:
        TypeError: If ``value`` is not a mapping.
    """

    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping, got {type(value).__name__}.")
    return MappingProxyType(dict(value))


def _freeze_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Copy metric records into an immutable sequence of read-only mappings.

    Args:
        records: Sequence of metric record mappings.

    Returns:
        Tuple of read-only record mapping proxies.

    Raises:
        TypeError: If ``records`` is not a sequence of mappings.
    """

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError(
            f"records must be a sequence of mappings, got {type(records).__name__}."
        )

    frozen_records: list[Mapping[str, object]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(
                "records must contain only mappings; "
                f"record {index} has type {type(record).__name__}."
            )
        frozen_records.append(MappingProxyType(dict(record)))
    return tuple(frozen_records)
