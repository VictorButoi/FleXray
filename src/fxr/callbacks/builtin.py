from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd
from tabulate import tabulate

from .contexts import BatchContext, MetricsContext


class BatchProgressLogger:
    """Stdout batch-progress logger for simple training loops.

    Attributes:
        every_n_batches: Positive interval used for periodic logging.
        phases: Optional tuple of phase names allowed to emit progress lines.
    """

    def __init__(
        self,
        every_n_batches: int = 50,
        phases: str | Iterable[str] | None = None,
    ) -> None:
        """Configure the batch logging interval and optional phase filter.

        Args:
            every_n_batches: Positive interval for periodic progress logs.
            phases: Optional phase name or iterable of phase names to log.

        Returns:
            ``None``.

        Raises:
            ValueError: If ``every_n_batches`` is less than one or ``phases``
                contains no phase names.
        """

        if every_n_batches < 1:
            raise ValueError("every_n_batches must be at least 1.")
        self.every_n_batches = every_n_batches
        self.phases = _normalize_phases(phases)

    def on_batch_end(self, context: BatchContext) -> None:
        """Print batch progress for the first, interval, and final batches.

        Args:
            context: Batch event context containing phase, epoch, batch index,
                and number of batches.

        Returns:
            ``None``.
        """

        if self.phases is not None and context.phase not in self.phases:
            return

        current_batch = context.batch_idx + 1
        is_first = context.batch_idx == 0
        is_interval = current_batch % self.every_n_batches == 0
        is_final = current_batch == context.num_batches
        if is_first or is_interval or is_final:
            print(
                f"{context.phase} epoch {context.epoch} "
                f"batch {current_batch}/{context.num_batches}"
            )


class MetricsTablePrinter:
    """Stdout metrics table printer for metric-record lists.

    Attributes:
        tablefmt: Tabulate table format used for printed tables.
        floatfmt: Tabulate float format used for numeric metric values.
    """

    def __init__(self, tablefmt: str = "github", floatfmt: str = ".4f") -> None:
        """Configure tabulate output formatting.

        Args:
            tablefmt: Format name passed to ``tabulate``.
            floatfmt: Float format passed to ``tabulate``.

        Returns:
            ``None``.
        """

        self.tablefmt = tablefmt
        self.floatfmt = floatfmt

    def on_metrics(self, context: MetricsContext) -> None:
        """Print a pivoted metrics table from metric records.

        Args:
            context: Metrics event context containing records with ``epoch``,
                ``phase``, optional ``dataset``, and metric columns.

        Returns:
            ``None``.
        """

        records = [dict(record) for record in context.records]
        if not records:
            print("No metrics to print.")
            return

        metric_columns = _metric_columns(records)
        if not metric_columns:
            print("No metric columns to print.")
            return

        rows = []
        phase_labels: list[str] = []
        for record in records:
            phase_label = _phase_label(record)
            if phase_label not in phase_labels:
                phase_labels.append(phase_label)
            row = {
                "epoch": record.get("epoch"),
                "phase_label": phase_label,
            }
            for column in metric_columns:
                row[column] = record.get(column)
            rows.append(row)

        frame = pd.DataFrame(rows)
        melted = frame.melt(
            id_vars=["epoch", "phase_label"],
            value_vars=metric_columns,
            var_name="metric",
            value_name="value",
        )
        pivot = (
            melted.pivot_table(
                index=["epoch", "metric"],
                columns="phase_label",
                values="value",
                aggfunc="last",
                dropna=False,
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
        ordered_columns = ["epoch", "metric", *phase_labels]
        table = pivot.reindex(columns=ordered_columns)
        print(
            tabulate(
                table,
                headers="keys",
                tablefmt=self.tablefmt,
                showindex=False,
                floatfmt=self.floatfmt,
                missingval="",
            )
        )


def _normalize_phases(phases: str | Iterable[str] | None) -> tuple[str, ...] | None:
    """Normalize an optional phase filter to a tuple of strings.

    Args:
        phases: Optional phase name or iterable of phase names.

    Returns:
        Tuple of phase names, or ``None`` when no filter is configured.

    Raises:
        ValueError: If an iterable filter is empty.
        TypeError: If a phase name is not a string.
    """

    if phases is None:
        return None
    if isinstance(phases, str):
        if not phases:
            raise ValueError("phases must contain non-empty phase names.")
        return (phases,)

    normalized = tuple(phases)
    if not normalized:
        raise ValueError("phases must contain at least one phase.")
    for phase in normalized:
        if not isinstance(phase, str) or not phase:
            raise TypeError("phases must contain only non-empty strings.")
    return normalized


def _metric_columns(records: list[Mapping[str, Any]]) -> list[str]:
    """Return metric columns in first-seen order.

    Args:
        records: Metric records to inspect.

    Returns:
        Ordered metric-column names, excluding event descriptor fields.
    """

    descriptor_columns = {"epoch", "phase", "dataset"}
    columns: list[str] = []
    for record in records:
        for key in record:
            if key in descriptor_columns or key in columns:
                continue
            columns.append(key)
    return columns


def _phase_label(record: Mapping[str, Any]) -> str:
    """Build a phase label, including dataset when present.

    Args:
        record: Metric record containing at least a phase field.

    Returns:
        Phase label, or ``phase/dataset`` when a dataset field is present.
    """

    phase = str(record.get("phase", ""))
    dataset = record.get("dataset")
    if dataset is None or dataset == "":
        return phase
    return f"{phase}/{dataset}"
