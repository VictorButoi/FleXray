from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import fxr
import fxr.callbacks as public_callbacks
from fxr.callbacks import (
    BatchContext,
    BatchProgressLogger,
    CallbackRunner,
    MetricsContext,
    MetricsTablePrinter,
    TrainContext,
    build_callback_runner,
)

NOT_A_CLASS = object()


class BuilderRecorder:
    """Small callback used by import-string builder tests.

    Attributes:
        name: Name supplied through callback config.
    """

    def __init__(self, name: str) -> None:
        """Store the configured callback name.

        Args:
            name: Name to store on the callback.

        Returns:
            ``None``.
        """

        self.name = name


class Recorder:
    """Callback that records train-start dispatches.

    Attributes:
        name: Callback name written into the shared event log.
        events: Shared event log.
    """

    def __init__(self, name: str, events: list[tuple[str, str, object]]) -> None:
        """Store the callback name and shared event log.

        Args:
            name: Callback name written into events.
            events: Mutable event log owned by the test.

        Returns:
            ``None``.
        """

        self.name = name
        self.events = events

    def on_train_start(self, context: TrainContext) -> None:
        """Record a train-start event.

        Args:
            context: Train event context passed by the runner.

        Returns:
            ``None``.
        """

        self.events.append((self.name, "train_start", context))


class FailingRecorder:
    """Callback that raises during train-start dispatch.

    Attributes:
        events: Shared event log.
    """

    def __init__(self, events: list[str]) -> None:
        """Store the shared event log.

        Args:
            events: Mutable event log owned by the test.

        Returns:
            ``None``.
        """

        self.events = events

    def on_train_start(self, context: TrainContext) -> None:
        """Record and raise a deterministic failure.

        Args:
            context: Train event context passed by the runner.

        Returns:
            ``None``.

        Raises:
            RuntimeError: Always raised for fail-fast tests.
        """

        del context
        self.events.append("failed")
        raise RuntimeError("callback boom")


def test_public_callback_api_exports_planned_names() -> None:
    assert "callbacks" in fxr.__all__
    assert public_callbacks.__all__ == [
        "BatchContext",
        "BatchProgressLogger",
        "CallbackRunner",
        "EpochContext",
        "MetricsContext",
        "MetricsTablePrinter",
        "TrainContext",
        "build_callback_runner",
    ]
    assert not hasattr(public_callbacks, "PrintLogged")


def test_callback_initializer_is_reexport_only() -> None:
    tree = ast.parse(Path(public_callbacks.__file__).read_text(encoding="utf-8"))
    disallowed = (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef)

    assert not any(isinstance(node, disallowed) for node in tree.body)


def test_runner_dispatches_in_order_and_ignores_missing_hooks() -> None:
    events: list[tuple[str, str, object]] = []
    context = TrainContext(metadata={"run_id": "abc"})
    runner = CallbackRunner(
        [
            Recorder("first", events),
            object(),
            Recorder("second", events),
        ]
    )

    runner.on_train_start(context)
    runner.on_train_end(context)

    assert runner.callbacks[0].name == "first"
    assert events == [
        ("first", "train_start", context),
        ("second", "train_start", context),
    ]


def test_runner_propagates_exceptions_without_running_later_callbacks() -> None:
    events: list[str] = []

    class LaterRecorder:
        """Callback that must not run after a previous callback fails.

        Attributes:
            None.
        """

        def on_train_start(self, context: TrainContext) -> None:
            """Record an event if dispatch reaches this callback.

            Args:
                context: Train event context passed by the runner.

            Returns:
                ``None``.
            """

            del context
            events.append("later")

    runner = CallbackRunner([FailingRecorder(events), LaterRecorder()])

    with pytest.raises(RuntimeError, match="callback boom"):
        runner.on_train_start(TrainContext())

    assert events == ["failed"]


def test_contexts_are_frozen_and_freeze_nested_mappings() -> None:
    batch_context = BatchContext(
        phase="train",
        epoch=2,
        batch_idx=0,
        num_batches=3,
        metadata={"dataset": "HipRay"},
    )
    metrics_context = MetricsContext(
        records=[{"epoch": 2, "phase": "val", "dice": 0.75}],
    )

    with pytest.raises(FrozenInstanceError):
        batch_context.phase = "val"  # type: ignore[misc]
    with pytest.raises(TypeError):
        batch_context.metadata["dataset"] = "MOOSE"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        metrics_context.records = ()  # type: ignore[misc]
    with pytest.raises(TypeError):
        metrics_context.records[0]["dice"] = 0.8  # type: ignore[index]


def test_build_callback_runner_instantiates_import_strings_in_order() -> None:
    runner = build_callback_runner(
        [
            {"_class": "tests.test_callbacks.BuilderRecorder", "name": "first"},
            {"_class": "tests.test_callbacks.BuilderRecorder", "name": "second"},
        ]
    )

    assert [callback.name for callback in runner.callbacks] == ["first", "second"]


def test_build_callback_runner_rejects_invalid_specs() -> None:
    with pytest.raises(TypeError, match="sequence"):
        build_callback_runner("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="spec 0"):
        build_callback_runner([object()])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="_class"):
        build_callback_runner([{"name": "missing"}])
    with pytest.raises(TypeError, match="dotted import string"):
        build_callback_runner([{"_class": 123}])
    with pytest.raises(ValueError, match="dotted import string"):
        build_callback_runner([{"_class": "BatchProgressLogger"}])
    with pytest.raises(TypeError, match="not a class"):
        build_callback_runner([{"_class": "tests.test_callbacks.NOT_A_CLASS"}])


def test_batch_progress_logger_logs_first_interval_and_final(capsys) -> None:
    logger = BatchProgressLogger(every_n_batches=3)

    for batch_idx in range(5):
        logger.on_batch_end(
            BatchContext(
                phase="train",
                epoch=2,
                batch_idx=batch_idx,
                num_batches=5,
            )
        )

    assert capsys.readouterr().out.splitlines() == [
        "train epoch 2 batch 1/5",
        "train epoch 2 batch 3/5",
        "train epoch 2 batch 5/5",
    ]


def test_batch_progress_logger_filters_phases(capsys) -> None:
    logger = BatchProgressLogger(every_n_batches=1, phases="val")

    logger.on_batch_end(
        BatchContext(phase="train", epoch=1, batch_idx=0, num_batches=1)
    )
    logger.on_batch_end(BatchContext(phase="val", epoch=1, batch_idx=0, num_batches=1))

    assert capsys.readouterr().out.splitlines() == ["val epoch 1 batch 1/1"]


def test_metrics_table_printer_pivots_records_and_dataset_labels(capsys) -> None:
    context = MetricsContext(
        records=[
            {"epoch": 1, "phase": "train", "loss": 1.23456, "dice": 0.5},
            {
                "epoch": 1,
                "phase": "val",
                "dataset": "HipRay",
                "loss": 0.98765,
                "dice": 0.75,
            },
            {
                "epoch": 1,
                "phase": "val",
                "dataset": "JRST",
                "loss": 0.87654,
                "dice": 0.7,
            },
        ]
    )

    MetricsTablePrinter(floatfmt=".2f").on_metrics(context)
    output = capsys.readouterr().out

    assert "|   epoch | metric" in output
    assert "train" in output
    assert "val/HipRay" in output
    assert "val/JRST" in output
    assert "dice" in output
    assert "loss" in output
    assert "1.23" in output
    assert "0.75" in output


def test_metrics_table_printer_handles_empty_records(capsys) -> None:
    MetricsTablePrinter().on_metrics(MetricsContext(records=[]))

    assert capsys.readouterr().out.strip() == "No metrics to print."
