"""Framework-neutral training callbacks for FleXray."""

from .builtin import BatchProgressLogger, MetricsTablePrinter
from .config import build_callback_runner
from .contexts import BatchContext, EpochContext, MetricsContext, TrainContext
from .runner import CallbackRunner

__all__ = [
    "BatchContext",
    "BatchProgressLogger",
    "CallbackRunner",
    "EpochContext",
    "MetricsContext",
    "MetricsTablePrinter",
    "TrainContext",
    "build_callback_runner",
]
