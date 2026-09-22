from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import runpy
import subprocess
import sys
import textwrap
from collections.abc import Callable

import pytest

from fxr.launch._extras import TRAINING_EXTRA_MESSAGE
from fxr.launch.cli import main as train_main
from fxr.launch.submit_cli import main as submit_main


def _hide_module(monkeypatch: pytest.MonkeyPatch, hidden_name: str) -> None:
    original_find_spec: Callable = importlib.util.find_spec

    def fake_find_spec(name: str, *args: object, **kwargs: object) -> object:
        if name == hidden_name:
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)


def test_fxr_train_reports_missing_training_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hide_module(monkeypatch, "kornia")

    with pytest.raises(SystemExit) as exc_info:
        train_main(["--base", "base"])

    assert str(exc_info.value) == TRAINING_EXTRA_MESSAGE


def test_fxr_submit_reports_missing_training_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hide_module(monkeypatch, "submitit")

    with pytest.raises(SystemExit) as exc_info:
        submit_main(
            [
                "--cluster-cfg",
                "cluster.yml",
                "--exp-config",
                "sweep.yml",
            ]
        )

    assert str(exc_info.value) == TRAINING_EXTRA_MESSAGE


def test_training_and_callbacks_explain_missing_wandb() -> None:
    """Missing WandB fails before trainer setup or callback prediction work."""
    code = textwrap.dedent("""
        import sys
        from types import SimpleNamespace
        sys.modules['wandb'] = None
        from fxr.experiment import (
            EvalSetMetricLogger, TrainExperiment, WandbSamplePredictionLogger,
        )
        from fxr.experiment._wandb import WANDB_EXTRA_MESSAGE

        experiment = SimpleNamespace(val_datasets={'sample': object()})
        sample_logger = WandbSamplePredictionLogger(experiment)
        eval_logger = EvalSetMetricLogger(experiment, data=None)
        eval_logger.datasets = {'sample': object()}
        for action in (
            lambda: TrainExperiment._init_wandb(experiment),
            lambda: sample_logger(epoch=1),
            lambda: eval_logger(epoch=1),
        ):
            try:
                action()
            except ImportError as exc:
                assert str(exc) == WANDB_EXTRA_MESSAGE
            else:
                raise AssertionError('Expected the WandB installation hint')

        # Disabled and empty callbacks do not need a logging backend.
        sample_logger.every = 0
        eval_logger.every = 0
        sample_logger(epoch=1)
        eval_logger(epoch=1)
        sample_logger.every = eval_logger.every = 1
        experiment.val_datasets = {}
        eval_logger.datasets = {}
        sample_logger(epoch=1)
        eval_logger(epoch=1)
    """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(Path("src").resolve())},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError("Missing WandB dependency", name="wandb_dependency"),
        ImportError("Incompatible WandB dependency"),
    ],
)
def test_wandb_import_preserves_broken_install_errors(monkeypatch, error) -> None:
    """Do not misreport a broken WandB installation as an absent extra."""
    import builtins

    original_import = builtins.__import__

    def import_with_broken_wandb(name, *args, **kwargs):
        if name == "wandb":
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_with_broken_wandb)
    with pytest.raises(type(error)) as exc_info:
        runpy.run_path("src/fxr/experiment/_wandb.py")
    assert exc_info.value is error
