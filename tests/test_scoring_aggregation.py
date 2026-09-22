"""Regression tests for Dice aggregation across partially ignored batches."""

from types import SimpleNamespace

import pytest
import torch

from fxr.experiment import EvalSetMetricLogger, FleXrayTrainExperiment


def _experiment():
    """Return a lightweight experiment that exercises real metric aggregation."""
    experiment = object.__new__(FleXrayTrainExperiment)
    experiment.model = torch.nn.Identity()
    experiment.ema = SimpleNamespace(policy=SimpleNamespace(enabled=False))
    experiment.run_step = lambda batch, **kwargs: batch
    return experiment


def _outputs():
    """Return a correct image, an ignored image, and an incorrect image."""
    return {
        "y_pred": torch.tensor([10., -10., -10.]).reshape(3, 1, 1, 1),
        "y_true": torch.tensor([1., 0., 1.]).reshape(3, 1, 1, 1),
        "loss": torch.tensor([2., 4., 6.]),
    }


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_epoch_dice_uses_eligible_counts_and_loss_uses_batch_size(batch_size):
    outputs = _outputs()
    experiment = _experiment()
    experiment.val_dl = [
        {**{key: value[start:start + batch_size] for key, value in outputs.items()},
         "dataset_name": "Toy"}
        for start in range(0, 3, batch_size)
    ]
    metrics = experiment.run_phase("val", 0)
    assert metrics["dice"] == pytest.approx(0.5, abs=1e-6)
    assert metrics["Toy/dice"] == pytest.approx(0.5, abs=1e-6)
    assert metrics["loss"] == pytest.approx(4.)
    assert metrics["Toy/loss"] == pytest.approx(4.)


def test_epoch_omits_dice_when_no_images_are_eligible():
    experiment = _experiment()
    outputs = _outputs()
    experiment.val_dl = [
        {**{key: value[1:2] for key, value in outputs.items()}, "dataset_name": "Empty"}
    ]
    assert experiment.run_phase("val", 0) == {"loss": 4., "Empty/loss": 4.}


def test_epoch_counts_each_dataset_independently():
    experiment = _experiment()
    outputs = _outputs()
    experiment.val_dl = [
        {**{key: value[:2] for key, value in outputs.items()}, "dataset_name": "A"},
        {**{key: value[2:] for key, value in outputs.items()}, "dataset_name": "B"},
    ]
    metrics = experiment.run_phase("val", 0)
    assert metrics["dice"] == pytest.approx(0.5, abs=1e-6)
    assert metrics["A/dice"] == pytest.approx(1.)
    assert metrics["B/dice"] == pytest.approx(0., abs=1e-6)


def test_background_only_image_remains_eligible_for_epoch_dice():
    experiment = _experiment()
    experiment.val_dl = [{
        "y_pred": torch.tensor([10., -10.]).reshape(1, 2, 1, 1),
        "y_true": torch.tensor([1., 0.]).reshape(1, 2, 1, 1),
        "loss": torch.tensor(0.),
    }]
    assert experiment.run_phase("val", 0)["dice"] == pytest.approx(1.)


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("min_area", [0., 0.25])
def test_evalset_dice_is_independent_of_batch_splitting(batch_size, min_area):
    experiment = _experiment()
    outputs = _outputs()
    dataset = [
        {
            "y_pred": torch.cat((torch.full_like(pred, -10.), pred)),
            "y_true": torch.cat((torch.zeros_like(true), true)),
        }
        for pred, true in zip(outputs["y_pred"], outputs["y_true"])
    ]
    experiment._prediction_step = lambda batch, **kwargs: batch.batch
    logger = EvalSetMetricLogger(
        experiment, data={"Xray": {"Toy": {}}}, batch_size=batch_size,
        min_ground_truth_area_fraction=min_area,
    )
    assert logger._evaluate_dataset("Toy", dataset) == pytest.approx(0.5, abs=1e-6)
    assert experiment.model.training


def test_evalset_empty_targets_keep_the_documented_zero_fallback():
    experiment = _experiment()
    experiment._prediction_step = lambda batch, **kwargs: batch.batch
    logger = EvalSetMetricLogger(experiment, data={"Xray": {"Toy": {}}})
    dataset = [{"y_pred": torch.zeros(1, 2, 2), "y_true": torch.zeros(1, 2, 2)}]
    assert logger._evaluate_dataset("Toy", dataset) == 0.
