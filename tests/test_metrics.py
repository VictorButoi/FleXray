from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

import fxr
import fxr.metrics as public_metrics
from fxr.losses import SoftDiceLoss
from fxr.metrics import dice_score, hd95, soft_dice_score


def test_public_metric_api_exports_dice_and_hd95_only() -> None:
    assert "metrics" in fxr.__all__
    assert public_metrics.__all__ == [
        "dice_score",
        "hd95",
        "soft_dice_score",
    ]
    assert not hasattr(public_metrics, "pixel_accuracy")
    assert not hasattr(public_metrics, "threshold_accuracy")


def test_metric_initializer_is_reexport_only() -> None:
    tree = ast.parse(Path(public_metrics.__file__).read_text(encoding="utf-8"))
    disallowed = (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef)

    assert not any(isinstance(node, disallowed) for node in tree.body)


def test_dice_binary_multichannel_keeps_overlapping_masks_independent() -> None:
    y_true = torch.tensor(
        [[[[1.0, 1.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]]]
    )
    y_pred = torch.tensor(
        [[[[1.0, 1.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]]]
    )

    score = dice_score(
        y_pred,
        y_true,
        mode="binary",
        reduction="none",
        batch_reduction="none",
        smooth=0.0,
    )

    assert torch.allclose(score, torch.tensor([[1.0, 2.0 / 3.0]]))


def test_dice_modes_auto_and_logits() -> None:
    logits = torch.tensor(
        [
            [
                [[8.0, -8.0], [-8.0, -8.0]],
                [[-8.0, 8.0], [-8.0, 8.0]],
                [[-8.0, -8.0], [8.0, -8.0]],
            ]
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([[[0, 1], [2, 1]]], dtype=torch.long)
    onehot = F.one_hot(labels, num_classes=3).permute(0, 3, 1, 2).float()

    binary_logits = torch.tensor([[[[8.0, -8.0], [-8.0, 8.0]]]])
    binary_true = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])

    onehot_score = dice_score(logits, onehot, mode="onehot", from_logits=True)
    auto_onehot_score = dice_score(logits, onehot, mode="auto", from_logits=True)
    multiclass_score = dice_score(logits, labels, mode="multiclass", from_logits=True)
    auto_multiclass_score = dice_score(logits, labels, mode="auto", from_logits=True)
    binary_score = dice_score(
        binary_logits,
        binary_true,
        mode="binary",
        from_logits=True,
    )

    assert onehot_score.item() == pytest.approx(1.0)
    assert torch.allclose(auto_onehot_score, onehot_score)
    assert torch.allclose(multiclass_score, onehot_score)
    assert torch.allclose(auto_multiclass_score, multiclass_score)
    assert binary_score.item() == pytest.approx(1.0)


def test_dice_weights_reductions_and_ignored_classes() -> None:
    y_true = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 1.0]]]])
    y_pred = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]]]])

    weighted = dice_score(
        y_pred,
        y_true,
        mode="binary",
        reduction="none",
        batch_reduction="none",
        weights=[1.0, 2.0, 3.0],
        ignore_index=[1],
        ignore_empty_labels=False,
        smooth=0.0,
    )

    assert torch.allclose(weighted, torch.tensor([[1.0, 0.0, 2.0]]))


def test_dice_ignores_empty_labels_and_background() -> None:
    empty_true = torch.tensor([[[[1.0, 0.0]], [[0.0, 0.0]]]])
    empty_pred = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]])

    with_empty = dice_score(
        empty_pred,
        empty_true,
        mode="binary",
        ignore_empty_labels=False,
        smooth=0.0,
    )
    without_empty = dice_score(
        empty_pred,
        empty_true,
        mode="binary",
        ignore_empty_labels=True,
        smooth=0.0,
    )

    bg_true = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]])
    bg_pred = torch.tensor([[[[0.0, 1.0]], [[1.0, 0.0]]]])
    with_background = dice_score(
        bg_pred,
        bg_true,
        mode="binary",
        ignore_empty_labels=True,
        ignore_background=False,
        smooth=0.0,
    )
    without_background = dice_score(
        bg_pred,
        bg_true,
        mode="binary",
        ignore_empty_labels=True,
        ignore_background=True,
        smooth=0.0,
    )

    assert with_empty.item() == pytest.approx(0.5)
    assert without_empty.item() == pytest.approx(1.0)
    assert with_background.item() == pytest.approx(0.5)
    assert without_background.item() == pytest.approx(1.0)


def test_soft_dice_score_matches_one_minus_soft_dice_loss() -> None:
    logits = torch.tensor(
        [
            [
                [[2.5, 0.3], [0.1, -0.2]],
                [[0.2, 1.4], [0.7, 2.0]],
                [[-1.1, -0.6], [1.8, 0.4]],
            ],
            [
                [[0.5, 2.0], [1.0, -0.3]],
                [[1.3, 0.1], [0.2, 0.4]],
                [[-0.5, -1.0], [0.8, 1.6]],
            ],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([[[0, 1], [2, 1]], [[1, 0], [2, 2]]], dtype=torch.long)
    onehot = F.one_hot(labels, num_classes=3).permute(0, 3, 1, 2).float()
    pixel_weights = torch.tensor(
        [[[1.0, 1.0], [0.0, 1.0]], [[1.0, 0.5], [1.0, 1.0]]]
    )

    score = soft_dice_score(
        logits,
        onehot,
        mode="onehot",
        from_logits=True,
        reduction="mean",
        batch_reduction="none",
        weights=[1.0, 0.5, 2.0],
        pixel_weights=pixel_weights,
    )
    loss = SoftDiceLoss(
        mode="onehot",
        from_logits=True,
        reduction="mean",
        batch_reduction="none",
        weights=[1.0, 0.5, 2.0],
        pixel_weights=pixel_weights,
    )(logits, onehot)

    assert torch.allclose(score, 1.0 - loss)


def test_hd95_perfect_empty_one_empty_and_shifted_masks() -> None:
    y_true = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
    y_true[:, :, 1:4, 1:4] = 1.0

    assert hd95(y_true.clone(), y_true, mode="binary").item() == pytest.approx(0.0)

    empty = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
    assert hd95(
        empty,
        empty,
        mode="binary",
        ignore_empty_labels=False,
    ).item() == pytest.approx(0.0)

    one_empty_true = torch.zeros((1, 1, 5, 4), dtype=torch.float32)
    one_empty_true[:, :, 2, 2] = 1.0
    one_empty_pred = torch.zeros_like(one_empty_true)
    assert hd95(one_empty_pred, one_empty_true, mode="binary").item() == pytest.approx(
        5.0
    )

    shifted_true = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
    shifted_pred = torch.zeros_like(shifted_true)
    shifted_true[:, :, 2, 2] = 1.0
    shifted_pred[:, :, 2, 3] = 1.0
    assert hd95(shifted_pred, shifted_true, mode="binary").item() == pytest.approx(1.0)


def test_hd95_ignores_empty_labels_background_and_indices() -> None:
    y_true = torch.zeros((1, 3, 5, 5), dtype=torch.float32)
    y_pred = torch.zeros_like(y_true)
    y_true[:, 0, 0, 0] = 1.0
    y_pred[:, 0, 4, 4] = 1.0
    y_true[:, 1, 2, 2] = 1.0
    y_pred[:, 1, 2, 2] = 1.0
    y_pred[:, 2, 4, 4] = 1.0

    ignore_background = hd95(
        y_pred,
        y_true,
        mode="binary",
        ignore_background=True,
        ignore_empty_labels=True,
    )
    ignore_empty_and_index = hd95(
        y_pred,
        y_true,
        mode="binary",
        ignore_empty_labels=True,
        ignore_index=[0],
    )

    assert ignore_background.item() == pytest.approx(0.0)
    assert ignore_empty_and_index.item() == pytest.approx(0.0)


@pytest.mark.parametrize("metric", [dice_score, soft_dice_score, hd95])
def test_metric_fractional_weights_are_invariant_to_positive_rescaling(metric):
    y_true = torch.tensor([[[[1., 0.]], [[1., 1.]]]])
    y_pred = torch.tensor([[[[1., 0.]], [[1., 0.]]]])
    fractional = metric(y_pred, y_true, mode="binary", weights=[0.1, 0.2])
    rescaled = metric(y_pred, y_true, mode="binary", weights=[1., 2.])
    torch.testing.assert_close(fractional, rescaled)


@pytest.mark.parametrize("metric", [dice_score, soft_dice_score, hd95])
def test_metric_mean_excludes_zero_weight_samples(metric):
    y_true = torch.tensor([[[[1., 0.]]], [[[1., 0.]]]])
    y_pred = torch.tensor([[[[0., 1.]]], [[[1., 0.]]]])
    actual = metric(
        y_pred, y_true, mode="binary", weights=torch.tensor([[0.2], [0.]])
    )
    expected = metric(y_pred[:1], y_true[:1], mode="binary")
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("metric", [dice_score, soft_dice_score, hd95])
def test_entirely_ignored_metric_batch_returns_zero(metric):
    y_true = torch.zeros(2, 2, 2, 2)
    actual = metric(torch.ones_like(y_true), y_true, mode="binary", ignore_empty_labels=True)
    assert actual.item() == 0.


@pytest.mark.parametrize("mode", ["auto", "onehot", "multiclass"])
@pytest.mark.parametrize("channels", [2, 3])
@pytest.mark.parametrize("metric", [dice_score, hd95])
def test_categorical_threshold_is_ignored_for_all_channel_counts(mode, channels, metric):
    y_pred = torch.zeros(1, channels, 2, 2)
    y_pred[:, 0] = 0.4
    y_pred[:, 1] = 0.6
    y_true = torch.zeros_like(y_pred)
    y_true[:, 1] = 1.
    if mode == "multiclass":
        y_true = y_true.argmax(dim=1)
    actual = metric(y_pred, y_true, mode=mode, threshold=0.9)
    assert actual.item() == pytest.approx(1. if metric is dice_score else 0.)


def test_binary_threshold_applies_to_every_overlapping_channel():
    y_pred = torch.full((1, 2, 2, 2), 0.6)
    y_true = torch.ones_like(y_pred)
    assert dice_score(y_pred, y_true, mode="binary", threshold=0.5).item() == 1.
    assert dice_score(
        y_pred, y_true, mode="binary", threshold=0.9, smooth=0.
    ).item() == 0.
    assert hd95(y_pred, y_true, mode="binary", threshold=0.5).item() == 0.
    assert hd95(
        y_pred, y_true, mode="binary", threshold=0.9
    ).item() == pytest.approx(2. ** 0.5)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("mode", ["binary", "onehot", "multiclass"])
@pytest.mark.parametrize("from_logits", [False, True])
def test_dice_rejects_nonfinite_predictions(value, mode, from_logits):
    y_pred = torch.zeros(1, 2, 2, 2)
    y_pred[0, 1, 0, 0] = value
    y_true = torch.zeros_like(y_pred)
    if mode == "multiclass":
        y_true = y_true.argmax(dim=1)
    with pytest.raises(ValueError, match="non-finite"):
        dice_score(y_pred, y_true, mode=mode, from_logits=from_logits)


def test_hd95_matches_paper_maximum_of_directed_percentiles():
    y_true = torch.zeros(1, 1, 1, 20)
    y_true[..., [1, 5, 9]] = 1.
    y_pred = y_true.clone()
    y_pred[..., 19] = 1.
    # Directed distances are [0, 0, 0, 10] and [0, 0, 0].
    # Their 95th percentiles are 8.5 and 0; pooling would give 7.
    assert hd95(y_pred, y_true, mode="binary").item() == pytest.approx(8.5)
    assert hd95(y_true, y_pred, mode="binary").item() == pytest.approx(8.5)
