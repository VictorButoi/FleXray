from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

import fxr
import fxr.losses as public_losses
from fxr.losses import (
    CombinedLoss,
    DatasetRoutedLoss,
    PixelCELoss,
    SoftDiceLoss,
    iter_leaf_loss_modules,
    set_batch_reduction,
)

_CONSTANT = "tests.loss_helpers.ConstantLoss"
_DIFFERENCE = "tests.loss_helpers.DifferenceLoss"


class _ConstantLoss(torch.nn.Module):
    """Local deterministic scalar loss used by loss tests.

    Attributes:
        value: Constant scalar returned by ``forward``.
    """

    def __init__(self, value: float) -> None:
        """Store the constant loss value.

        Args:
            value: Scalar value returned by ``forward``.

        Returns:
            ``None``.
        """

        super().__init__()
        self.value = value

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        **loss_kwargs: object,
    ) -> torch.Tensor:
        """Return the configured constant as a tensor on the output device.

        Args:
            outputs: Output tensor used for device and dtype placement.
            targets: Ignored target tensor.
            **loss_kwargs: Ignored per-call loss keyword arguments.

        Returns:
            Scalar tensor containing ``value``.
        """

        del targets, loss_kwargs
        return outputs.new_tensor(self.value)


def _combo_profile(values: dict[str, float], weights: dict[str, float]) -> dict:
    combo = {}
    for name, value in values.items():
        combo[name] = {
            "_class": _CONSTANT,
            "value": value,
            "weight": weights[name],
        }
    return {"_combo_class": combo}


def _single_profile(value: float) -> dict:
    return {"_class": _CONSTANT, "value": value}


def test_public_loss_api_exports_planned_names() -> None:
    assert "losses" in fxr.__all__
    assert {"datasets", "models", "protocols"}.issubset(set(fxr.__all__))
    assert public_losses.__all__ == [
        "CombinedLoss",
        "DatasetRoutedLoss",
        "PixelCELoss",
        "SoftDiceLoss",
        "iter_leaf_loss_modules",
        "set_batch_reduction",
    ]
    assert not hasattr(public_losses, "CoveredCrossEntropy")
    assert not hasattr(public_losses, "PixelFocalLoss")
    assert SoftDiceLoss.__name__ == "SoftDiceLoss"
    assert PixelCELoss.__name__ == "PixelCELoss"


def test_loss_initializer_is_reexport_only() -> None:
    tree = ast.parse(Path(public_losses.__file__).read_text(encoding="utf-8"))
    disallowed = (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef)

    assert not any(isinstance(node, disallowed) for node in tree.body)


def test_combined_loss_tracks_raw_component_breakdown_and_clones() -> None:
    loss_func = CombinedLoss(
        fn_dict={
            "SoftDiceLoss": _ConstantLoss(2.0),
            "CrossEntropyLoss": _ConstantLoss(3.0),
        },
        fn_weights={
            "SoftDiceLoss": 0.5,
            "CrossEntropyLoss": 2.0,
        },
        fn_label_types={
            "SoftDiceLoss": None,
            "CrossEntropyLoss": None,
        },
    )

    total_loss = loss_func(torch.zeros(1), torch.zeros(1))
    breakdown = loss_func.get_last_loss_breakdown()
    breakdown["SoftDiceLoss"].add_(100.0)

    assert torch.isclose(total_loss, torch.tensor(7.0))
    assert torch.isclose(
        loss_func.get_last_loss_breakdown()["SoftDiceLoss"], torch.tensor(2.0)
    )
    assert torch.isclose(
        loss_func.get_last_loss_breakdown()["CrossEntropyLoss"], torch.tensor(3.0)
    )


def test_combined_loss_label_type_routes_outputs_and_targets() -> None:
    loss_func = CombinedLoss(
        fn_dict={"aux": importlib.import_module("tests.loss_helpers").DifferenceLoss()},
        fn_weights={"aux": 2.0},
        fn_label_types={"aux": "model"},
    )

    total = loss_func(
        {"model": torch.tensor([3.0]), "other": torch.tensor([100.0])},
        {"model": torch.tensor([1.0]), "other": torch.tensor([100.0])},
    )

    assert torch.isclose(total, torch.tensor(4.0))


def test_dataset_routed_loss_routes_to_correct_profile() -> None:
    router = DatasetRoutedLoss(
        losses={
            "standard": _combo_profile({"A": 2.0, "B": 3.0}, {"A": 1.0, "B": 1.0}),
            "covered": _combo_profile({"A": 5.0, "B": 7.0}, {"A": 1.0, "B": 1.0}),
        },
        dataset_losses={"MOOSE3D": "standard", "FluXray": "covered"},
    )

    outputs = torch.zeros(1)
    targets = torch.zeros(1)

    assert torch.isclose(
        router(outputs, targets, dataset_name="MOOSE3D"), torch.tensor(5.0)
    )
    assert torch.isclose(
        router(outputs, targets, dataset_name="FluXray"), torch.tensor(12.0)
    )


def test_dataset_routed_loss_accepts_homogeneous_dataset_name_batch() -> None:
    router = DatasetRoutedLoss(
        losses={"standard": _combo_profile({"A": 2.0}, {"A": 1.0})},
        dataset_losses={"MOOSE3D": "standard"},
    )

    loss = router(torch.zeros(1), torch.zeros(1), dataset_name=["MOOSE3D", "MOOSE3D"])

    assert torch.isclose(loss, torch.tensor(2.0))


def test_dataset_routed_loss_rejects_unknown_or_missing_routes() -> None:
    router = DatasetRoutedLoss(
        losses={"only": _combo_profile({"X": 1.0}, {"X": 1.0})},
        dataset_losses={"MOOSE3D": "only"},
    )

    with pytest.raises(KeyError, match="No loss profile registered"):
        router(torch.zeros(1), torch.zeros(1), dataset_name="Nope")
    with pytest.raises(ValueError, match="requires `dataset_name`"):
        router(torch.zeros(1), torch.zeros(1), dataset_name=None)
    with pytest.raises(ValueError, match="homogeneous batch"):
        router(torch.zeros(1), torch.zeros(1), dataset_name=["MOOSE3D", "Nope"])


def test_dataset_routed_loss_init_and_validate_errors() -> None:
    with pytest.raises(ValueError, match="unknown profile"):
        DatasetRoutedLoss(
            losses={"a": _combo_profile({"X": 1.0}, {"X": 1.0})},
            dataset_losses={"MOOSE3D": "nonexistent"},
        )
    with pytest.raises(ValueError, match="non-empty `losses`"):
        DatasetRoutedLoss(losses={}, dataset_losses={"M": "a"})
    with pytest.raises(ValueError, match="non-empty `dataset_losses`"):
        DatasetRoutedLoss(
            losses={"a": _combo_profile({"X": 1.0}, {"X": 1.0})},
            dataset_losses={},
        )
    with pytest.raises(ValueError, match="must define either"):
        DatasetRoutedLoss(losses={"bad": {"nonsense": 1}}, dataset_losses={"M": "bad"})

    router = DatasetRoutedLoss(
        losses={"a": _combo_profile({"X": 1.0}, {"X": 1.0})},
        dataset_losses={"MOOSE3D": "a"},
    )
    with pytest.raises(ValueError, match="missing entries"):
        router.validate_routing({"MOOSE3D", "FluXray"})
    with pytest.raises(ValueError, match="not present"):
        DatasetRoutedLoss(
            losses={"a": _combo_profile({"X": 1.0}, {"X": 1.0})},
            dataset_losses={"MOOSE3D": "a", "Ghost": "a"},
        ).validate_routing({"MOOSE3D"})

    router.validate_routing({"MOOSE3D"})


def test_dataset_routed_loss_single_and_combo_profiles_work() -> None:
    router = DatasetRoutedLoss(
        losses={
            "bare": _single_profile(4.0),
            "combo": _combo_profile({"X": 1.0}, {"X": 2.0}),
        },
        dataset_losses={"D1": "bare", "D2": "combo"},
    )

    assert torch.isclose(
        router(torch.zeros(1), torch.zeros(1), dataset_name="D1"), torch.tensor(4.0)
    )
    assert torch.isclose(
        router(torch.zeros(1), torch.zeros(1), dataset_name="D2"), torch.tensor(2.0)
    )


def test_dataset_routed_loss_breakdown_uses_stable_profile_prefixes() -> None:
    router = DatasetRoutedLoss(
        losses={
            "standard": _combo_profile({"A": 2.0, "B": 3.0}, {"A": 1.0, "B": 1.0}),
            "covered": _combo_profile({"A": 5.0, "C": 7.0}, {"A": 1.0, "C": 1.0}),
        },
        dataset_losses={"M": "standard", "F": "covered"},
    )

    router(torch.zeros(1), torch.zeros(1), dataset_name="F")
    breakdown = router.get_last_loss_breakdown()
    breakdown["covered/A"].add_(100.0)

    assert set(breakdown) == {"standard/A", "standard/B", "covered/A", "covered/C"}
    assert torch.isclose(
        router.get_last_loss_breakdown()["covered/A"], torch.tensor(5.0)
    )
    assert torch.isclose(
        router.get_last_loss_breakdown()["covered/C"], torch.tensor(7.0)
    )
    assert torch.isclose(
        router.get_last_loss_breakdown()["standard/A"], torch.tensor(0.0)
    )
    assert torch.isclose(
        router.get_last_loss_breakdown()["standard/B"], torch.tensor(0.0)
    )


def test_dataset_routed_loss_single_profile_breakdown_uses_class_name() -> None:
    router = DatasetRoutedLoss(
        losses={
            "bare": _single_profile(4.0),
            "combo": _combo_profile({"X": 1.0}, {"X": 1.0}),
        },
        dataset_losses={"D1": "bare", "D2": "combo"},
    )

    router(torch.zeros(1), torch.zeros(1), dataset_name="D1")
    breakdown = router.get_last_loss_breakdown()

    assert set(breakdown) == {"bare/ConstantLoss", "combo/X"}
    assert torch.isclose(breakdown["bare/ConstantLoss"], torch.tensor(4.0))
    assert torch.isclose(breakdown["combo/X"], torch.tensor(0.0))


def test_set_batch_reduction_recurses_through_routed_and_combo_profiles() -> None:
    router = DatasetRoutedLoss(
        losses={
            "combo": {
                "_combo_class": {
                    "dice": {"_class": "fxr.losses.SoftDiceLoss", "mode": "binary"},
                    "ce": {"_class": "fxr.losses.PixelCELoss", "mode": "binary"},
                }
            },
            "single": {"_class": "fxr.losses.SoftDiceLoss", "mode": "binary"},
        },
        dataset_losses={"D1": "combo", "D2": "single"},
    )

    set_batch_reduction(router, "none")

    leaves = list(iter_leaf_loss_modules(router))
    assert [module.batch_reduction for _, module in leaves] == ["none", "none", "none"]


def test_set_batch_reduction_rejects_unwrapped_leaf() -> None:
    with pytest.raises(ValueError, match="does not expose"):
        set_batch_reduction(_ConstantLoss(1.0), "none")


def test_pixel_ce_binary_multichannel_ignores_empty_labels_per_channel() -> None:
    logits = torch.randn(2, 3, 2, 2)
    y_true = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ],
            [
                [[1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
        ],
        dtype=torch.float32,
    )

    actual = PixelCELoss(
        mode="binary",
        from_logits=True,
        ignore_empty_labels=True,
    )(logits, y_true)

    flat_logits = logits.reshape(2 * 3, 1, 2, 2)
    flat_true = y_true.reshape(2 * 3, 1, 2, 2)
    per_channel = (
        F.binary_cross_entropy_with_logits(
            flat_logits,
            flat_true,
            reduction="none",
        )
        .squeeze(1)
        .mean(dim=(1, 2))
    )
    keep = flat_true.sum(dim=(1, 2, 3)) > 0
    expected = per_channel[keep].mean()

    assert torch.allclose(actual, expected)


def test_pixel_ce_binary_ignore_background_drops_background_when_foreground_exists() -> (
    None
):
    logits = torch.tensor(
        [[[[5.0, 5.0], [5.0, 5.0]], [[-5.0, -5.0], [-5.0, -5.0]]]],
        dtype=torch.float32,
    )
    y_true = torch.tensor(
        [[[[0.0, 1.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]]]],
        dtype=torch.float32,
    )

    actual = PixelCELoss(
        mode="binary",
        from_logits=True,
        ignore_background=True,
    )(logits, y_true)
    expected = F.binary_cross_entropy_with_logits(
        logits[:, 1:],
        y_true[:, 1:],
        reduction="none",
    ).mean()

    assert torch.allclose(actual, expected)


def test_pixel_ce_reduction_shapes_and_pixel_weights() -> None:
    logits = torch.tensor([[[[2.0, -1.0], [0.5, -0.5]]]], dtype=torch.float32)
    y_true = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32)
    pixel_weights = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=torch.float32)

    loss = PixelCELoss(
        mode="binary",
        from_logits=True,
        reduction="none",
        batch_reduction="none",
    )(logits, y_true)
    weighted = PixelCELoss(
        mode="binary",
        from_logits=True,
        reduction="mean",
        batch_reduction="mean",
        pixel_weights=pixel_weights,
    )(logits, y_true)
    expected = (
        F.binary_cross_entropy_with_logits(
            logits[..., :1, :1],
            y_true[..., :1, :1],
            reduction="none",
        )
        .squeeze(1)
        .mean()
    )

    assert loss.shape == (1, 2, 2)
    assert torch.allclose(weighted, expected)


def test_pixel_ce_onehot_multiclass_and_auto_modes() -> None:
    logits = torch.tensor(
        [
            [
                [[2.5, 0.3], [0.1, -0.2]],
                [[0.2, 1.4], [0.7, 2.0]],
                [[-1.1, -0.6], [1.8, 0.4]],
            ]
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([[[0, 1], [2, 1]]], dtype=torch.long)
    onehot = F.one_hot(labels, num_classes=3).permute(0, 3, 1, 2).float()
    log_probs = F.log_softmax(logits, dim=1)

    expected = F.cross_entropy(logits, labels, reduction="none").mean()
    onehot_loss = PixelCELoss(mode="onehot", from_logits=True)(logits, onehot)
    auto_onehot_loss = PixelCELoss(mode="auto", from_logits=True)(logits, onehot)
    multiclass_loss = PixelCELoss(mode="multiclass", from_logits=False)(
        log_probs, labels
    )
    auto_multiclass_loss = PixelCELoss(mode="auto", from_logits=True)(logits, labels)

    assert torch.allclose(onehot_loss, expected)
    assert torch.allclose(auto_onehot_loss, onehot_loss)
    assert torch.allclose(multiclass_loss, expected)
    assert torch.allclose(auto_multiclass_loss, expected)


def test_soft_dice_binary_from_logits_ignore_empty_and_background() -> None:
    logits = torch.tensor(
        [
            [
                [[6.0, 6.0], [6.0, 6.0]],
                [[-6.0, -6.0], [-6.0, -6.0]],
            ]
        ],
        dtype=torch.float32,
    )
    y_true = torch.tensor(
        [
            [
                [[0.0, 1.0], [1.0, 1.0]],
                [[1.0, 0.0], [0.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )

    with_background = SoftDiceLoss(
        mode="binary",
        from_logits=True,
        ignore_empty_labels=True,
        ignore_background=False,
    )(logits, y_true)
    ignore_background = SoftDiceLoss(
        mode="binary",
        from_logits=True,
        ignore_empty_labels=True,
        ignore_background=True,
    )(logits, y_true)

    assert with_background.item() < 0.6
    assert ignore_background.item() > 0.99


def test_soft_dice_ignore_background_keeps_background_for_all_background_targets() -> (
    None
):
    logits = torch.tensor(
        [
            [
                [[6.0, 6.0], [6.0, 6.0]],
                [[-6.0, -6.0], [-6.0, -6.0]],
            ]
        ],
        dtype=torch.float32,
    )
    y_true = torch.tensor(
        [
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )

    loss = SoftDiceLoss(
        mode="binary",
        from_logits=True,
        ignore_empty_labels=True,
        ignore_background=True,
    )(logits, y_true)

    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_soft_dice_reduction_shapes_and_pixel_weights() -> None:
    logits = torch.tensor(
        [
            [
                [[6.0, -6.0], [-6.0, -6.0]],
                [[-6.0, 6.0], [6.0, -6.0]],
            ]
        ],
        dtype=torch.float32,
    )
    y_true = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    pixel_weights = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]], dtype=torch.float32)

    unreduced = SoftDiceLoss(
        mode="binary",
        from_logits=True,
        reduction="none",
        batch_reduction="none",
    )(logits, y_true)
    class_reduced = SoftDiceLoss(
        mode="binary",
        from_logits=True,
        reduction="mean",
        batch_reduction="none",
    )(logits, y_true)
    batch_reduced = SoftDiceLoss(
        mode="binary",
        from_logits=True,
        reduction="none",
        batch_reduction="mean",
    )(logits, y_true)
    masked = SoftDiceLoss(
        mode="binary",
        from_logits=True,
        pixel_weights=pixel_weights,
    )(logits, y_true)
    expected_masked = SoftDiceLoss(mode="binary", from_logits=True)(
        logits[..., :1, :],
        y_true[..., :1, :],
    )

    assert unreduced.shape == (1, 2)
    assert class_reduced.shape == (1,)
    assert batch_reduced.shape == (2,)
    assert torch.allclose(masked, expected_masked, atol=1e-6)


def test_soft_dice_onehot_multiclass_and_auto_modes() -> None:
    logits = torch.tensor(
        [
            [
                [[4.0, -4.0], [-4.0, -4.0]],
                [[-4.0, 4.0], [-4.0, 4.0]],
                [[-4.0, -4.0], [4.0, -4.0]],
            ]
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([[[0, 1], [2, 1]]], dtype=torch.long)
    onehot = F.one_hot(labels, num_classes=3).permute(0, 3, 1, 2).float()

    onehot_loss = SoftDiceLoss(mode="onehot", from_logits=True)(logits, onehot)
    auto_onehot_loss = SoftDiceLoss(mode="auto", from_logits=True)(logits, onehot)
    multiclass_loss = SoftDiceLoss(mode="multiclass", from_logits=True)(logits, labels)
    auto_multiclass_loss = SoftDiceLoss(mode="auto", from_logits=True)(logits, labels)

    assert onehot_loss.item() < 0.01
    assert torch.allclose(auto_onehot_loss, onehot_loss)
    assert torch.allclose(multiclass_loss, onehot_loss)
    assert torch.allclose(auto_multiclass_loss, multiclass_loss)


def _partial_labeled_profile() -> dict:
    return {
        "_combo_class": {
            "dice": {
                "_class": "SoftDiceLoss",
                "from_logits": True,
                "mode": "binary",
                "ignore_empty_labels": True,
                "weight": 1.0,
            },
            "ce": {
                "_class": "PixelCELoss",
                "from_logits": True,
                "mode": "binary",
                "ignore_empty_labels": True,
                "weight": 1.0,
            },
        }
    }


def _two_empty_channel_targets() -> torch.Tensor:
    return torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]],
            [[[1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]],
        ],
        dtype=torch.float32,
    )


def test_pixel_ce_supervise_empty_label_ids_penalizes_selected_empty_channels() -> None:
    logits = torch.randn(2, 3, 2, 2)
    y_true = _two_empty_channel_targets()
    loss = PixelCELoss(mode="binary", from_logits=True, ignore_empty_labels=True)

    actual = loss(logits, y_true, supervise_empty_label_ids=[1])

    flat_logits = logits.reshape(6, 1, 2, 2)
    flat_true = y_true.reshape(6, 1, 2, 2)
    per_channel = F.binary_cross_entropy_with_logits(
        flat_logits, flat_true, reduction="none"
    ).mean(dim=(1, 2, 3))
    keep = flat_true.sum(dim=(1, 2, 3)) > 0
    keep[1] = True  # sample 0 / channel 1 is empty but supervised as a known negative
    assert keep[5].item() is False  # sample 1 / channel 2 stays ignored
    assert torch.allclose(actual, per_channel[keep].mean())

    with pytest.raises(ValueError, match="ignore_empty_labels"):
        PixelCELoss(mode="binary", from_logits=True)(
            logits, y_true, supervise_empty_label_ids=[1]
        )
    with pytest.raises(ValueError, match="unique"):
        loss(logits, y_true, supervise_empty_label_ids=[1, 1])
    with pytest.raises(ValueError, match="unique"):
        loss(logits, y_true, supervise_empty_label_ids=[3])


def test_dataset_routed_loss_passes_supervise_empty_ids_only_to_partial_pixel_ce() -> None:
    router = DatasetRoutedLoss(
        losses={"partial": _partial_labeled_profile()},
        dataset_losses={"ElbowCT": "partial", "Other": "partial"},
    )
    router.configure_supervise_empty_label_ids({"ElbowCT": [1], "Other": []})
    logits = torch.randn(2, 3, 2, 2)
    y_true = _two_empty_channel_targets()

    assert router.dataset_supervise_empty_label_ids == {"ElbowCT": (1,)}
    expected = SoftDiceLoss(
        from_logits=True, mode="binary", ignore_empty_labels=True
    )(logits, y_true) + PixelCELoss(
        from_logits=True, mode="binary", ignore_empty_labels=True
    )(logits, y_true, supervise_empty_label_ids=[1])
    assert torch.allclose(router(logits, y_true, dataset_name="ElbowCT"), expected)
    plain = router(logits, y_true, dataset_name="Other")
    assert not torch.allclose(plain, expected)


def test_dataset_routed_loss_supervise_empty_requires_binary_pixel_ce() -> None:
    dice_only = {"_class": "SoftDiceLoss", "from_logits": True, "mode": "binary"}
    router = DatasetRoutedLoss(losses={"p": dice_only}, dataset_losses={"A": "p"})
    with pytest.raises(AssertionError, match="PixelCELoss"):
        router.configure_supervise_empty_label_ids({"A": [1]})

    onehot_ce = {
        "_class": "PixelCELoss",
        "from_logits": True,
        "mode": "onehot",
        "ignore_empty_labels": True,
    }
    router = DatasetRoutedLoss(losses={"p": onehot_ce}, dataset_losses={"A": "p"})
    with pytest.raises(AssertionError, match="binary"):
        router.configure_supervise_empty_label_ids({"A": [1]})


@pytest.mark.parametrize("log_loss", [False, True])
def test_soft_dice_fractional_weights_preserve_loss_and_gradients(log_loss):
    y_true = torch.tensor([[[[1., 0.]]], [[[0., 0.]]]])
    losses, gradients = [], []
    for scale in (0.1, 1., 10.):
        y_pred = torch.tensor(
            [[[[0.8, 0.1]]], [[[0.2, 0.7]]]], requires_grad=True
        )
        loss = SoftDiceLoss(
            mode="binary", weights=[scale], ignore_empty_labels=True,
            log_loss=log_loss,
        )(y_pred, y_true)
        loss.backward()
        assert torch.isfinite(y_pred.grad).all()
        assert torch.equal(y_pred.grad[1], torch.zeros_like(y_pred.grad[1]))
        reference = SoftDiceLoss(mode="binary", log_loss=log_loss)(
            y_pred[:1].detach(), y_true[:1]
        )
        torch.testing.assert_close(loss.detach(), reference)
        losses.append(loss.detach())
        gradients.append(y_pred.grad.clone())
    for loss, gradient in zip(losses[1:], gradients[1:]):
        torch.testing.assert_close(loss, losses[0])
        torch.testing.assert_close(gradient, gradients[0])
