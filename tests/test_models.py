from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest
import torch
import torch.nn as torch_nn

import fxr
import fxr.models as public_models
import fxr.models.nn as public_nn
from fxr.models import UNet
from fxr.models.nn import ConvBlock


def _first_module(
    module: torch_nn.Module, kind: type[torch_nn.Module]
) -> torch_nn.Module:
    return next(child for child in module.modules() if isinstance(child, kind))


def test_public_model_api_exports_planned_names() -> None:
    assert fxr.__all__ == [
        "augmentation",
        "callbacks",
        "config",
        "datasets",
        "drr",
        "experiment",
        "inference",
        "losses",
        "metrics",
        "models",
        "protocols",
    ]
    assert public_models.__all__ == ["UNet"]
    assert public_nn.__all__ == ["ConvBlock", "get_activation"]
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("fxr.nn")
    assert UNet.__name__ == "UNet"
    assert ConvBlock.__name__ == "ConvBlock"


def test_model_nn_initializer_is_reexport_only() -> None:
    tree = ast.parse(Path(public_nn.__file__).read_text(encoding="utf-8"))
    disallowed = (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef)

    assert not any(isinstance(node, disallowed) for node in tree.body)


def test_unet_forward_maps_input_to_logit_shape() -> None:
    model = UNet(in_channels=1, out_channels=3, filters=(4, 8), norm="none")
    x = torch.randn(2, 1, 32, 32)

    logits = model(x)

    assert logits.shape == (2, 3, 32, 32)


def test_unet_preserves_odd_spatial_size() -> None:
    model = UNet(in_channels=1, out_channels=2, filters=(4, 8, 16))
    x = torch.randn(2, 1, 31, 35)

    logits = model(x)

    assert logits.shape == (2, 2, 31, 35)


def test_unet_skip_connections_can_be_disabled() -> None:
    model = UNet(
        in_channels=1,
        out_channels=2,
        filters=(4, 8, 16),
        up_filters=(8, 4),
        skip_connections=False,
    )
    x = torch.randn(2, 1, 33, 37)

    logits = model(x)

    assert logits.shape == (2, 2, 33, 37)


def test_unet_residual_blocks_add_shortcuts_and_preserve_logit_shape() -> None:
    model = UNet(in_channels=1, out_channels=3, filters=(4, 8), residual=True)
    x = torch.randn(2, 1, 32, 32)

    logits = model(x)

    assert logits.shape == (2, 3, 32, 32)
    assert all(block.shortcut is not None for block in model.down_blocks)
    assert all(block.shortcut is not None for block in model.up_blocks)


def test_unet_conversion_options_use_named_projection_shortcuts() -> None:
    model = UNet(
        in_channels=1,
        out_channels=3,
        filters=(4, 8),
        norm="instance",
        residual=True,
        norm_after_activation=True,
        upsample_align_corners=True,
        residual_projection="always",
        residual_shortcut_norm=True,
    )

    assert model.upsample_align_corners is True
    assert isinstance(model.down_blocks[0].layers[1], torch_nn.LeakyReLU)
    assert isinstance(model.down_blocks[0].layers[2], torch_nn.InstanceNorm2d)
    assert isinstance(model.down_blocks[0].shortcut, torch_nn.Sequential)
    assert list(model.down_blocks[0].shortcut._modules) == ["conv", "instancenorm"]
    assert "down_blocks.0.shortcut.conv.weight" in model.state_dict()
    assert "down_blocks.0.shortcut.instancenorm.weight" in model.state_dict()
    assert model(torch.randn(2, 1, 32, 32)).shape == (2, 3, 32, 32)


def test_unet_dropout_path_constructs_and_runs_in_training_mode() -> None:
    model = UNet(in_channels=1, out_channels=2, filters=(4, 8), dropout=0.25)
    model.train()

    assert any(isinstance(module, torch_nn.Dropout2d) for module in model.modules())
    assert model(torch.randn(2, 1, 32, 32)).shape == (2, 2, 32, 32)


def test_unet_output_layer_returns_raw_logits() -> None:
    model = UNet(in_channels=1, out_channels=2, filters=(4,))

    assert isinstance(model.output_conv, torch_nn.Conv2d)
    assert model.output_conv.kernel_size == (1, 1)
    assert list(model.output_conv.children()) == []

    with torch.no_grad():
        model.output_conv.weight.zero_()
        model.output_conv.bias.copy_(torch.tensor([-3.0, 2.0]))

    logits = model(torch.ones(1, 1, 8, 8))

    assert torch.allclose(logits[:, 0], torch.full((1, 8, 8), -3.0))
    assert torch.allclose(logits[:, 1], torch.full((1, 8, 8), 2.0))


def test_conv_block_kaiming_initializes_conv_weights_and_zero_bias() -> None:
    block = ConvBlock(1, 4)
    conv = _first_module(block, torch_nn.Conv2d)

    assert torch.count_nonzero(conv.weight).item() > 0
    assert torch.allclose(conv.bias, torch.zeros_like(conv.bias))

    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.fill_(1.0)
    block.reset_parameters()

    assert torch.count_nonzero(conv.weight).item() > 0
    assert torch.allclose(conv.bias, torch.zeros_like(conv.bias))


def test_conv_block_group_norm_selects_valid_group_counts() -> None:
    divisible = ConvBlock(3, 64, norm="group")
    divisible_norm = _first_module(divisible, torch_nn.GroupNorm)
    assert divisible_norm.num_groups == 32

    fallback = ConvBlock(3, 7, norm="group")
    fallback_norm = _first_module(fallback, torch_nn.GroupNorm)
    assert fallback_norm.num_groups >= 1
    assert fallback_norm.num_channels % fallback_norm.num_groups == 0


def test_conv_block_residual_shortcut_handles_same_and_different_channels() -> None:
    same_channels = ConvBlock(4, 4, residual=True)
    different_channels = ConvBlock(4, 6, residual=True)

    assert isinstance(same_channels.shortcut, torch_nn.Identity)
    assert isinstance(different_channels.shortcut, torch_nn.Conv2d)
    assert same_channels(torch.randn(2, 4, 8, 8)).shape == (2, 4, 8, 8)
    assert different_channels(torch.randn(2, 4, 8, 8)).shape == (2, 6, 8, 8)


def test_unet_and_block_validation_errors_are_clear() -> None:
    with pytest.raises(ValueError, match="filters"):
        UNet(in_channels=1, out_channels=1, filters=())
    with pytest.raises(ValueError, match="up_filters"):
        UNet(in_channels=1, out_channels=1, filters=(4, 8), up_filters=(4, 4))
    with pytest.raises(ValueError, match="Unsupported norm"):
        ConvBlock(1, 4, norm="sync")
    with pytest.raises(ValueError, match="Unsupported activation"):
        ConvBlock(1, 4, activation="DefinitelyNotActivation")
    with pytest.raises(ValueError, match="residual_projection"):
        ConvBlock(1, 4, residual=True, residual_projection="sometimes")  # type: ignore[arg-type]


def test_unet_rejects_non_2d_tensor_shapes() -> None:
    model = UNet(in_channels=1, out_channels=2, filters=(4,))

    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(1, 1, 8, 8, 8))
