"""Behavior tests for model-only initialization sources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from safetensors.torch import save_file

import fxr.experiment.initialization as initialization_module
import fxr.inference.artifacts as artifacts_module
from fxr.experiment.initialization import initialize_model


def _model_config(
    *, out_channels: int = 2, kernel_size: int = 1
) -> dict[str, Any]:
    """Return a small serializable convolution model config.

    Returns:
        Direct model constructor config.
    """

    return {
        "_class": "torch.nn.Conv2d",
        "in_channels": 1,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "bias": False,
    }


def _save_bundle(
    root: Path,
    state: dict[str, torch.Tensor],
    *,
    model_config: dict[str, Any],
    labels: tuple[str, ...],
) -> None:
    """Write the minimal pretrained-bundle artifacts used by tests.

    Args:
        root: Destination bundle directory.
        state: Model state mapping saved as safetensors.
        model_config: Architecture metadata stored in ``config.yml``.
        labels: Ordered output labels stored in ``label_schema.json``.

    Returns:
        ``None``.
    """

    root.mkdir()
    save_file(state, str(root / "model.safetensors"))
    (root / "config.yml").write_text(
        yaml.safe_dump({"model": model_config}, sort_keys=False),
        encoding="utf-8",
    )
    (root / "label_schema.json").write_text(
        json.dumps({"label_names": list(labels)}),
        encoding="utf-8",
    )


def test_local_bundle_loads_strict_weights_and_ignores_compile_metadata(
    tmp_path: Path,
) -> None:
    source = torch.nn.Conv2d(1, 2, 1, bias=False)
    with torch.no_grad():
        source.weight.fill_(3.0)
    model_config = _model_config()
    model_config["compile_cfg"] = {"enabled": False}
    bundle = tmp_path / "bundle"
    labels = ("background", "bone")
    _save_bundle(
        bundle,
        source.state_dict(),
        model_config=model_config,
        labels=labels,
    )
    target = torch.nn.Conv2d(1, 2, 1, bias=False)
    with torch.no_grad():
        target.weight.zero_()
    provenance = initialize_model(
        target,
        {"kind": "pretrained", "source": str(bundle)},
        expected_label_names=labels,
        expected_model_config={**_model_config(), "compile_cfg": {"enabled": True}},
    )
    assert provenance["kind"] == "pretrained"
    assert provenance["artifact"] == "local_bundle"
    assert torch.equal(target.weight, source.weight)


def test_bundle_metadata_requires_exact_labels_and_architecture(tmp_path: Path) -> None:
    source = torch.nn.Conv2d(1, 2, 1, bias=False)
    bundle = tmp_path / "bundle"
    labels = ("background", "bone")
    _save_bundle(
        bundle,
        source.state_dict(),
        model_config=_model_config(),
        labels=labels,
    )

    with pytest.raises(ValueError, match="label order"):
        initialize_model(
            torch.nn.Conv2d(1, 2, 1, bias=False),
            {"kind": "pretrained", "source": str(bundle)},
            expected_label_names=tuple(reversed(labels)),
            expected_model_config=_model_config(),
        )

    with pytest.raises(ValueError, match="model config"):
        initialize_model(
            torch.nn.Conv2d(1, 2, 1, bias=False),
            {"kind": "pretrained", "source": str(bundle)},
            expected_label_names=labels,
            expected_model_config=_model_config(kernel_size=3),
        )


def test_trusted_run_loads_only_model_state(tmp_path: Path) -> None:
    source = torch.nn.Conv2d(1, 2, 1, bias=False)
    with torch.no_grad():
        source.weight.fill_(4.0)
    run_dir = tmp_path / "source-run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "config.yml").write_text(
        yaml.safe_dump({"model": _model_config()}, sort_keys=False),
        encoding="utf-8",
    )
    torch.save(
        {"model": source.state_dict(), "optim": {"ignored": True}, "epoch": 99},
        run_dir / "checkpoints" / "last.pt",
    )
    target = torch.nn.Conv2d(1, 2, 1, bias=False)
    provenance = initialize_model(
        target,
        {"kind": "run", "source": str(run_dir), "checkpoint": "last"},
        expected_label_names=None,
        expected_model_config=_model_config(),
    )

    assert provenance["kind"] == "run"
    assert torch.equal(target.weight, source.weight)
    assert provenance["checkpoint"] == "last.pt"


def test_huggingface_bundle_threads_pinned_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = torch.nn.Conv2d(1, 2, 1, bias=False)
    bundle = tmp_path / "bundle"
    labels = ("background", "bone")
    _save_bundle(
        bundle,
        source.state_dict(),
        model_config=_model_config(),
        labels=labels,
    )
    files = {
        "model.safetensors": bundle / "model.safetensors",
        "config.yml": bundle / "config.yml",
        "label_schema.json": bundle / "label_schema.json",
    }
    calls: list[tuple[str, str, str | None]] = []

    def fake_download(
        *,
        repo_id: str,
        filename: str,
        subfolder: str | None = None,
        revision: str | None,
    ) -> str:
        filename = f"{subfolder}/{filename}" if subfolder else filename
        calls.append((repo_id, filename, revision))
        if filename not in files:
            raise FileNotFoundError(filename)
        return str(files[filename])

    monkeypatch.setattr(artifacts_module, "hf_hub_download", fake_download)
    target = torch.nn.Conv2d(1, 2, 1, bias=False)
    provenance = initialize_model(
        target,
        {
            "kind": "pretrained",
            "source": "example/flexray",
            "revision": "abc123",
        },
        expected_label_names=labels,
        expected_model_config=_model_config(),
    )

    assert provenance["revision"] == "abc123"
    assert torch.equal(target.weight, source.weight)
    assert all(call[2] == "abc123" for call in calls)


def test_standalone_safetensors_requires_explicit_unverified_label_opt_in(
    tmp_path: Path,
) -> None:
    """Bare weights are usable only after persisting the label-order risk."""

    source = torch.nn.Conv2d(1, 2, 1, bias=False)
    weights_path = tmp_path / "weights.safetensors"
    save_file(source.state_dict(), str(weights_path))
    target = torch.nn.Conv2d(1, 2, 1, bias=False)

    with pytest.raises(ValueError, match="Standalone safetensors cannot verify"):
        initialize_model(
            target,
            {"kind": "pretrained", "source": str(weights_path)},
            expected_label_names=("background", "bone"),
            expected_model_config=_model_config(),
        )

    provenance = initialize_model(
        target,
        {
            "kind": "pretrained",
            "source": str(weights_path),
            "allow_unverified_label_order": True,
        },
        expected_label_names=("background", "bone"),
        expected_model_config=_model_config(),
    )

    assert provenance == {
        "kind": "pretrained",
        "source": str(weights_path.resolve()),
        "artifact": "standalone_safetensors",
        "label_order_verified": False,
    }
    assert torch.equal(target.weight, source.weight)


def test_bundle_initialization_requires_model_and_ordered_label_metadata(
    tmp_path: Path,
) -> None:
    """A bundle cannot silently fall back to shape-only compatibility."""

    source = torch.nn.Conv2d(1, 2, 1, bias=False)
    labels = ("background", "bone")
    missing_labels = tmp_path / "missing-labels"
    _save_bundle(
        missing_labels, source.state_dict(), model_config=_model_config(), labels=labels
    )
    (missing_labels / "label_schema.json").unlink()

    with pytest.raises(FileNotFoundError, match="label_schema.json"):
        initialize_model(
            torch.nn.Conv2d(1, 2, 1, bias=False),
            {"kind": "pretrained", "source": str(missing_labels)},
            expected_label_names=labels,
            expected_model_config=_model_config(),
        )

    missing_config = tmp_path / "missing-config"
    _save_bundle(
        missing_config, source.state_dict(), model_config=_model_config(), labels=labels
    )
    (missing_config / "config.yml").unlink()

    with pytest.raises(FileNotFoundError, match="missing model config"):
        initialize_model(
            torch.nn.Conv2d(1, 2, 1, bias=False),
            {"kind": "pretrained", "source": str(missing_config)},
            expected_label_names=labels,
            expected_model_config=_model_config(),
        )


def test_unverified_label_opt_in_is_rejected_for_bundles(tmp_path: Path) -> None:
    """The escape hatch is deliberately confined to one bare local file."""

    source = torch.nn.Conv2d(1, 2, 1, bias=False)
    bundle = tmp_path / "bundle"
    labels = ("background", "bone")
    _save_bundle(
        bundle, source.state_dict(), model_config=_model_config(), labels=labels
    )

    with pytest.raises(ValueError, match="applies only to a standalone"):
        initialize_model(
            torch.nn.Conv2d(1, 2, 1, bias=False),
            {
                "kind": "pretrained",
                "source": str(bundle),
                "allow_unverified_label_order": True,
            },
            expected_label_names=labels,
            expected_model_config=_model_config(),
        )


def test_bundle_rejects_schema_marked_as_unverified(tmp_path: Path) -> None:
    """An explicit unverified marker cannot masquerade as release metadata."""

    source = torch.nn.Conv2d(1, 2, 1, bias=False)
    bundle = tmp_path / "bundle"
    labels = ("background", "bone")
    _save_bundle(
        bundle, source.state_dict(), model_config=_model_config(), labels=labels
    )
    (bundle / "label_schema.json").write_text(
        json.dumps(
            {
                "label_names": list(labels),
                "label_order_verified": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="marks its label order as unverified"):
        initialize_model(
            torch.nn.Conv2d(1, 2, 1, bias=False),
            {"kind": "pretrained", "source": str(bundle)},
            expected_label_names=labels,
            expected_model_config=_model_config(),
        )


def _unet_config(out_channels: int) -> dict[str, Any]:
    return {
        "_class": "fxr.models.UNet",
        "in_channels": 1,
        "out_channels": out_channels,
        "filters": [4, 8],
    }


def _unet(out_channels: int, filters: tuple[int, int] = (4, 8)):
    from fxr.models import UNet

    return UNet(in_channels=1, out_channels=out_channels, filters=list(filters))


def test_replace_head_loads_backbone_and_reinitializes_output_conv(tmp_path: Path) -> None:
    torch.manual_seed(0)
    source = _unet(3)
    bundle = tmp_path / "bundle"
    _save_bundle(bundle, source.state_dict(), model_config=_unet_config(3), labels=("background", "a", "b"))
    target = _unet(2)
    fresh_head = target.output_conv.weight.clone()

    with pytest.raises(ValueError, match="label order"):
        initialize_model(
            target,
            {"kind": "pretrained", "source": str(bundle)},
            expected_label_names=("background", "z"),
            expected_model_config=_unet_config(2),
        )
    provenance = initialize_model(
        target,
        {"kind": "pretrained", "source": str(bundle), "replace_head": True},
        expected_label_names=("background", "z"),
        expected_model_config=_unet_config(2),
    )

    assert provenance["replace_head"] is True and provenance["source_out_channels"] == 3
    for key, value in source.down_blocks.state_dict().items():
        assert torch.equal(target.down_blocks.state_dict()[key], value)
    assert torch.equal(target.output_conv.weight, fresh_head)
    assert all(p.requires_grad for p in target.parameters())


def test_replace_head_rejects_backbone_mismatch_and_freezes_backbone(tmp_path: Path) -> None:
    source = _unet(3)
    bundle = tmp_path / "bundle"
    _save_bundle(bundle, source.state_dict(), model_config=_unet_config(3), labels=("background", "a", "b"))

    wider = {**_unet_config(2), "filters": [8, 16]}
    with pytest.raises(ValueError, match="does not exactly match"):
        initialize_model(
            _unet(2, filters=(8, 16)),
            {"kind": "pretrained", "source": str(bundle), "replace_head": True},
            expected_label_names=("background", "z"),
            expected_model_config=wider,
        )

    target = _unet(2)
    initialize_model(
        target,
        {"kind": "pretrained", "source": str(bundle), "replace_head": True, "freeze_backbone": True},
        expected_label_names=("background", "z"),
        expected_model_config=_unet_config(2),
    )
    trainable = {name for name, p in target.named_parameters() if p.requires_grad}
    assert trainable == {"output_conv.weight", "output_conv.bias"}

    with pytest.raises(ValueError, match="freeze_backbone requires"):
        initialize_model(
            _unet(2),
            {"kind": "pretrained", "source": str(bundle), "freeze_backbone": True},
            expected_label_names=("background", "z"),
            expected_model_config=_unet_config(2),
        )
