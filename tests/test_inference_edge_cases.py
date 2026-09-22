"""Regression coverage for local bundles, seeded inference, and batch failures."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fxr.inference import FleXraySegmenter, ImagePreprocessing, InferenceRunner, predict_with_tta
from fxr.inference import artifacts
from fxr.inference.cli import main, predict_from_paths
from fxr.models import UNet
from tests.test_inference_tta import _IdentityModel
from tests.test_inference_cli import (
    _write_ensemble_repo,
    _write_hf_bundle,
    _write_rgb_image,
)


@pytest.fixture
def local_bundle(tmp_path, monkeypatch):
    """Make a tiny real bundle and forbid network fallback."""
    def no_download(**kwargs):
        pytest.fail(f"Unexpected network request: {kwargs}")

    monkeypatch.setattr(artifacts, "hf_hub_download", no_download)
    return _write_hf_bundle(tmp_path, image_size=(8, 8))


def test_local_bundle_loads_without_network(local_bundle):
    """Portable bundle artifacts load directly without mocking artifact paths."""
    model = FleXraySegmenter.from_pretrained(str(local_bundle), device="cpu")
    result = model.predict(np.ones((8, 8), dtype=np.uint8))
    assert result.logits.shape == (1, 2, 8, 8)
    assert torch.isfinite(result.logits).all()


@pytest.mark.parametrize("choice", ["flagship", "member", "ensemble"])
def test_local_ensemble_resolution(local_bundle, tmp_path, choice, capsys):
    """Local directories support the same member selection as remote bundles."""
    subfolders = ("members/a", "members/b")
    repo = _write_ensemble_repo(tmp_path / "repo", local_bundle, subfolders)
    kwargs = {"ensemble": True} if choice == "ensemble" else {}
    if choice == "member":
        kwargs["subfolder"] = subfolders[1]
    model = FleXraySegmenter.from_pretrained(str(repo), device="cpu", **kwargs)
    assert len(model.models) == (2 if choice == "ensemble" else 1)
    if choice == "flagship":
        assert "loading the flagship bundle members/a" in capsys.readouterr().err


@pytest.mark.parametrize("device", [None, "auto", "cpu"])
def test_from_pretrained_device_selection(local_bundle, device):
    """Auto chooses available CUDA and an explicit CPU request stays on CPU."""
    model = FleXraySegmenter.from_pretrained(str(local_bundle), device=device)
    expected = "cuda" if device != "cpu" and torch.cuda.is_available() else "cpu"
    assert model.device.type == expected


def test_cli_continues_after_corrupt_image(local_bundle, tmp_path, capsys):
    """An unreadable image must not prevent the next valid image being saved."""
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "a_bad.png").write_bytes(b"not a png")
    _write_rgb_image(inputs / "b_good.png")
    output = tmp_path / "output"
    code = main([
        "--model-id", str(local_bundle), "--input", str(inputs),
        "--output-dir", str(output), "--device", "cpu",
    ])
    assert code == 1
    assert np.load(output / "b_good_masks.npy").shape == (2, 8, 8)
    assert not (output / "a_bad_masks.npy").exists()
    assert json.loads((output / "label_names.json").read_text()) == ["background", "bone"]
    assert "Skipped 1 unreadable input(s)" in capsys.readouterr().err


def test_invalid_binary_label_is_a_command_error(local_bundle, tmp_path):
    """A bad label is an invalid command, not an unreadable image."""
    image = _write_rgb_image(tmp_path / "input.png")
    with pytest.raises(SystemExit) as exc:
        main([
            "--model-id", str(local_bundle), "--input", str(image),
            "--output-dir", str(tmp_path / "output"), "--binary", "typo",
            "--device", "cpu",
        ])
    assert exc.value.code == 2


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
def test_model_error_is_not_swallowed(local_bundle, tmp_path, monkeypatch, error_type):
    """A model failure must propagate instead of being reported as corrupt input."""
    image = _write_rgb_image(tmp_path / "input.png")

    def broken_forward(self, images):
        raise error_type("model execution failed")

    monkeypatch.setattr(UNet, "forward", broken_forward)
    with pytest.raises(error_type, match="model execution failed"):
        predict_from_paths(
            model_id=str(local_bundle), input_path=image,
            output_dir=tmp_path / "output", device="cpu",
            on_error=lambda *args: pytest.fail("Model error was skipped"),
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("first_label,second_label", [
    ("bone", None), (None, "bone"), ("bone", "background"),
])
def test_conflicting_labels_preserve_existing_outputs(
    local_bundle, tmp_path, monkeypatch, first_label, second_label,
):
    """A different channel count or order must fail before inference or writes."""
    first = _write_rgb_image(tmp_path / "first.png")
    second = _write_rgb_image(tmp_path / "second.png")
    output = tmp_path / "output"
    predict_from_paths(
        model_id=str(local_bundle), input_path=first, output_dir=output,
        binary_label=first_label, device="cpu",
    )
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    monkeypatch.setattr(UNet, "forward", lambda *args: pytest.fail("Model ran"))
    with pytest.raises(ValueError, match="conflicts"):
        predict_from_paths(
            model_id=str(local_bundle), input_path=second, output_dir=output,
            binary_label=second_label, device="cpu",
        )
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


def test_failed_label_request_preserves_existing_metadata(local_bundle, tmp_path):
    """An invalid request should not rewrite metadata for existing valid results."""
    image = _write_rgb_image(tmp_path / "input.png")
    output = tmp_path / "output"
    predict_from_paths(
        model_id=str(local_bundle), input_path=image, output_dir=output, device="cpu",
    )
    metadata = output / "label_names.json"
    before = metadata.read_bytes()
    try:
        predict_from_paths(
            model_id=str(local_bundle), input_path=image, output_dir=output,
            binary_label="typo", device="cpu",
        )
    except ValueError:
        pass
    assert metadata.read_bytes() == before


def test_negative_tta_rejected_before_model_loading(monkeypatch, tmp_path):
    """The CLI must reject a negative pass count before loading anything."""
    monkeypatch.setattr(
        FleXraySegmenter, "from_pretrained", lambda **kw: pytest.fail("Model loaded")
    )
    with pytest.raises(SystemExit) as exc:
        main(["--input", "missing.png", "--output-dir", str(tmp_path), "--tta-samples", "-1"])
    assert exc.value.code == 2


def test_output_file_rejected_before_model_loading(tmp_path, monkeypatch):
    """Invalid output paths should fail before any download or model allocation."""
    image = _write_rgb_image(tmp_path / "input.png")
    output = tmp_path / "output"
    output.write_text("sentinel")
    monkeypatch.setattr(
        FleXraySegmenter, "from_pretrained", lambda **kw: pytest.fail("Model loaded")
    )
    with pytest.raises(NotADirectoryError):
        predict_from_paths(input_path=image, output_dir=output, device="cpu")
    assert output.read_text() == "sentinel"


@pytest.mark.parametrize("entry_point", ["function", "runner", "segmenter"])
def test_seeded_tta_repeatability_and_cpu_rng_preservation(entry_point):
    """Seeded augmented predictions repeat and leave the CPU generator untouched."""
    image = torch.linspace(0, 1, 64).reshape(1, 1, 8, 8)
    if entry_point == "function":
        def predict(seed):
            return predict_with_tta(
                lambda x: x, image, mode="multilabel", tta_samples=5, seed=seed,
            )[1]
    else:
        model = _IdentityModel()
        predictor = (
            InferenceRunner(model) if entry_point == "runner" else
            FleXraySegmenter(
                model, label_names=("pixel",),
                preprocessing=ImagePreprocessing(image_size=(8, 8)),
            )
        )

        def predict(seed):
            return predictor.predict(image, tta_samples=5, seed=seed).probabilities

    before = torch.get_rng_state().clone()
    first, second, other = predict(7), predict(7), predict(8)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("image_device", ["cpu", "cuda"])
def test_seeded_tta_preserves_cuda_rng(image_device):
    """Even CPU inference must not reset the caller's CUDA random streams."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    devices = list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(1234)
        image = torch.linspace(0, 1, 64, device=image_device).reshape(1, 1, 8, 8)
        before = torch.cuda.get_rng_state_all()
        predict_with_tta(lambda x: x, image, mode="multilabel", tta_samples=3, seed=7)
        after = torch.cuda.get_rng_state_all()
        assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_mcp_device_option_reaches_model_loader(monkeypatch):
    """The server's device option should reach each lazily cached model."""
    import fxr.mcp.cli as cli
    import fxr.mcp.inference_tools as inference
    import fxr.mcp.server as server

    monkeypatch.setattr(inference, "_SEGMENTER_CACHE", {})
    monkeypatch.setattr(inference, "_SERVER_DEVICE", None)
    seen = []
    monkeypatch.setattr(
        FleXraySegmenter, "from_pretrained",
        lambda **kw: seen.append(kw["device"]) or object(),
    )
    monkeypatch.setattr(cli, "require_mcp_extra", lambda *args: None)
    monkeypatch.setattr(server, "create_server", lambda: SimpleNamespace(
        run=lambda: inference._get_segmenter("org/model", None, None, False)
    ))
    assert cli.main(["--device", "cpu"]) == 0
    assert seen == ["cpu"]


def test_local_corrupt_weights_error_does_not_blame_hf_cache(local_bundle):
    """Local bundle failures need local recovery instructions."""
    (local_bundle / "model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError) as exc:
        FleXraySegmenter.from_pretrained(str(local_bundle), device="cpu")
    assert "Failed to read model weights" in str(exc.value)
    assert "Restore the local model.safetensors file from the original bundle" in str(exc.value)
    assert "Hugging Face cache" not in str(exc.value)


def test_cli_continues_after_dicom_with_missing_rows(local_bundle, tmp_path):
    """Malformed DICOM metadata must not abort the remaining good image inputs."""
    pydicom = pytest.importorskip("pydicom")
    from tests.test_inference_cli import _write_dicom

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    broken = inputs / "a_bad.dcm"
    _write_dicom(broken, np.zeros((4, 4), dtype=np.uint16))
    dataset = pydicom.dcmread(broken)
    del dataset.Rows
    dataset.save_as(broken)
    _write_rgb_image(inputs / "b_good.png")
    output = tmp_path / "output"
    code = main([
        "--model-id", str(local_bundle), "--input", str(inputs),
        "--output-dir", str(output), "--device", "cpu",
    ])
    assert code == 1
    assert (output / "b_good_masks.npy").exists()


def test_api_returns_list_and_only_skips_reads_with_callback(local_bundle, tmp_path):
    """The original list and fail-fast contract remains; skipping is opt-in."""
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    bad = inputs / "a_bad.png"
    bad.write_bytes(b"not a png")
    _write_rgb_image(inputs / "b_good.png")
    output = tmp_path / "output"
    kwargs = dict(
        model_id=str(local_bundle), input_path=inputs, output_dir=output, device="cpu",
    )
    with pytest.raises(OSError):
        predict_from_paths(**kwargs)
    assert not output.exists()
    failures = []
    written = predict_from_paths(**kwargs, on_error=lambda path, err: failures.append((path, err)))
    assert isinstance(written, list)
    assert len(written) == 1
    assert set(written[0]) == {"logits", "probabilities", "masks"}
    assert len(failures) == 1 and failures[0][0] == bad
    assert isinstance(failures[0][1], OSError)


@pytest.mark.parametrize("labels", [None, "bone"])
def test_matching_output_labels_can_be_reused(local_bundle, tmp_path, labels):
    """The same channel selection can append outputs without rewriting metadata."""
    output = tmp_path / "output"
    for name in ("first.png", "second.png"):
        predict_from_paths(
            model_id=str(local_bundle), input_path=_write_rgb_image(tmp_path / name),
            output_dir=output, binary_label=labels, device="cpu",
        )
    expected = ["background", "bone"] if labels is None else [labels]
    assert json.loads((output / "label_names.json").read_text()) == expected
    assert len(list(output.glob("*_masks.npy"))) == 2


@pytest.mark.parametrize("metadata", [None, b"not json", b"null", b"\xff"])
def test_unverifiable_output_metadata_preserves_files(local_bundle, tmp_path, metadata):
    """Named results cannot silently label legacy arrays or replace bad metadata."""
    output = tmp_path / "output"
    output.mkdir()
    np.save(output / "old_masks.npy", np.zeros((2, 8, 8), dtype=np.uint8))
    if metadata is not None:
        (output / "label_names.json").write_bytes(metadata)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ValueError, match="metadata"):
        predict_from_paths(
            model_id=str(local_bundle), input_path=_write_rgb_image(tmp_path / "input.png"),
            output_dir=output, device="cpu",
        )
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


def test_all_unreadable_inputs_leave_no_output_metadata(local_bundle, tmp_path):
    """A batch without a successful prediction must not leave a label sidecar."""
    image = tmp_path / "bad.png"
    image.write_bytes(b"broken")
    output = tmp_path / "output"
    assert main([
        "--model-id", str(local_bundle), "--input", str(image),
        "--output-dir", str(output), "--device", "cpu",
    ]) == 1
    assert not output.exists()


def test_preprocessing_failure_is_not_an_unreadable_image(local_bundle, tmp_path, monkeypatch):
    """Valid pixels reaching broken preprocessing must abort, not be skipped."""
    def fail(self, image):
        raise RuntimeError("normalization failed")

    monkeypatch.setattr(ImagePreprocessing, "prepare", fail)
    with pytest.raises(RuntimeError, match="normalization failed"):
        predict_from_paths(
            model_id=str(local_bundle), input_path=_write_rgb_image(tmp_path / "input.png"),
            output_dir=tmp_path / "output", device="cpu",
            on_error=lambda *args: pytest.fail("Preprocessing error was skipped"),
        )


def test_python_negative_tta_fails_before_loading(monkeypatch, tmp_path):
    """Direct Python callers receive early pass-count validation too."""
    monkeypatch.setattr(
        FleXraySegmenter, "from_pretrained", lambda **kw: pytest.fail("Model loaded")
    )
    with pytest.raises(ValueError, match="tta_samples"):
        predict_from_paths(input_path="missing.png", output_dir=tmp_path, tta_samples=-1)


def test_from_pretrained_auto_falls_back_to_cpu(local_bundle, monkeypatch):
    """An inference-only CPU host must be able to use the default device."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert FleXraySegmenter.from_pretrained(str(local_bundle)).device.type == "cpu"


@pytest.mark.parametrize("device", ["not-a-device", "cuda"])
def test_invalid_api_device_fails_before_downloading(monkeypatch, device):
    """The API must share the CLI's early device validation."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(artifacts, "hf_hub_download", lambda **kw: pytest.fail("Downloaded"))
    with pytest.raises(ValueError, match="device|Device"):
        FleXraySegmenter.from_pretrained("org/model", device=device)


def test_seeded_tta_never_seeds_accelerators(monkeypatch):
    """CPU-only CI also guards against torch.manual_seed reseeding all devices."""
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *args: pytest.fail("CUDA reseeded"))
    predict_with_tta(
        lambda x: x, torch.ones((1, 1, 8, 8)),
        mode="multilabel", tta_samples=3, seed=7,
    )


def test_seeded_tta_restores_cpu_rng_when_forward_raises():
    """A failed augmented forward must leave the CPU random stream unchanged."""
    calls = []

    def forward(images):
        calls.append(images)
        if len(calls) > 1:
            raise RuntimeError("augmented pass failed")
        return images

    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="augmented pass failed"):
        predict_with_tta(
            forward, torch.ones((1, 1, 8, 8)), mode="multilabel", tta_samples=3, seed=7,
        )
    assert torch.equal(torch.get_rng_state(), before)


def test_cli_seed_repeats_actual_augmented_predictions(local_bundle, tmp_path):
    """The CLI seed must reach augmented inference, not just argument parsing."""
    from safetensors.torch import save_file

    model = UNet(in_channels=1, out_channels=2, filters=[2], convs_per_block=1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.1)
    save_file(model.state_dict(), str(local_bundle / "model.safetensors"))
    image = _write_rgb_image(tmp_path / "input.png")
    predictions = []
    for index, seed in enumerate((7, 7, 8)):
        output = tmp_path / f"output_{index}"
        assert main([
            "--model-id", str(local_bundle), "--input", str(image),
            "--output-dir", str(output), "--device", "cpu",
            "--tta-samples", "5", "--seed", str(seed),
        ]) == 0
        predictions.append(np.load(output / "input_probabilities.npy"))
    assert np.array_equal(predictions[0], predictions[1])
    assert not np.array_equal(predictions[0], predictions[2])


def test_unnamed_outputs_do_not_inherit_existing_labels(local_bundle, tmp_path):
    """A model without names cannot append to a directory advertising labels."""
    image = _write_rgb_image(tmp_path / "input.png")
    output = tmp_path / "named_output"
    predict_from_paths(
        model_id=str(local_bundle), input_path=image, output_dir=output, device="cpu",
    )
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    (local_bundle / "label_schema.json").write_text("{}")
    with pytest.raises(ValueError, match="conflicts"):
        predict_from_paths(
            model_id=str(local_bundle), input_path=image,
            output_dir=output, device="cpu",
        )
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before
    unnamed_output = tmp_path / "unnamed_output"
    written = predict_from_paths(
        model_id=str(local_bundle), input_path=image,
        output_dir=unnamed_output, device="cpu",
    )
    assert len(written) == 1
    assert not (unnamed_output / "label_names.json").exists()
