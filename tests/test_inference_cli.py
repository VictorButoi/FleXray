from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from fxr.inference import (
    DEFAULT_MODEL_ID,
    ENSEMBLE_MEMBER_SUBFOLDERS,
    FLAGSHIP_SUBFOLDER,
)
from fxr.inference.segmenter import FleXraySegmenter
from fxr.inference.cli import (
    _build_parser,
    collect_input_images,
    main,
    predict_from_paths,
    preprocess_image_file,
    resolve_device,
)
from fxr.models import UNet


def _write_hf_bundle(
    tmp_path: Path,
    *,
    image_size: tuple[int, int] = (256, 256),
) -> Path:
    bundle_dir = tmp_path / "hf_bundle"
    bundle_dir.mkdir()
    (bundle_dir / "config.yml").write_text(
        "model:\n"
        "  _class: fxr.models.UNet\n"
        "  in_channels: 1\n"
        "  out_channels: 2\n"
        "  filters: [2]\n"
        "  convs_per_block: 1\n",
        encoding="utf-8",
    )
    (bundle_dir / "label_schema.json").write_text(
        json.dumps({"label_names": ["background", "bone"], "num_labels": 2}),
        encoding="utf-8",
    )
    (bundle_dir / "preprocessing.json").write_text(
        json.dumps(
            {
                "image_size": list(image_size),
                "color_mode": "grayscale",
                "pad_to_square": True,
                "scale": "zero_one",
                "probability_mode": "multilabel",
            }
        ),
        encoding="utf-8",
    )
    model = UNet(in_channels=1, out_channels=2, filters=[2], convs_per_block=1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.output_conv.bias.copy_(torch.tensor([-1.0, 1.0]))
    save_file(model.state_dict(), str(bundle_dir / "model.safetensors"))
    return bundle_dir


def _patch_hf_download(
    monkeypatch: pytest.MonkeyPatch,
    bundle_dir: Path | dict[str, Path],
    calls: list[tuple[str, str, str | None]] | None = None,
) -> None:
    """Serve one bundle directory, or route by repo id when given a mapping."""

    import fxr.inference.artifacts as artifacts_module

    def fake_hf_hub_download(
        *,
        repo_id: str,
        filename: str,
        subfolder: str | None = None,
        revision: str | None = None,
    ) -> str:
        filename = f"{subfolder}/{filename}" if subfolder else filename
        if calls is not None:
            calls.append((repo_id, filename, revision))
        root = bundle_dir[repo_id] if isinstance(bundle_dir, dict) else bundle_dir
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(filename)
        return str(path)

    monkeypatch.setattr(artifacts_module, "hf_hub_download", fake_hf_hub_download)


def _write_ensemble_repo(
    repo_dir: Path, bundle_dir: Path, subfolders: tuple[str, ...]
) -> Path:
    """Lay out ``bundle_dir`` under every subfolder plus an ``ensemble.json``."""

    for subfolder in subfolders:
        shutil.copytree(bundle_dir, repo_dir / subfolder)
    (repo_dir / "ensemble.json").write_text(
        json.dumps(
            {
                "flagship": subfolders[0],
                "members": [{"subfolder": name} for name in subfolders],
            }
        ),
        encoding="utf-8",
    )
    return repo_dir


def _write_rgb_image(path: Path, *, width: int = 5, height: int = 3) -> Path:
    array = np.full((height, width, 3), 255, dtype=np.uint8)
    Image.fromarray(array).save(path)
    return path


def test_collect_input_images_accepts_file_and_sorted_directory(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    b_path = _write_rgb_image(image_dir / "b.jpg")
    a_path = _write_rgb_image(image_dir / "a.png")
    (image_dir / "notes.txt").write_text("not an image", encoding="utf-8")

    assert collect_input_images(a_path) == (a_path,)
    assert collect_input_images(image_dir) == (a_path, b_path)


def test_preprocess_image_file_converts_to_batched_grayscale_square_tensor(
    tmp_path: Path,
) -> None:
    image_path = _write_rgb_image(tmp_path / "wide.png", width=4, height=2)

    tensor = preprocess_image_file(image_path)

    assert tensor.shape == (1, 1, 256, 256)
    assert tensor.dtype == torch.float32
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0
    assert tensor[0, 0, 0, 0].item() == pytest.approx(0.0, abs=1.0e-6)
    assert tensor[0, 0, 128, 128].item() > 0.9


def test_predict_cli_uses_default_model_id_and_writes_prediction_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = _write_hf_bundle(tmp_path)
    calls: list[tuple[str, str, str | None]] = []
    _patch_hf_download(monkeypatch, bundle_dir, calls)
    image_path = _write_rgb_image(tmp_path / "input.png")
    output_dir = tmp_path / "predictions"

    exit_code = main(
        [
            "--input",
            str(image_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert calls[:2] == [
        (DEFAULT_MODEL_ID, "ensemble.json", None),
        (DEFAULT_MODEL_ID, "config.yml", None),
    ]
    assert "Wrote 1 prediction" in capsys.readouterr().out
    logits = np.load(output_dir / "input_logits.npy")
    probabilities = np.load(output_dir / "input_probabilities.npy")
    masks = np.load(output_dir / "input_masks.npy")

    assert logits.shape == (2, 256, 256)
    assert probabilities.shape == (2, 256, 256)
    assert masks.shape == (2, 256, 256)
    assert masks.dtype == np.uint8
    assert np.allclose(logits[0], -1.0)
    assert np.allclose(logits[1], 1.0)
    assert np.allclose(probabilities[0], 1.0 / (1.0 + np.exp(1.0)))
    assert np.all(masks[0] == 0)
    assert np.all(masks[1] == 1)


def test_predict_cli_binary_label_writes_single_channel_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = _write_hf_bundle(tmp_path)
    _patch_hf_download(monkeypatch, bundle_dir)
    image_path = _write_rgb_image(tmp_path / "input.png")
    output_dir = tmp_path / "predictions"

    exit_code = main(
        [
            "--input",
            str(image_path),
            "--output-dir",
            str(output_dir),
            "--binary",
            "bone",
        ]
    )

    assert exit_code == 0
    logits = np.load(output_dir / "input_logits.npy")
    masks = np.load(output_dir / "input_masks.npy")
    assert logits.shape == (1, 256, 256)
    assert masks.shape == (1, 256, 256)
    assert np.allclose(logits[0], 1.0)
    assert np.all(masks[0] == 1)


def test_predict_from_paths_honors_explicit_model_id_revision_and_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = _write_hf_bundle(tmp_path)
    calls: list[tuple[str, str, str | None]] = []
    _patch_hf_download(monkeypatch, bundle_dir, calls)
    image_path = _write_rgb_image(tmp_path / "input.png")
    output_dir = tmp_path / "predictions"

    written = predict_from_paths(
        model_id="vbutoi/custom",
        revision="abc123",
        input_path=image_path,
        output_dir=output_dir,
        threshold=0.8,
    )

    assert calls[:2] == [
        ("vbutoi/custom", "ensemble.json", "abc123"),
        ("vbutoi/custom", "config.yml", "abc123"),
    ]
    masks = np.load(written[0]["masks"])
    assert masks.dtype == np.uint8
    assert np.all(masks == 0)


def test_cli_parser_repeats_model_id() -> None:
    parser = _build_parser()
    base_args = ["--input", "image.png", "--output-dir", "out"]

    assert parser.parse_args(base_args).model_id is None
    assert parser.parse_args([*base_args, "--model-id", "org/a"]).model_id == ["org/a"]
    assert parser.parse_args(
        [*base_args, "--model-id", "org/a", "--model-id", "org/b"]
    ).model_id == ["org/a", "org/b"]


def test_cli_parser_ensemble_and_subfolder_are_exclusive() -> None:
    parser = _build_parser()
    base_args = ["--input", "image.png", "--output-dir", "out"]

    assert parser.parse_args(base_args).ensemble is False
    assert parser.parse_args(base_args).subfolder is None
    assert parser.parse_args([*base_args, "--ensemble"]).ensemble is True
    assert parser.parse_args([*base_args, "--subfolder", "members/a"]).subfolder == (
        "members/a"
    )
    with pytest.raises(SystemExit):
        parser.parse_args([*base_args, "--ensemble", "--subfolder", "members/a"])


def test_cli_ensemble_rejects_several_model_ids(tmp_path: Path) -> None:
    image_path = _write_rgb_image(tmp_path / "input.png")
    base_args = ["--input", str(image_path), "--output-dir", str(tmp_path / "out")]

    with pytest.raises(SystemExit):
        main([*base_args, "--ensemble", "--model-id", "org/a", "--model-id", "org/b"])
    with pytest.raises(SystemExit):
        main([*base_args, "--subfolder", "x", "--model-id", "org/a", "--model-id", "org/b"])


def test_ensemble_member_subfolders_start_with_flagship_and_are_unique() -> None:
    assert ENSEMBLE_MEMBER_SUBFOLDERS[0] == FLAGSHIP_SUBFOLDER
    assert len(ENSEMBLE_MEMBER_SUBFOLDERS) == 5
    assert len(set(ENSEMBLE_MEMBER_SUBFOLDERS)) == len(ENSEMBLE_MEMBER_SUBFOLDERS)


def test_predict_cli_ensemble_with_tta_loads_every_published_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _write_hf_bundle(tmp_path)
    repo = _write_ensemble_repo(tmp_path / "repo", bundle, ENSEMBLE_MEMBER_SUBFOLDERS)
    calls: list[tuple[str, str, str | None]] = []
    _patch_hf_download(monkeypatch, {DEFAULT_MODEL_ID: repo}, calls)
    image_path = _write_rgb_image(tmp_path / "input.png")
    output_dir = tmp_path / "predictions"

    exit_code = main(
        [
            "--ensemble",
            "--tta-samples",
            "2",
            "--input",
            str(image_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert {call[0] for call in calls} == {DEFAULT_MODEL_ID}
    assert {call[1] for call in calls if call[1].endswith("model.safetensors")} == {
        f"{subfolder}/model.safetensors" for subfolder in ENSEMBLE_MEMBER_SUBFOLDERS
    }
    assert np.load(output_dir / "input_masks.npy").shape == (2, 256, 256)


def test_predict_cli_subfolder_loads_only_that_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _write_hf_bundle(tmp_path)
    repo = _write_ensemble_repo(tmp_path / "repo", bundle, ENSEMBLE_MEMBER_SUBFOLDERS)
    calls: list[tuple[str, str, str | None]] = []
    _patch_hf_download(monkeypatch, {DEFAULT_MODEL_ID: repo}, calls)
    image_path = _write_rgb_image(tmp_path / "input.png")

    exit_code = main(
        [
            "--subfolder",
            "members/flux000",
            "--input",
            str(image_path),
            "--output-dir",
            str(tmp_path / "predictions"),
        ]
    )

    assert exit_code == 0
    assert {call[1] for call in calls if call[1].endswith("model.safetensors")} == {
        "members/flux000/model.safetensors"
    }
    assert not any(call[1] == "ensemble.json" for call in calls)


def test_predict_cli_with_two_model_ids_loads_both_and_averages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    first = _write_hf_bundle(tmp_path / "first")
    second = _write_hf_bundle(tmp_path / "second")
    model = UNet(in_channels=1, out_channels=2, filters=[2], convs_per_block=1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.output_conv.bias.copy_(torch.tensor([1.0, -1.0]))
    save_file(model.state_dict(), str(second / "model.safetensors"))
    calls: list[tuple[str, str, str | None]] = []
    _patch_hf_download(monkeypatch, {"org/first": first, "org/second": second}, calls)
    image_path = _write_rgb_image(tmp_path / "input.png")
    output_dir = tmp_path / "predictions"

    exit_code = main(
        [
            "--model-id",
            "org/first",
            "--model-id",
            "org/second",
            "--input",
            str(image_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert {call[0] for call in calls} == {"org/first", "org/second"}
    probabilities = np.load(output_dir / "input_probabilities.npy")
    assert np.allclose(probabilities, 0.5)


def test_predict_reports_missing_hf_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = _write_hf_bundle(tmp_path)
    (bundle_dir / "model.safetensors").unlink()
    _patch_hf_download(monkeypatch, bundle_dir)
    image_path = _write_rgb_image(tmp_path / "input.png")

    with pytest.raises(SystemExit) as missing_weights:
        main(
            [
                "--input",
                str(image_path),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
    assert missing_weights.value.code == 2
    assert "model.safetensors" in capsys.readouterr().err

    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        predict_from_paths(
            input_path=image_path,
            output_dir=tmp_path / "out",
        )


def test_resolve_device_auto_prefers_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == torch.device("cuda")
    assert resolve_device(None) == torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")


def test_resolve_device_rejects_cuda_without_cuda_and_bad_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cpu") == torch.device("cpu")
    with pytest.raises(ValueError, match="CUDA is not available"):
        resolve_device("cuda:1")
    with pytest.raises(ValueError, match="Unrecognized device"):
        resolve_device("not-a-device")


def test_resolve_device_rejects_out_of_range_cuda_ordinal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert resolve_device("cuda:1") == torch.device("cuda:1")
    with pytest.raises(ValueError, match="out of range"):
        resolve_device("cuda:2")


def test_cli_parser_device_defaults_to_auto() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--input", "a.png", "--output-dir", "out"])
    assert args.device == "auto"
    args = parser.parse_args(
        ["--device", "cuda:1", "--input", "a.png", "--output-dir", "out"]
    )
    assert args.device == "cuda:1"


def test_predict_from_paths_passes_device_to_from_pretrained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = _write_hf_bundle(tmp_path)
    _patch_hf_download(monkeypatch, bundle_dir)
    image_path = _write_rgb_image(tmp_path / "input.png")
    seen: list[torch.device | str | None] = []
    original = FleXraySegmenter.from_pretrained.__func__

    def spy(cls, *args, **kwargs):
        seen.append(kwargs.get("device"))
        return original(cls, *args, **kwargs)

    monkeypatch.setattr(FleXraySegmenter, "from_pretrained", classmethod(spy))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    predict_from_paths(
        input_path=image_path, output_dir=tmp_path / "out", device="cpu"
    )
    predict_from_paths(input_path=image_path, output_dir=tmp_path / "out2")

    assert seen == [torch.device("cpu"), torch.device("cpu")]


def _write_dicom(path: Path, pixels: np.ndarray, photometric: str = "MONOCHROME2") -> None:
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = pydicom.uid.ComputedRadiographyImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    ds.save_as(str(path), enforce_file_format=True)


def test_preprocess_dicom_file_scales_by_image_range_and_inverts_monochrome1(
    tmp_path: Path,
) -> None:
    pixels = np.arange(16, dtype=np.uint16).reshape(4, 4) * 100  # 12-bit-ish range
    mono2 = tmp_path / "scan.dcm"
    mono1 = tmp_path / "inverted"  # extension-less, detected by DICM magic
    _write_dicom(mono2, pixels)
    _write_dicom(mono1, pixels, photometric="MONOCHROME1")

    tensor = preprocess_image_file(mono2, size=(4, 4))
    inverted = preprocess_image_file(mono1, size=(4, 4))

    assert tensor.shape == (1, 1, 4, 4)
    assert tensor.dtype == torch.float32
    assert float(tensor.min()) == pytest.approx(0.0)
    assert float(tensor.max()) == pytest.approx(1.0)
    assert torch.allclose(inverted, 1.0 - tensor, atol=1e-6)
    assert collect_input_images(tmp_path) == (mono1, mono2)
