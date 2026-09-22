from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

import fxr.mcp as public_mcp
from fxr.mcp import (
    MODEL_REGISTRY,
    describe_dataset,
    describe_model,
    describe_protocol,
    explain_mapping,
    list_datasets,
    list_models,
    list_protocols,
    segment_image,
)
from fxr.mcp._extras import MCP_EXTRA_MESSAGE, require_mcp_extra
from fxr.mcp.inference_tools import _SEGMENTER_CACHE, _clear_segmenter_cache
from fxr.models import UNet

PROTOCOL_NAME = "all_structures_flexray_v4"


@pytest.fixture(autouse=True)
def _reset_segmenter_cache() -> None:
    _clear_segmenter_cache()
    yield
    _clear_segmenter_cache()


def _write_hf_bundle(tmp_path: Path, *, image_size: tuple[int, int] = (4, 4)) -> Path:
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
        json.dumps(
            {
                "label_names": ["background", "bone"],
                "num_labels": 2,
                "weights_license": "CC-BY-NC-4.0",
            }
        ),
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
    model = UNet(in_channels=1, out_channels=2, filters=[2])
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.output_conv.bias.copy_(torch.tensor([-1.0, 1.0]))
    save_file(model.state_dict(), str(bundle_dir / "model.safetensors"))
    return bundle_dir


def _patch_hf_download(
    monkeypatch: pytest.MonkeyPatch,
    bundle_dir: Path,
    calls: list[tuple[str, str, str | None]] | None = None,
) -> None:
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
        path = bundle_dir / filename
        if not path.exists():
            raise FileNotFoundError(filename)
        return str(path)

    monkeypatch.setattr(artifacts_module, "hf_hub_download", fake_hf_hub_download)


def _write_gray_image(path: Path, *, width: int = 4, height: int = 4) -> Path:
    image = Image.new("L", (width, height), color=128)
    image.save(path)
    return path


def test_model_registry_lists_the_inference_ensemble_in_order() -> None:
    from fxr.inference import ENSEMBLE_MEMBER_SUBFOLDERS

    assert tuple(entry.subfolder for entry in MODEL_REGISTRY) == ENSEMBLE_MEMBER_SUBFOLDERS
    assert {entry.model_id for entry in MODEL_REGISTRY} == {public_mcp.DEFAULT_MODEL_ID}


def test_default_model_id_matches_inference_default() -> None:
    from fxr.inference import DEFAULT_MODEL_ID as inference_default

    assert public_mcp.DEFAULT_MODEL_ID == inference_default
    assert MODEL_REGISTRY[0].model_id == inference_default


def test_list_models_returns_static_registry_without_loading() -> None:
    result = list_models()

    assert [entry["model_id"] for entry in result["models"]] == [
        entry.model_id for entry in MODEL_REGISTRY
    ]
    assert result["models"][0]["protocol_name"] == PROTOCOL_NAME
    assert "Hugging Face repo id" in result["note"]


def test_list_and_describe_protocols_match_packaged_configs() -> None:
    from fxr.protocols import list_protocol_names, load_protocol_by_name

    assert list_protocols() == {"protocols": list(list_protocol_names())}

    described = describe_protocol(PROTOCOL_NAME)
    protocol = load_protocol_by_name(PROTOCOL_NAME)
    assert described["protocol_name"] == PROTOCOL_NAME
    assert described["num_labels"] == len(protocol.labels)
    assert described["labels"][0] == {"label": "background", "channel": 0}
    assert [entry["label"] for entry in described["labels"]] == list(protocol.labels)


def test_list_datasets_includes_aliases_and_describe_resolves_them() -> None:
    result = list_datasets()

    names = [entry["name"] for entry in result["datasets"]]
    assert len(names) == len(set(names))
    moose = next(entry for entry in result["datasets"] if entry["name"] == "MOOSE")
    assert "FluXray" in moose["aliases"]

    described = describe_dataset("FluXray")
    assert described["dataset_name"] == "MOOSE"
    assert described["stored_labels"][0] == {"native_id": 0, "label": "background"}


def test_protocol_tools_reject_unknown_and_path_like_names() -> None:
    with pytest.raises(FileNotFoundError):
        describe_protocol("no_such_protocol")
    with pytest.raises(ValueError):
        describe_dataset("../MOOSE")


def test_explain_mapping_agrees_with_compiled_lut() -> None:
    from fxr.protocols import (
        compile_training_lut_by_name,
        load_dataset_spec_by_name,
    )

    result = explain_mapping(PROTOCOL_NAME, "HipRay")
    lut = compile_training_lut_by_name(PROTOCOL_NAME, "HipRay")
    stored_labels = load_dataset_spec_by_name("HipRay").stored_labels

    assert result["protocol_labels"] == list(lut.protocol_labels)
    assert result["label_lut"] == list(lut.label_lut)
    assert len(result["rows"]) == len(lut.native_id_to_protocol_id)
    for row in result["rows"]:
        native_id = row["native_id"]
        assert row["native_label"] == stored_labels[native_id]
        assert row["protocol_channel"] == lut.native_id_to_protocol_id[native_id]
        assert (
            row["protocol_label"]
            == lut.native_id_to_protocol_label[native_id]
        )


def test_segment_image_writes_artifacts_and_reports_statistics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = _write_hf_bundle(tmp_path)
    _patch_hf_download(monkeypatch, bundle_dir)
    image_path = _write_gray_image(tmp_path / "sample.png")
    output_dir = tmp_path / "out"

    result = segment_image(str(image_path), str(output_dir))

    assert result["image_count"] == 1
    entry = result["results"][0]
    assert entry["image_path"] == str(image_path)
    for kind in ("logits", "probabilities", "masks"):
        saved = np.load(entry["artifacts"][kind])
        assert saved.shape == (2, 4, 4)
    masks = np.load(entry["artifacts"]["masks"])
    assert masks.dtype == np.uint8

    labels = entry["labels"]
    assert [stat["label"] for stat in labels] == ["background", "bone"]
    assert [stat["channel"] for stat in labels] == [0, 1]
    for stat in labels:
        channel_mask = masks[stat["channel"]]
        assert stat["pixel_count"] == int(channel_mask.sum())
        assert stat["pixel_fraction"] == stat["pixel_count"] / 16
        assert 0.0 <= stat["mean_probability"] <= 1.0
        assert stat["mean_probability"] <= stat["max_probability"]
    assert labels[0]["pixel_count"] == 0
    assert labels[1]["pixel_count"] == 16


def test_segment_image_directory_input_dedupes_stems(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = _write_hf_bundle(tmp_path)
    _patch_hf_download(monkeypatch, bundle_dir)
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    _write_gray_image(input_dir / "sample.jpg")
    _write_gray_image(input_dir / "sample.png")
    output_dir = tmp_path / "out"

    result = segment_image(str(input_dir), str(output_dir))

    assert result["image_count"] == 2
    stems = [
        Path(entry["artifacts"]["masks"]).name for entry in result["results"]
    ]
    assert stems == ["sample_masks.npy", "sample_2_masks.npy"]


def test_segment_image_label_filter_reports_true_channel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = _write_hf_bundle(tmp_path)
    _patch_hf_download(monkeypatch, bundle_dir)
    image_path = _write_gray_image(tmp_path / "sample.png")
    output_dir = tmp_path / "out"

    result = segment_image(str(image_path), str(output_dir), label="bone")

    entry = result["results"][0]
    saved = np.load(entry["artifacts"]["masks"])
    assert saved.shape == (1, 4, 4)
    assert entry["labels"] == [
        {
            "label": "bone",
            "channel": 1,
            "pixel_count": 16,
            "pixel_fraction": 1.0,
            "mean_probability": entry["labels"][0]["mean_probability"],
            "max_probability": entry["labels"][0]["max_probability"],
        }
    ]


def test_segment_image_rejects_invalid_threshold(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="threshold"):
        segment_image(str(tmp_path), str(tmp_path / "out"), threshold=1.5)


def test_segmenter_cache_loads_each_model_revision_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = _write_hf_bundle(tmp_path)
    calls: list[tuple[str, str, str | None]] = []
    _patch_hf_download(monkeypatch, bundle_dir, calls)
    image_path = _write_gray_image(tmp_path / "sample.png")

    segment_image(str(image_path), str(tmp_path / "out_a"))
    first_load_calls = len(calls)
    assert first_load_calls > 0

    segment_image(str(image_path), str(tmp_path / "out_b"))
    describe_model()
    assert len(calls) == first_load_calls

    describe_model(revision="other")
    assert len(calls) > first_load_calls
    assert calls[-1][2] == "other"


def test_segment_image_accepts_model_id_list_and_caches_by_member_tuple(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = _write_hf_bundle(tmp_path)
    calls: list[tuple[str, str, str | None]] = []
    _patch_hf_download(monkeypatch, bundle_dir, calls)
    image_path = _write_gray_image(tmp_path / "sample.png")

    with pytest.raises(ValueError, match="repeat"):
        segment_image(str(image_path), str(tmp_path / "dup"), model_id=["a/x", "a/x"])

    result = segment_image(str(image_path), str(tmp_path / "out"), model_id=["a/x", "a/y"])
    loaded_repos = [call[0] for call in calls]
    first_load_calls = len(calls)

    assert result["model_id"] == ["a/x", "a/y"]
    assert {"a/x", "a/y"} <= set(loaded_repos)
    described = describe_model(["a/x", "a/y"])
    assert described["model_id"] == ["a/x", "a/y"]
    assert len(calls) == first_load_calls
    assert ("a/x", "a/y") in {key[0] for key in _SEGMENTER_CACHE}


def test_describe_model_reports_bundle_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = _write_hf_bundle(tmp_path)
    _patch_hf_download(monkeypatch, bundle_dir)

    result = describe_model("some/other-repo")

    assert result["model_id"] == "some/other-repo"
    assert result["revision"] is None
    assert result["labels"] == [
        {"label": "background", "channel": 0},
        {"label": "bone", "channel": 1},
    ]
    assert result["probability_mode"] == "multilabel"
    assert result["preprocessing"]["image_size"] == [4, 4]


def test_require_mcp_extra_raises_clean_message_for_missing_module() -> None:
    with pytest.raises(SystemExit) as excinfo:
        require_mcp_extra("definitely_not_an_installed_module")
    assert str(excinfo.value) == MCP_EXTRA_MESSAGE


def test_mcp_package_imports_stay_light() -> None:
    code = (
        "import sys\n"
        "import fxr.mcp\n"
        "import fxr.mcp.cli\n"
        "assert 'mcp' not in sys.modules, 'MCP SDK imported eagerly'\n"
        "assert 'torch' not in sys.modules, 'torch imported eagerly'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_server_registers_planned_tool_schemas() -> None:
    pytest.importorskip("mcp")
    import anyio

    from fxr.mcp.server import create_server

    server = create_server()
    tools = anyio.run(server.list_tools)
    assert {tool.name for tool in tools} == {
        "segment_image",
        "list_models",
        "describe_model",
        "list_protocols",
        "describe_protocol",
        "list_datasets",
        "describe_dataset",
        "explain_mapping",
    }

    schema = next(t for t in tools if t.name == "segment_image").inputSchema
    assert set(schema["required"]) == {"input_path", "output_dir"}
    assert set(schema["properties"]) == {
        "input_path",
        "output_dir",
        "model_id",
        "revision",
        "label",
        "threshold",
        "tta_samples",
        "subfolder",
        "ensemble",
    }
    assert schema["properties"]["threshold"]["default"] == 0.5
    assert schema["properties"]["tta_samples"]["default"] == 1
    assert schema["properties"]["ensemble"]["default"] is False
    model_id_types = {
        option["type"] for option in schema["properties"]["model_id"]["anyOf"]
    }
    assert model_id_types == {"string", "array"}


def test_server_round_trip_serves_list_protocols_in_memory() -> None:
    pytest.importorskip("mcp")
    import anyio

    from mcp.shared.memory import create_connected_server_and_client_session
    from fxr.mcp.server import create_server

    async def _round_trip() -> None:
        server = create_server()
        async with create_connected_server_and_client_session(
            server._mcp_server
        ) as session:
            listed = await session.list_tools()
            assert any(tool.name == "list_protocols" for tool in listed.tools)
            result = await session.call_tool("list_protocols", {})
            assert not result.isError
            payload = json.loads(result.content[0].text)
            assert PROTOCOL_NAME in payload["protocols"]

    anyio.run(_round_trip)


@pytest.mark.parametrize(
    "width,height,orientation,box",
    [
        (8, 4, 1, [1, 0, 3, 4]),
        (8, 4, 6, [0, 1, 4, 3]),
        (1000, 1, 1, [2, 0, 3, 4]),
    ],
)
@pytest.mark.parametrize("label", [None, "bone"])
def test_segment_image_statistics_follow_oriented_input_footprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    width: int,
    height: int,
    orientation: int,
    box: list[int],
    label: str | None,
) -> None:
    """MCP counts content pixels after orientation, including very thin images."""
    bundle_dir = _write_hf_bundle(tmp_path)
    _patch_hf_download(monkeypatch, bundle_dir)
    image = Image.fromarray(
        (np.arange(width * height) % 256).astype(np.uint8).reshape(height, width)
    )
    image.getexif()[274] = orientation
    image_path = tmp_path / "input.png"
    image.save(image_path, exif=image.getexif())

    result = segment_image(str(image_path), str(tmp_path / "out"), label=label)

    entry = result["results"][0]
    assert entry["content_box"] == box
    bone = next(stat for stat in entry["labels"] if stat["label"] == "bone")
    top, left, bottom, right = box
    assert bone["pixel_count"] == (bottom - top) * (right - left)
    assert bone["pixel_fraction"] == 1.0
    assert np.load(entry["artifacts"]["masks"]).shape[-2:] == (4, 4)


def test_label_statistics_exclude_padding_from_probabilities() -> None:
    """Padding is excluded from probability summaries as well as mask counts."""
    from fxr.inference import FleXrayPrediction
    from fxr.mcp.inference_tools import _label_statistics

    probabilities = torch.full((1, 2, 4, 4), 0.9)
    probabilities[:, 0, 1:3] = torch.tensor([0.1, 0.4]).view(2, 1)
    probabilities[:, 1, 1:3] = torch.tensor([0.6, 0.8]).view(2, 1)
    prediction = FleXrayPrediction(
        logits=torch.logit(probabilities),
        probabilities=probabilities,
        masks=(probabilities >= 0.5).to(torch.uint8),
    )

    stats = _label_statistics(
        prediction=prediction,
        label_names=("background", "bone"),
        label=None,
        content_box=[1, 0, 3, 4],
    )

    assert [stat["pixel_count"] for stat in stats] == [0, 8]
    assert [stat["pixel_fraction"] for stat in stats] == [0.0, 1.0]
    assert [stat["mean_probability"] for stat in stats] == pytest.approx([0.25, 0.7])
    assert [stat["max_probability"] for stat in stats] == pytest.approx([0.4, 0.8])
