from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from safetensors.torch import save_file

import fxr
import fxr.inference as public_inference
import fxr.inference.artifacts as artifact_module
from fxr.inference.artifacts import build_model_from_config
from fxr.models import UNet
from fxr.inference import (
    DEFAULT_MODEL_ID,
    FleXrayPrediction,
    FleXraySegmenter,
    ImagePreprocessing,
    InferenceBatchResult,
    InferencePostprocessResult,
    InferenceRunner,
    apply_inference_postprocessing,
    dequantize_saved_probabilities,
    get_inference_postprocessing_metadata,
    probabilities_from_logits,
    quantized_probabilities,
)


def _logits_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return torch.logit(probs.clamp(1e-6, 1.0 - 1e-6))


class _ScaledEchoModel(torch.nn.Module):
    """Test model that scales images and records call state.

    Attributes:
        scale: Scalar parameter used to infer device and scale outputs.
        seen_device: Device of the most recent input received by ``forward``.
        seen_training: Training flag observed during the most recent call.
    """

    def __init__(self, scale: float = 1.0) -> None:
        """Create a scalar-parameter model.

        Args:
            scale: Multiplicative scale applied to input images.

        Returns:
            None.
        """

        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(float(scale)))
        self.seen_device: torch.device | None = None
        self.seen_training: bool | None = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Scale the input images and record call state.

        Args:
            images: Batched image tensor.

        Returns:
            Input tensor multiplied by ``scale``.
        """

        self.seen_device = images.device
        self.seen_training = bool(self.training)
        return images * self.scale


class _BufferOnlyModel(torch.nn.Module):
    """Test model whose device is inferred from a registered buffer.

    Attributes:
        offset: Scalar buffer added to input images.
        seen_device: Device of the most recent input received by ``forward``.
    """

    def __init__(self, offset: float = 0.0) -> None:
        """Create a model with no parameters and one scalar buffer.

        Args:
            offset: Additive offset applied to input images.

        Returns:
            None.
        """

        super().__init__()
        self.register_buffer("offset", torch.tensor(float(offset)))
        self.seen_device: torch.device | None = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Add the buffer offset to input images.

        Args:
            images: Batched image tensor.

        Returns:
            Input tensor plus ``offset``.
        """

        self.seen_device = images.device
        return images + self.offset


class _StaticOutputModel(torch.nn.Module):
    """Test model that returns a preconfigured output object.

    Attributes:
        anchor: Scalar parameter used for device inference.
        output: Object returned from ``forward``.
    """

    def __init__(self, output: object) -> None:
        """Create a model that returns ``output``.

        Args:
            output: Object returned whenever the model is called.

        Returns:
            None.
        """

        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.output = output

    def forward(self, images: torch.Tensor) -> object:
        """Return the configured output and ignore input contents.

        Args:
            images: Batched image tensor supplied by the runner.

        Returns:
            The preconfigured output object.
        """

        del images
        return self.output


class _StatelessEchoModel(torch.nn.Module):
    """Test model with no parameters or buffers.

    Attributes:
        No public attributes are defined because the model is intentionally
        stateless.
    """

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return input images unchanged.

        Args:
            images: Batched image tensor.

        Returns:
            The same tensor passed to the method.
        """

        return images


def test_public_inference_api_exports_planned_names() -> None:
    assert "inference" in fxr.__all__
    assert public_inference.__all__ == [
        "DEFAULT_MODEL_ID",
        "ENSEMBLE_MANIFEST_FILENAME",
        "ENSEMBLE_MEMBER_SUBFOLDERS",
        "FLAGSHIP_SUBFOLDER",
        "FleXrayPrediction",
        "FleXraySegmenter",
        "ImagePreprocessing",
        "InferenceBatchResult",
        "InferencePostprocessResult",
        "InferenceRunner",
        "apply_inference_postprocessing",
        "dequantize_saved_probabilities",
        "get_inference_postprocessing_metadata",
        "predict_with_tta",
        "probabilities_from_logits",
        "quantized_probabilities",
    ]
    assert DEFAULT_MODEL_ID == "VictorButoi/flexray"
    assert FleXrayPrediction.__name__ == "FleXrayPrediction"
    assert FleXraySegmenter.__name__ == "FleXraySegmenter"
    assert ImagePreprocessing.__name__ == "ImagePreprocessing"
    assert InferenceBatchResult.__name__ == "InferenceBatchResult"
    assert InferencePostprocessResult.__name__ == "InferencePostprocessResult"
    assert InferenceRunner.__name__ == "InferenceRunner"
    assert not hasattr(public_inference, "InferenceExperiment")
    assert not hasattr(public_inference, "PredictionCallback")


def test_inference_initializer_is_reexport_only() -> None:
    tree = ast.parse(Path(public_inference.__file__).read_text(encoding="utf-8"))
    disallowed = (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef)

    assert not any(isinstance(node, disallowed) for node in tree.body)


def test_inference_runner_predict_returns_logits_and_probabilities() -> None:
    model = _ScaledEchoModel(scale=2.0)
    images = torch.tensor(
        [[[[0.0, 1.0], [2.0, -1.0]], [[1.0, -1.0], [0.5, 0.0]]]],
        dtype=torch.float32,
    )

    result = InferenceRunner(model, probability_mode="sigmoid").predict(images)

    expected_logits = images * 2.0
    assert isinstance(result, InferenceBatchResult)
    assert torch.allclose(result.logits, expected_logits)
    assert torch.allclose(result.probabilities, torch.sigmoid(expected_logits))
    assert result.logits.device == model.scale.device
    assert result.probabilities.device == model.scale.device


def test_inference_runner_honors_constructor_probability_modes() -> None:
    model = _ScaledEchoModel(scale=1.0)
    images = torch.tensor([[[[0.0]], [[1.0]]]], dtype=torch.float32)

    sigmoid = InferenceRunner(model, probability_mode="sigmoid").predict(images)
    softmax = InferenceRunner(model, probability_mode="softmax").predict(images)

    assert torch.allclose(sigmoid.probabilities, torch.sigmoid(images))
    assert torch.allclose(softmax.probabilities, torch.softmax(images, dim=1))
    assert not torch.allclose(sigmoid.probabilities, softmax.probabilities)
    with pytest.raises(ValueError, match="probability mode"):
        InferenceRunner(model, probability_mode="not-a-mode")


def test_inference_runner_restores_model_eval_mode_after_prediction() -> None:
    model = _ScaledEchoModel()
    runner = InferenceRunner(model)
    images = torch.zeros((1, 1, 2, 2), dtype=torch.float32)

    model.train()
    runner.predict(images)

    assert model.seen_training is False
    assert model.training is True

    model.eval()
    runner.predict(images)

    assert model.seen_training is False
    assert model.training is False


def test_inference_runner_moves_inputs_to_parameter_device_on_cpu() -> None:
    model = _ScaledEchoModel()
    images = torch.zeros((1, 1, 2, 2), dtype=torch.float32)

    InferenceRunner(model).predict(images)

    assert model.seen_device == model.scale.device
    assert model.seen_device == torch.device("cpu")


def test_inference_runner_infers_device_from_buffer_when_no_parameters() -> None:
    model = _BufferOnlyModel(offset=3.0)
    images = torch.zeros((1, 1, 2, 2), dtype=torch.float32)

    result = InferenceRunner(model).predict(images)

    assert list(model.parameters()) == []
    assert model.seen_device == model.offset.device
    assert torch.allclose(result.logits, torch.full_like(images, 3.0))
    assert torch.allclose(result.probabilities, torch.sigmoid(result.logits))


def test_inference_runner_rejects_empty_and_mixed_device_ensembles() -> None:
    with pytest.raises(ValueError, match="at least one model"):
        InferenceRunner([])
    with pytest.raises(TypeError, match="torch.nn.Module"):
        InferenceRunner([torch.nn.Identity(), "not-a-module"])

    cpu_model = UNet(in_channels=1, out_channels=2, filters=[2])
    meta_model = UNet(in_channels=1, out_channels=2, filters=[2]).to("meta")
    runner = InferenceRunner([cpu_model, meta_model])
    with pytest.raises(ValueError, match="share one device"):
        runner.predict(torch.zeros((1, 1, 4, 4)))
    with pytest.raises(ValueError, match="use .models"):
        _ = runner.model


def test_inference_runner_rejects_models_without_parameters_or_buffers() -> None:
    with pytest.raises(ValueError, match="Could not infer model device"):
        InferenceRunner(_StatelessEchoModel()).predict(
            torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        )


@pytest.mark.parametrize(
    "output",
    [
        pytest.param((torch.zeros((1, 1, 2, 2)),), id="tuple"),
        pytest.param([torch.zeros((1, 1, 2, 2))], id="list"),
        pytest.param({"logits": torch.zeros((1, 1, 2, 2))}, id="dict"),
    ],
)
def test_inference_runner_rejects_container_model_outputs(output: object) -> None:
    model = _StaticOutputModel(output)

    with pytest.raises(TypeError, match="tuple, list, and dict outputs"):
        InferenceRunner(model).predict(
            torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        )


def test_inference_runner_validates_input_and_logit_shapes() -> None:
    runner = InferenceRunner(_ScaledEchoModel())

    with pytest.raises(TypeError, match="images.*torch.Tensor"):
        runner.predict(np.zeros((1, 1, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="images.*BxCxHxW"):
        runner.predict(torch.zeros((1, 1, 2), dtype=torch.float32))

    invalid_output = _StaticOutputModel(torch.zeros((1, 1, 2), dtype=torch.float32))
    with pytest.raises(ValueError, match="model output logits.*BxCxHxW"):
        InferenceRunner(invalid_output).predict(
            torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        )


def test_probabilities_from_logits_modes_and_single_channel_outputs() -> None:
    logits = torch.tensor(
        [[[[0.0, 2.0]], [[1.0, -1.0]]]],
        dtype=torch.float32,
    )

    assert torch.allclose(
        probabilities_from_logits(logits, mode="binary"),
        torch.sigmoid(logits),
    )
    assert torch.allclose(
        probabilities_from_logits(logits, mode="multilabel"),
        torch.sigmoid(logits),
    )

    softmax_probs = probabilities_from_logits(logits, mode="onehot")
    assert torch.allclose(softmax_probs, torch.softmax(logits, dim=1))
    assert torch.allclose(
        softmax_probs.sum(dim=1),
        torch.ones_like(softmax_probs[:, 0]),
    )

    single_channel = torch.tensor([[[[0.0, 2.0]]]], dtype=torch.float32)
    assert torch.allclose(
        probabilities_from_logits(single_channel, mode="softmax"),
        torch.sigmoid(single_channel),
    )


def test_probabilities_from_logits_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="Unsupported probability mode"):
        probabilities_from_logits(torch.zeros((1, 2, 3)), mode="unknown")
    with pytest.raises(ValueError, match="channel dimensions"):
        probabilities_from_logits(torch.zeros((3,)), mode="binary")


def test_probability_sidecar_quantization_and_uint8_dequantization() -> None:
    probs = torch.tensor([[[0.0, 0.5, 1.0, 1.2, -0.1]]], dtype=torch.float32)

    quantized = quantized_probabilities(probs)
    restored = dequantize_saved_probabilities(quantized)

    assert quantized.dtype == np.uint8
    assert quantized.shape == (1, 1, 5)
    assert quantized.tolist() == [[[0, 128, 255, 255, 0]]]
    assert restored.dtype == np.float32
    assert np.allclose(restored, quantized.astype(np.float32) / 255.0)


def test_dequantize_saved_probabilities_warns_for_legacy_compatible_dtypes() -> None:
    with pytest.warns(UserWarning, match="bool sidecars"):
        bool_probs = dequantize_saved_probabilities(
            np.array([[[True, False]]], dtype=bool)
        )
    with pytest.warns(UserWarning, match="integer sidecars in"):
        binary_int_probs = dequantize_saved_probabilities(
            np.array([[[0, 1]]], dtype=np.int16)
        )
    with pytest.warns(UserWarning, match="non-uint8 integer dtype"):
        int_probs = dequantize_saved_probabilities(
            np.array([[[0, 128, 255]]], dtype=np.int16)
        )
    with pytest.warns(UserWarning, match="floating-point sidecars"):
        float_probs = dequantize_saved_probabilities(
            np.array([[[0.25, 1.0]]], dtype=np.float64)
        )

    assert bool_probs.dtype == np.float32
    assert bool_probs.tolist() == [[[1.0, 0.0]]]
    assert binary_int_probs.tolist() == [[[0.0, 1.0]]]
    assert np.allclose(int_probs, np.array([[[0.0, 128.0 / 255.0, 1.0]]]))
    assert float_probs.dtype == np.float32
    assert float_probs.tolist() == [[[0.25, 1.0]]]


def test_dequantize_saved_probabilities_rejects_bad_shape_ranges_and_dtype() -> None:
    empty = dequantize_saved_probabilities(np.empty((0, 2, 3), dtype=np.int16))
    assert empty.dtype == np.float32
    assert empty.shape == (0, 2, 3)

    with pytest.raises(ValueError, match="shape CxHxW"):
        dequantize_saved_probabilities(np.zeros((1, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="Integer probability sidecars"):
        dequantize_saved_probabilities(np.array([[[-1]]], dtype=np.int16))
    with pytest.raises(ValueError, match="Floating-point probability sidecars"):
        dequantize_saved_probabilities(np.array([[[1.1]]], dtype=np.float32))
    with pytest.raises(TypeError, match="Unsupported probability sidecar dtype"):
        dequantize_saved_probabilities(np.array([[[1.0 + 0.0j]]], dtype=np.complex64))


def test_postprocessing_metadata_reports_identity_enabled_and_skip_states() -> None:
    unknown = get_inference_postprocessing_metadata(
        dataset_name="Toy",
        label_names=("background",),
    )
    missing_names = get_inference_postprocessing_metadata(
        dataset_name="HipRay",
        label_names=None,
    )
    missing_required = get_inference_postprocessing_metadata(
        dataset_name="MendeleyCXR",
        label_names=("background", "lungs"),
    )
    enabled = get_inference_postprocessing_metadata(
        dataset_name="ShoulderMonch",
        label_names=("background", "clavicles", "humeri"),
    )

    assert unknown["status"] == "identity"
    assert unknown["registered"] is False
    assert missing_names["status"] == "skipped_missing_label_names"
    assert missing_required["status"] == "skipped_missing_required_labels"
    assert missing_required["detail"] == "missing labels: liver"
    assert enabled["status"] == "enabled"
    assert enabled["applied"] is True
    assert enabled["changed_label_names"] == ["clavicles"]


def test_apply_postprocessing_returns_identity_for_unknown_or_missing_labels() -> None:
    probs = torch.rand((1, 2, 3, 3), dtype=torch.float32)
    logits = _logits_from_probs(probs)

    unknown = apply_inference_postprocessing(
        logits=logits,
        probabilities=probs,
        dataset_name="Toy",
        label_names=("background", "thing"),
    )
    missing = apply_inference_postprocessing(
        logits=logits,
        probabilities=probs,
        dataset_name="HipRay",
        label_names=("background", "femurs"),
    )

    assert unknown.metadata["name"] == "identity"
    assert unknown.logits is logits
    assert unknown.probabilities is probs
    assert missing.metadata["status"] == "skipped_missing_required_labels"
    assert missing.metadata["detail"] == "missing labels: hips"
    assert missing.logits is logits
    assert missing.probabilities is probs


def test_hipray_removes_hips_near_femurs_and_updates_only_hips() -> None:
    label_names = ("background", "femurs", "hips")
    femur_idx = label_names.index("femurs")
    hip_idx = label_names.index("hips")
    probs = torch.zeros((1, len(label_names), 11, 11), dtype=torch.float32)
    probs[:, femur_idx] = 0.2
    probs[0, femur_idx, 5, 5] = 0.9
    probs[0, hip_idx, 5, 9] = 0.8
    probs[0, hip_idx, 5, 10] = 0.8
    logits = _logits_from_probs(probs)

    result = apply_inference_postprocessing(
        logits=logits,
        probabilities=probs,
        label_names=label_names,
        dataset_name="HipRay",
        threshold=0.5,
    )

    assert result.metadata["status"] == "applied"
    assert result.metadata["changed_label_names"] == ["hips"]
    assert torch.equal(result.probabilities[:, femur_idx], probs[:, femur_idx])
    assert torch.equal(result.logits[:, femur_idx], logits[:, femur_idx])
    assert result.probabilities[0, hip_idx, 5, 9].item() == pytest.approx(0.0)
    assert result.probabilities[0, hip_idx, 5, 10].item() == pytest.approx(1.0)
    expected_logits = _logits_from_probs(result.probabilities[:, hip_idx].unsqueeze(1))
    assert torch.allclose(result.logits[:, hip_idx], expected_logits[:, 0])


def test_mendeleycxr_masks_lungs_where_liver_exceeds_threshold() -> None:
    label_names = ("background", "liver", "lungs")
    liver_idx = label_names.index("liver")
    lungs_idx = label_names.index("lungs")
    probs = torch.full((1, len(label_names), 2, 2), 0.1, dtype=torch.float32)
    probs[0, liver_idx] = torch.tensor(
        [[0.8, 0.5], [0.51, 0.49]],
        dtype=torch.float32,
    )
    probs[0, lungs_idx] = torch.tensor(
        [[0.7, 0.6], [0.4, 0.2]],
        dtype=torch.float32,
    )
    logits = _logits_from_probs(probs)

    result = apply_inference_postprocessing(
        logits=logits,
        probabilities=probs,
        label_names=label_names,
        dataset_name="MendeleyCXR",
        threshold=0.5,
    )

    expected_lungs = torch.tensor(
        [[[0.0, 0.6], [0.0, 0.2]]],
        dtype=torch.float32,
    )
    assert result.metadata["name"] == "MendeleyCXR"
    assert result.metadata["required_label_names"] == ["liver", "lungs"]
    assert result.metadata["changed_label_names"] == ["lungs"]
    assert torch.equal(result.probabilities[:, liver_idx], probs[:, liver_idx])
    assert torch.equal(result.logits[:, liver_idx], logits[:, liver_idx])
    assert torch.equal(result.probabilities[:, lungs_idx], expected_lungs)
    expected_logits = _logits_from_probs(expected_lungs.unsqueeze(1))
    assert torch.allclose(result.logits[:, lungs_idx], expected_logits[:, 0])


def test_shouldermonch_selects_closest_non_tiny_clavicle_component() -> None:
    label_names = ("background", "clavicles", "humeri")
    clavicle_idx = label_names.index("clavicles")
    humerus_idx = label_names.index("humeri")
    probs = torch.zeros((1, len(label_names), 64, 64), dtype=torch.float32)
    label = torch.zeros_like(probs)

    probs[0, clavicle_idx, 10:30, 10:30] = 0.9
    probs[0, clavicle_idx, 45:51, 45:51] = 0.9
    probs[0, clavicle_idx, 55:58, 55:58] = 0.9
    label[0, humerus_idx, 54:58, 54:58] = 1.0
    logits = _logits_from_probs(probs)

    result = apply_inference_postprocessing(
        logits=logits,
        probabilities=probs,
        label=label,
        label_names=label_names,
        dataset_name="ShoulderMonch",
    )

    expected = torch.zeros((64, 64), dtype=torch.float32)
    expected[45:51, 45:51] = 1.0
    assert result.metadata["status"] == "applied"
    assert torch.equal(result.probabilities[0, clavicle_idx], expected)
    assert torch.equal(result.probabilities[0, humerus_idx], probs[0, humerus_idx])
    expected_logits = _logits_from_probs(expected.unsqueeze(0).unsqueeze(0))
    assert torch.allclose(result.logits[0, clavicle_idx], expected_logits[0, 0])


def test_shouldermonch_skips_without_ground_truth_and_partially_applies() -> None:
    label_names = ("background", "clavicles", "humeri")
    clavicle_idx = label_names.index("clavicles")
    humerus_idx = label_names.index("humeri")
    probs = torch.zeros((2, len(label_names), 10, 10), dtype=torch.float32)
    label = torch.zeros_like(probs)

    probs[0, clavicle_idx, 0:2, 0:2] = 0.9
    probs[0, clavicle_idx, 6, 6] = 0.9
    probs[0, clavicle_idx, 7, 7] = 0.9
    probs[0, clavicle_idx, 8, 8] = 0.9
    label[0, humerus_idx, 8, 9] = 1.0
    probs[1, clavicle_idx, 2, 2] = 0.73
    logits = _logits_from_probs(probs)

    no_gt = apply_inference_postprocessing(
        logits=logits,
        probabilities=probs,
        label=None,
        label_names=label_names,
        dataset_name="ShoulderMonch",
    )
    partial = apply_inference_postprocessing(
        logits=logits,
        probabilities=probs,
        label=label,
        label_names=label_names,
        dataset_name="ShoulderMonch",
    )

    expected_cleaned = torch.zeros((10, 10), dtype=torch.float32)
    expected_cleaned[6, 6] = 1.0
    expected_cleaned[7, 7] = 1.0
    expected_cleaned[8, 8] = 1.0
    assert no_gt.metadata["status"] == "skipped_missing_ground_truth"
    assert no_gt.probabilities is probs
    assert partial.metadata["status"] == "partially_applied"
    assert partial.metadata["sample_statuses"] == [
        "applied",
        "skipped_missing_humeri_ground_truth",
    ]
    assert partial.metadata["skipped_batch_indices"] == [1]
    assert partial.metadata["detail"] == (
        "postprocessed 1/2 samples; skipped 1 without GT humeri"
    )
    assert torch.equal(partial.probabilities[0, clavicle_idx], expected_cleaned)
    assert torch.equal(partial.probabilities[1, clavicle_idx], probs[1, clavicle_idx])
    assert torch.equal(partial.logits[1, clavicle_idx], logits[1, clavicle_idx])


def test_postprocessing_validates_prediction_and_label_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        apply_inference_postprocessing(
            logits=torch.zeros((1, 3, 2, 2)),
            probabilities=torch.zeros((1, 3, 2, 3)),
            label_names=("background", "femurs", "hips"),
            dataset_name="HipRay",
        )
    with pytest.raises(ValueError, match="BxCxHxW"):
        apply_inference_postprocessing(
            logits=torch.zeros((3, 2, 2)),
            probabilities=torch.zeros((3, 2, 2)),
            label_names=("background", "femurs", "hips"),
            dataset_name="HipRay",
        )
    with pytest.raises(ValueError, match="does not match prediction shape"):
        apply_inference_postprocessing(
            logits=torch.zeros((1, 3, 2, 2)),
            probabilities=torch.zeros((1, 3, 2, 2)),
            label=torch.zeros((1, 3, 3, 2)),
            label_names=("background", "clavicles", "humeri"),
            dataset_name="ShoulderMonch",
        )
    with pytest.raises(ValueError, match="expected GT channel"):
        apply_inference_postprocessing(
            logits=torch.zeros((1, 3, 2, 2)),
            probabilities=torch.zeros((1, 3, 2, 2)),
            label=torch.zeros((1, 2, 2, 2)),
            label_names=("background", "clavicles", "humeri"),
            dataset_name="ShoulderMonch",
        )


@pytest.mark.parametrize(
    "model_config",
    [
        pytest.param(
            {"_class": "builtins.dict"},
            id="arbitrary-root-constructor",
        ),
        pytest.param(
            {
                "_class": "fxr.models.UNet",
                "in_channels": 1,
                "out_channels": 2,
                "filters": [2],
                "activation": {"_class": "builtins.dict"},
            },
            id="nested-class",
        ),
        pytest.param(
            {
                "_class": "fxr.models.UNet",
                "in_channels": 1,
                "out_channels": 2,
                "filters": [2],
                "compile_cfg": {"hook": {"_fn": "os.system"}},
            },
            id="nested-function-in-runtime-config",
        ),
    ],
)
def test_portable_model_configs_reject_executable_directives_without_importing(
    model_config: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fxr.config.imports as config_imports

    imported: list[str] = []

    def reject_import(path: str) -> object:
        """Record any attempted generic constructor import."""

        imported.append(path)
        raise AssertionError(f"Unexpected generic import: {path}")

    monkeypatch.setattr(config_imports, "absolute_import", reject_import)

    with pytest.raises(ValueError, match="constructor|executable directive"):
        artifact_module.build_model_from_config(
            {"model": model_config},
            label_names=None,
        )

    assert imported == []


def test_portable_unet_schema_rejects_unknown_constructor_arguments() -> None:
    config = {
        "model": {
            "_class": "fxr.models.UNet",
            "in_channels": 1,
            "out_channels": 2,
            "filters": [2],
            "unexpected": "value",
        }
    }

    with pytest.raises(ValueError, match="Unsupported fxr.models.UNet config keys"):
        artifact_module.build_model_from_config(config, label_names=None)


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


def test_flexray_segmenter_from_pretrained_loads_hf_bundle_and_predicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = _write_hf_bundle(tmp_path, image_size=(4, 4))
    calls: list[tuple[str, str, str | None]] = []
    _patch_hf_download(monkeypatch, bundle_dir, calls)

    segmenter = FleXraySegmenter.from_pretrained(device="cpu")
    image = np.full((3, 5, 3), 255, dtype=np.uint8)
    prediction = segmenter.predict(image, threshold=0.8)

    assert segmenter.repo_id == DEFAULT_MODEL_ID
    assert segmenter.subfolders == (None,)
    assert segmenter.label_names == ("background", "bone")
    assert calls[:2] == [
        (DEFAULT_MODEL_ID, "ensemble.json", None),
        (DEFAULT_MODEL_ID, "config.yml", None),
    ]
    assert prediction.logits.shape == (1, 2, 4, 4)
    assert prediction.probabilities.shape == (1, 2, 4, 4)
    assert prediction.masks.shape == (1, 2, 4, 4)
    assert prediction.masks.dtype == torch.uint8
    assert isinstance(prediction, FleXrayPrediction)
    assert torch.allclose(prediction.logits[:, 0], torch.full((1, 4, 4), -1.0))
    assert torch.allclose(prediction.logits[:, 1], torch.full((1, 4, 4), 1.0))
    assert torch.allclose(prediction.probabilities, torch.sigmoid(prediction.logits))
    assert torch.all(prediction.masks[:, 0] == 0)
    assert torch.all(prediction.masks[:, 1] == 0)


def test_flexray_segmenter_predict_label_filters_to_one_channel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = _write_hf_bundle(tmp_path, image_size=(4, 4))
    _patch_hf_download(monkeypatch, bundle_dir)

    segmenter = FleXraySegmenter.from_pretrained(device="cpu")
    image = np.full((4, 4), 255, dtype=np.uint8)
    prediction = segmenter.predict(image, label="bone")

    assert prediction.logits.shape == (1, 1, 4, 4)
    assert prediction.probabilities.shape == (1, 1, 4, 4)
    assert prediction.masks.shape == (1, 1, 4, 4)
    assert torch.allclose(prediction.logits[:, 0], torch.full((1, 4, 4), 1.0))

    with pytest.raises(ValueError, match="Unknown label"):
        segmenter.predict(image, label="femurs")


def test_flexray_segmenter_from_pretrained_honors_repo_revision_and_missing_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = _write_hf_bundle(tmp_path, image_size=(4, 4))
    calls: list[tuple[str, str, str | None]] = []
    _patch_hf_download(monkeypatch, bundle_dir, calls)

    FleXraySegmenter.from_pretrained("vbutoi/custom", revision="abc123")

    assert calls[:2] == [
        ("vbutoi/custom", "ensemble.json", "abc123"),
        ("vbutoi/custom", "config.yml", "abc123"),
    ]

    (bundle_dir / "label_schema.json").unlink()
    with pytest.raises(FileNotFoundError, match="label_schema.json"):
        FleXraySegmenter.from_pretrained("vbutoi/custom")


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


def test_flexray_segmenter_resolves_flagship_members_and_subfolder_from_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = _write_hf_bundle(tmp_path)
    subfolders = ("members/flagship", "members/other")
    repo = _write_ensemble_repo(tmp_path / "repo", bundle_dir, subfolders)
    calls: list[tuple[str, str, str | None]] = []
    _patch_hf_download(monkeypatch, {"org/repo": repo}, calls)

    flagship = FleXraySegmenter.from_pretrained("org/repo")
    ensemble = FleXraySegmenter.from_pretrained("org/repo", ensemble=True)
    member = FleXraySegmenter.from_pretrained("org/repo", subfolder="members/other")

    assert flagship.subfolders == ("members/flagship",)
    assert len(flagship.models) == 1
    assert ensemble.subfolders == subfolders
    assert len(ensemble.models) == 2
    assert member.subfolders == ("members/other",)
    weights = [call[1] for call in calls if call[1].endswith("model.safetensors")]
    assert weights == [
        "members/flagship/model.safetensors",
        "members/flagship/model.safetensors",
        "members/other/model.safetensors",
        "members/other/model.safetensors",
    ]


def test_flexray_segmenter_rejects_invalid_subfolder_and_ensemble_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = _write_hf_bundle(tmp_path)
    repo = _write_ensemble_repo(tmp_path / "repo", bundle_dir, ("members/a",))
    _patch_hf_download(monkeypatch, {"org/repo": repo, "org/root": bundle_dir})

    with pytest.raises(ValueError, match="mutually exclusive"):
        FleXraySegmenter.from_pretrained("org/repo", subfolder="members/a", ensemble=True)
    with pytest.raises(ValueError, match="single repo_id"):
        FleXraySegmenter.from_pretrained(["org/repo", "org/root"], ensemble=True)
    with pytest.raises(FileNotFoundError, match="ensemble.json"):
        FleXraySegmenter.from_pretrained("org/root", ensemble=True)

    (repo / "ensemble.json").write_text(
        json.dumps({"flagship": "members/a", "members": [{"subfolder": "members/a"}] * 2}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="repeat"):
        FleXraySegmenter.from_pretrained("org/repo", ensemble=True)


def _write_member_bundles(tmp_path: Path) -> dict[str, Path]:
    """Write two bundles whose output biases are mirror images of each other."""

    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    first = _write_hf_bundle(tmp_path / "first", image_size=(4, 4))
    second = _write_hf_bundle(tmp_path / "second", image_size=(4, 4))
    model = UNet(in_channels=1, out_channels=2, filters=[2])
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.output_conv.bias.copy_(torch.tensor([1.0, -1.0]))
    save_file(model.state_dict(), str(second / "model.safetensors"))
    return {"org/first": first, "org/second": second}


def test_flexray_segmenter_from_pretrained_repo_id_list_averages_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundles = _write_member_bundles(tmp_path)
    calls: list[tuple[str, str, str | None]] = []
    _patch_hf_download(monkeypatch, bundles, calls)

    segmenter = FleXraySegmenter.from_pretrained(
        ["org/first", "org/second"], revision="r1", device="cpu"
    )
    prediction = segmenter.predict(np.full((4, 4), 255, dtype=np.uint8))

    assert segmenter.repo_id == ("org/first", "org/second")
    assert len(segmenter.models) == 2
    assert {call[0] for call in calls} == {"org/first", "org/second"}
    assert all(call[2] == "r1" for call in calls)
    assert torch.allclose(prediction.probabilities, torch.full((1, 2, 4, 4), 0.5))
    assert torch.allclose(prediction.logits, torch.zeros((1, 2, 4, 4)), atol=1e-5)
    with pytest.raises(ValueError, match="use .models"):
        _ = segmenter.model


def test_flexray_segmenter_ensemble_rejects_mismatched_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundles = _write_member_bundles(tmp_path)
    _patch_hf_download(monkeypatch, bundles)
    second = bundles["org/second"]

    (second / "label_schema.json").write_text(
        json.dumps({"label_names": ["background", "femurs"], "num_labels": 2}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"'org/second'.*label_names"):
        FleXraySegmenter.from_pretrained(["org/first", "org/second"])

    (second / "label_schema.json").write_text(
        json.dumps({"label_names": ["background", "bone"], "num_labels": 2}),
        encoding="utf-8",
    )
    preprocessing = json.loads((second / "preprocessing.json").read_text(encoding="utf-8"))
    preprocessing["image_size"] = [8, 8]
    (second / "preprocessing.json").write_text(json.dumps(preprocessing), encoding="utf-8")
    with pytest.raises(ValueError, match=r"'org/second'.*preprocessing"):
        FleXraySegmenter.from_pretrained(["org/first", "org/second"])


def test_flexray_segmenter_rejects_empty_and_duplicate_repo_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hf_download(monkeypatch, _write_member_bundles(tmp_path))

    with pytest.raises(ValueError, match="at least one"):
        FleXraySegmenter.from_pretrained([])
    with pytest.raises(ValueError, match="repeat"):
        FleXraySegmenter.from_pretrained(["org/first", "org/first"])


def test_percentile_preprocessing_matches_training_normalizer() -> None:
    """Bundle percentile metadata reproduces the train-time tensor transform."""

    from fxr.augmentation import PercentileMinMaxNormalize

    image = torch.tensor(
        [0.0, 1.0, 2.0, 100.0, -5.0, 0.0, 5.0, 10.0]
    ).reshape(2, 1, 2, 2)
    preprocessing = ImagePreprocessing.from_metadata(
        {
            "image_size": [2, 2],
            "normalization": {
                "scheme": "percentile_minmax",
                "percentiles": [25.0, 75.0],
                "eps": 1.0e-7,
            },
        }
    )

    actual = preprocessing.prepare(image)
    expected = PercentileMinMaxNormalize(
        percentiles=(25.0, 75.0), eps=1.0e-7
    )(image)

    assert torch.allclose(actual, expected)
    assert preprocessing.to_metadata()["normalization"] == {
        "scheme": "percentile_minmax",
        "percentiles": [25.0, 75.0],
        "eps": 1.0e-7,
    }


def test_uint16_array_and_file_preprocessing_preserve_dtype_range(
    tmp_path: Path,
) -> None:
    """Sixteen-bit grayscale inputs are scaled without 8-bit clipping."""

    from PIL import Image

    pixels = np.array(
        [[0, 257], [32768, 65535]],
        dtype=np.uint16,
    )
    preprocessing = ImagePreprocessing(
        image_size=(2, 2),
        pad_to_square=False,
    )
    image_path = tmp_path / "sixteen-bit.png"
    Image.fromarray(pixels).save(image_path)
    expected = torch.from_numpy(pixels.astype(np.float32) / 65535.0)[
        None, None
    ]

    assert torch.allclose(preprocessing.prepare(pixels), expected, atol=1.0e-6)
    assert torch.allclose(preprocessing.prepare(image_path), expected, atol=1.0e-6)
    assert preprocessing.prepare(pixels)[0, 0, 0, 1].item() < 0.01
