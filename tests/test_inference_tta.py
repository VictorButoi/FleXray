from __future__ import annotations

import pytest
import torch

import fxr.inference.tta as tta
from fxr.inference import InferenceRunner, predict_with_tta
from fxr.inference.cli import _build_parser


class _IdentityModel(torch.nn.Module):
    """Test model that returns its input as logits.

    Attributes:
        scale: Scalar parameter used only for device inference.
    """

    def __init__(self) -> None:
        """Create the identity model with one device-carrying parameter.

        Returns:
            None.
        """

        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return the input unchanged.

        Args:
            images: Batched image tensor.

        Returns:
            ``images`` multiplied by the unit scale parameter.
        """

        return images * self.scale


class _ConstantModel(torch.nn.Module):
    """Test model that ignores its input and returns fixed logits.

    Attributes:
        logits: Constant ``BxCxHxW`` logits returned by every forward call.
    """

    def __init__(self, logits: torch.Tensor) -> None:
        """Store the constant logits.

        Args:
            logits: Tensor returned by ``forward`` regardless of input.

        Returns:
            None.
        """

        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return the stored constant logits.

        Args:
            images: Ignored batched image tensor.

        Returns:
            The constant logits.
        """

        return self.logits.clone()


def _disable_intensity_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero every intensity-op probability so only the flip can fire."""

    for name in (
        "_INVERT_PROB",
        "_CLAHE_PROB",
        "_GAMMA_PROB",
        "_CONTRAST_PROB",
        "_SHARPNESS_PROB",
        "_GAUSSIAN_NOISE_PROB",
    ):
        monkeypatch.setattr(tta, name, 0.0)


def test_flip_is_exactly_inverted_on_logits(monkeypatch: pytest.MonkeyPatch) -> None:
    """A guaranteed flip with an identity model must merge to the plain result."""

    _disable_intensity_ops(monkeypatch)
    monkeypatch.setattr(tta, "_FLIP_PROB", 1.0)
    torch.manual_seed(0)
    images = torch.rand((2, 1, 8, 8))
    assert not torch.equal(images, torch.flip(images, dims=(-1,)))

    result = InferenceRunner(_IdentityModel()).predict(images, tta_samples=4)

    # Identity model + exact unflip means every view's probabilities equal the
    # plain pass, so the TTA mean must match sigmoid(images) exactly.
    assert torch.allclose(result.probabilities, torch.sigmoid(images), atol=1e-6)


def test_tta_samples_one_is_bit_identical_to_plain_predict() -> None:
    """``tta_samples`` of 0 or 1 must not change the non-TTA output."""

    torch.manual_seed(0)
    images = torch.rand((2, 3, 8, 8))
    runner = InferenceRunner(_IdentityModel())

    plain = runner.predict(images)
    for tta_samples in (0, 1):
        result = runner.predict(images, tta_samples=tta_samples)
        assert torch.equal(result.logits, plain.logits)
        assert torch.equal(result.probabilities, plain.probabilities)
    assert plain.probabilities.shape == (2, 3, 8, 8)
    assert plain.probabilities.dtype == torch.float32


def test_merge_is_mean_probability_and_relogitized() -> None:
    """A constant, flip-invariant model must merge to its own probabilities."""

    torch.manual_seed(0)
    # Constant per channel, so a horizontal flip leaves the logits unchanged.
    constant_logits = torch.tensor([0.5, -1.0, 2.0]).view(1, 3, 1, 1).expand(2, 3, 8, 8)
    model = _ConstantModel(constant_logits.contiguous())
    images = torch.rand((2, 3, 8, 8))

    result = InferenceRunner(model).predict(images, tta_samples=5)

    expected_probabilities = torch.sigmoid(constant_logits)
    assert torch.allclose(result.probabilities, expected_probabilities, atol=1e-6)
    assert torch.allclose(result.logits, constant_logits, atol=1e-4)


def test_ensemble_members_share_each_augmented_view() -> None:
    """Every member must see the plain view and the same drawn augmented views."""

    torch.manual_seed(0)
    images = torch.rand((2, 1, 8, 8))
    seen: dict[str, list[torch.Tensor]] = {"a": [], "b": []}

    def _recorder(name: str):
        def forward(view: torch.Tensor) -> torch.Tensor:
            seen[name].append(view.clone())
            return view

        return forward

    predict_with_tta(
        [_recorder("a"), _recorder("b")], images, mode="multilabel", tta_samples=3
    )

    assert len(seen["a"]) == len(seen["b"]) == 3
    assert torch.equal(seen["a"][0], images)
    for view_a, view_b in zip(seen["a"], seen["b"]):
        assert torch.equal(view_a, view_b)
    assert not torch.equal(seen["a"][1], seen["a"][2])


def test_ensemble_mean_is_over_members_times_views() -> None:
    """Two constant members must merge to the mean of their probabilities."""

    torch.manual_seed(0)
    logits_a = torch.tensor([0.5, -1.0, 2.0]).view(1, 3, 1, 1).expand(2, 3, 8, 8).contiguous()
    logits_b = torch.tensor([-0.5, 1.0, 0.0]).view(1, 3, 1, 1).expand(2, 3, 8, 8).contiguous()
    runner = InferenceRunner([_ConstantModel(logits_a), _ConstantModel(logits_b)])
    images = torch.rand((2, 3, 8, 8))

    result = runner.predict(images, tta_samples=4)

    expected = (torch.sigmoid(logits_a) + torch.sigmoid(logits_b)) / 2
    assert torch.allclose(result.probabilities, expected, atol=1e-6)
    assert torch.allclose(torch.sigmoid(result.logits), expected, atol=1e-4)
    assert len(runner.models) == 2


def test_ensemble_of_one_is_bit_identical_to_single_model() -> None:
    """Wrapping one model in a list must not change outputs, with or without TTA."""

    images = torch.rand((2, 1, 8, 8))
    model = _IdentityModel()
    single = InferenceRunner(model)
    ensemble = InferenceRunner([model])

    for tta_samples in (1, 3):
        torch.manual_seed(0)
        expected = single.predict(images, tta_samples=tta_samples)
        torch.manual_seed(0)
        actual = ensemble.predict(images, tta_samples=tta_samples)
        assert torch.equal(actual.logits, expected.logits)
        assert torch.equal(actual.probabilities, expected.probabilities)
    assert ensemble.model is model


def test_tta_v3_preset_values_are_pinned() -> None:
    """Keep public inference aligned with the released ensemble TTA recipe."""

    assert tta._FLIP_PROB == 0.5
    assert tta._INVERT_PROB == 0.5
    assert tta._CLAHE_PROB == 0.1
    assert tta._CLAHE_CLIP_LIMIT_RANGE == (1.0, 2.0)
    assert tta._GAMMA_PROB == 0.25
    assert tta._GAMMA_RANGE == (0.9, 1.1)
    assert tta._GAIN_RANGE == (0.9, 1.1)
    assert tta._CONTRAST_PROB == 0.25
    assert tta._CONTRAST_RANGE == (0.7, 1.3)
    assert tta._SHARPNESS_PROB == 0.5
    assert tta._SHARPNESS_RANGE == (0.7, 1.3)
    assert tta._GAUSSIAN_NOISE_PROB == 0.25
    assert tta._GAUSSIAN_NOISE_STD == 0.01
    assert not hasattr(tta, "_INTENSITY_SCALE_PROB")
    assert not hasattr(tta, "_BRIGHTNESS_PROB")


def test_clahe_is_deterministic_and_preserves_tensor_contract() -> None:
    """The dependency-light CLAHE branch must retain shape, dtype, and range."""

    image = torch.linspace(0.0, 1.0, 64 * 64).reshape(1, 64, 64)
    first = tta._clahe(image, clip_limit=1.5, grid_size=(8, 8))
    second = tta._clahe(image, clip_limit=1.5, grid_size=(8, 8))

    assert torch.equal(first, second)
    assert first.shape == image.shape
    assert first.dtype == image.dtype
    assert float(first.min()) >= 0.0
    assert float(first.max()) <= 1.0
    assert not torch.equal(first, image)


def test_augmented_views_receive_bundle_normalizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bundle normalization must run after tta_v3 intensity augmentation."""

    images = torch.full((1, 1, 4, 4), 0.75)
    seen: list[torch.Tensor] = []

    def forward(view: torch.Tensor) -> torch.Tensor:
        seen.append(view.clone())
        return view

    monkeypatch.setattr(
        tta,
        "_augment_batch",
        lambda batch: (
            batch * 2.0,
            torch.zeros(batch.shape[0], dtype=torch.bool, device=batch.device),
        ),
    )
    tta._predict_with_tta(
        forward,
        images,
        mode="multilabel",
        tta_samples=2,
        normalizer=lambda batch: batch.clamp(0.0, 1.0),
    )

    assert torch.equal(seen[0], images)
    assert torch.equal(seen[1], torch.ones_like(images))


def test_negative_tta_samples_rejected() -> None:
    """Negative pass counts must fail loudly in both public entry points."""

    images = torch.rand((1, 1, 8, 8))
    with pytest.raises(ValueError, match="tta_samples"):
        InferenceRunner(_IdentityModel()).predict(images, tta_samples=-1)
    with pytest.raises(ValueError, match="tta_samples"):
        predict_with_tta(lambda x: x, images, mode="multilabel", tta_samples=-2)


def test_cli_parser_accepts_tta_samples() -> None:
    """``flexify`` must expose ``--tta-samples`` with a default of 1."""

    parser = _build_parser()
    base_args = ["--input", "image.png", "--output-dir", "out"]
    assert parser.parse_args(base_args).tta_samples == 1
    assert parser.parse_args([*base_args, "--tta-samples", "8"]).tta_samples == 8
