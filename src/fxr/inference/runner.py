from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from .probabilities import _KNOWN_MODES, ProbabilityMode
from .tta import _predict_with_tta


@dataclass(frozen=True)
class InferenceBatchResult:
    """Batched tensors returned by ``InferenceRunner.predict``.

    Attributes:
        logits: Raw model output tensor with shape ``BxCxHxW`` on the model
            device.
        probabilities: Probability tensor with shape ``BxCxHxW`` on the model
            device, derived from ``logits`` with the requested probability mode.
    """

    logits: Tensor
    probabilities: Tensor


class InferenceRunner:
    """Minimal PyTorch model runner for already-prepared image tensors.

    Attributes:
        models: PyTorch modules invoked for prediction, one per ensemble member
            (a single-model runner has exactly one). The caller owns checkpoint
            loading, preprocessing, normalization, postprocessing, iteration, and
            persistence.
        model: The sole member for single-model runners; raises for ensembles.
        probability_mode: Logits-to-probability conversion mode applied to every
            prediction. It is a property of the model head, not of one call.
        _tta_normalizer: Optional bundle-specific normalizer applied after each
            augmented TTA view and before member inference.
    """

    def __init__(
        self,
        model: torch.nn.Module | Sequence[torch.nn.Module],
        probability_mode: ProbabilityMode | str = "multilabel",
    ) -> None:
        """Store the model(s) and their probability conversion mode.

        Args:
            model: PyTorch module whose forward call returns logits, or a
                sequence of such modules that are averaged as an ensemble.
            probability_mode: Conversion mode forwarded to
                ``probabilities_from_logits`` for every prediction.

        Returns:
            None.

        Raises:
            TypeError: If any member is not a ``torch.nn.Module``.
            ValueError: If ``probability_mode`` is unknown or no model is given.
        """

        resolved_mode = str(probability_mode).strip().lower()
        if resolved_mode not in _KNOWN_MODES:
            raise ValueError(
                "Unsupported probability mode "
                f"{probability_mode!r}; expected one of {sorted(_KNOWN_MODES)}."
            )
        self.models = _member_models(model)
        self.probability_mode = resolved_mode
        self._tta_normalizer: Callable[[Tensor], Tensor] | None = None

    def _configure_tta_normalizer(
        self, normalizer: Callable[[Tensor], Tensor]
    ) -> None:
        """Configure post-augmentation normalization for a loaded bundle.

        Args:
            normalizer: Callable mapping an augmented ``BxCxHxW`` tensor to
                the model input range.

        Returns:
            None.
        """

        self._tta_normalizer = normalizer

    @property
    def model(self) -> torch.nn.Module:
        """Return the single wrapped model.

        Returns:
            The sole member module.

        Raises:
            ValueError: If the runner wraps an ensemble; use ``models`` instead.
        """

        if len(self.models) != 1:
            raise ValueError(
                f"InferenceRunner wraps {len(self.models)} ensemble members; use .models."
            )
        return self.models[0]

    def predict(
        self,
        images: Tensor,
        *,
        tta_samples: int = 1,
        seed: int | None = None,
    ) -> InferenceBatchResult:
        """Run the model(s) on one batched tensor and return logits/probabilities.

        Args:
            images: Input image tensor with shape ``BxCxHxW``. It is moved to the
                inferred model device before the models are called.
            tta_samples: Total number of test-time-augmentation forward passes
                per member. Values ``<= 1`` run one plain forward pass. Larger
                values add ``tta_samples - 1`` randomly augmented passes, shared
                across members, whose probabilities are averaged with the plain
                pass (see ``fxr.inference.tta``).
            seed: Optional seed for the test-time-augmentation draws. ``None``
                draws from the global torch RNG, so repeated runs differ.

        Returns:
            ``InferenceBatchResult`` containing logits and probabilities on the
            model device; ensembles return the mean over members.

        Raises:
            TypeError: If ``images`` is not a tensor or a model output is not a
                tensor.
            ValueError: If ``images`` or model logits are not shaped
                ``BxCxHxW``, if ``tta_samples`` is negative, if a member device
                cannot be inferred from a parameter or buffer, or if members sit
                on different devices.
        """

        _require_batched_tensor(images, name="images")
        device = _infer_shared_device(self.models)
        forward_fns = [functools.partial(self._member_forward, model) for model in self.models]

        training_flags = [bool(model.training) for model in self.models]
        try:
            for model in self.models:
                model.eval()
            with torch.inference_mode():
                logits, probabilities = _predict_with_tta(
                    forward_fns,
                    images.to(device),
                    mode=self.probability_mode,
                    tta_samples=tta_samples,
                    normalizer=self._tta_normalizer,
                    seed=seed,
                )
                return InferenceBatchResult(
                    logits=logits,
                    probabilities=probabilities,
                )
        finally:
            for model, was_training in zip(self.models, training_flags):
                model.train(was_training)

    @staticmethod
    def _member_forward(model: torch.nn.Module, images: Tensor) -> Tensor:
        """Run one validated forward pass of one member returning ``BxCxHxW`` logits.

        Args:
            model: Member module to call.
            images: Input tensor already on the model device.

        Returns:
            Model logits with shape ``BxCxHxW``.

        Raises:
            TypeError: If the model output is not a tensor.
            ValueError: If the model logits are not shaped ``BxCxHxW``.
        """

        output = model(images)
        logits = _require_tensor_logits(output)
        _require_batched_tensor(logits, name="model output logits")
        return logits


def _member_models(
    model: torch.nn.Module | Sequence[torch.nn.Module],
) -> tuple[torch.nn.Module, ...]:
    """Normalize one module or a sequence of modules to a non-empty tuple.

    Args:
        model: Single module or sequence of ensemble members.

    Returns:
        Tuple of member modules.

    Raises:
        TypeError: If any member is not a ``torch.nn.Module``.
        ValueError: If the sequence is empty.
    """

    models = (model,) if isinstance(model, torch.nn.Module) else tuple(model)
    if not models:
        raise ValueError("InferenceRunner requires at least one model.")
    for member in models:
        if not isinstance(member, torch.nn.Module):
            raise TypeError(
                f"Ensemble members must be torch.nn.Module instances; got {type(member).__name__}."
            )
    return models


def _infer_shared_device(models: Sequence[torch.nn.Module]) -> torch.device:
    """Infer the single device shared by every member.

    Args:
        models: Member modules.

    Returns:
        The common member device.

    Raises:
        ValueError: If a member has no parameters or buffers, or members sit on
            different devices.
    """

    devices = [_infer_model_device(model) for model in models]
    if any(device != devices[0] for device in devices):
        raise ValueError(
            f"Ensemble members must share one device; got {sorted({str(d) for d in devices})}."
        )
    return devices[0]


def _infer_model_device(model: torch.nn.Module) -> torch.device:
    """Infer a model device from the first parameter, then first buffer.

    Args:
        model: PyTorch module inspected for parameters and buffers.

    Returns:
        Device of the first parameter, or the first buffer when the module has no
        parameters.

    Raises:
        ValueError: If the module has no parameters or buffers.
    """

    for parameter in model.parameters():
        return parameter.device
    for buffer in model.buffers():
        return buffer.device
    raise ValueError(
        "Could not infer model device from parameters or buffers; "
        "register a parameter or buffer before calling predict."
    )


def _require_tensor_logits(output: object) -> Tensor:
    """Return tensor model output or raise a clear type error.

    Args:
        output: Raw object returned by the model.

    Returns:
        The output as a tensor.

    Raises:
        TypeError: If the model returned a tuple, list, dict, or other non-tensor
            object.
    """

    if isinstance(output, (tuple, list, dict)):
        raise TypeError(
            "InferenceRunner model output must be a torch.Tensor with shape "
            "BxCxHxW; tuple, list, and dict outputs are not supported."
        )
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            "InferenceRunner model output must be a torch.Tensor with shape "
            f"BxCxHxW; got {type(output).__name__}."
        )
    return output


def _require_batched_tensor(tensor: object, *, name: str) -> None:
    """Validate one tensor as a batched channel-first 2D tensor.

    Args:
        tensor: Object expected to be a tensor.
        name: Human-readable tensor name for error messages.

    Returns:
        None.

    Raises:
        TypeError: If ``tensor`` is not a PyTorch tensor.
        ValueError: If ``tensor`` is not shaped ``BxCxHxW``.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"InferenceRunner.predict expects {name} to be a torch.Tensor with "
            f"shape BxCxHxW; got {type(tensor).__name__}."
        )
    if tensor.ndim != 4:
        raise ValueError(
            f"InferenceRunner.predict expects {name} with shape BxCxHxW; "
            f"got shape {tuple(tensor.shape)}."
        )


__all__ = [
    "InferenceBatchResult",
    "InferenceRunner",
]
