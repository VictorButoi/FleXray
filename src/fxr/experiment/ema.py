"""Small, config-driven exponential moving average for local training."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Iterator

import torch


@dataclass(frozen=True)
class EmaPolicy:
    """Validated exponential-moving-average settings for a training run.

    Attributes:
        enabled: Whether model weights are tracked.
        decay: Previous-shadow weight used by each moving-average update.
        start_after_steps: Optimizer steps completed before the first update.
        update_every: Eligible optimizer-step interval between updates.
    """

    enabled: bool
    decay: float = 0.9999
    start_after_steps: int = 0
    update_every: int = 1

    def as_dict(self) -> dict[str, bool | float | int]:
        """Return the policy as checkpoint-safe scalar values.

        Args:
            None.

        Returns:
            Plain policy mapping.
        """

        return {
            "enabled": self.enabled,
            "decay": self.decay,
            "start_after_steps": self.start_after_steps,
            "update_every": self.update_every,
        }


def resolve_ema_policy(config: Any) -> EmaPolicy:
    """Resolve and validate the ``train.ema`` configuration.

    Args:
        config: Full experiment config mapping or Config-like object.

    Returns:
        Validated EMA policy. A missing section disables EMA.

    Raises:
        TypeError: If a section or setting has an invalid type.
        ValueError: If a setting is unknown or outside its valid range.
    """

    config_dict = _plain_mapping(config, name="config")
    model_cfg = _plain_mapping(config_dict.get("model", {}), name="model")
    if "ema" in model_cfg:
        raise ValueError(
            "model.ema is unsupported; configure moving averages under train.ema."
        )

    train_cfg = _plain_mapping(config_dict.get("train", {}), name="train")
    if "ema" not in train_cfg:
        return EmaPolicy(enabled=False)

    ema_cfg = _plain_mapping(train_cfg["ema"], name="train.ema")
    allowed = {"enabled", "decay", "start_after_steps", "update_every"}
    unknown = sorted(set(ema_cfg) - allowed)
    if unknown:
        raise ValueError(f"Unexpected train.ema keys: {unknown}.")

    enabled = ema_cfg.get("enabled", True)
    if not isinstance(enabled, bool):
        raise TypeError("train.ema.enabled must be a bool.")

    decay = ema_cfg.get("decay", 0.9999)
    if isinstance(decay, bool) or not isinstance(decay, Real):
        raise TypeError("train.ema.decay must be a number strictly between 0 and 1.")
    decay = float(decay)
    if not 0.0 < decay < 1.0:
        raise ValueError(f"train.ema.decay must satisfy 0 < decay < 1; got {decay}.")

    start_after_steps = ema_cfg.get("start_after_steps", 0)
    if isinstance(start_after_steps, bool) or not isinstance(
        start_after_steps, Integral
    ):
        raise TypeError("train.ema.start_after_steps must be an integer >= 0.")
    start_after_steps = int(start_after_steps)
    if start_after_steps < 0:
        raise ValueError("train.ema.start_after_steps must be >= 0.")

    update_every = ema_cfg.get("update_every", 1)
    if isinstance(update_every, bool) or not isinstance(update_every, Integral):
        raise TypeError("train.ema.update_every must be an integer >= 1.")
    update_every = int(update_every)
    if update_every < 1:
        raise ValueError("train.ema.update_every must be >= 1.")

    return EmaPolicy(
        enabled=enabled,
        decay=decay,
        start_after_steps=start_after_steps,
        update_every=update_every,
    )


class ExponentialMovingAverage:
    """Track a model state while leaving the original module object in place.

    Attributes:
        model: Raw, uncompiled model whose state is tracked.
        policy: Validated update schedule and decay.
        initialized: Whether the first eligible optimizer step seeded the shadow.
        num_updates: Number of completed shadow initializations/updates.
        _shadow_state: Float32 shadow copy of tracked model state.
        _raw_backup: Raw model state saved during an EMA weight swap.
        _swap_depth: Nesting depth for active EMA weight contexts.
        using_ema_weights: Whether EMA values are installed in ``model``.
    """

    STATE_VERSION = 1

    def __init__(self, model: torch.nn.Module, policy: EmaPolicy) -> None:
        """Allocate shadow state for an optionally enabled EMA policy.

        Args:
            model: Raw model whose parameters and persistent buffers are tracked.
            policy: Validated EMA behavior.

        Returns:
            ``None``.
        """

        self.model = model
        self.policy = policy
        self.initialized = False
        self.num_updates = 0
        self._shadow_state: dict[str, torch.Tensor] = {}
        self._raw_backup: dict[str, torch.Tensor] | None = None
        self._swap_depth = 0
        if policy.enabled:
            self._copy_model_to_shadow()

    @property
    def using_ema_weights(self) -> bool:
        """Return whether EMA values are currently installed in ``model``.

        Args:
            None.

        Returns:
            ``True`` inside an active :meth:`use_ema_weights` context.
        """

        return self._swap_depth > 0

    def update_after_optimizer_step(self, completed_optimizer_steps: int) -> bool:
        """Initialize or update shadows after one completed optimizer step.

        Args:
            completed_optimizer_steps: One-based total optimizer-step count.

        Returns:
            Whether this call initialized or updated the EMA state.
        """

        if not self.policy.enabled or not self._is_update_step(
            completed_optimizer_steps
        ):
            return False
        if not self.initialized:
            self._copy_model_to_shadow()
            self.initialized = True
            self.num_updates += 1
            return True

        current = self.model.state_dict()
        one_minus_decay = 1.0 - self.policy.decay
        with torch.no_grad():
            for name, shadow in self._shadow_state.items():
                value = current[name].detach()
                if shadow.is_floating_point():
                    shadow.mul_(self.policy.decay).add_(
                        value.to(device=shadow.device, dtype=shadow.dtype),
                        alpha=one_minus_decay,
                    )
                else:
                    shadow.copy_(value.to(device=shadow.device))
        self.num_updates += 1
        return True

    def deployment_model_state(self) -> dict[str, torch.Tensor]:
        """Return portable EMA weights, or raw weights before initialization.

        Args:
            None.

        Returns:
            Cloned full model state suitable for deployment in ``state['model']``.
        """

        if not self.policy.enabled or not self.initialized:
            return self.raw_model_state()
        return _clone_state(self._shadow_state, floating_dtype=None)

    def raw_model_state(self) -> dict[str, torch.Tensor]:
        """Return raw optimizer-owned model weights even while EMA is active.

        Args:
            None.

        Returns:
            Cloned full raw model state suitable for training resume.
        """

        if self.using_ema_weights:
            if self._raw_backup is None:
                raise RuntimeError("EMA weights are active without a raw backup.")
            return _clone_state(self._raw_backup, floating_dtype=None)
        return _clone_state(self.model.state_dict(), floating_dtype=None)

    def checkpoint_metadata(self) -> dict[str, Any]:
        """Return the scalar EMA metadata stored beside checkpoint weights.

        Args:
            None.

        Returns:
            Versioned metadata including initialization state and policy.
        """

        return {
            "version": self.STATE_VERSION,
            "initialized": self.initialized,
            "num_updates": self.num_updates,
            **{key: value for key, value in self.policy.as_dict().items() if key != "enabled"},
        }

    def load_checkpoint_state(
        self,
        deployment_state: Mapping[str, torch.Tensor],
        metadata: Any,
    ) -> None:
        """Restore shadows and counters from an EMA-aware checkpoint.

        Args:
            deployment_state: Full deployable model state stored in ``model``.
            metadata: Versioned EMA checkpoint metadata.

        Returns:
            ``None``.

        Raises:
            TypeError: If checkpoint metadata has invalid scalar types.
            ValueError: If checkpoint version or policy differs from this run.
        """

        if not self.policy.enabled:
            raise RuntimeError("Cannot restore EMA state while EMA is disabled.")
        state = _plain_mapping(metadata, name="checkpoint _ema_state")
        if state.get("version") != self.STATE_VERSION:
            raise ValueError(
                "Unsupported EMA checkpoint version "
                f"{state.get('version')!r}; expected {self.STATE_VERSION}."
            )
        saved_policy = {
            "decay": state.get("decay"),
            "start_after_steps": state.get("start_after_steps"),
            "update_every": state.get("update_every"),
        }
        current_policy = {
            "decay": self.policy.decay,
            "start_after_steps": self.policy.start_after_steps,
            "update_every": self.policy.update_every,
        }
        if saved_policy != current_policy:
            raise ValueError(
                "EMA policy mismatch while resuming: checkpoint has "
                f"{saved_policy}, current train.ema resolves to {current_policy}."
            )

        initialized = state.get("initialized")
        if not isinstance(initialized, bool):
            raise TypeError("checkpoint _ema_state.initialized must be a bool.")
        num_updates = state.get("num_updates")
        if isinstance(num_updates, bool) or not isinstance(num_updates, Integral):
            raise TypeError("checkpoint _ema_state.num_updates must be an integer >= 0.")
        if int(num_updates) < 0:
            raise ValueError("checkpoint _ema_state.num_updates must be >= 0.")

        self._shadow_state = _clone_state_like(
            deployment_state,
            self.model.state_dict(),
            floating_dtype=torch.float32,
        )
        self.initialized = initialized
        self.num_updates = int(num_updates)

    def seed_from_model(self, completed_optimizer_steps: int) -> None:
        """Seed EMA when resuming a legacy checkpoint without EMA metadata.

        Args:
            completed_optimizer_steps: Restored optimizer-step count.

        Returns:
            ``None``.
        """

        if not self.policy.enabled:
            return
        self._copy_model_to_shadow()
        self.num_updates = 0
        self.initialized = completed_optimizer_steps >= self.policy.start_after_steps

    @contextmanager
    def use_ema_weights(self) -> Iterator[None]:
        """Temporarily install initialized EMA values in the existing model.

        Args:
            None.

        Returns:
            Context manager that restores raw values on exit.
        """

        if not self.policy.enabled or not self.initialized:
            yield
            return
        if self.using_ema_weights:
            self._swap_depth += 1
            try:
                yield
            finally:
                self._swap_depth -= 1
            return

        self._raw_backup = _clone_state(self.model.state_dict(), floating_dtype=None)
        self._swap_depth = 1
        try:
            self.model.load_state_dict(self._shadow_state, strict=True)
            yield
        finally:
            try:
                if self._raw_backup is None:
                    raise RuntimeError("EMA raw backup disappeared during weight swap.")
                self.model.load_state_dict(self._raw_backup, strict=True)
            finally:
                self._raw_backup = None
                self._swap_depth = 0

    def _copy_model_to_shadow(self) -> None:
        """Replace shadow tensors with a float32 copy of current model state.

        Returns:
            ``None``.
        """

        self._shadow_state = _clone_state(
            self.model.state_dict(), floating_dtype=torch.float32
        )

    def _is_update_step(self, completed_optimizer_steps: int) -> bool:
        """Return whether a completed optimizer step is eligible for EMA.

        Args:
            completed_optimizer_steps: One-based completed optimizer-step count.

        Returns:
            Whether the configured EMA schedule updates on this step.
        """

        eligible_step = completed_optimizer_steps - self.policy.start_after_steps
        return eligible_step >= 1 and (eligible_step - 1) % self.policy.update_every == 0


def _clone_state(
    state: Mapping[str, torch.Tensor],
    *,
    floating_dtype: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    """Clone a model state with an optional floating-point shadow dtype.

    Args:
        state: Model-state tensor mapping.
        floating_dtype: Optional dtype for floating tensors; ``None`` preserves it.

    Returns:
        Detached cloned state mapping.
    """

    cloned: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        tensor = value.detach()
        if floating_dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype=floating_dtype)
        cloned[name] = tensor.clone()
    return cloned


def _clone_state_like(
    state: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    *,
    floating_dtype: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    """Clone checkpoint tensors onto their corresponding model-state devices.

    Args:
        state: Checkpoint model-state tensor mapping.
        reference: Live model state whose devices the clones should follow.
        floating_dtype: Optional dtype for floating tensors.

    Returns:
        Detached cloned state mapping colocated with ``reference``.

    Raises:
        KeyError: If checkpoint and live model state keys differ.
    """

    state_keys = set(state)
    reference_keys = set(reference)
    if state_keys != reference_keys:
        missing = sorted(reference_keys - state_keys)
        unexpected = sorted(state_keys - reference_keys)
        raise KeyError(
            "EMA deployment state does not match the live model: "
            f"missing={missing}, unexpected={unexpected}."
        )
    cloned: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        tensor = value.detach()
        if floating_dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(device=reference[name].device, dtype=floating_dtype)
        else:
            tensor = tensor.to(device=reference[name].device)
        cloned[name] = tensor.clone()
    return cloned


def _plain_mapping(value: Any, *, name: str) -> dict[str, Any]:
    """Return a Config-like value as a plain mapping.

    Args:
        value: Mapping or object exposing ``to_dict``.
        name: Config/checkpoint path used in validation errors.

    Returns:
        Plain dictionary copy.

    Raises:
        TypeError: If ``value`` is not mapping-like.
    """

    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}.")
    return dict(value)
