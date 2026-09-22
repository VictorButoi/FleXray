from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn

from ._config import build_loss_from_config
from .combo import CombinedLoss
from .segmentation import PixelCELoss


class DatasetRoutedLoss(nn.Module):
    """Route loss computation to a profile selected by ``dataset_name``.

    Attributes:
        _profiles: Named loss profile modules built from config bodies.
        _dataset_to_profile: Mapping from dataset name to profile name.
        _last_loss_breakdown: Detached breakdown from the most recent forward
            call, including zero values for inactive profiles.
        _dataset_supervise_empty_label_ids: Per-dataset model channel ids that
            stay supervised as known negatives (see
            :meth:`configure_supervise_empty_label_ids`).
        fn_module_dict: Public view of the named profile modules.
        dataset_supervise_empty_label_ids: Public copy of the configured
            per-dataset known-negative channel ids.
    """

    def __init__(
        self,
        losses: Mapping[str, Mapping[str, Any]],
        dataset_losses: Mapping[str, str],
    ) -> None:
        """Initialize routed profiles and dataset-to-profile mapping.

        Args:
            losses: Mapping from profile name to loss config body.
            dataset_losses: Mapping from dataset name to profile name.

        Returns:
            ``None``.

        Raises:
            ValueError: If mappings are empty or reference unknown profiles.
            TypeError: If a profile body is not a mapping.
        """

        super().__init__()

        if not isinstance(losses, Mapping) or not losses:
            raise ValueError(
                "DatasetRoutedLoss requires a non-empty `losses` mapping of "
                "profile_name -> profile_body."
            )
        if not isinstance(dataset_losses, Mapping) or not dataset_losses:
            raise ValueError(
                "DatasetRoutedLoss requires a non-empty `dataset_losses` mapping "
                "of dataset_name -> profile_name."
            )

        self._profiles = nn.ModuleDict(
            {name: self._build_profile(name, body) for name, body in losses.items()}
        )
        unknown_targets = {
            dataset_name: profile_name
            for dataset_name, profile_name in dataset_losses.items()
            if profile_name not in self._profiles
        }
        if unknown_targets:
            raise ValueError(
                f"dataset_losses references unknown profile(s): {unknown_targets}. "
                f"Known profiles: {sorted(self._profiles.keys())}."
            )

        self._dataset_to_profile = dict(dataset_losses)
        self._last_loss_breakdown: dict[str, torch.Tensor] = {}
        self._dataset_supervise_empty_label_ids: dict[str, tuple[int, ...]] = {}

    @staticmethod
    def _build_profile(name: str, body: Mapping[str, Any]) -> nn.Module:
        """Build one loss profile module from a config body.

        Args:
            name: Profile name used in error messages.
            body: Loss profile config mapping.

        Returns:
            Instantiated loss module for the profile.

        Raises:
            TypeError: If ``body`` is not a mapping.
            ValueError: If the body does not declare ``_class`` or
                ``_combo_class``.
        """

        if not isinstance(body, Mapping):
            raise TypeError(
                f"Profile {name!r} body must be a mapping, got {type(body).__name__}."
            )
        body_copy = deepcopy(dict(body))
        if "_combo_class" in body_copy or "_class" in body_copy:
            return build_loss_from_config(body_copy)
        raise ValueError(
            f"Profile {name!r} must define either `_combo_class` or `_class`. "
            f"Got keys: {sorted(body_copy.keys())}."
        )

    @property
    def fn_module_dict(self) -> nn.ModuleDict:
        """Return the routed profile module dictionary.

        Returns:
            ``ModuleDict`` keyed by profile name.
        """

        return self._profiles

    @property
    def dataset_supervise_empty_label_ids(self) -> dict[str, tuple[int, ...]]:
        """Return the configured per-dataset known-negative channel ids.

        Returns:
            Copy of the dataset-name to channel-id mapping.
        """

        return dict(self._dataset_supervise_empty_label_ids)

    def configure_supervise_empty_label_ids(
        self, mapping: Mapping[str, Sequence[int]]
    ) -> None:
        """Register model channels supervised as known negatives per dataset.

        The ids reach every ``PixelCELoss`` component of the routed profile that
        ignores empty labels; those components must run in binary mode.

        Args:
            mapping: Dataset name to model channel ids (empty sequences are
                ignored).

        Returns:
            ``None``.
        """

        normalized: dict[str, tuple[int, ...]] = {}
        for dataset_name, label_ids in mapping.items():
            ids = tuple(int(label_id) for label_id in label_ids)
            if not ids:
                continue
            assert dataset_name in self._dataset_to_profile, (
                f"supervise_empty_labels dataset {dataset_name!r} has no loss route."
            )
            assert len(set(ids)) == len(ids) and min(ids) >= 0, (
                f"supervise_empty label ids for {dataset_name!r} must be unique and >= 0."
            )
            profile_name = self._dataset_to_profile[dataset_name]
            components = self._supervisable_components(self._profiles[profile_name])
            assert components, (
                f"Loss profile {profile_name!r} routed for {dataset_name!r} has no "
                "PixelCELoss with ignore_empty_labels=True to apply supervise_empty_labels."
            )
            normalized[dataset_name] = ids
        self._dataset_supervise_empty_label_ids = normalized

    @staticmethod
    def _supervisable_components(profile: nn.Module) -> dict[str | None, nn.Module]:
        """Find ``PixelCELoss`` components that ignore empty labels.

        Args:
            profile: Bare loss module or ``CombinedLoss`` profile.

        Returns:
            Mapping from component name (``None`` for a bare profile) to module.
        """

        if isinstance(profile, CombinedLoss):
            candidates = dict(profile.fn_module_dict.items())
        else:
            candidates = {None: profile}
        selected: dict[str | None, nn.Module] = {}
        for name, module in candidates.items():
            if not isinstance(module, PixelCELoss) or not module.ignore_empty_labels:
                continue
            assert module.mode == "binary", (
                "PixelCELoss with ignore_empty_labels=True must set mode='binary' "
                "to use supervise_empty_labels."
            )
            selected[name] = module
        return selected

    def _supervise_empty_kwargs(
        self, dataset_key: str, profile: nn.Module, loss_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Attach configured known-negative ids for one routed batch.

        Args:
            dataset_key: Routed dataset name.
            profile: Loss profile module selected for the batch.
            loss_kwargs: Caller-provided keyword arguments.

        Returns:
            ``loss_kwargs`` extended with ``supervise_empty_label_ids`` routing.
        """

        ids = self._dataset_supervise_empty_label_ids.get(dataset_key)
        if not ids:
            return loss_kwargs
        components = self._supervisable_components(profile)
        if None in components:
            return {**loss_kwargs, "supervise_empty_label_ids": list(ids)}
        component_kwargs = {
            name: {"supervise_empty_label_ids": list(ids)} for name in components
        }
        return {**loss_kwargs, "component_kwargs": component_kwargs}

    def validate_routing(self, configured_dataset_names: Iterable[str]) -> None:
        """Validate that configured datasets exactly match routed datasets.

        Args:
            configured_dataset_names: Dataset names present in a training run.

        Returns:
            ``None``.

        Raises:
            ValueError: If any configured dataset is missing a route or any
                route references a dataset absent from the run.
        """

        configured = set(configured_dataset_names)
        missing = sorted(configured - self._dataset_to_profile.keys())
        if missing:
            raise ValueError(
                "DatasetRoutedLoss.dataset_losses is missing entries for "
                f"configured dataset(s): {missing}. "
                f"Routed datasets: {sorted(self._dataset_to_profile.keys())}."
            )

        stray = sorted(self._dataset_to_profile.keys() - configured)
        if stray:
            raise ValueError(
                "DatasetRoutedLoss.dataset_losses references dataset(s) not "
                f"present in the run's configured datasets: {stray}. "
                f"Configured datasets: {sorted(configured)}."
            )

    def forward(
        self,
        outputs: Any,
        targets: Any,
        *,
        dataset_name: str | Sequence[str],
        **loss_kwargs: Any,
    ) -> torch.Tensor:
        """Route one homogeneous batch to its configured loss profile.

        Args:
            outputs: Model outputs passed to the selected profile.
            targets: Training targets passed to the selected profile.
            dataset_name: Dataset name string, or homogeneous sequence of the
                same dataset name for batched samples.
            **loss_kwargs: Additional keyword arguments forwarded to the profile.

        Returns:
            Loss tensor returned by the selected profile.

        Raises:
            KeyError: If no profile is registered for ``dataset_name``.
            TypeError: If ``dataset_name`` has an invalid type.
            ValueError: If a dataset-name sequence is empty or heterogeneous.
        """

        dataset_key = self._normalize_dataset_name(dataset_name)
        if dataset_key not in self._dataset_to_profile:
            raise KeyError(
                f"No loss profile registered for dataset {dataset_key!r}. "
                f"Routed datasets: {sorted(self._dataset_to_profile.keys())}."
            )

        profile_name = self._dataset_to_profile[dataset_key]
        profile = self._profiles[profile_name]
        loss_kwargs = self._supervise_empty_kwargs(dataset_key, profile, loss_kwargs)
        total_loss = profile(outputs, targets, **loss_kwargs)
        self._record_breakdown(profile_name, profile, reference=total_loss)
        return total_loss

    @staticmethod
    def _normalize_dataset_name(dataset_name: str | Sequence[str]) -> str:
        """Normalize a batch dataset-name value to one dataset key.

        Args:
            dataset_name: Dataset name string or homogeneous sequence of dataset
                names.

        Returns:
            Single dataset name used for routing.

        Raises:
            TypeError: If the value is not a string or non-empty string sequence.
            ValueError: If a sequence contains multiple dataset names.
        """

        if dataset_name is None:
            raise ValueError(
                "DatasetRoutedLoss.forward requires `dataset_name`; got None."
            )
        if isinstance(dataset_name, str):
            return dataset_name
        if isinstance(dataset_name, Sequence) and dataset_name:
            first = dataset_name[0]
            if not isinstance(first, str):
                raise TypeError("dataset_name sequences must contain strings.")
            if any(name != first for name in dataset_name):
                raise ValueError(
                    "DatasetRoutedLoss expects a homogeneous batch; "
                    f"got dataset names {list(dataset_name)!r}."
                )
            return first
        raise TypeError(
            "DatasetRoutedLoss.forward requires `dataset_name` as a string or "
            "non-empty homogeneous sequence of strings."
        )

    def _record_breakdown(
        self,
        active_profile: str,
        active_module: nn.Module,
        reference: torch.Tensor,
    ) -> None:
        """Record active and inactive profile loss breakdown values.

        Args:
            active_profile: Name of the profile used for the current batch.
            active_module: Loss module used for the current batch.
            reference: Loss tensor used to shape inactive-profile zero values.

        Returns:
            ``None``. ``_last_loss_breakdown`` is replaced in place.
        """

        def zero_like_reference() -> torch.Tensor | float:
            return (
                torch.zeros_like(reference)
                if isinstance(reference, torch.Tensor)
                else 0.0
            )

        breakdown: dict[str, torch.Tensor] = {}
        for profile_name, module in self._profiles.items():
            if profile_name == active_profile:
                active_breakdown = self._extract_active_breakdown(module, reference)
                for component_name, value in active_breakdown.items():
                    breakdown[f"{profile_name}/{component_name}"] = value
            else:
                for component_name in self._profile_component_names(module):
                    breakdown[f"{profile_name}/{component_name}"] = (
                        zero_like_reference()
                    )

        self._last_loss_breakdown = breakdown

    @staticmethod
    def _extract_active_breakdown(
        module: nn.Module, reference: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Read a profile's component breakdown after a forward call.

        Args:
            module: Active loss profile module.
            reference: Fallback loss tensor when the module has no breakdown API.

        Returns:
            Mapping from component name to detached loss tensor.
        """

        get_breakdown = getattr(module, "get_last_loss_breakdown", None)
        if callable(get_breakdown):
            return get_breakdown()
        return {type(module).__name__: reference.detach()}

    @staticmethod
    def _profile_component_names(module: nn.Module) -> list[str]:
        """Return stable component names for a profile module.

        Args:
            module: Loss profile module to inspect.

        Returns:
            Component names for combined losses, otherwise the module class name.
        """

        if isinstance(module, CombinedLoss):
            return list(module.fn_module_dict.keys())
        return [type(module).__name__]

    def get_last_loss_breakdown(self) -> dict[str, torch.Tensor]:
        """Return the routed breakdown from the previous forward call.

        Returns:
            Dictionary keyed by ``profile/component`` with cloned tensor values.
        """

        return {
            name: value.clone() if isinstance(value, torch.Tensor) else value
            for name, value in self._last_loss_breakdown.items()
        }
