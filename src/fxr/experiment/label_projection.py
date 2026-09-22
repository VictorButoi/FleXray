"""Project native integer training labels into model-output channels.

Training datasets emit dataset-native integer label maps. X-ray batches are
projected into channel-first model masks at forward time. CT batches keep dense
native ids through the DRR subject build, then projected foreground masks are
collapsed into the configured model-output channels using the same compiled
``TrainingLabelRemap`` (native id -> model id LUT).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
from torch import Tensor

from fxr.datasets import TrainingLabelRemap


@dataclass(frozen=True)
class TrainingLabelProjection:
    """Map native integer label maps to model-channel binary masks.

    Attributes:
        label_lut: Tuple indexed by native label id holding the model label id,
            or ``-1`` for native ids the dataset spec never declares. Declared
            labels that are dropped for this protocol map to ``0``.
        label_names: Ordered model label names produced by the projection.
        num_classes: Number of model output channels (``len(label_names)``).
    """

    label_lut: tuple[int, ...]
    label_names: tuple[str, ...]
    num_classes: int

    @classmethod
    def from_label_remap(
        cls, label_remap: TrainingLabelRemap
    ) -> "TrainingLabelProjection":
        """Build a projection from a compiled training label remap.

        Args:
            label_remap: Compiled remap carrying a dense native->model LUT.

        Returns:
            A ``TrainingLabelProjection`` wrapping the remap LUT and labels.
        """

        label_lut = tuple(int(value) for value in label_remap.label_lut.tolist())
        return cls(
            label_lut=label_lut,
            label_names=tuple(label_remap.label_names),
            num_classes=int(label_remap.num_classes),
        )

    def project(self, label: Tensor, *, source: str) -> Tensor:
        """Project a native integer label map into model-channel masks.

        Args:
            label: Native integer label map of shape ``Bx1x...``, ``Bx...``, or
                ``...`` (a single map). Values are dataset-native label ids.
            source: Human-readable label source used in error messages.

        Returns:
            Float channel-first mask of shape ``(B, num_classes, *spatial)`` with
            background recomputed as the foreground complement.
        """

        native_ids = self.validated_native_ids(label, source=source)
        lut = torch.as_tensor(self.label_lut, dtype=torch.long, device=label.device)
        model_ids = lut[native_ids]

        target_dtype = label.dtype if torch.is_floating_point(label) else torch.float32
        projected = torch.zeros(
            int(native_ids.shape[0]),
            self.num_classes,
            *tuple(int(dim) for dim in native_ids.shape[1:]),
            dtype=target_dtype,
            device=label.device,
        )
        for model_id in range(1, self.num_classes):
            projected[:, model_id] = (model_ids == model_id).to(dtype=target_dtype)
        return self._with_background(projected)

    def validated_native_ids(
        self,
        label: Tensor,
        *,
        source: str,
        require_mapped: bool = False,
    ) -> Tensor:
        """Return a dense native-id tensor after validating its values.

        Args:
            label: Native integer label map of shape ``Bx1x...``, ``Bx...``, or
                ``...`` (a single map). Values are dataset-native label ids.
            source: Human-readable label source used in error messages.
            require_mapped: Whether every observed native id must have a
                declared LUT entry. Declared drops that map to background remain
                valid.

        Returns:
            Long tensor shaped ``(B, *spatial)`` containing native label ids.

        Raises:
            ValueError: If labels are non-integral, outside the LUT range, or
                unmapped while ``require_mapped`` is ``True``. Unmapped ids
                warn instead when ``require_mapped`` is ``False``.
        """

        native_ids = self._to_native_ids(label)
        native_ids = self._validate_native_ids(native_ids, source=source)
        if require_mapped:
            self._validate_mapped_ids(native_ids, source=source)
        else:
            self._warn_unmapped_ids(native_ids, source=source)
        return native_ids

    def foreground_collapse_map(self) -> tuple[int, ...]:
        """Return native foreground channel targets for DRR mask collapse.

        Returns:
            Tuple whose element ``i`` maps rendered native foreground channel
            ``i`` (native id ``i + 1``) to a model-output channel id. A value of
            ``0`` drops that native foreground into background.
        """

        return tuple(max(0, int(model_id)) for model_id in self.label_lut[1:])

    def native_foreground_label_ids(self) -> tuple[int, ...]:
        """Return native foreground ids that map to model foreground labels.

        Returns:
            Tuple of native label ids greater than zero whose remapped model id
            is also greater than zero.
        """

        return tuple(
            native_id
            for native_id, model_id in enumerate(self.label_lut)
            if native_id > 0 and int(model_id) > 0
        )

    @staticmethod
    def _to_native_ids(label: Tensor) -> Tensor:
        """Normalize a native integer label tensor to shape ``Bx...``.

        Args:
            label: Native dense label tensor with an optional singleton channel.

        Returns:
            Label tensor with a leading batch dimension and no channel dimension.

        Raises:
            ValueError: If ``label`` does not have a supported dense-map shape.
        """
        if label.ndim >= 4 and int(label.shape[1]) == 1:
            return label[:, 0]
        if label.ndim == 3:
            return label
        if label.ndim == 2:
            return label.unsqueeze(0)
        raise ValueError(
            "Expected a native integer label map with shape Bx1x..., Bx..., or "
            f"..., got {tuple(label.shape)}."
        )

    def _validate_native_ids(self, native_ids: Tensor, *, source: str) -> Tensor:
        """Validate native ids are integral and within the known LUT range.

        Args:
            native_ids: Native dense label ids to validate.
            source: Human-readable source name used in validation errors.

        Returns:
            Validated label ids converted to ``torch.long``.

        Raises:
            ValueError: If ids are fractional, negative, or outside the LUT.
        """
        if native_ids.numel() == 0:
            return native_ids.to(dtype=torch.long)

        rounded = native_ids.round()
        if not torch.equal(native_ids.to(rounded.dtype), rounded):
            raise ValueError(f"{source} contains non-integer native label values.")
        native_ids = rounded.to(dtype=torch.long)

        min_id = int(native_ids.min().item())
        max_id = int(native_ids.max().item())
        if min_id < 0:
            raise ValueError(f"{source} contains negative native label id {min_id}.")
        if max_id >= len(self.label_lut):
            raise ValueError(
                f"{source} contains unknown native label id {max_id}; known ids are "
                f"in [0, {len(self.label_lut) - 1}]."
            )
        return native_ids

    def _validate_mapped_ids(self, native_ids: Tensor, *, source: str) -> None:
        """Validate observed native ids have non-negative LUT entries.

        Args:
            native_ids: Previously range-validated native label ids.
            source: Human-readable source name used in validation errors.

        Returns:
            ``None``.

        Raises:
            ValueError: If an observed id has no model-label mapping.
        """
        lut = torch.as_tensor(
            self.label_lut, dtype=torch.long, device=native_ids.device
        )
        mapped = lut[native_ids]
        if torch.any(mapped < 0):
            bad_ids = torch.unique(native_ids[mapped < 0]).tolist()
            raise ValueError(
                f"{source} contains native label id(s) missing from the remap LUT: "
                f"{bad_ids}."
            )

    def _warn_unmapped_ids(self, native_ids: Tensor, *, source: str) -> None:
        """Warn about observed native ids that have no LUT entry.

        Args:
            native_ids: Previously range-validated native label ids.
            source: Human-readable source name used in the warning.

        Returns:
            ``None``.
        """
        lut = torch.as_tensor(
            self.label_lut, dtype=torch.long, device=native_ids.device
        )
        unmapped = lut[native_ids] < 0
        if not bool(torch.any(unmapped)):
            return
        bad_ids = torch.unique(native_ids[unmapped]).tolist()
        warnings.warn(
            f"{source} contains native label id(s) the dataset spec never "
            f"declares: {bad_ids}; they are projected to background.",
            stacklevel=3,
        )

    @staticmethod
    def _with_background(projected: Tensor) -> Tensor:
        """Recompute channel 0 as the complement of the foreground union.

        Args:
            projected: Channel-first projected mask with background at channel 0.

        Returns:
            The input mask after updating its background channel in place.
        """
        return recompute_background_channel(projected)


def recompute_background_channel(projected: Tensor) -> Tensor:
    """Recompute channel 0 of a batch of masks as the foreground complement.

    Args:
        projected: Channel-first mask batch ``(B, C, *spatial)`` with background
            at channel 0.

    Returns:
        The input mask after updating its background channel in place.
    """

    if int(projected.shape[1]) <= 0:
        return projected
    if int(projected.shape[1]) == 1:
        projected[:, 0] = 1.0
        return projected
    foreground_union = projected[:, 1:].amax(dim=1, keepdim=True)
    projected[:, :1] = (foreground_union <= 0).to(dtype=projected.dtype)
    return projected
