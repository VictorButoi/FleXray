"""Camera pose sampling utilities for DRR rendering."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

__all__ = ["PoseSampler", "RandomPoseGenerator"]


class PoseSampler(nn.Module):
    """Sample fixed, cycled, or random camera poses for nanoDRR rendering.

    Returned rotations are ZXY Euler angles in degrees. Returned translations are
    camera positions in world units relative to the selected isocenter.

    Attributes:
        FIXED_PRESETS: Built-in named rotation presets in ZXY Euler degrees.
        preset: Requested fixed preset, ``"random"``, ``"fixed"``, or ordered
            multi-preset schedule.
        camera_displacement: Default source displacement used for fixed poses.
        fixed_params: Explicit ``rot`` and ``xyz`` values for ``preset="fixed"``.
        sample_params: Axis ranges used by random pose sampling.
        seed: Optional random seed for reproducible preset choices and random
            pose generation.
        _preset_idx: Current position in a cyclic multi-preset schedule.
        _preset_generator: Optional seeded generator for multi-preset choices.
        _is_multi_preset: Whether ``preset`` defines a multi-preset schedule.
        multi_preset_slots: Normalized multi-preset schedule entries.
        random_pose_generator: Random pose generator created for
            ``preset="random"``.
    """

    FIXED_PRESETS: dict[str, tuple[float, float, float]] = {
        "frontal": (0.0, 0.0, 0.0),
        "lateral": (90.0, 0.0, 0.0),
        "lateral_left": (-90.0, 0.0, 0.0),
        "offangle": (45.0, 0.0, 0.0),
        "above": (0.0, 45.0, 0.0),
    }

    def __init__(
        self,
        preset: str | list[Any] | tuple[Any, ...],
        camera_displacement: float = 1200.0,
        *,
        fixed_params: Mapping[str, Any] | None = None,
        sample_params: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize fixed, multi-preset, or random pose sampling state.

        Args:
            preset: Fixed preset name, ``"fixed"``, ``"random"``, or a sequence
                of preset names and nested choice lists.
            camera_displacement: Default y translation for fixed named presets.
            fixed_params: Optional ``rot`` and ``xyz`` values for
                ``preset="fixed"``.
            sample_params: Random axis ranges used when ``preset="random"``.
            seed: Optional seed for deterministic random choices.

        Returns:
            ``None``.

        Raises:
            ValueError: If ``preset`` is unsupported or random/fixed parameters
                are invalid.
            TypeError: If multi-preset entries have unsupported types.
        """

        super().__init__()
        if preset == "optax":
            raise ValueError("preset='optax' is not part of the FleXray DRR API.")

        self.preset = preset
        self.camera_displacement = float(camera_displacement)
        self.fixed_params = dict(fixed_params or {})
        self.sample_params = dict(sample_params or {})
        self.seed = seed
        self._preset_idx = 0
        self._preset_generator: torch.Generator | None = None
        self._is_multi_preset = isinstance(preset, (list, tuple))
        self.multi_preset_slots: list[tuple[str, ...]] = []

        if self._is_multi_preset and seed is not None:
            self._preset_generator = torch.Generator()
            self._preset_generator.manual_seed(seed)

        self._build_pose_state()

    def _build_pose_state(self) -> None:
        """Create buffers or helper objects needed by the configured preset.

        Returns:
            ``None``. The sampler is modified in place.

        Raises:
            ValueError: If a multi-preset schedule is empty or a preset is
                unknown.
        """

        if self._is_multi_preset:
            self.multi_preset_slots = [
                self._normalize_multi_preset_slot(slot) for slot in self.preset
            ]
            if not self.multi_preset_slots:
                raise ValueError("Multi-preset pose lists cannot be empty.")
            return

        if self.preset == "random":
            self.random_pose_generator = RandomPoseGenerator(
                self.sample_params,
                camera_displacement=self.camera_displacement,
                seed=self.seed,
            )
            return

        rot, xyz = self._resolve_fixed_pose(str(self.preset))
        self.register_buffer("fixed_rot", rot)
        self.register_buffer("fixed_xyz", xyz)

    def _resolve_fixed_pose(self, preset_name: str) -> tuple[Tensor, Tensor]:
        """Resolve one fixed preset into rotation and translation tensors.

        Args:
            preset_name: Built-in preset name or ``"fixed"`` for explicit
                ``fixed_params``.

        Returns:
            Pair of ``(rot, xyz)`` tensors, each with shape ``(1, 3)``.

        Raises:
            ValueError: If the preset name or explicit pose values are invalid.
        """

        if preset_name == "fixed":
            rot = _triple(self.fixed_params.get("rot", (0.0, 0.0, 0.0)), "rot")
            xyz = _triple(
                self.fixed_params.get(
                    "xyz", (0.0, self.camera_displacement, 0.0)
                ),
                "xyz",
            )
        else:
            if preset_name not in self.FIXED_PRESETS:
                valid = sorted((*self.FIXED_PRESETS, "fixed", "random"))
                raise ValueError(
                    f"Unknown DRR pose preset {preset_name!r}. Expected one of {valid}."
                )
            rot = self.FIXED_PRESETS[preset_name]
            xyz = (0.0, self.camera_displacement, 0.0)

        return (
            torch.tensor(rot, dtype=torch.float32).reshape(1, 3),
            torch.tensor(xyz, dtype=torch.float32).reshape(1, 3),
        )

    def _normalize_multi_preset_slot(self, slot: Any) -> tuple[str, ...]:
        """Normalize one multi-preset schedule entry.

        Args:
            slot: Preset name or non-empty sequence of preset-name choices.

        Returns:
            Tuple of valid preset names for that schedule slot.

        Raises:
            TypeError: If ``slot`` or its choices are not preset names.
            ValueError: If a choice list is empty or contains an unknown preset.
        """

        if isinstance(slot, str):
            self._resolve_fixed_pose(slot)
            return (slot,)
        if isinstance(slot, (list, tuple)):
            if not slot:
                raise ValueError("Nested multi-preset choices cannot be empty.")
            normalized: list[str] = []
            for choice in slot:
                if not isinstance(choice, str):
                    raise TypeError(
                        "Nested multi-preset choices must contain preset names."
                    )
                self._resolve_fixed_pose(choice)
                normalized.append(choice)
            return tuple(normalized)
        raise TypeError("Multi-preset entries must be preset names or choice lists.")

    def _sample_multi_preset_name(self, choices: tuple[str, ...]) -> str:
        """Choose one preset name from a normalized multi-preset slot.

        Args:
            choices: Non-empty tuple of preset names for a schedule slot.

        Returns:
            Selected preset name, randomly chosen when more than one choice is
            available.
        """

        if len(choices) == 1:
            return choices[0]
        idx = int(
            torch.randint(
                len(choices),
                (1,),
                generator=self._preset_generator,
            ).item()
        )
        return choices[idx]

    def _sample_multi_preset_poses(self, num_poses: int) -> tuple[Tensor, Tensor]:
        """Sample poses from the cyclic multi-preset schedule.

        Args:
            num_poses: Number of poses to sample.

        Returns:
            Pair of rotation and translation tensors with shape
            ``(num_poses, 3)``.
        """

        rots: list[Tensor] = []
        xyzs: list[Tensor] = []
        for offset in range(num_poses):
            slot_idx = (self._preset_idx + offset) % len(self.multi_preset_slots)
            preset_name = self._sample_multi_preset_name(
                self.multi_preset_slots[slot_idx]
            )
            rot, xyz = self._resolve_fixed_pose(preset_name)
            rots.append(rot[0])
            xyzs.append(xyz[0])
        self._preset_idx += num_poses
        return torch.stack(rots), torch.stack(xyzs)

    @staticmethod
    def _expand_pose_batch(rot: Tensor, xyz: Tensor, num_poses: int) -> tuple[Tensor, Tensor]:
        """Repeat a single fixed pose to a requested batch size.

        Args:
            rot: Rotation tensor with shape ``(1, 3)``.
            xyz: Translation tensor with shape ``(1, 3)``.
            num_poses: Number of repeated poses to return.

        Returns:
            Pair of rotation and translation tensors with shape
            ``(num_poses, 3)``.
        """

        if num_poses == 1:
            return rot, xyz
        return rot.repeat(num_poses, 1), xyz.repeat(num_poses, 1)

    def sample(
        self,
        num_poses: int = 1,
        sampled_sdd: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Sample rotation and translation tensors.

        Args:
            num_poses: Number of poses to sample.
            sampled_sdd: Optional source-to-detector distances used by random
                sampling to keep y translation no larger than SDD.

        Returns:
            Pair of rotation and translation tensors with shape
            ``(num_poses, 3)``.

        Raises:
            ValueError: If ``num_poses`` is less than one or random sampling
                receives incompatible SDD values.
        """

        if num_poses < 1:
            raise ValueError(f"num_poses must be >= 1. Got {num_poses}.")
        if self._is_multi_preset:
            return self._sample_multi_preset_poses(num_poses)
        if self.preset == "random":
            return self.random_pose_generator.generate_poses(
                num_poses,
                sampled_sdd=sampled_sdd,
            )
        return self._expand_pose_batch(self.fixed_rot, self.fixed_xyz, num_poses)

    def forward(
        self,
        num_poses: int = 1,
        sampled_sdd: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Delegate module calls to ``sample``.

        Args:
            num_poses: Number of poses to sample.
            sampled_sdd: Optional source-to-detector distances for random
                sampling constraints.

        Returns:
            Pair of rotation and translation tensors with shape
            ``(num_poses, 3)``.
        """

        return self.sample(num_poses=num_poses, sampled_sdd=sampled_sdd)


class RandomPoseGenerator:
    """Sample random DRR rotations and translations from explicit axis ranges.

    Attributes:
        camera_displacement: Default fixed-pose displacement retained for API
            compatibility with pose samplers.
        rot_range: Mapping from rotation axis name to inclusive low/high bounds.
        xyz_range: Mapping from translation axis name to inclusive low/high
            bounds.
        _generator: Optional seeded generator used for random pose samples.
    """

    def __init__(
        self,
        sample_params: Mapping[str, Any],
        camera_displacement: float = 1200.0,
        *,
        seed: int | None = None,
    ) -> None:
        """Initialize random pose ranges and optional generator state.

        Args:
            sample_params: Mapping containing ``rot_range`` and ``xyz_range``
                axis-bound mappings.
            camera_displacement: Default displacement retained for sampler API
                compatibility.
            seed: Optional torch generator seed.

        Returns:
            ``None``.

        Raises:
            ValueError: If required axis ranges are missing or invalid.
        """

        self.camera_displacement = float(camera_displacement)
        self._generator: torch.Generator | None = None
        if seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(seed)

        self.rot_range = self._require_axis_ranges(
            sample_params=sample_params,
            range_name="rot_range",
            axes=("alpha", "beta", "gamma"),
        )
        self.xyz_range = self._require_axis_ranges(
            sample_params=sample_params,
            range_name="xyz_range",
            axes=("x", "y", "z"),
        )

    @staticmethod
    def _require_axis_ranges(
        *,
        sample_params: Mapping[str, Any],
        range_name: str,
        axes: tuple[str, ...],
    ) -> dict[str, tuple[float, float]]:
        """Validate and normalize one named axis-range mapping.

        Args:
            sample_params: Random pose configuration mapping.
            range_name: Name of the range mapping to read.
            axes: Required axis names within the range mapping.

        Returns:
            Mapping from axis name to ``(low, high)`` float bounds.

        Raises:
            ValueError: If the mapping or any required axis is missing or has
                invalid bounds.
        """

        ranges = sample_params.get(range_name)
        if not isinstance(ranges, Mapping):
            raise ValueError(
                f"preset='random' requires {range_name} to be a mapping "
                f"with keys {axes}."
            )

        missing = [axis for axis in axes if axis not in ranges]
        if missing:
            raise ValueError(
                f"preset='random' requires explicit {range_name} entries "
                f"for {missing}."
            )
        return {
            axis: _range_pair(ranges[axis], f"{range_name}.{axis}") for axis in axes
        }

    def _rand(self, *shape: int) -> Tensor:
        """Draw uniform random values from the optional local generator.

        Args:
            *shape: Output tensor shape.

        Returns:
            Tensor of random values in ``[0, 1)`` with the requested shape.
        """

        return torch.rand(*shape, generator=self._generator)

    def generate_poses(
        self,
        n: int = 1,
        sampled_sdd: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Generate random pose batches from configured axis ranges.

        Args:
            n: Number of poses to generate.
            sampled_sdd: Optional source-to-detector distances. When provided,
                sampled y translation and SDD are sorted per pose so translation
                is no larger than SDD.

        Returns:
            Pair of rotation and translation tensors with shape ``(n, 3)``.

        Raises:
            ValueError: If ``n`` is less than one or ``sampled_sdd`` does not
                contain one value per pose.
        """

        if n < 1:
            raise ValueError(f"n must be >= 1. Got {n}.")

        rot_lows = torch.tensor(
            [self.rot_range[axis][0] for axis in ("alpha", "beta", "gamma")],
            dtype=torch.float32,
        )
        rot_highs = torch.tensor(
            [self.rot_range[axis][1] for axis in ("alpha", "beta", "gamma")],
            dtype=torch.float32,
        )
        rot = rot_lows + (rot_highs - rot_lows) * self._rand(n, 3)

        xyz_lows = torch.tensor(
            [self.xyz_range[axis][0] for axis in ("x", "y", "z")],
            dtype=torch.float32,
        )
        xyz_highs = torch.tensor(
            [self.xyz_range[axis][1] for axis in ("x", "y", "z")],
            dtype=torch.float32,
        )
        xyz = xyz_lows + (xyz_highs - xyz_lows) * self._rand(n, 3)

        if sampled_sdd is not None:
            sdd = torch.as_tensor(sampled_sdd, dtype=torch.float32).reshape(-1)
            if sdd.numel() != n:
                raise ValueError(
                    "sampled_sdd must have one value per sampled pose; got "
                    f"{sdd.numel()} values for n={n}."
                )
            sdd = sdd.to(device=xyz.device)
            sorted_y_sdd = torch.sort(torch.stack((xyz[:, 1], sdd), dim=1), dim=1)
            xyz[:, 1] = sorted_y_sdd.values[:, 0]
            sorted_sdd = sorted_y_sdd.values[:, 1]
            if isinstance(sampled_sdd, Tensor):
                with torch.no_grad():
                    sampled_sdd.reshape(-1).copy_(
                        sorted_sdd.to(
                            device=sampled_sdd.device,
                            dtype=sampled_sdd.dtype,
                        )
                    )

        return rot, xyz

    def generate_pose(self) -> tuple[Tensor, Tensor]:
        """Generate one random pose.

        Returns:
            Pair of rotation and translation tensors, each with shape ``(1, 3)``.
        """

        return self.generate_poses(1)


def _triple(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be a 3-value list or tuple.")
    triple = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in triple):
        raise ValueError(f"{name} values must be finite.")
    return triple


def _range_pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a 2-value list or tuple.")
    low = float(value[0])
    high = float(value[1])
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError(f"{name} bounds must be finite.")
    if high < low:
        raise ValueError(f"{name} has invalid bounds [{low}, {high}].")
    return low, high
