"""CT->DRR forward-pass routing for segmentation training.

``DrrForwardPipeline`` owns one :class:`SegmentationDRRRuntime` per configured CT
dataset and renders a CT batch into DRR images and projected labels, selecting
the per-dataset runtime by source name.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from fxr.models.camera import CTDRRRenderResult, SegmentationDRRRuntime


@dataclass
class DrrForwardPipeline:
    """Route CT batches to their per-dataset DRR rendering runtime.

    Attributes:
        runtimes_by_dataset: Mapping from CT dataset name to its render runtime.
    """

    runtimes_by_dataset: dict[str, SegmentationDRRRuntime]

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        device: torch.device,
    ) -> "DrrForwardPipeline | None":
        """Build a pipeline from an experiment config.

        Args:
            config: Full experiment config mapping.
            device: Device used for runtime pose-sampler state.

        Returns:
            A pipeline with one runtime per CT dataset, or ``None`` when no
            ``drr_model`` block is configured.
        """

        runtimes = SegmentationDRRRuntime.build_runtimes_from_config(
            config, device=device
        )
        if not runtimes:
            return None
        return cls(runtimes_by_dataset=dict(runtimes))

    def get_render_size(self) -> tuple[int, int] | None:
        """Return the shared detector ``(height, width)`` across CT runtimes.

        Returns:
            Shared detector size, or ``None`` when the pipeline has no runtimes.
        """
        if not self.runtimes_by_dataset:
            return None
        return next(iter(self.runtimes_by_dataset.values())).get_render_size()

    def render(
        self,
        *,
        volume: Tensor,
        label: Tensor,
        affine: Tensor,
        dataset_name: str | None,
        fg_centroids_ijk: Tensor | None = None,
        render_kwargs: Mapping[str, Any] | None = None,
        foreground_collapse_map: tuple[int, ...] | Tensor | None = None,
        attenuated_label_ids: tuple[int, ...] | Tensor | None = None,
    ) -> CTDRRRenderResult:
        """Render a CT subject through the runtime for ``dataset_name``.

        Args:
            volume: CT volume for a single subject.
            label: Dense integer label aligned with ``volume``.
            affine: Voxel-to-world affine for the subject.
            dataset_name: Source CT dataset name selecting the render runtime.
            fg_centroids_ijk: Per-class voxel centroids for ``random_label``
                isocenter sampling.
            render_kwargs: Optional per-call nanoDRR render overrides.
            foreground_collapse_map: Optional native-foreground to model-channel
                collapse map used after projection.
            attenuated_label_ids: Optional native foreground ids eligible for
                per-label attenuation.

        Returns:
            The rendered DRR images and projected labels.
        """

        runtime = self._select_runtime(dataset_name)
        return runtime.render(
            volume=volume,
            label=label,
            affine=affine,
            fg_centroids_ijk=fg_centroids_ijk,
            render_kwargs=render_kwargs,
            foreground_collapse_map=foreground_collapse_map,
            attenuated_label_ids=attenuated_label_ids,
        )

    def _select_runtime(self, dataset_name: str | None) -> SegmentationDRRRuntime:
        """Select the DRR runtime for a CT dataset name.

        Args:
            dataset_name: Source CT dataset name.

        Returns:
            Runtime configured for ``dataset_name``.

        Raises:
            ValueError: If no dataset name is provided or the name is unknown.
        """
        known = ", ".join(sorted(self.runtimes_by_dataset))
        if dataset_name is None:
            raise ValueError(
                f"CT DRR rendering requires a dataset_name; configured CT datasets: {known}."
            )
        runtime = self.runtimes_by_dataset.get(str(dataset_name))
        if runtime is None:
            raise ValueError(
                f"No CT DRR profile configured for dataset {dataset_name!r}; "
                f"configured CT datasets: {known}."
            )
        return runtime
