"""Offline CT->DRR sample rendering (the ``fxr-render`` data-engine command).

Renders a packed ``ct-seg`` dataset through the same label projection, DRR
runtime, and per-view intensity scaling that training uses, and writes the
results as image/mask arrays plus a ``schema_version: 1`` ``xray-seg`` manifest
that ``fxr-dataset pack`` turns into a trainable X-ray package. This is the raw
DRR path of the data engine (no diffusion refinement).
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from fxr.datasets import (
    build_training_dataset,
    compile_training_label_remap_by_name,
    compile_training_label_remap_from_stored_labels,
)
from fxr.experiment._collation import _ct_safe_collate
from fxr.experiment.batch_inputs import resolve_batch_inputs
from fxr.experiment.drr_forward import DrrForwardPipeline
from fxr.experiment.label_projection import TrainingLabelProjection
from fxr.experiment.segmentation import supervised_foreground_centroids
from fxr.models.camera import scale_rendered_images
from fxr.models.camera.config import resolve_num_views
from fxr.protocols import (
    resolve_run_output_label_names,
    resolve_run_protocol_spec,
)


@dataclass(frozen=True)
class OfflineRenderRequest:
    """Inputs of one offline rendering job.

    Attributes:
        dataset_path: Packed ``ct-seg`` ThunderDB directory.
        dataset_name: Dataset name (also used to look up a packaged dataset
            spec when the package declares no ``stored_labels``).
        config: Training config supplying ``protocol`` and ``drr_model``.
        profile: Key under ``drr_model.datasets`` whose DRR profile to render.
        output: Output directory for arrays and the manifest.
        split: Split of the CT package to render.
        renders: Camera draws per CT volume (each yields ``num_views`` images).
        seed: Seed for Python, NumPy, and Torch random streams.
        device: Torch device used for rendering.
        png: Whether to also write 8-bit PNG previews.
        air_clamp_hu: HU threshold below which voxels are treated as air.
        output_name: Dataset name written to the rendered manifest.
    """

    dataset_path: Path
    dataset_name: str
    config: Mapping[str, Any]
    profile: str
    output: Path
    split: str = "train"
    renders: int = 1
    seed: int = 0
    device: torch.device = torch.device("cpu")
    png: bool = False
    air_clamp_hu: float | None = -900.0
    output_name: str | None = None


@dataclass(frozen=True)
class OfflineRenderReport:
    """Summary of a completed offline rendering job.

    Attributes:
        manifest: Written ``xray-seg`` manifest path.
        num_volumes: CT volumes (records) rendered.
        num_samples: Rendered images written.
        label_names: Ordered mask channel names of the rendered masks.
    """

    manifest: Path
    num_volumes: int
    num_samples: int
    label_names: tuple[str, ...]


def render_dataset(request: OfflineRenderRequest) -> OfflineRenderReport:
    """Render every volume of a packed CT dataset into DRR image/mask samples.

    Args:
        request: Rendering job description.

    Returns:
        Summary with the manifest path to feed into ``fxr-dataset pack``.
    """

    random.seed(request.seed)
    np.random.seed(request.seed)
    torch.manual_seed(request.seed)
    config = _render_config(request)
    dataset = build_training_dataset(
        request.dataset_name,
        request.split,
        "ct",
        cfg={
            "path": str(request.dataset_path),
            "air_clamp_hu": request.air_clamp_hu,
            "compute_fg_centroids": True,
        },
    )
    try:
        projection = _projection(dataset, config, request.dataset_name)
        pipeline = DrrForwardPipeline.from_config(config, device=request.device)
        assert pipeline is not None, "fxr-render requires a drr_model block in the config."
        output_label_names = resolve_run_output_label_names(config)
        assert output_label_names is not None
        writer = _SampleWriter(request.output, png=request.png)
        for index in range(len(dataset)):
            inputs = resolve_batch_inputs(_ct_safe_collate([dataset[index]]), modality="ct")
            native = projection.validated_native_ids(
                inputs.label, source=f"CT dataset {request.dataset_name!r}", require_mapped=True
            )
            record = dataset.records[index]
            for render_index in range(request.renders):
                rendered = pipeline.render(
                    volume=inputs.image.to(request.device),
                    label=native.to(request.device),
                    affine=inputs.affine.to(request.device),
                    dataset_name=request.dataset_name,
                    fg_centroids_ijk=supervised_foreground_centroids(inputs, projection),
                    foreground_collapse_map=projection.foreground_collapse_map(),
                    attenuated_label_ids=projection.native_foreground_label_ids(),
                )
                writer.write(
                    scale_rendered_images(rendered.images).cpu(),
                    rendered.labels.cpu(),
                    request=rendered.request,
                    sample_id=record.data_id,
                    subject_id=record.subject_id,
                    render_index=render_index,
                    split=request.split,
                )
        manifest = writer.write_manifest(
            dataset_name=request.output_name or f"{request.dataset_name}_drr",
            protocol_name=str(config["protocol"]["name"]),
            label_names=output_label_names,
        )
    finally:
        dataset.close()
    return OfflineRenderReport(
        manifest=manifest,
        num_volumes=len(dataset),
        num_samples=len(writer.samples),
        label_names=output_label_names,
    )


def _render_config(request: OfflineRenderRequest) -> dict[str, Any]:
    """Build a one-dataset training config using the requested DRR profile.

    Args:
        request: Rendering job description.

    Returns:
        Config with ``protocol``, ``data.CT.<name>``, ``dataloader``, and a
        ``drr_model`` whose only dataset entry is the selected profile.
    """

    config = dict(request.config)
    assert resolve_run_protocol_spec(config) is not None, "config needs protocol.name."
    drr_model = config.get("drr_model") or {}
    profiles = drr_model.get("datasets") or {}
    assert request.profile in profiles, (
        f"drr_model.datasets has no profile {request.profile!r}; known: {sorted(profiles)}."
    )
    default = drr_model.get("default") or {}
    profile = dict(profiles[request.profile] or {})
    profile["num_views"] = resolve_num_views({**default, **profile})
    return {
        "protocol": config["protocol"],
        "data": {"CT": {request.dataset_name: {}}},
        "dataloader": {"batch_size": 1},
        "drr_model": {
            "default": default,
            "datasets": {request.dataset_name: profile},
        },
    }


def _projection(dataset: Any, config: Mapping[str, Any], dataset_name: str) -> TrainingLabelProjection:
    """Resolve the native-to-protocol projection for a packed CT dataset.

    Packages declaring ``stored_labels`` use those names; otherwise the
    packaged dataset spec for ``dataset_name`` is compiled.

    Args:
        dataset: Built CT training dataset.
        config: Render config with ``protocol.name``.
        dataset_name: Dataset name for spec lookup.

    Returns:
        Label projection for the run protocol.
    """

    protocol_name = str(config["protocol"]["name"])
    config_root = config["protocol"].get("config_root")
    stored = getattr(dataset.backend, "attrs", {}).get("stored_labels")
    if stored is not None:
        remap = compile_training_label_remap_from_stored_labels(
            protocol_name, dataset_name, stored, config_root=config_root
        )
    else:
        remap = compile_training_label_remap_by_name(
            protocol_name, dataset_name, config_root=config_root
        )
    return TrainingLabelProjection.from_label_remap(remap)


class _SampleWriter:
    """Write rendered views as arrays plus a packaging manifest.

    Attributes:
        output: Output directory.
        png: Whether PNG previews are written.
        samples: Manifest sample entries written so far.
    """

    def __init__(self, output: Path, *, png: bool) -> None:
        """Create the output layout.

        Args:
            output: Output directory (created; ``images/`` and ``masks/`` inside).
            png: Whether to write PNG previews under ``previews/``.

        Returns:
            ``None``.
        """

        self.output = Path(output)
        self.png = png
        self.samples: list[dict[str, Any]] = []
        for name in ("images", "masks") + (("previews",) if png else ()):
            (self.output / name).mkdir(parents=True, exist_ok=True)

    def write(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        *,
        request: Any,
        sample_id: str,
        subject_id: str,
        render_index: int,
        split: str,
    ) -> None:
        """Write every view of one render and record its manifest entries.

        Args:
            images: Scaled images shaped ``(V, 1, H, W)``.
            labels: Projected masks shaped ``(V, C, H, W)``.
            request: Sampled camera request (may be ``None``).
            sample_id: Source CT sample id.
            subject_id: Source subject id.
            render_index: Camera draw index for this volume.
            split: Split recorded for every view.

        Returns:
            ``None``.
        """

        for view in range(int(images.shape[0])):
            stem = f"{sample_id}__r{render_index:03d}_v{view:02d}"
            image = images[view].numpy().astype(np.float32)
            mask = labels[view].numpy()
            hard = bool(np.array_equal(mask, np.rint(mask)))
            mask = mask.astype(np.uint8) if hard else mask.astype(np.float16)
            np.save(self.output / "images" / f"{stem}.npy", image)
            np.save(self.output / "masks" / f"{stem}.npy", mask)
            if self.png:
                Image.fromarray((image[0] * 255).round().astype(np.uint8)).save(
                    self.output / "previews" / f"{stem}.png"
                )
            self.samples.append(
                {
                    "sample_id": stem,
                    "subject_id": subject_id,
                    "split": split,
                    "image": f"images/{stem}.npy",
                    "label": f"masks/{stem}.npy",
                    "metadata": {
                        "source_sample_id": sample_id,
                        "render_index": render_index,
                        "view_index": view,
                        **_camera_metadata(request, view),
                    },
                }
            )

    def write_manifest(
        self, *, dataset_name: str, protocol_name: str, label_names: tuple[str, ...]
    ) -> Path:
        """Write the ``xray-seg`` manifest for the rendered samples.

        Args:
            dataset_name: Manifest dataset name.
            protocol_name: Protocol whose label order the masks follow.
            label_names: Ordered mask channel names.

        Returns:
            Manifest path.
        """

        manifest = {
            "schema_version": 1,
            "dataset_name": dataset_name,
            "dataset_type": "xray-seg",
            "protocol_name": protocol_name,
            "label_names": list(label_names),
            "samples": self.samples,
        }
        path = self.output / "manifest.yml"
        path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        return path


def _camera_metadata(request: Any, view: int) -> dict[str, Any]:
    """Extract JSON-compatible camera parameters of one view from a request."""
    if request is None:
        return {}
    intrinsics = dict(request.intrinsics) if isinstance(request.intrinsics, Mapping) else {}
    per_view = {
        key: value[view]
        for key, value in intrinsics.items()
        if isinstance(value, (list, tuple)) and len(value) > view
    }
    return {
        "rot_deg": [float(v) for v in torch.as_tensor(request.rot)[view].reshape(-1).tolist()],
        "xyz_mm": [float(v) for v in torch.as_tensor(request.xyz)[view].reshape(-1).tolist()],
        "orthographic": bool(request.orthographic),
        **{key: float(value) for key, value in per_view.items()},
    }
