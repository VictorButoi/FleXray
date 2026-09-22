"""Pretrained-inference tools for the FleXray MCP server.

Every function is a plain JSON-in/JSON-out callable; `fxr.mcp.server` registers
them as MCP tools, so each docstring doubles as the tool description shown to
AI clients. `fxr.inference` (and therefore torch) is imported lazily inside
the functions that need a model, keeping server startup and registry-only
usage fast.

Loaded segmenters run on the device `set_server_device` selected at startup
(CUDA when available) and are cached for the lifetime of the process, keyed by
`(model ids, revision, subfolder, ensemble)`, with no eviction: the server is a local, single-user
stdio process and the cache grows only with the number of distinct models
(or ensembles) requested.
"""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any

from .models import ARBITRARY_REPO_NOTE, DEFAULT_MODEL_ID, MODEL_REGISTRY

_SEGMENTER_CACHE: dict[tuple[tuple[str, ...], str | None, str | None, bool], Any] = {}
_SEGMENTER_CACHE_LOCK = threading.Lock()
_SERVER_DEVICE: str | None = None


def set_server_device(device: str | None) -> None:
    """Choose the torch device every model loaded by this process is placed on.

    Args:
        device: Device name; `"auto"` (the `fxr-mcp` default) and `None` select
            CUDA when available. Set before serving, so it never invalidates a
            cached segmenter.

    Returns:
        None.
    """

    global _SERVER_DEVICE
    _SERVER_DEVICE = device


def list_models() -> dict:
    """List the published FleXray pretrained segmentation models.

    Returns the static registry of known Hugging Face model bundles. This does
    not touch the network or load any model; use `describe_model` for the
    loaded bundle's exact label space and preprocessing.

    Returns:
        Dictionary with `models`, a list of `{model_id, subfolder, description,
        protocol_name, labels_summary}` entries, and `note` explaining that
        arbitrary Hugging Face repo ids are also accepted.
    """

    return {
        "models": [
            {
                "model_id": entry.model_id,
                "subfolder": entry.subfolder,
                "description": entry.description,
                "protocol_name": entry.protocol_name,
                "labels_summary": entry.labels_summary,
            }
            for entry in MODEL_REGISTRY
        ],
        "note": ARBITRARY_REPO_NOTE,
    }


def describe_model(
    model_id: str | list[str] = DEFAULT_MODEL_ID,
    revision: str | None = None,
    subfolder: str | None = None,
    ensemble: bool = False,
) -> dict:
    """Describe a pretrained FleXray model bundle's outputs and preprocessing.

    The first call for a model downloads its artifacts from Hugging Face and
    loads the model; later calls reuse the in-process cache.

    Args:
        model_id: Hugging Face repo id containing FleXray bundle artifacts, or
            a list of repo ids loaded together as an ensemble. A repo with an
            `ensemble.json` loads its flagship bundle by default.
        revision: Optional Hugging Face branch, tag, or commit id.
        subfolder: Optional bundle directory inside a single `model_id`.
        ensemble: Average every member declared by a single `model_id`'s
            `ensemble.json`.

    Returns:
        Dictionary with `model_id`, `revision`, `subfolder`, `ensemble`,
        `labels` (a list of `{label,
        channel}` entries in output-channel order, empty if the bundle declares
        no label schema), `probability_mode`, and `preprocessing` (the bundle's
        image preprocessing metadata).

    Raises:
        FileNotFoundError: If required bundle artifacts are missing.
        TypeError: If bundle metadata has an unsupported type.
        ValueError: If bundle config, labels, or preprocessing are invalid.
    """

    segmenter = _get_segmenter(model_id, revision, subfolder, ensemble)
    label_names = segmenter.label_names or ()
    return {
        "model_id": model_id,
        "revision": revision,
        "subfolder": subfolder,
        "ensemble": ensemble,
        "labels": [
            {"label": label, "channel": channel}
            for channel, label in enumerate(label_names)
        ],
        "probability_mode": segmenter.probability_mode,
        "preprocessing": segmenter.preprocessing.to_metadata(),
    }


def segment_image(
    input_path: str,
    output_dir: str,
    model_id: str | list[str] = DEFAULT_MODEL_ID,
    revision: str | None = None,
    subfolder: str | None = None,
    ensemble: bool = False,
    label: str | None = None,
    threshold: float = 0.5,
    tta_samples: int = 1,
) -> dict:
    """Segment anatomical structures in X-ray image files.

    Accepts one image file or a directory of image files (png/jpg/jpeg/bmp/
    tif/tiff, plus DICOM as dcm/dicom or an extension-less file with a DICM
    header, non-recursive; DICOM needs the `dicom` or `full` extra, which
    `mcp` does not pull in). For each input image, writes three NumPy
    artifacts into `output_dir` — `{stem}_logits.npy`, `{stem}_probabilities.npy`
    (float32, CxHxW), and `{stem}_masks.npy` (uint8, CxHxW) — matching the
    `flexify` CLI output format, and returns per-label summary statistics so
    results are interpretable without reading the arrays. The first call for a
    model downloads its artifacts from Hugging Face and loads the model; later
    calls reuse the in-process cache.

    Args:
        input_path: Image file or directory of image files.
        output_dir: Directory for written artifacts; created if missing.
        model_id: Hugging Face repo id containing FleXray bundle artifacts, or
            a list of repo ids whose predictions are averaged as an ensemble.
            A repo with an `ensemble.json` loads its flagship bundle by default.
        revision: Optional Hugging Face branch, tag, or commit id.
        subfolder: Optional bundle directory inside a single `model_id`.
        ensemble: Average every member declared by a single `model_id`'s
            `ensemble.json` (the published FleXray ensemble).
        label: Optional output label name. `None` keeps every model channel; a
            label name restricts artifacts and statistics to that channel.
        threshold: Probability threshold for the uint8 masks, in `[0, 1]`.
        tta_samples: Total test-time-augmentation forward passes per image.
            Values `<= 1` disable TTA.

    Returns:
        Dictionary with `model_id`, `revision`, `subfolder`, `ensemble`,
        `threshold`, `tta_samples`,
        `image_count`, and `results` — one entry per image with `image_path`,
        `artifacts` (paths keyed `logits`/`probabilities`/`masks`),
        `content_box` (`[top, left, bottom, right]` canvas pixels the input
        occupies after pad-to-square and resize; `None` for DICOM inputs), and
        `labels`, a list of `{label, channel, pixel_count, pixel_fraction,
        mean_probability, max_probability}` computed from the thresholded mask
        and probability channels inside `content_box`, so `pixel_count` is in
        canvas pixels and `pixel_fraction` divides by the rounded content-box
        area. Bilinear interpolation can mix input and padding at the boundary.

    Raises:
        FileNotFoundError: If the input path or required bundle artifacts are
            missing, or a directory contains no supported images.
        TypeError: If bundle metadata has an unsupported type.
        ValueError: If the threshold, label, model metadata, or input paths
            are invalid.
    """

    _validate_threshold(threshold)
    from fxr.inference.cli import collect_input_images

    image_paths = collect_input_images(input_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    segmenter = _get_segmenter(model_id, revision, subfolder, ensemble)

    results: list[dict] = []
    used_stems: set[str] = set()
    for index, image_path in enumerate(image_paths):
        prediction = segmenter.predict(
            image_path,
            label=label,
            threshold=threshold,
            tta_samples=tta_samples,
        )
        stem = _unique_stem(image_path, index=index, used=used_stems)
        artifacts = _write_prediction_artifacts(
            output_dir=output_path,
            stem=stem,
            prediction=prediction,
        )
        content_box = _content_box(segmenter, image_path)
        results.append(
            {
                "image_path": str(image_path),
                "artifacts": artifacts,
                "content_box": content_box,
                "labels": _label_statistics(
                    prediction=prediction,
                    label_names=segmenter.label_names,
                    label=label,
                    content_box=content_box,
                ),
            }
        )
    return {
        "model_id": model_id,
        "revision": revision,
        "subfolder": subfolder,
        "ensemble": ensemble,
        "threshold": threshold,
        "tta_samples": tta_samples,
        "image_count": len(results),
        "results": results,
    }


def _get_segmenter(
    model_id: str | list[str],
    revision: str | None,
    subfolder: str | None,
    ensemble: bool,
) -> Any:
    """Return a cached `FleXraySegmenter`, loading it on first request.

    Args:
        model_id: Hugging Face repo id, or list of ensemble member repo ids.
        revision: Optional Hugging Face branch, tag, or commit id.
        subfolder: Optional bundle directory inside a single `model_id`.
        ensemble: Whether to load every `ensemble.json` member.

    Returns:
        The cached segmenter for `(model ids, revision, subfolder, ensemble)`,
        loaded on the device `set_server_device` selected.
    """

    model_ids = (model_id,) if isinstance(model_id, str) else tuple(model_id)
    key = (model_ids, revision, subfolder, ensemble)
    with _SEGMENTER_CACHE_LOCK:
        segmenter = _SEGMENTER_CACHE.get(key)
        if segmenter is None:
            from fxr.inference import FleXraySegmenter

            segmenter = FleXraySegmenter.from_pretrained(
                repo_id=model_id,
                revision=revision,
                subfolder=subfolder,
                ensemble=ensemble,
                device=_SERVER_DEVICE,
            )
            _SEGMENTER_CACHE[key] = segmenter
    return segmenter


def _clear_segmenter_cache() -> None:
    """Drop every cached segmenter. Intended for tests.

    Returns:
        None.
    """

    with _SEGMENTER_CACHE_LOCK:
        _SEGMENTER_CACHE.clear()


def _validate_threshold(threshold: float) -> None:
    """Validate a probability threshold.

    Args:
        threshold: Candidate threshold.

    Returns:
        None.

    Raises:
        ValueError: If the threshold is not finite or outside `[0, 1]`.
    """

    value = float(threshold)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(
            f"threshold must be a finite value in [0, 1], got {threshold!r}."
        )


def _unique_stem(image_path: Path, *, index: int, used: set[str]) -> str:
    """Return a deterministic artifact stem that avoids collisions.

    Args:
        image_path: Source image path.
        index: Zero-based input position.
        used: Mutable set of stems already returned.

    Returns:
        Unique filename stem for prediction artifacts.
    """

    base = image_path.stem or f"image_{index:04d}"
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _write_prediction_artifacts(
    *,
    output_dir: Path,
    stem: str,
    prediction: Any,
) -> dict[str, str]:
    """Write logits, probabilities, and masks for one prediction.

    Args:
        output_dir: Destination directory.
        stem: Filename stem used for all artifacts.
        prediction: `FleXrayPrediction` with batched `1xCxHxW` tensors.

    Returns:
        Mapping from artifact kind to written path as strings.
    """

    import numpy as np

    logits_path = output_dir / f"{stem}_logits.npy"
    probabilities_path = output_dir / f"{stem}_probabilities.npy"
    masks_path = output_dir / f"{stem}_masks.npy"
    np.save(logits_path, prediction.logits[0].detach().float().cpu().numpy())
    np.save(
        probabilities_path,
        prediction.probabilities[0].detach().float().cpu().numpy(),
    )
    np.save(masks_path, prediction.masks[0].detach().cpu().numpy())
    return {
        "logits": str(logits_path),
        "probabilities": str(probabilities_path),
        "masks": str(masks_path),
    }


def _content_box(segmenter: Any, image_path: Path) -> list[int] | None:
    """Return the canvas box one input occupies, or `None` when its size is unknown.

    Args:
        segmenter: Loaded segmenter whose preprocessing defines the canvas.
        image_path: Input image path.

    Returns:
        `[top, left, bottom, right]` canvas pixel bounds, or `None` for DICOM
        inputs, whose statistics then cover the whole canvas.
    """

    from PIL import Image

    from fxr.inference.dicom import is_dicom_path

    if is_dicom_path(image_path):
        return None
    with Image.open(image_path) as image:
        width, height = image.size
        if image.getexif().get(0x0112, 1) in {5, 6, 7, 8}:
            width, height = height, width
    return list(segmenter.preprocessing.content_box(height, width))


def _label_statistics(
    *,
    prediction: Any,
    label_names: tuple[str, ...] | None,
    label: str | None,
    content_box: list[int] | None = None,
) -> list[dict]:
    """Summarize one prediction's mask and probability channels.

    Args:
        prediction: `FleXrayPrediction` with batched `1xCxHxW` tensors.
        label_names: Bundle label names in channel order, or `None`.
        label: Optional label filter already applied to the prediction.
        content_box: Optional `[top, left, bottom, right]` canvas bounds of the
            input; statistics are restricted to it so padding is not counted.

    Returns:
        List of per-channel statistic dictionaries with `label`, `channel`,
        `pixel_count`, `pixel_fraction`, `mean_probability`, and
        `max_probability`.
    """

    masks = prediction.masks[0]
    probabilities = prediction.probabilities[0]
    if content_box is not None:
        top, left, bottom, right = content_box
        masks = masks[:, top:bottom, left:right]
        probabilities = probabilities[:, top:bottom, left:right]
    pixels = masks.shape[-2] * masks.shape[-1]
    if label is not None:
        entries = [(label, list(label_names or ()).index(label))]
    elif label_names:
        entries = list(zip(label_names, range(len(label_names))))
    else:
        entries = [
            (f"channel_{index}", index) for index in range(masks.shape[0])
        ]

    statistics: list[dict] = []
    for tensor_index, (name, channel) in enumerate(entries):
        mask = masks[tensor_index if label is not None else channel]
        probability = probabilities[
            tensor_index if label is not None else channel
        ]
        pixel_count = int(mask.sum().item())
        statistics.append(
            {
                "label": name,
                "channel": channel,
                "pixel_count": pixel_count,
                "pixel_fraction": pixel_count / pixels,
                "mean_probability": float(probability.mean().item()),
                "max_probability": float(probability.max().item()),
            }
        )
    return statistics


__all__ = [
    "describe_model",
    "list_models",
    "segment_image",
    "set_server_device",
]
