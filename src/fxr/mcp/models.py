"""Static registry of published FleXray pretrained model bundles.

This module intentionally avoids importing `fxr.inference` (and therefore
torch) so registry lookups stay cheap for MCP clients. The default model id
and member subfolders here must match `fxr.inference.DEFAULT_MODEL_ID` and
`fxr.inference.ENSEMBLE_MEMBER_SUBFOLDERS`; a test enforces that.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MODEL_ID = "VictorButoi/flexray"
_LABELS_SUMMARY = (
    "All-structures FleXray v4 label space: background plus "
    "full-body anatomical structures across body regions."
)

ARBITRARY_REPO_NOTE = (
    "Any Hugging Face repo id containing FleXray bundle artifacts "
    "(model.safetensors, config, label_schema.json, preprocessing.json) "
    "is accepted by segment_image and describe_model. A repo whose bundles "
    "live in subfolders declares them in ensemble.json: its flagship loads by "
    "default, subfolder picks one member, and ensemble=true averages them all."
)


@dataclass(frozen=True)
class KnownModel:
    """One published FleXray pretrained model bundle.

    Attributes:
        model_id: Hugging Face repo id holding the bundle.
        subfolder: Bundle directory inside the repo (`""` for the repo root).
        description: One-sentence summary of what the model segments.
        protocol_name: FleXray protocol whose label space the model outputs.
        labels_summary: Short human-readable summary of the output labels.
    """

    model_id: str
    subfolder: str
    description: str
    protocol_name: str
    labels_summary: str


MODEL_REGISTRY: tuple[KnownModel, ...] = (
    KnownModel(
        model_id=DEFAULT_MODEL_ID,
        subfolder="members/flux0375",
        description=(
            "Flagship FleXray full-body X-ray segmentation model (FluXray "
            "proportion 0.375) trained on real X-ray, online CT-to-DRR, and "
            "generated FluXray sources; loaded by default."
        ),
        protocol_name="all_structures_flexray_v4",
        labels_summary=_LABELS_SUMMARY,
    ),
    *(
        KnownModel(
            model_id=DEFAULT_MODEL_ID,
            subfolder=f"members/flux{suffix}",
            description=(
                "FluXray-proportion ablation member trained with generated "
                f"FluXray proportion {proportion}; same architecture, labels, "
                "and preprocessing as the flagship, averaged with it by "
                "ensemble=true."
            ),
            protocol_name="all_structures_flexray_v4",
            labels_summary=_LABELS_SUMMARY,
        )
        for suffix, proportion in (("000", 0.0), ("025", 0.25), ("050", 0.5), ("075", 0.75))
    ),
)


__all__ = [
    "ARBITRARY_REPO_NOTE",
    "DEFAULT_MODEL_ID",
    "KnownModel",
    "MODEL_REGISTRY",
]
