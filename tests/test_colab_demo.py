from __future__ import annotations

import ast
import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_NOTEBOOK_PATH = _REPOSITORY_ROOT / "examples" / "flexray_colab_demo.ipynb"
_WEBSITE_ASSET_URL = (
    "https://flexray.csail.mit.edu/assets/hero-mosaic/dark/originals"
)


def _notebook_cells() -> tuple[dict[str, object], ...]:
    """Return every serialized notebook cell."""
    notebook = json.loads(_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return tuple(notebook["cells"])


def _notebook_sources() -> tuple[str, ...]:
    """Return the concatenated source for every notebook cell."""
    return tuple("".join(cell.get("source", ())) for cell in _notebook_cells())


def _cell_source(marker: str) -> str:
    """Return the source of the unique cell containing marker."""
    matches = tuple(source for source in _notebook_sources() if marker in source)
    assert len(matches) == 1, (
        f"Expected one notebook cell containing {marker!r}; found {len(matches)}."
    )
    return matches[0]


def _literal_assignments(source: str, names: set[str]) -> dict[str, object]:
    """Return literal values assigned to the requested top-level names."""
    module = ast.parse(source)
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in names
    }


def test_colab_demo_code_cells_parse() -> None:
    """Keep every ordinary Python cell syntactically valid."""
    for cell in _notebook_cells():
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", ()))
        if source.lstrip().startswith("%"):
            continue
        ast.parse(source)


def test_colab_demo_installs_public_releases_without_github_credentials() -> None:
    """Keep a fresh public runtime independent of private GitHub source access."""
    command = shlex.split(_cell_source("%pip install"))

    assert command[:3] == ["%pip", "install", "-q"]
    requirements = command[3:]
    assert any(requirement.startswith("flexray==") for requirement in requirements)
    assert "nanodrr>=0.1.5" in requirements
    assert all("git+" not in requirement for requirement in requirements)


def test_colab_demo_uses_current_nanodrr_subject_keyword() -> None:
    """Keep the DeepFluoro demo call compatible with nanoDRR 0.1.5."""
    source = _cell_source("download_deepfluoro")

    assert "download_deepfluoro(subject_id=1)" in source
    assert "download_deepfluoro(subject=1)" not in source


def test_colab_demo_reorders_volume_and_affine_axes_together() -> None:
    """Prevent the tensor/affine mismatch that rotated every rendered view."""
    source = _cell_source("img.data.permute")

    assert "img.data.permute(0, 3, 2, 1)" in source
    assert "mask_img.data.permute(0, 3, 2, 1)" in source
    assert "img.affine[:, [2, 1, 0, 3]]" in source
    assert "rot90" not in "\n".join(_notebook_sources())


def test_colab_demo_uses_wide_drr_geometry_and_nonempty_projection_threshold() -> None:
    """Keep the pelvis framed and retain every intersected CT-label ray."""
    comparison = _cell_source("DRR_PIXEL_SPACING =")
    assignments = _literal_assignments(
        comparison,
        {
            "DRR_SDD",
            "DRR_PIXEL_SPACING",
            "DRR_CAMERA_DISPLACEMENT",
            "DRR_SIZE",
        },
    )

    assert assignments == {
        "DRR_SDD": 1200.0,
        "DRR_PIXEL_SPACING": 2.2,
        "DRR_CAMERA_DISPLACEMENT": 900.0,
        "DRR_SIZE": 256,
    }
    assert "seg_threshold=0.0" in comparison

    interactive = _cell_source("DETECTOR_SPACING =")
    assert "DETECTOR_SPACING = 2.2" in interactive
    assert "seg_threshold=0.0" in interactive
    assert "sample_params=random_params" in interactive


def test_colab_demo_uses_published_radiopaedia_assets() -> None:
    """Keep sample URLs aligned with two published, attributed website images."""
    source = _cell_source("SAMPLE_IMAGES =")
    assignments = _literal_assignments(source, {"ASSET_URL", "SAMPLE_IMAGES"})

    assert assignments["ASSET_URL"] == _WEBSITE_ASSET_URL
    assert assignments["SAMPLE_IMAGES"] == {
        "chest": "slide_07.png",
        "pelvis": "slide_22.png",
    }

    attribution = _cell_source("Radiographs courtesy")
    assert "Radiographs courtesy of [Radiopaedia]" in attribution


def test_colab_demo_blends_overlapping_channels_without_a_winner() -> None:
    """Verify two active labels both contribute color at the same pixel."""
    source = _cell_source("def blend_mask_layers")
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "blend_mask_layers"
    )
    namespace = {"np": np}
    compiled = compile(
        ast.Module(body=[function], type_ignores=[]),
        "<cell>",
        "exec",
    )
    exec(compiled, namespace)

    overlay = namespace["blend_mask_layers"](
        np.zeros((1, 1), dtype=np.float32),
        np.ones((2, 1, 1), dtype=np.float32),
        np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
        opacity=1.0,
    )

    np.testing.assert_allclose(overlay[0, 0], (0.5, 0.0, 0.5))
    assert ".argmax(" not in "\n".join(_notebook_sources())


def test_colab_demo_all_label_panel_uses_visible_fills_and_two_pixel_contours() -> None:
    """Keep filled masks, two-pixel contours, and only drawn legend labels."""
    initial_source = _cell_source("def mask_boundary")
    helper_source = _cell_source("def prediction_filled_contour_overlay")
    function_names = {
        "mask_boundary",
        "blend_mask_layers",
        "prediction_filled_contour_overlay",
    }
    functions = [
        node
        for cell_source in (initial_source, helper_source)
        for node in ast.parse(cell_source).body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    namespace = {
        "np": np,
        "binary_dilation": binary_dilation,
        "binary_erosion": binary_erosion,
        "PALETTE": {
            "bone": (255, 0, 0),
            "empty": (0, 0, 255),
        },
    }
    compiled = compile(
        ast.Module(body=functions, type_ignores=[]),
        "<cells>",
        "exec",
    )
    exec(compiled, namespace)

    probabilities = torch.zeros((1, 3, 9, 9), dtype=torch.float32)
    probabilities[0, 1, 2:7, 2:7] = 1.0
    prediction = SimpleNamespace(probabilities=probabilities)
    overlay, visible = namespace["prediction_filled_contour_overlay"](
        np.zeros((9, 9), dtype=np.float32),
        prediction,
        ("background", "bone", "empty"),
    )

    assert visible == ("bone",)
    np.testing.assert_allclose(overlay[4, 4], (0.58, 0.0, 0.0))
    np.testing.assert_allclose(overlay[1, 4], (0.0, 0.0, 0.0))
    np.testing.assert_allclose(overlay[2, 4], (0.979, 0.0, 0.0))
    np.testing.assert_allclose(overlay[3, 4], (0.979, 0.0, 0.0))
    np.testing.assert_allclose(overlay[0, 0], (0.0, 0.0, 0.0))

    assert "fill_opacity=0.58" in helper_source
    assert "boundary_opacity=0.95" in helper_source
    assert "boundary_pixels=2" in helper_source
    assert "boundary_outline_opacity=0.85" in helper_source
    assert "boundary_outline_pixels=0" in helper_source
    assert "fig.legend(" in helper_source
    assert "for name in visible" in helper_source
    assert 'title="Predicted labels"' in helper_source
    assert 'set_title("all labels (filled + 2 px contours)")' in helper_source


def test_colab_demo_has_separate_cached_chest_and_pelvis_triptychs() -> None:
    """Keep one editable individual-label view per bundled radiograph."""
    helper = _cell_source("def show_prediction_triptych")
    assert "plt.subplots(1, 3" in helper
    assert "whole_overlay" in helper
    assert "selected_labels=(individual_label,)" in helper
    assert "probabilities >= threshold" in helper
    assert helper.count("prediction_filled_contour_overlay(") == 3
    assert "fill_opacity=opacity" in helper
    assert 'axes[2].imshow(single_overlay, interpolation="nearest")' in helper
    assert "prediction_overlay(" not in helper

    downloads = _cell_source("sample_predictions =")
    assert "segmenter.predict(str(path))" in downloads

    chest = _cell_source("CHEST_LABEL =")
    assert 'CHEST_LABEL = "lungs"  # @param {type:"string"}' in chest
    assert 'sample_predictions["chest"]' in chest
    assert "segmenter.predict" not in chest

    pelvis = _cell_source("PELVIS_LABEL =")
    assert 'PELVIS_LABEL = "hips"  # @param {type:"string"}' in pelvis
    assert 'sample_predictions["pelvis"]' in pelvis
    assert "segmenter.predict" not in pelvis
