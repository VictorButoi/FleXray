from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import pytest
import yaml

_TRAINING_ONLY = {
    "kornia",
    "nanodrr",
    "pandas",
    "pydantic",
    "submitit",
    "tabulate",
    "thunderpack",
    "wandb",
    "zstd",
}
_BASE_DEPENDENCIES = {
    "huggingface_hub",
    "numpy",
    "pillow",
    "pyyaml",
    "safetensors",
    "scipy",
    "torch",
    "torchvision",
}
_PYPI_CLASSIFIERS = {
    "Environment :: Console",
    "Intended Audience :: Healthcare Industry",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Image Processing",
    "Topic :: Scientific/Engineering :: Medical Science Apps.",
}
_PYPI_KEYWORDS = {
    "deep-learning",
    "drr",
    "medical-imaging",
    "pytorch",
    "segmentation",
    "x-ray",
}
_PROJECT_URLS = {
    "Homepage": "https://flexray.csail.mit.edu/",
    "Documentation": "https://github.com/VictorButoi/FleXray/tree/main/docs",
    "Repository": "https://github.com/VictorButoi/FleXray",
    "Issues": "https://github.com/VictorButoi/FleXray/issues",
}


def _pyproject() -> dict:
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))


def _release_workflow() -> dict:
    """Load the release workflow without YAML 1.1 boolean coercion."""
    return yaml.load(
        Path(".github/workflows/release.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )


def _requirement_name(requirement: str) -> str:
    return re.split(r"[<>=!~; \\[]", requirement, maxsplit=1)[0].lower()


def _requirement_names(requirements: list[str]) -> set[str]:
    return {_requirement_name(requirement) for requirement in requirements}


def test_base_dependencies_cover_inference_commands() -> None:
    project = _pyproject()["project"]
    base = _requirement_names(project["dependencies"])

    assert _BASE_DEPENDENCIES <= base
    assert base.isdisjoint(_TRAINING_ONLY)


def test_train_and_full_extras_carry_training_dependencies() -> None:
    optional = _pyproject()["project"]["optional-dependencies"]
    train = _requirement_names(optional["train"])
    full = _requirement_names(optional["full"])
    test = _requirement_names(optional["test"])

    assert _TRAINING_ONLY <= train
    assert train <= full
    assert "pytest" not in train
    assert "pytest" in test


def test_mcp_extra_carries_sdk_without_touching_base_dependencies() -> None:
    project = _pyproject()["project"]
    optional = project["optional-dependencies"]

    assert "mcp" in _requirement_names(optional["mcp"])
    assert "mcp" in _requirement_names(optional["full"])
    assert "mcp" in _requirement_names(optional["test"])
    assert "mcp" not in _requirement_names(project["dependencies"])


def test_console_scripts_resolve_to_callable_public_commands() -> None:
    scripts = _pyproject()["project"]["scripts"]
    required = {
        "flexify": "fxr.inference.cli:main",
        "fxr-dataset": "fxr.datasets.cli:main",
        "fxr-train": "fxr.launch.cli:main",
        "fxr-submit": "fxr.launch.submit_cli:main",
        "fxr-mcp": "fxr.mcp.cli:main",
        "fxr-render": "fxr.launch.render_cli:main",
        "fxr-protocol": "fxr.protocols.cli:main",
    }

    assert required.items() <= scripts.items()
    for target in required.values():
        module_name, attribute = target.split(":", maxsplit=1)
        assert callable(getattr(importlib.import_module(module_name), attribute))


def test_pypi_metadata_describes_the_public_project() -> None:
    """PyPI metadata identifies FleXray, its license, and its public links."""
    metadata = _pyproject()
    project = metadata["project"]

    assert any(
        requirement.startswith("setuptools")
        for requirement in metadata["build-system"]["requires"]
    )
    assert project["license"] == "MIT"
    assert "LICENSE" in project["license-files"]
    assert _PYPI_KEYWORDS <= set(project["keywords"])
    assert _PYPI_CLASSIFIERS <= set(project["classifiers"])
    assert _PROJECT_URLS.items() <= project["urls"].items()


def test_pypi_release_workflow_uses_trusted_publishing() -> None:
    """A published release builds, validates, and uploads through OIDC."""
    workflow = _release_workflow()
    build = workflow["jobs"]["build"]
    publish = workflow["jobs"]["publish"]

    assert build["if"] == (
        "github.repository == 'VictorButoi/FleXray' && "
        "github.event.repository.private == false"
    )
    assert workflow["on"]["release"]["types"] == ["published"]
    build_commands = "\n".join(
        step["run"] for step in build["steps"] if "run" in step
    )
    assert "uv build" in build_commands
    assert "twine check dist/*" in build_commands
    assert any(
        step.get("uses") == "actions/upload-artifact@v5"
        for step in build["steps"]
    )

    assert publish["needs"] == ["build"]
    assert publish["environment"] == {
        "name": "pypi",
        "url": "https://pypi.org/p/flexray",
    }
    assert publish["permissions"] == {"id-token": "write"}
    assert [step.get("uses") for step in publish["steps"]] == [
        "actions/download-artifact@v6",
        "pypa/gh-action-pypi-publish@release/v1",
    ]


def test_readme_repository_links_are_absolute() -> None:
    """README links remain usable when its Markdown is rendered by PyPI."""
    readme = Path("README.md").read_text(encoding="utf-8")
    relative_targets = {
        "CITATION.cff",
        "LICENSE",
        "docs/inference.md",
        "docs/training.md",
    }

    for target in relative_targets:
        assert f"]({target})" not in readme
        assert f"https://github.com/VictorButoi/FleXray/blob/main/{target}" in readme

    link_targets = re.findall(r"\]\(([^)\s]+)", readme)
    relative_links = [
        target
        for target in link_targets
        if not target.startswith(("https://", "http://", "mailto:", "#"))
    ]
    assert not relative_links, f"README links must also work on PyPI: {relative_links}"


@pytest.mark.parametrize(
    "module_name", ["fxr.inference", "fxr.mcp", "fxr.models", "fxr.config"]
)
def test_base_public_surface_import_smoke(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert module is not None


def test_inference_runs_without_training_dependencies() -> None:
    """Block training imports in a fresh process and exercise CPU inference."""
    from tests.inference_install_smoke import TRAINING_MODULES

    code = (
        "import runpy, sys\n"
        f"sys.modules.update(dict.fromkeys({TRAINING_MODULES!r}))\n"
        "runpy.run_module('tests.inference_install_smoke', run_name='__main__')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(Path("src").resolve())},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
