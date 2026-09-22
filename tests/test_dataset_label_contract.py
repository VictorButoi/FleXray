from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from fxr.datasets import pack_dataset, validate_dataset_manifest
from fxr.datasets.cli import main as dataset_main


def _dense_manifest(
    root: Path,
    *,
    stored_labels: dict[int, str] | None,
    label: np.ndarray | None = None,
) -> Path:
    """Write a minimal dense X-ray manifest for label-contract tests.

    Args:
        root: Test directory that owns all payloads.
        stored_labels: Optional native id-to-name declaration.
        label: Optional dense label array; defaults to all background.

    Returns:
        Written YAML manifest path.
    """

    np.save(root / "image.npy", np.ones((64, 64), dtype=np.float32))
    np.save(
        root / "label.npy",
        np.zeros((64, 64), dtype=np.uint8) if label is None else label,
    )
    body: dict[str, object] = {
        "schema_version": 1,
        "dataset_name": "DeclaredLabels",
        "dataset_type": "xray-seg",
        "samples": [
            {
                "sample_id": "sample",
                "subject_id": "patient",
                "split": "train",
                "image": "image.npy",
                "label": "label.npy",
            }
        ],
    }
    if stored_labels is not None:
        body["stored_labels"] = stored_labels
    path = root / "dataset.yml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("stored_labels", "match"),
    [
        ({1: "anatomy"}, "contiguous"),
        ({0: "air"}, "id 0"),
        ({0: "background", 1: "background"}, "unique"),
    ],
)
def test_dense_manifest_validates_native_label_declarations(
    tmp_path: Path,
    stored_labels: dict[int, str],
    match: str,
) -> None:
    manifest = _dense_manifest(tmp_path, stored_labels=stored_labels)

    with pytest.raises(ValueError, match=match):
        validate_dataset_manifest(manifest)


def test_dense_manifest_requires_every_observed_id_to_be_declared(
    tmp_path: Path,
) -> None:
    label = np.zeros((64, 64), dtype=np.uint8)
    label[0, 1] = 1
    manifest = _dense_manifest(
        tmp_path,
        stored_labels={0: "background"},
        label=label,
    )

    with pytest.raises(ValueError, match="ids absent from stored_labels"):
        validate_dataset_manifest(manifest)


def test_dense_manifest_requires_stored_labels(tmp_path: Path) -> None:
    manifest = _dense_manifest(tmp_path, stored_labels=None)

    with pytest.raises(ValueError, match="require stored_labels"):
        validate_dataset_manifest(manifest)


def test_channel_manifest_rejects_dense_label_declarations(tmp_path: Path) -> None:
    np.save(tmp_path / "image.npy", np.ones((64, 64), dtype=np.float32))
    np.save(tmp_path / "label.npy", np.zeros((1, 64, 64), dtype=np.uint8))
    path = tmp_path / "dataset.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "dataset_name": "NamedChannels",
                "dataset_type": "xray-seg",
                "label_names": ["lungs"],
                "stored_labels": {0: "background"},
                "samples": [
                    {
                        "sample_id": "sample",
                        "subject_id": "patient",
                        "split": "train",
                        "image": "image.npy",
                        "label": "label.npy",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reject stored_labels"):
        validate_dataset_manifest(path)


def test_dataset_cli_formats_expected_input_failures_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.yml"

    with pytest.raises(SystemExit) as exc_info:
        dataset_main(["validate", str(missing)])

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert str(missing) in stderr
    assert "Traceback" not in stderr


@pytest.mark.parametrize("output", [Path("/"), Path(".")])
def test_pack_rejects_broad_destructive_output_targets(
    tmp_path: Path,
    output: Path,
) -> None:
    manifest = _dense_manifest(
        tmp_path,
        stored_labels={0: "background"},
    )

    with pytest.raises(ValueError, match="dedicated non-root"):
        pack_dataset(manifest, output, overwrite=True)


def test_pack_rejects_symbolic_link_output(tmp_path: Path) -> None:
    manifest_root = tmp_path / "inputs"
    manifest_root.mkdir()
    manifest = _dense_manifest(
        manifest_root,
        stored_labels={0: "background"},
    )
    target = tmp_path / "target"
    target.mkdir()
    output = tmp_path / "output-link"
    output.symlink_to(target, target_is_directory=True)

    with pytest.raises(FileExistsError, match="symbolic link"):
        pack_dataset(manifest, output, overwrite=True)
