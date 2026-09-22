"""Generate a tiny synthetic hip X-ray dataset for the custom-dataset walkthrough.

Writes six 64x64 PNG image/mask pairs plus a ``schema_version: 1`` packaging
manifest. Masks use the dataset's *native* ids (see ``CustomHips.yml``):
``1 hip_left``, ``2 hip_right``, ``3 femurs``, ``4 implant``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

SIZE = 64
STORED_LABELS = {0: "background", 1: "hip_left", 2: "hip_right", 3: "femurs", 4: "implant"}


def make_sample(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw one synthetic radiograph and its native-id mask.

    Args:
        rng: Random generator controlling structure placement.

    Returns:
        ``(image, mask)`` as ``uint8`` arrays shaped ``(64, 64)``.
    """

    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    for label, cx in ((1, 20), (2, 44)):
        cy = 22 + int(rng.integers(-3, 4))
        mask[(yy - cy) ** 2 + (xx - cx - int(rng.integers(-2, 3))) ** 2 < 8**2] = label
    mask[34:60, 14:22] = 3
    mask[34:60, 42:50] = 3
    ix, iy = int(rng.integers(26, 34)), int(rng.integers(6, 12))
    mask[iy : iy + 4, ix : ix + 4] = 4
    image = rng.normal(70, 8, size=(SIZE, SIZE))
    image[mask > 0] += 90
    image[mask == 4] = 255
    return np.clip(image, 0, 255).astype(np.uint8), mask


def write_dataset(output: Path, *, num_train: int = 4, num_val: int = 2, seed: int = 0) -> Path:
    """Write images, masks, and the packaging manifest.

    Args:
        output: Output directory (created).
        num_train: Number of training samples.
        num_val: Number of validation samples.
        seed: Generator seed.

    Returns:
        Path of the written ``dataset.yml`` manifest.
    """

    rng = np.random.default_rng(seed)
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "masks").mkdir(parents=True, exist_ok=True)
    samples = []
    for index in range(num_train + num_val):
        image, mask = make_sample(rng)
        stem = f"hips_{index:02d}"
        Image.fromarray(image).save(output / "images" / f"{stem}.png")
        Image.fromarray(mask).save(output / "masks" / f"{stem}.png")
        samples.append(
            {
                "sample_id": stem,
                "subject_id": f"subject_{index:02d}",
                "split": "train" if index < num_train else "val",
                "image": f"images/{stem}.png",
                "label": f"masks/{stem}.png",
            }
        )
    manifest = {
        "schema_version": 1,
        "dataset_name": "CustomHips",
        "dataset_type": "xray-seg",
        "stored_labels": STORED_LABELS,
        "samples": samples,
    }
    path = output / "dataset.yml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Optional arguments excluding the program name.

    Returns:
        Exit status ``0``.
    """

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("work/data"), help="output directory")
    parser.add_argument("--seed", type=int, default=0, help="generator seed")
    args = parser.parse_args(argv)
    manifest = write_dataset(args.output, seed=args.seed)
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
