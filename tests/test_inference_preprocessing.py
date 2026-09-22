"""Regression coverage for image I/O at the public inference boundary."""

from pathlib import Path
import warnings

import numpy as np
import pytest
import torch
from PIL import Image, ImageOps

from fxr.inference import ImagePreprocessing


@pytest.mark.parametrize(
    "dtype,offset",
    [
        (np.int32, 2**24),
        (np.uint32, 2**24),
        (np.int64, 2**60),
        (np.int64, -(2**60)),
        (np.uint64, 2**63 + 2**60),
    ],
)
def test_wide_integer_contrast_survives_array_and_tensor_conversion(
    dtype, offset
) -> None:
    """Low-contrast pixels with large offsets scale before float32 conversion."""
    pixels = np.arange(6, dtype=dtype).reshape(2, 3) + dtype(offset)
    preprocessing = ImagePreprocessing(image_size=(2, 3), pad_to_square=False)
    expected = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3) / 5

    torch.testing.assert_close(preprocessing.prepare(pixels), expected)
    torch.testing.assert_close(preprocessing.prepare(torch.from_numpy(pixels)), expected)


def test_wide_integer_scaling_uses_each_batch_member_range() -> None:
    """One batch member's offset and contrast do not affect another member."""
    pixels = torch.arange(6, dtype=torch.int64).reshape(1, 1, 2, 3)
    batch = torch.cat([pixels + 2**60, pixels * 10 - 2**60])
    preprocessing = ImagePreprocessing(image_size=(2, 3), pad_to_square=False)
    expected = pixels.float().expand(2, -1, -1, -1) / 5

    torch.testing.assert_close(preprocessing.prepare(batch), expected)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_wide_integer_full_range_does_not_overflow(dtype) -> None:
    """Scaling supports both small differences and the entire integer range."""
    limits = np.iinfo(dtype)
    pixels = np.array(
        [[limits.min, 0], [limits.max // 2, limits.max]], dtype=dtype
    )
    expected = torch.tensor(
        [
            [
                (int(value) - int(limits.min)) / (int(limits.max) - int(limits.min))
                for value in row
            ]
            for row in pixels
        ],
        dtype=torch.float32,
    )[None, None]
    preprocessing = ImagePreprocessing(image_size=(2, 2), pad_to_square=False)

    torch.testing.assert_close(preprocessing.prepare(pixels), expected)
    torch.testing.assert_close(preprocessing.prepare(torch.from_numpy(pixels)), expected)


@pytest.mark.parametrize("pad_to_square", [False, True])
@pytest.mark.parametrize("size", [(3, 5), (9, 7)])
def test_file_array_and_tensor_share_grayscale_geometry(
    tmp_path: Path, size, pad_to_square
) -> None:
    """Odd padding and both resize directions agree across supported input types."""
    pixels = np.arange(40, dtype=np.uint16).reshape(5, 8) * 1000
    path = tmp_path / "image.tiff"
    Image.fromarray(pixels).save(path)
    preprocessing = ImagePreprocessing(image_size=size, pad_to_square=pad_to_square)
    expected = preprocessing.prepare(path)

    torch.testing.assert_close(preprocessing.prepare(pixels), expected)
    torch.testing.assert_close(
        preprocessing.prepare(torch.from_numpy(pixels)),
        expected,
        atol=2e-7,
        rtol=1e-6,
    )


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.int32, np.float32])
@pytest.mark.parametrize("scale", ["zero_one", "none"])
def test_color_arrays_and_tensors_agree(channels, dtype, scale) -> None:
    """Alpha compositing and luminance use the same order before pixel scaling."""
    pixels = (np.arange(60).reshape(5, 4, 3) * 3 + 10).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        pixels /= 255
    if channels == 4:
        alpha = np.linspace(0, 1, 20).reshape(5, 4, 1)
        if np.issubdtype(dtype, np.integer):
            maximum = np.iinfo(dtype).max if np.dtype(dtype).itemsize <= 2 else 255
            alpha *= maximum
        pixels = np.concatenate([pixels, alpha.astype(dtype)], axis=-1)
    preprocessing = ImagePreprocessing(
        image_size=(5, 4), pad_to_square=False, scale=scale
    )
    expected = preprocessing.prepare(pixels)
    chw = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)

    torch.testing.assert_close(preprocessing.prepare(chw), expected)
    torch.testing.assert_close(preprocessing.prepare(torch.from_numpy(pixels)), expected)


@pytest.mark.parametrize("mode", ["RGBA", "LA", "P"])
def test_pillow_transparency_composites_over_black(tmp_path: Path, mode) -> None:
    """Transparent white pixels become black for direct images and saved files."""
    rgba = np.full((2, 2, 4), 255, dtype=np.uint8)
    rgba[0, 0, 3] = 0
    image = Image.fromarray(rgba).convert(mode)
    path = tmp_path / "transparent.png"
    image.save(path)
    preprocessing = ImagePreprocessing(image_size=(2, 2), pad_to_square=False)
    expected = torch.tensor([[[[0.0, 1.0], [1.0, 1.0]]]])

    torch.testing.assert_close(preprocessing.prepare(image), expected)
    torch.testing.assert_close(preprocessing.prepare(path), expected)


@pytest.mark.parametrize("orientation", range(2, 9))
def test_pillow_exif_orientation_matches_preoriented_pixels(
    tmp_path: Path, orientation
) -> None:
    """Rotations and reflections are applied before computing padding geometry."""
    image = Image.fromarray(np.arange(15, dtype=np.uint8).reshape(3, 5) * 16)
    image.getexif()[274] = orientation
    path = tmp_path / "oriented.png"
    image.save(path, exif=image.getexif())
    preprocessing = ImagePreprocessing(image_size=(8, 8))
    expected = preprocessing.prepare(np.asarray(ImageOps.exif_transpose(image)))

    torch.testing.assert_close(preprocessing.prepare(image), expected)
    torch.testing.assert_close(preprocessing.prepare(path), expected)


@pytest.mark.parametrize("as_tensor", [False, True])
def test_nonfinite_pixels_are_replaced_before_resizing(as_tensor) -> None:
    """A bad source pixel cannot contaminate neighboring pixels during resize."""
    pixels = np.arange(24, dtype=np.float32).reshape(4, 6) / 24
    pixels[0, :3] = [np.nan, np.inf, -np.inf]
    clean = np.nan_to_num(pixels, nan=0, posinf=0, neginf=0)
    preprocessing = ImagePreprocessing(image_size=(7, 9))
    expected = preprocessing.prepare(clean)

    with pytest.warns(RuntimeWarning, match="3 non-finite"):
        actual = preprocessing.prepare(torch.from_numpy(pixels) if as_tensor else pixels)

    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-6)
    assert np.isnan(pixels[0, 0])  # Caller-owned inputs are unchanged.


@pytest.mark.parametrize("as_tensor", [False, True])
def test_constant_image_warns_before_padding_adds_contrast(as_tensor) -> None:
    """Letterboxing cannot conceal a constant input image."""
    pixels = np.ones((2, 7), dtype=np.float32)
    preprocessing = ImagePreprocessing(image_size=(8, 8))

    with pytest.warns(RuntimeWarning, match="no contrast"):
        actual = preprocessing.prepare(torch.from_numpy(pixels) if as_tensor else pixels)

    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("channels", [3, 4])
def test_ambiguous_tensor_layout_has_explicit_alternatives(channels) -> None:
    """Color images and grayscale batches use distinct explicit layouts."""
    pixels = torch.arange(channels * 5 * 6).reshape(channels, 5, 6).float()
    preprocessing = ImagePreprocessing(image_size=(5, 6), pad_to_square=False)

    with pytest.raises(ValueError, match="ambiguous"):
        preprocessing.prepare(pixels)
    assert preprocessing.prepare(pixels.unsqueeze(0)).shape == (1, 1, 5, 6)
    assert preprocessing.prepare(pixels.unsqueeze(1)).shape == (channels, 1, 5, 6)


@pytest.mark.parametrize("shape", [(0, 4), (1, 0, 4, 4), (0, 1, 4, 4)])
def test_empty_images_raise_clear_error(shape) -> None:
    """Empty images fail at the input boundary instead of during a reduction."""
    with pytest.raises(ValueError, match="non-empty"):
        ImagePreprocessing().prepare(torch.empty(shape))


@pytest.mark.parametrize("height,width", [(1, 1000), (1000, 1), (3, 7), (7, 3)])
def test_content_box_is_nonempty_and_within_canvas(height, width) -> None:
    """Even subpixel footprints have a usable region for MCP statistics."""
    preprocessing = ImagePreprocessing(image_size=(4, 6))
    top, left, bottom, right = preprocessing.content_box(height, width)

    assert 0 <= top < bottom <= 4
    assert 0 <= left < right <= 6


@pytest.mark.parametrize("height,width", [(0, 1), (1, 0), (-1, 2)])
def test_content_box_rejects_invalid_dimensions(height, width) -> None:
    """Invalid input dimensions fail before any coordinate arithmetic."""
    with pytest.raises(ValueError, match="positive"):
        ImagePreprocessing().content_box(height, width)


@pytest.mark.parametrize("readonly", [False, True])
def test_reversed_float_array_remains_a_valid_input(readonly) -> None:
    """Negative strides and read-only views do not change preprocessing."""
    pixels = (np.arange(30, dtype=np.float32) / 30).reshape(5, 6)[::-1, ::-1]
    if readonly:
        pixels.setflags(write=False)
    preprocessing = ImagePreprocessing(image_size=(7, 9))
    expected = preprocessing.prepare(np.array(pixels, copy=True))

    with warnings.catch_warnings(record=True) as caught:
        actual = preprocessing.prepare(pixels)

    assert not caught
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("as_tensor", [False, True])
def test_nonfinite_alpha_is_reported_and_becomes_black(as_tensor) -> None:
    """Clamping valid alpha cannot hide NaN or infinite transparency values."""
    pixels = np.full((5, 4, 4), 0.5, dtype=np.float32)
    pixels[..., 3] = 1
    pixels[0, :3, 3] = [np.nan, np.inf, -np.inf]
    expected = torch.full((1, 1, 5, 4), 0.5)
    expected[0, 0, 0, :3] = 0
    preprocessing = ImagePreprocessing(image_size=(5, 4), pad_to_square=False)

    with pytest.warns(RuntimeWarning, match="3 non-finite"):
        actual = preprocessing.prepare(torch.from_numpy(pixels) if as_tensor else pixels)

    torch.testing.assert_close(actual, expected)
