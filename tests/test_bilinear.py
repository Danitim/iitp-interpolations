import numpy as np
import pytest

from methods.bilinear import bilinear_interpolation


def test_bilinear_grayscale_upscale() -> None:
    image = np.array([[10, 20], [30, 40]], dtype=np.float32)

    result = bilinear_interpolation(image, 4, 4)

    assert result.shape == (4, 4)
    assert result.dtype == np.uint8
    assert np.isclose(result[0, 0], 10, atol=2)
    assert np.isclose(result[-1, -1], 40, atol=2)


def test_bilinear_rgb_upscale() -> None:
    image = np.array(
        [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]],
        dtype=np.float32,
    )

    result = bilinear_interpolation(image, 4, 4)

    assert result.shape == (4, 4, 3)
    assert result.dtype == np.uint8
    assert np.allclose(result[0, 0], [10, 20, 30], atol=2)
    assert np.allclose(result[-1, -1], [100, 110, 120], atol=2)


def test_bilinear_downscale_grayscale() -> None:
    image = np.array(
        [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
        dtype=np.float32,
    )

    result = bilinear_interpolation(image, 2, 2)

    assert result.shape == (2, 2)
    assert result.dtype == np.uint8


def test_bilinear_identity() -> None:
    image = np.array([[100, 150], [200, 250]], dtype=np.float32)

    bilinear_interpolation(image, 2, 2)
    assert np


def test_bilinear_invalid_ndim_raises_valueerror() -> None:
    image = np.zeros((2, 2, 2, 2), dtype=np.uint8)  # ndim == 4
    with pytest.raises(ValueError, match="Unsupported image dimensions"):
        bilinear_interpolation(image, 4, 4)


@pytest.mark.parametrize("empty_shape", [(0, 10), (10, 0)])
def test_bilinear_empty_image_shape_raises(empty_shape: tuple[int, int]) -> None:
    image = np.zeros(empty_shape, dtype=np.uint8)
    with pytest.raises(ValueError, match="Empty image."):
        bilinear_interpolation(image, 4, 4)
