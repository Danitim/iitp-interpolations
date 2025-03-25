import numpy as np
import pytest

from methods.lanczos import lanczos_interpolation


def test_lanczos_grayscale_shape_and_values() -> None:
    # Исходное grayscale изображение 2x2
    image = np.array([[10, 20], [30, 40]], dtype=np.float32)

    new_h, new_w = 4, 4
    result = lanczos_interpolation(image, new_h, new_w, a=2)

    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 4)
    assert result.dtype == np.uint8

    assert np.isclose(result[0, 0], 10, atol=2)
    assert np.isclose(result[-1, -1], 40, atol=2)


def test_lanczos_rgb_shape_and_channels() -> None:
    # RGB изображение 2x2x3
    image = np.array(
        [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]],
        dtype=np.float32,
    )

    new_h, new_w = 4, 4
    result = lanczos_interpolation(image, new_h, new_w, a=2)

    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 4, 3)
    assert result.dtype == np.uint8

    assert np.allclose(result[0, 0], [10, 20, 30], atol=2)
    assert np.allclose(result[-1, -1], [100, 110, 120], atol=2)


@pytest.mark.parametrize(
    "input_shape",
    [
        (2,),
        (2, 2, 2, 2),
    ],
)
def test_lanczos_invalid_input_shape_raises(input_shape: tuple[int, ...]) -> None:
    image = np.zeros(input_shape, dtype=np.float32)

    with pytest.raises(ValueError):
        lanczos_interpolation(image, 4, 4)


def test_lanczos_same_size_returns_similar_image() -> None:
    image = np.array([[100, 150], [200, 250]], dtype=np.float32)

    result = lanczos_interpolation(image, 2, 2, a=3)
    assert result.shape == (2, 2)
    assert np.allclose(result, image, atol=5)
