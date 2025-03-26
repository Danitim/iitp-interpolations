import numpy as np
import pytest

from methods.spline import cubic_kernel, spline_interpolation


def test_cubic_kernel_at_zero() -> None:
    assert np.isclose(cubic_kernel(0.0), 1.0)


def test_cubic_kernel_at_one() -> None:
    result = cubic_kernel(1.0)
    assert 0.0 < result < 1.0


def test_cubic_kernel_at_two() -> None:
    assert np.isclose(cubic_kernel(2.0), 0.0)


def test_cubic_kernel_out_of_range() -> None:
    for x in [2.1, 3, 10, 100]:
        assert np.isclose(cubic_kernel(x), 0.0)
        assert np.isclose(cubic_kernel(-x), 0.0)


def test_cubic_kernel_symmetry() -> None:
    for x in np.linspace(0, 2, num=50):
        assert np.isclose(cubic_kernel(x), cubic_kernel(-x), atol=1e-6)


def test_spline_grayscale_upscale() -> None:
    image = np.array([[50, 100], [150, 200]], dtype=np.float32)

    result = spline_interpolation(image, 4, 4)

    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 4)
    assert result.dtype == np.uint8

    assert np.isclose(result[0, 0], 50, atol=10)
    assert np.isclose(result[-1, -1], 200, atol=10)


def test_spline_rgb_upscale() -> None:
    image = np.array(
        [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]],
        dtype=np.float32,
    )

    result = spline_interpolation(image, 4, 4)

    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 4, 3)
    assert result.dtype == np.uint8

    assert np.allclose(result[0, 0], [10, 20, 30], atol=10)
    assert np.allclose(result[-1, -1], [100, 110, 120], atol=10)


def test_spline_grayscale_same_size() -> None:
    image = np.array([[10, 20], [30, 40]], dtype=np.float32)

    result = spline_interpolation(image, 2, 2)

    assert result.shape == (2, 2)
    assert np.allclose(result, image, atol=10)


def test_spline_rgb_same_size() -> None:
    image = np.array(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        dtype=np.float32,
    )

    result = spline_interpolation(image, 2, 2)

    assert result.shape == (2, 2, 3)
    assert np.allclose(result, image, atol=10)


@pytest.mark.parametrize(
    "invalid_shape",
    [
        (2,),  # 1D
        (2, 2, 2, 2),  # 4D
        (0, 0),  # empty image
    ],
)
def test_spline_invalid_shape_raises(invalid_shape: tuple[int, ...]) -> None:
    image = np.zeros(invalid_shape, dtype=np.float32)
    with pytest.raises(ValueError):
        spline_interpolation(image, 4, 4)
