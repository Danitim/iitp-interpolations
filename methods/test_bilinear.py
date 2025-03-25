import numpy as np
import pytest

from methods.bilinear import (
    create_resized_image,
    generate_interpolation_grid,
    get_surrounding_pixels,
    compute_interpolation_weights,
    bilinear_interpolate_channel,
    bilinear_interpolation,
)


def test_create_resized_image():
    image = np.ones((5, 5, 3), dtype=np.uint8)
    resized = create_resized_image(image, 10, 10)
    assert resized.shape == (10, 10, 3)
    assert np.all(resized == 0)


def test_generate_interpolation_grid():
    x_grid, y_grid = generate_interpolation_grid(5, 5, 3, 3)
    assert x_grid.shape == (3, 3)
    assert y_grid.shape == (3, 3)
    assert x_grid[0, 0] == 0 and x_grid[-1, -1] == 4
    assert y_grid[0, 0] == 0 and y_grid[-1, -1] == 4


def test_get_surrounding_pixels():
    x_grid, y_grid = generate_interpolation_grid(5, 5, 3, 3)
    x0, y0, x1, y1 = get_surrounding_pixels(x_grid, y_grid, 5, 5)
    assert np.all(x0 >= 0) and np.all(x1 < 5)
    assert np.all(y0 >= 0) and np.all(y1 < 5)


def test_compute_interpolation_weights():
    x_grid, y_grid = generate_interpolation_grid(5, 5, 3, 3)
    x0, y0, _, _ = get_surrounding_pixels(x_grid, y_grid, 5, 5)
    dx, dy = compute_interpolation_weights(x_grid, y_grid, x0, y0)
    assert np.all((dx >= 0) & (dx <= 1))
    assert np.all((dy >= 0) & (dy <= 1))


def test_bilinear_interpolate_channel():
    channel = np.array([[0, 10], [20, 30]], dtype=np.float32)
    x_grid, y_grid = generate_interpolation_grid(2, 2, 3, 3)
    x0, y0, x1, y1 = get_surrounding_pixels(x_grid, y_grid, 2, 2)
    dx, dy = compute_interpolation_weights(x_grid, y_grid, x0, y0)
    result = bilinear_interpolate_channel(channel, x0, y0, x1, y1, dx, dy)
    assert result.shape == (3, 3)


def test_bilinear_interpolation_rgb_image():
    # 2x2 RGB изображение (3 канала)
    image = np.array(
        [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]],
        dtype=np.float32,
    )

    x_new, y_new = 4, 4
    result = bilinear_interpolation(image, x_new, y_new)

    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 4, 3)

    assert np.allclose(result[0, 0], [10, 20, 30])
    assert np.allclose(result[0, -1], [40, 50, 60])
    assert np.allclose(result[-1, 0], [70, 80, 90])
    assert np.allclose(result[-1, -1], [100, 110, 120])


def test_bilinear_interpolation_grayscale_image():
    # 2x2 изображение, только один канал (grayscale)
    image = np.array([[10, 20], [30, 40]], dtype=np.float32)

    x_new, y_new = 4, 4
    result = bilinear_interpolation(image, x_new, y_new)

    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 4)

    assert np.isclose(result[0, 0], 10.0)
    assert np.isclose(result[0, -1], 20.0)
    assert np.isclose(result[-1, 0], 30.0)
    assert np.isclose(result[-1, -1], 40.0)


if __name__ == "__main__":
    pytest.main()  # pragma: no cover
