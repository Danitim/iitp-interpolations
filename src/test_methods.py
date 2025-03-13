import numpy as np
import pytest

from methods import (
    create_resized_image,
    generate_interpolation_grid,
    get_surrounding_pixels,
    compute_interpolation_weights,
    bilinear_interpolate_channel,
    bilinear_interpolation
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
    channel = np.array(
        [[0, 10], [20, 30]], dtype=np.float32
    )
    x_grid, y_grid = generate_interpolation_grid(2, 2, 3, 3)
    x0, y0, x1, y1 = get_surrounding_pixels(x_grid, y_grid, 2, 2)
    dx, dy = compute_interpolation_weights(x_grid, y_grid, x0, y0)
    result = bilinear_interpolate_channel(channel, x0, y0, x1, y1, dx, dy)
    assert result.shape == (3, 3)

def test_bilinear_interpolation():
    image = np.array(
        [[[0, 0, 0], [10, 10, 10]], [[20, 20, 20], [30, 30, 30]]], dtype=np.uint8
    )
    resized = bilinear_interpolation(image, 4, 4)
    assert resized.shape == (4, 4, 3)
    assert resized.dtype == np.uint8

if __name__ == "__main__":
    pytest.main()