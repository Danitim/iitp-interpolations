import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def create_resized_image(image, x_new, y_new):
    """Initialize an empty resized image with the same type and number of channels."""
    return np.zeros((y_new, x_new, *image.shape[2:]), dtype=image.dtype)

def generate_interpolation_grid(width, height, x_new, y_new):
    """Generate the interpolation grid for bilinear interpolation."""
    x_grid, y_grid = np.meshgrid(
        np.linspace(0, width - 1, x_new), np.linspace(0, height - 1, y_new)
    )
    return x_grid, y_grid

def get_surrounding_pixels(x_grid, y_grid, width, height):
    """Get integer coordinates of surrounding pixels and ensure they stay in bounds."""
    x0, y0 = np.floor(x_grid).astype(int), np.floor(y_grid).astype(int)
    x1, y1 = np.clip(x0 + 1, 0, width - 1), np.clip(y0 + 1, 0, height - 1)
    return x0, y0, x1, y1

def compute_interpolation_weights(x_grid, y_grid, x0, y0):
    """Compute weights for bilinear interpolation."""
    dx, dy = x_grid - x0, y_grid - y0
    return dx, dy

def bilinear_interpolate_channel(channel, x0, y0, x1, y1, dx, dy):
    """Perform bilinear interpolation for a single channel."""
    Ia, Ib, Ic, Id = channel[y0, x0], channel[y1, x0], channel[y0, x1], channel[y1, x1]
    top = Ia + dx * (Ic - Ia)
    bottom = Ib + dx * (Id - Ib)
    return top + dy * (bottom - top)

def bilinear_interpolation(image, x_new, y_new):
    """Perform bilinear interpolation on an image."""
    height, width = image.shape[:2]
    resized_image = create_resized_image(image, x_new, y_new)
    x_grid, y_grid = generate_interpolation_grid(width, height, x_new, y_new)
    x0, y0, x1, y1 = get_surrounding_pixels(x_grid, y_grid, width, height)
    dx, dy = compute_interpolation_weights(x_grid, y_grid, x0, y0)
    
    if image.ndim == 3:
        for c in range(image.shape[2]):
            resized_image[..., c] = bilinear_interpolate_channel(image[..., c], x0, y0, x1, y1, dx, dy)
    else:
        resized_image = bilinear_interpolate_channel(image, x0, y0, x1, y1, dx, dy)
    
    return resized_image.astype(image.dtype)
