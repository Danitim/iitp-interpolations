"""Lancsoz interpolation method."""

import numpy as np
from tqdm import tqdm


def sinc(x: np.ndarray) -> np.ndarray:
    """Computes the normalized sinc function: sin(pi * x) / (pi * x).

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Array of sinc values for each element in x.
    """
    x = np.where(x == 0, 1e-10, x)
    return np.sin(np.pi * x) / (np.pi * x)


def lanczos_kernel(x: np.ndarray, a: int) -> np.ndarray:
    """Computes the Lanczos windowed sinc kernel.

    Args:
        x (np.ndarray): Distance(s) from the interpolation center.
        a (int): Size of the Lanczos window (typically 2 or 3).

    Returns:
        np.ndarray: Weight values for each element in x.
    """
    return np.where(np.abs(x) < a, sinc(x) * sinc(x / a), 0.0)


def lanczos_interpolation(
    image: np.ndarray,
    new_height: int,
    new_width: int,
    a: int = 3,
) -> np.ndarray:
    """Performs Lanczos interpolation on a grayscale or RGB image.

    Args:
        image (np.ndarray): Input image (2D grayscale or 3D RGB).
        new_height (int): Target height of the output image.
        new_width (int): Target width of the output image.
        a (int, optional): Size of the Lanczos window. Defaults to 3.

    Returns:
        np.ndarray: Interpolated image.

    Raises:
        ValueError: If the input image has unsupported dimensions or invalid size.
    """
    if image.size == 0 or new_height <= 0 or new_width <= 0:
        msg = "Invalid image or output dimensions"
        raise ValueError(msg)

    if image.ndim == 2:
        return _lanczos_gray(image, new_height, new_width, a)
    if image.ndim == 3:
        return np.stack(
            [_lanczos_gray(image[..., c], new_height, new_width, a, channel=c) for c in range(image.shape[2])],
            axis=-1,
        )
    msg = "Unsupported image dimensions"
    raise ValueError(msg)


def _lanczos_gray(
    image: np.ndarray,
    new_h: int,
    new_w: int,
    a: int,
    channel: int | None = None,
) -> np.ndarray:
    """Performs Lanczos interpolation on a single grayscale image channel.

    Args:
        image (np.ndarray): 2D array representing the grayscale image.
        new_h (int): Desired height of the output image.
        new_w (int): Desired width of the output image.
        a (int): Lanczos kernel window size (radius).
        channel (int | None, optional): Channel index for logging. Defaults to None.

    Returns:
        np.ndarray: Interpolated 2D image of shape (new_h, new_w).
    """
    h, w = image.shape
    _ = h / new_h
    _ = w / new_w

    out = np.zeros((new_h, new_w), dtype=np.float32)

    x_coords = np.linspace(0, h - 1, new_h)
    y_coords = np.linspace(0, w - 1, new_w)

    bar_desc = f"Lanczos Interpolation{' (channel ' + str(channel) + ')' if channel is not None else ''}"

    for i, x in enumerate(tqdm(x_coords, desc=bar_desc, unit="line")):
        x_int = int(np.floor(x))
        x_range = np.arange(x_int - a + 1, x_int + a)
        x_range = np.clip(x_range, 0, h - 1)
        wx = lanczos_kernel(x - x_range[:, None], a)

        for j, y in enumerate(y_coords):
            y_int = int(np.floor(y))
            y_range = np.arange(y_int - a + 1, y_int + a)
            y_range = np.clip(y_range, 0, w - 1)
            wy = lanczos_kernel(y - y_range, a)[None, :]

            patch = image[np.ix_(x_range, y_range)]
            weights = wx[: len(x_range)] * wy[:, : len(y_range)]

            norm = np.sum(weights)
            value = np.sum(patch * weights)
            out[i, j] = value / norm if norm != 0 else 0.0

    return np.clip(out, 0, 255).astype(np.uint8)
