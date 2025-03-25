import numpy as np
from tqdm import tqdm


def sinc(x):
    x = np.where(x == 0, 1e-10, x)
    return np.sin(np.pi * x) / (np.pi * x)


def lanczos_kernel(x, a):
    return np.where(np.abs(x) < a, sinc(x) * sinc(x / a), 0)


def lanczos_interpolation(image, new_height, new_width, a=3):
    """
    Optimized Lanczos interpolation for grayscale or RGB image.
    """
    if image.ndim == 2:
        return _lanczos_interpolate_gray_fast(image, new_height, new_width, a)
    elif image.ndim == 3:
        return np.stack(
            [
                _lanczos_interpolate_gray_fast(
                    image[..., c], new_height, new_width, a
                )
                for c in range(image.shape[2])
            ],
            axis=-1,
        )
    else:
        raise ValueError("Unsupported image dimensions")


def _lanczos_interpolate_gray_fast(image, new_h, new_w, a):
    h, w = image.shape
    scale_x = h / new_h
    scale_y = w / new_w

    x_coords = np.linspace(0, h - 1, new_h)
    y_coords = np.linspace(0, w - 1, new_w)

    out = np.zeros((new_h, new_w), dtype=np.float32)

    for i, x in enumerate(tqdm(x_coords, desc="Interpolating", unit="line")):
        x_int = int(np.floor(x))
        dx = np.arange(x_int - a + 1, x_int + a)
        dx = dx[(dx >= 0) & (dx < h)]
        kx = lanczos_kernel(x - dx[:, None], a)

        for j, y in enumerate(y_coords):
            y_int = int(np.floor(y))
            dy = np.arange(y_int - a + 1, y_int + a)
            dy = dy[(dy >= 0) & (dy < w)]
            ky = lanczos_kernel(y - dy, a)

            patch = image[np.ix_(dx, dy)]
            weights = np.outer(
                kx[: len(dx)].flatten(), ky[: len(dy)].flatten()
            )
            norm = np.sum(weights)

            out[i, j] = (patch * weights).sum() / norm if norm != 0 else 0

    return np.clip(out, 0, 255).astype(np.uint8)
