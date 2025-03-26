import numpy as np


def bilinear_interpolation(
    image: np.ndarray,
    new_height: int,
    new_width: int,
) -> np.ndarray:
    if image.ndim == 2:
        return _bilinear_gray(image, new_height, new_width, 0)
    if image.ndim == 3:
        return np.stack(
            [_bilinear_gray(image[..., c], new_height, new_width, c) for c in range(image.shape[2])],
            axis=-1,
        )
    msg = "Unsupported image dimensions"
    raise ValueError(msg)


def _bilinear_gray(
    image: np.ndarray,
    new_h: int,
    new_w: int,
    channel: int,
) -> np.ndarray:
    h, w = image.shape
    if h == 0 or w == 0:
        msg = "Empty image."
        raise ValueError(msg)

    _ = h / new_h
    _ = w / new_w

    # Сетка новых координат
    x = np.linspace(0, h - 1, new_h)
    y = np.linspace(0, w - 1, new_w)
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")

    x0 = np.floor(x_grid).astype(int)
    y0 = np.floor(y_grid).astype(int)
    x1 = np.clip(x0 + 1, 0, h - 1)
    y1 = np.clip(y0 + 1, 0, w - 1)

    dx = x_grid - x0
    dy = y_grid - y0

    Ia = image[x0, y0]
    Ib = image[x0, y1]
    Ic = image[x1, y0]
    Id = image[x1, y1]

    wa = (1 - dx) * (1 - dy)
    wb = (1 - dx) * dy
    wc = dx * (1 - dy)
    wd = dx * dy

    result = wa * Ia + wb * Ib + wc * Ic + wd * Id
    
    print(f"Bilinear interpolation (channel {channel}) complete!")
    
    return np.clip(result, 0, 255).astype(np.uint8)
