import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def polynomial_basis(mesh, degree):
    """
    Constructs a polynomial basis for interpolation in multiple
    dimensions for a given mesh.

    Args:
        mesh (np.ndarray): Mesh.
        degree (int): Degree of the polynomial basis.

    Returns:
        np.ndarray: Polynomial basis.
    """
    # if mesh is one-dimensional, reshape it to two-dimensional
    if mesh.ndim == 1:
        mesh = mesh[:, np.newaxis]

    poly = PolynomialFeatures(degree)
    A = poly.fit_transform(mesh)
    return A


def node_wise_polynomial_interpolation(mesh_old, field_old, mesh_new, degree=1):
    """
    Interpolates a multidimensional field from an old mesh to a
    new mesh using the node-wise polynomial interpolation method.

    Args:
        mesh_old (np.ndarray): Old mesh.
        field_old (np.ndarray): Field on the old mesh.
        mesh_new (np.ndarray): New mesh.

    Returns:
        np.ndarray: Interpolated field.
    """
    A_old = polynomial_basis(mesh_old, degree)
    coeffs, _, _, _ = np.linalg.lstsq(A_old, field_old)

    A_new = polynomial_basis(mesh_new, degree)
    field_new = A_new @ coeffs

    return field_new


def bilinear_interpolation(image, x_new, y_new):
    """
    Perform bilinear interpolation on an image.

    Args:
        image (np.ndarray): Image.
        x_new (int): New width.
        y_new (int): New height.

    Returns:
        np.ndarray: Resized image.
    """
    height, width = image.shape[:2]
    resized_image = np.zeros((y_new, x_new, *image.shape[2:]), dtype=image.dtype)

    x_grid, y_grid = np.meshgrid(
        np.linspace(0, width - 1, x_new), np.linspace(0, height - 1, y_new)
    )
    x0, y0 = np.floor(x_grid).astype(int), np.floor(y_grid).astype(int)
    x1, y1 = np.clip(x0 + 1, 0, width - 1), np.clip(y0 + 1, 0, height - 1)

    dx, dy = x_grid - x0, y_grid - y0

    for c in range(image.shape[2] if image.ndim == 3 else 1):
        channel = image[..., c] if image.ndim == 3 else image
        Ia, Ib, Ic, Id = (
            channel[y0, x0],
            channel[y1, x0],
            channel[y0, x1],
            channel[y1, x1],
        )
        top = Ia + dx * (Ic - Ia)
        bottom = Ib + dx * (Id - Ib)
        resized_image[..., c] = (
            (top + dy * (bottom - top))
            if image.ndim == 3
            else (top + dy * (bottom - top))
        )

    return resized_image.astype(image.dtype)
