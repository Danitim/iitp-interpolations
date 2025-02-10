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
    
        