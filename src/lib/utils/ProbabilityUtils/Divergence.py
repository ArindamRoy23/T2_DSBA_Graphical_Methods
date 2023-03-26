
import numpy as np 
import cupy as cp
import math

from numba import cuda
from scipy import sparse
from scipy.sparse._coo import coo_matrix

def make_derivative_matrix( 
        width: int, 
        height: int
    ) -> coo_matrix:
    """Creates a matrixform of the first order differencing.
    Args:
        w: width of the image to be differentiated
        h: height of the image to be differentiated
    Returns:
        First order differencing in matrix form.
    """
    def __generate_D_matrix(n):
        e = np.ones([2,n])
        e[1,:] = -e[1,:] 
        return sparse.spdiags(e, [0,1], n, n)

    Dy = sparse.kron(sparse.eye(width), __generate_D_matrix(height)) 
    Dx = sparse.kron(__generate_D_matrix(width),sparse.eye(height)) 

    D = sparse.vstack([Dy, Dx])
    return D ##???

def divergence(
        array: np.ndarray
    ) -> np.ndarray:
    """Creates the divergence of an input array.
    Args: 
        array: input with dimension: [2, class, height, width]
    Returns:
        Divergence [2, class, height, width] of the input array.
    """
    n_classes, height, width = array.shape[1:]
    deriv = make_derivative_matrix(
        width,
        height
    )
    dxy = deriv.T @ np.transpose(array, (0, 3, 2, 1)).reshape(-1, n_classes)
    dxy = np.transpose(dxy.reshape(width, height, n_classes), (2, 1, 0))
    return -dxy

def derivative(
        array: np.ndarray
    ) -> np.ndarray:
    n_channels, height, width = array.shape
    deriv = make_derivative_matrix(
        width, 
        height
    )
    dxy = deriv @ np.transpose(array, (2, 1, 0)).reshape(-1, n_channels)
    dxy = np.transpose(dxy.reshape(2, width, height, n_channels), (0, 3, 2, 1))
    return dxy