from numba import cuda
import numpy as np
import cupy as cp
import math 

@cuda.jit(device = True)
def init_diagonal_matrix(
        matrix: np.ndarray, 
        value: float
    ) -> None:
    i, j = cuda.grid(2)
    # Set the diagonal elements to ones
    if i == j:
        matrix[i, j] = value

@cuda.jit(device = True)
def get_determinant_diagonal_matrix(
        value: float,
        n_dim: int
    ) -> float:
    determinant = 1.0
    for dim in range(n_dim):
        determinant *= value.item()
    return determinant

@cuda.jit(device = True)
def dot_product_diagonal_matrix(
        vector: np.ndarray,
        value: float,
        output_array: np.ndarray 
    ) -> None:
    n_dimensions = vector.shape[0]
    for dimension in range(n_dimensions):
        output_array[dimension] = vector[dimension] * value

@cuda.jit(device = True)
def vector_dot_product(
        vector1: np.ndarray, 
        vector2: np.ndarray
    ) -> float:
    n_dimensions = vector1.shape[0]
    product = 0.0
    for dimension in range(n_dimensions):
        product += vector1[dimension] * vector2[dimension]
    return product

@cuda.jit(device = True)
def find_scribble_point_with_minimum_distance(
        x_coord: int, 
        y_coord: int, 
        scribble_coordinates: cp.ndarray
    ) -> float:
    l2_distance = lambda x1, x2, y1, y2: ((x1 - x2)**2 + (y1 - y2)**2)**(1/2) 
    min_distance = math.inf
    n_scribble_pixels = scribble_coordinates.shape[0] # flat vector, only one element
    x_coord = np.int64(x_coord)
    y_coord = np.int64(y_coord)
    for idx in range(n_scribble_pixels):
        x_coord_scribble, y_coord_scribble = scribble_coordinates[idx]
        # l2 distance
        x_coord_scribble = np.int64(x_coord_scribble)
        y_coord_scribble = np.int64(y_coord_scribble)
        distance = l2_distance(
            x_coord, 
            x_coord_scribble, 
            y_coord, 
            y_coord_scribble
        )
        if distance < min_distance:
            min_distance = distance
    return min_distance

@cuda.jit(device = True)
def find_scribble_pixel_color_intensity_values(
        image_array: np.ndarray, 
        scribble_coordinates: np.ndarray,
        output_array: np.ndarray
    ) -> None:
    n_channels = image_array.shape[0]
    n_scribble_pixels = scribble_coordinates.shape[0] # flat vector, only one element !!!! NO LONGER THE CASE
    target_shape = (n_scribble_pixels, n_channels)
    for idx in range(0, n_scribble_pixels): # Change back to range(0, n_scribble_pixels - 1, 2) in case of flat vector
        x_coord,  y_coord = scribble_coordinates[idx]
        for channel in range(n_channels):
            output_array[idx, channel] = image_array[channel, x_coord, y_coord]

@cuda.jit(device = True)
def multivariate_gaussian_kernel(
        x: np.ndarray, 
        mu: np.ndarray, 
        sigma: np.ndarray,
        #covariance_matrix: np.ndarray,  
        #inv_covariance: np.ndarray,
        exponent_offset: np.ndarray, 
        exponent: np.ndarray, 
        idx: int, 
        output_array: np.ndarray, 
    ) -> None:
    n_dimensions = x.shape[0] # either 2 for spatial kernels or 3 for chromo ones
    #init_diagonal_matrix(covariance_matrix, sigma)
    #init_diagonal_matrix(inv_covariance, 1 / sigma)
    det_covariance = get_determinant_diagonal_matrix(sigma, n_dimensions) ## DEFINE
    for dimension in range(n_dimensions):
        exponent_offset[dimension] = x[dimension] - mu[dimension]
    dot_product_diagonal_matrix(exponent_offset, 1.0 / sigma.item(),  exponent)
    exponent_ = vector_dot_product(exponent, exponent_offset)
    exponent_ = -0.5 * exponent_
    norm_denominator = math.sqrt(det_covariance) * (2 * math.pi)**(n_dimensions / 2)
    norm = 1 / norm_denominator
    output_array[idx] = norm * math.exp(exponent_)

@cuda.jit(device = True)
def pixel_multivariate_gaussian_kernel(
        x: np.ndarray, 
        scribble_coordinates: np.ndarray, 
        sigma: float,
        kernel_argument: np.ndarray,
        #covariance_matrix: np.ndarray, 
        #inv_covariance: np.ndarray,
        exponent_offset: np.ndarray,
        exponent: np.ndarray,   
        output_array: np.ndarray
    ) -> None:
    n_scribble_points = scribble_coordinates.shape[0]
    #x = x.astype(np.float32)
    dimensions = x.shape[0]
    for idx in range(n_scribble_points):
        x_scribble = scribble_coordinates[idx]
        for dimension in range(dimensions):
            kernel_argument[dimension] = x[dimension] - x_scribble[dimension]
        multivariate_gaussian_kernel(
            kernel_argument, 
            x, 
            sigma,
            #covariance_matrix, 
            #inv_covariance, 
            exponent_offset, 
            exponent,
            idx, 
            output_array
        )