"""
ProbaUtils.py
    Contains some functions needed for computing the Likelihood.
    These functions will be called inside the Likelihood (wrapper) methods
"""

import numpy as np
import cupy as cp


def __find_scribble_point_with_minimum_distance(
        int: x_coord, 
        int: y_coord,
        ndarray: scribble_coordinates
    ) -> float():
    """"
    __find_scribble_point_with_minimum_distance(
            self, 
            x_coord, 
            y_coord, 
            scribble_coordinates
        ):
        Given a pixel's coordinates and the scribble_coordinates array
        finds the l2 distance to the closest scribble point
    """
    l2_distance = lambda x1, x2, y1, y2: ((x1 - x2)**2 + (y1 - y2)**2)**(1/2) 
    min_distance = float("inf")
    n_scribble_pixels = scribble_coordinates.shape[0] # flat vector, only one element
    for idx in range(0, n_scribble_pixels - 1, 2):
        x_coord_scribble = scribble_coordinates[idx]
        y_coord_scribble = scribble_coordinates[idx + 1]
        # l2 distance
        distance = lp_distance(
            x_coord, 
            x_coord_scribble, 
            y_coord, 
            y_coord_scribble
        )
        if distance < min_distance:
            min_distance = distance
    return min_distance

def __multivariate_gaussian_kernel(
        ndarray: x, 
        ndarray: mu, 
        ndarray: sigma, # width of the kernel (Covariance matrix of gaussian)
        bool: on_gpu
    ) -> ndarray:
    """
    __multivariate_gaussian_kernel(x, mu, sigma, on_gpu):
        computes the multivariate gaussian kernel 
        at the given x value, centered in mu and with kernel
        width of sigma. If on_gpu it will be computed using cupy on 
        the Gpu for parallelization
    """
    n_dimensions = x.shape[0] # either 2 for spatial kernels or 3 for chromo ones
    covariance_matrix = cp.identity(n_dimensions) if on_gpu \
        else np.identity(n_dimensions)
    covariance_matrix = sigma * covariance_matrix
    det_covariance = cp.linalg.det(covariance_matrix) if on_gpu \
        else np.linalg.det(covariance_matrix)
    inv_covariance = cp.linalg.inv(covariance_matrix) if on_gpu \
        else np.linalg.inv(covariance_matrix)
    exponent_offset = x - mu
    exponent = cp.dot(exponent_offset.T, inv_covariance) if on_gpu \
        else np.dot(exponent_offset.T, inv_covariance)
    exponent = cp.dot(exponent, exponent_offset) if on_gpu \
        else np.dot(exponent, exponent_offset)
    exponent = -0.5 * exponent
    norm_denominator = cp.sqrt(det_covariance) * (2 * cp.pi)**(n_dimensions / 2) if on_gpu \
        else np.sqrt(det_covariance) * (2 * np.pi)**(n_dimensions / 2)
    norm = 1/norm_denominator
    kernel_val = norm * cp.exp(exponent) if on_gpu \
        else norm * np.exp(exponent) 
    return kernel_val

def __pixelwise_multivariate_gaussian_kernel(
        ndarray: x, # target pixel information: either of shape (n_channels, ) for chromatic k or (2, ) for the spatial one
        ndarray: scribble_coordinates, # coordinates of the scribble points
        bool: on_gpu, # whether to compile the function using cuda
        bool: spatial = True, # If true, computes the spatial kernel else the chromatic one,
        **kwargs 
    ) -> ndarray: # output shape (1, n_scribble_points)
    """
    __pixel_multivariate_gaussian_kernel(
            self, 
            x, 
            scribble_coord inates, 
            pixelwise_kernel_width,
            spatial
            **kwargs
        ):
        Computes the multivariate gaussian kernel for a given class and a given pixel.
        Basically, at the given pixel, computes a kernel for each of the scribble pixels in scribble_coordinates
        The output shape should be of (1, n_scribble_points)
    """
    # if we are computing the spatial kernel x must contain the coordinate informations of the pixel
    # and the args should be empty (since we have already the scribble pixel information in the scribble_coordinates argument)
    ##assert x.shape[0] == 2 and spatial and not args
    # if we are computing the chromatic kernel x must contain the channel informations of the pixel
    # and the args shall be passed as argument (since we need the chromatic information of the scribble pixels)
    ##assert x.shape[0] == 3 and not spatial and args # probably not good, we need to dynamically check the number of channels
    n_scribble_points = scribble_coordinates.shape[0] // 2
    gaussian_kernels = cp.empty(n_scribble_points) if on_gpu \
        else np.empty(n_scribble_points)
    for idx in range(0, scribble_coordinates.shape[0], 2):
        x_coord = scribble_coordinates[idx]
        y_coord = scribble_coordinates[idx + 1]
        # getting either the coordinates or the chromatic values of the scribble pixel
        x_scribble = (x_coord, y_coord) if spatial else args[0][idx // 2, :]
        ## TODO: COMPUTE KERNEL VALUE
        kernel_argument = x - x_scribble
        # get kernel width either from pixelwise 
        kernel_width = kwargs["pixelwise_kernel_width"][x] if spatial \
            else kwargs["sigma"]
        kernel_val = __multivariate_gaussian_kernel(
            kernel_argument, 
            x, 
            kernel_width
        )
        ##
        gaussian_kernels[idx] = kernel_val
    return gaussian_kernels