import numba.cuda as cuda
import numba as nb
import numpy as np
import cupy as cp
import math

from numba import float64 as float64
from numba import int64 as int64
from typing import Final
from numba.cuda.cudadrv.devicearray import DeviceNDArray

@cuda.jit(device = True)
def __compute_gaussian_kernel(
        x: Final[cuda.local.array], 
        mu: Final[cuda.local.array], 
        sigma: float64, 
        spatial: bool,
        debug: int64
    ) -> float64:
    """
    __compute_gaussian_kernel(
        x: Final[cuda.local.array], -> device array containing the argument of the kernel
        mu: Final[cuda.local.array], -> device array containing the mean of the kernel
        sigma: float64, -> device array containing the width of the kernel
        spatial: bool -> boolean indicating whether to compute the spatial or chromatic kernels
    )-> float64

    Device function used to compute the multivariate gaussian kernel. 
    This function should be used to compute the k(x - x_ij) and k(I - I_ij) in equation (9)
    for a given pixel and a given class.

    Returs: a scalar, representing the value of the kernel computed over the given pixel and a single scribble point
    
    """
    sigma = sigma.item()
    if not sigma:
        if debug == 0 or debug == 1:
            return 1.0
        if debug == 3:
            return 0.0
        if debug == 5:
            n_dim = 2 if spatial else 3
            return (2 * math.pi)**(n_dim / 2)
        if debug == 6:
            return sigma
    kernel_argument = 0.0
    n_dim = 2 if spatial else 3
    for dim in range(n_dim):
        kernel_argument  += ((x[dim] - mu[dim]) / sigma) **2   
    cov_det = sigma **2 if spatial else sigma **3 
    norm_denominator = cov_det * (2 * math.pi)**(n_dim / 2)
    norm = 1 / norm_denominator
    if debug == 0 or debug == 1 or debug == 2:
        return norm * math.exp(-0.5 * kernel_argument) ## This is the correct line, commented out for debugging
    if debug == 3 or debug == 4:
        return -0.5 * kernel_argument
    if debug == 5:
        return norm
    if debug == 6:
        return sigma

@cuda.jit(device = True)
def __pixel_factored_multivariate_gaussian_kernel(
        d_image_array: DeviceNDArray,
        target_pixel_color_intensities: DeviceNDArray, 
        target_pixel_coordinates: DeviceNDArray, 
        d_scribble_coordinates: DeviceNDArray,
        d_scribble_color_intensity_values: DeviceNDArray, 
        spatial_kernel_width: DeviceNDArray,
        d_sigma: DeviceNDArray,
        debug: int64
    ) -> float64:
    """
    __pixel_factored_multivariate_gaussian_kernel(
        d_image_array: DeviceNDArray, -> device array containing the target image
        target_pixel_color_intensities: DeviceNDArray, -> device array containing the target pixel's color intensities
        target_pixel_coordinates: DeviceNDArray, -> device array containing the target pixel's coordinates
        d_scribble_coordinates: DeviceNDArray, -> device array containing the scribble pixels's coordinates
        d_scribble_color_intensity_values: DeviceNDArray, -> device array containing the scribble pixel's color intensities
        spatial_kernel_width: DeviceNDArray, -> device array containing the spatial kernel width
        d_sigma: DeviceNDArray -> device array containing the chromatic kernel width
    ) -> float64

    Device function used to loop over the scribble points and compute the gaussian kernel density estimation

    Returns: a scalar, representing the result of kernel density estimation over a pixel, as of equation (9) of the paper

    """
    total_kernel_value = cuda.local.array(shape = 1, dtype = nb.float64)
    n_scribble_points = d_scribble_coordinates.shape[0]
    for scribble_point in range(n_scribble_points):
        ## NEED TO FIX THIS FOR DYNAMICALLY ALLOCATING THE SPACE
        ## IT WORKS FOR NOW SO LETS KEEP IT LIKE THIS
        scribble_point_color_intensities = cuda.local.array(shape = 3, dtype = nb.float64)
        for channel in range(3):
            scribble_point_color_intensities[channel] = d_scribble_color_intensity_values[scribble_point, channel]
        ## NEED TO FIX THIS FOR DYNAMICALLY ALLOCATING THE SPACE
        ## IT WORKS FOR NOW SO LETS KEEP IT LIKE THIS
        scribble_point_coordinates = cuda.local.array(shape = 2, dtype = nb.int64)
        for dim in range(2):
            scribble_point_coordinates[channel] = d_scribble_coordinates[scribble_point, dim]
        spatial_kernel = __compute_gaussian_kernel(
            target_pixel_coordinates, 
            scribble_point_coordinates,
            spatial_kernel_width,
            True,
            debug
        )
        chromo_kernel = __compute_gaussian_kernel(
            target_pixel_color_intensities, 
            scribble_point_color_intensities,
            d_sigma,
            False,
            debug
        )
        if debug == 0:
            total_kernel_value[0] += (spatial_kernel + 1e-13) * chromo_kernel ## This is the correct line, commented out for debugging
        if debug == 1 or debug == 3 or debug == 5:
            total_kernel_value[0] += spatial_kernel
        if debug == 2 or debug == 4:
            total_kernel_value[0] += chromo_kernel
        if debug == 7:
            total_kernel_value[0] += spatial_kernel_width

    total_kernel_value[0] /= n_scribble_points
    return total_kernel_value.item()

@cuda.jit(device = True)
def find_scribble_point_with_minimum_distance(
        x_coord: Final[int64], 
        y_coord: Final[int64], 
        scribble_coordinates: Final[DeviceNDArray]
    ) -> float64:
    """
    find_scribble_point_with_minimum_distance(
        x_coord: Final[int64], -> integer representing the x_coordinate of the target pixel
        y_coord: Final[int64], -> integer representing the y_cooridnate of the target pixel
        scribble_coordinates: Final[DeviceNDArray] -> device array containing the coordinates of the scribble pixels for a given class
    ) -> float64

    Device function used to retrieve the minimum distance to the closest scribble of a given class 
    for computing equation (14) of the paper.

    Returns: A scalar, representing the distance to the closests class's scribble point 
    
    TESTED AND DOUBLE CHECKED
    """
    l2_distance = lambda x1, x2, y1, y2: math.sqrt(((x1 - x2)**2 + (y1 - y2)**2)) 
    min_distance = math.inf
    n_scribble_pixels = scribble_coordinates.shape[0] 
    x_coord = np.int64(x_coord)
    y_coord = np.int64(y_coord)
    for idx in range(n_scribble_pixels):
        x_coord_scribble = scribble_coordinates[idx, 0]
        y_coord_scribble = scribble_coordinates[idx, 1]
        distance = (x_coord - x_coord_scribble) * (x_coord - x_coord_scribble) + \
            (y_coord - y_coord_scribble) * (y_coord - y_coord_scribble)
        distance = math.sqrt(distance)
        """distance = l2_distance(
            x_coord, 
            x_coord_scribble, 
            y_coord, 
            y_coord_scribble
        )
        """
        if distance < min_distance:
            min_distance = distance
    return min_distance

@cuda.jit(device = True)
def pixel_factored_multivariate_gaussian_kernel(
        d_image_array: DeviceNDArray, 
        x_coord: DeviceNDArray, 
        y_coord: DeviceNDArray,
        d_scribble_coordinates: DeviceNDArray,
        d_scribble_color_intensity_values: DeviceNDArray, 
        spatial_kernel_width: DeviceNDArray,
        d_sigma: DeviceNDArray,
        debug: int64
    ) -> float64:
    """
    pixel_factored_multivariate_gaussian_kernel(
        d_image_array: DeviceNDArray, -> device array containing the target image
        x_coord: DeviceNDArray, -> device array containing the x coordinate of the target pixel
        y_coord: DeviceNDArray, -> device array containing the y cooridnate of the target pixel
        d_scribble_coordinates: DeviceNDArray, -> device array containing the scribble pixels's coordinates
        d_scribble_color_intensity_values: DeviceNDArray, -> device array containing the scribble pixel's color intensities
        spatial_kernel_width: DeviceNDArray, -> device array containing the spatial kernel width
        d_sigma: DeviceNDArray -> device array containing the chromatic kernel width
    ) -> float64

    Device Function that, given the x and y coordinates of the target pixel, encodes them into
    arrays representing the spatial coordinates and the color intensity values for passing them as 
    input of __pixel_factored_multivariate_gaussian_kernel, which will be the function actually computing 
    equation (9) of the paper

    It is invoked by each thread once over each pixel. For the computations the threads will not share any 
    memory in order for them to be performed correctly

    Returns: a scalar, representing the result of kernel density estimation over a pixel, as of equation (9) of the paper
    
    """
    target_pixel_color_intensities = cuda.local.array(shape = 3, dtype = nb.uint8)
    target_pixel_coordinates = cuda.local.array(shape = 2, dtype = nb.uint8)
    for channel in range(3):
        target_pixel_color_intensities[channel] = d_image_array[channel, x_coord, y_coord]
    target_pixel_coordinates[0] = x_coord
    target_pixel_coordinates[1] = y_coord
    return __pixel_factored_multivariate_gaussian_kernel(
        d_image_array,
        target_pixel_color_intensities, 
        target_pixel_coordinates, 
        d_scribble_coordinates,
        d_scribble_color_intensity_values, 
        spatial_kernel_width,
        d_sigma,
        debug
    )
