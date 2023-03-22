import numba.cuda as cuda
import numba as nb
import numpy as np
import cupy as cp
import math 
from typing import Final


@cuda.jit(device = True)
def find_scribble_point_with_minimum_distance(
        x_coord: Final[int], 
        y_coord: Final[int], 
        scribble_coordinates: Final[cp.ndarray]
    ) -> float:
    l2_distance = lambda x1, x2, y1, y2: math.sqrt(((x1 - x2)**2 + (y1 - y2)**2)) 
    min_distance = math.inf
    n_scribble_pixels = scribble_coordinates.shape[0] 
    x_coord = np.int64(x_coord)
    y_coord = np.int64(y_coord)
    for idx in range(n_scribble_pixels):
        x_coord_scribble = scribble_coordinates[idx, 0]
        y_coord_scribble = scribble_coordinates[idx, 1]
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
def compute_gaussian_kernel(
        x: Final[cuda.local.array], 
        mu: Final[cuda.local.array], 
        sigma: float, 
        spatial: bool
    ) -> float:
    sigma = sigma.item()
    if not sigma:
        sigma = 1.0
    kernel_argument = 0.0
    n_dim = 2 if spatial else 3
    for dim in range(n_dim):
        kernel_argument  += (x[dim] - mu[dim]) **2 / sigma.item()
    cov_det = sigma.item() **2 if spatial else sigma.item() **3
    norm_denominator = math.sqrt(cov_det) * (2 * math.pi)**(n_dim / 2)
    norm = 1 / norm_denominator
    return norm * math.exp(-0.5 * kernel_argument) 

@cuda.jit(device = True)
def pixel_factored_multivariate_gaussian_kernel_(
        d_image_array,
        target_pixel_color_intensities, 
        target_pixel_coordinates, 
        d_scribble_coordinates,
        d_scribble_color_intensity_values, 
        spatial_kernel_width,
        d_sigma
    ) -> float:
    total_kernel_value = cuda.local.array(shape = 1, dtype = nb.float64)
    n_scribble_points = d_scribble_coordinates.shape[0]
    for scribble_point in range(n_scribble_points):
        scribble_point_color_intensities = cuda.local.array(shape = 3, dtype = nb.float64)
        for channel in range(3):
            scribble_point_color_intensities[channel] = d_scribble_color_intensity_values[scribble_point, channel]
        scribble_point_coordinates = cuda.local.array(shape = 2, dtype = nb.int64)
        for dim in range(2):
            scribble_point_coordinates[channel] = d_scribble_coordinates[scribble_point, dim]
        spatial_kernel = compute_gaussian_kernel(
            target_pixel_coordinates, 
            scribble_point_coordinates,
            spatial_kernel_width,
            True
        )
        chromo_kernel = compute_gaussian_kernel(
            target_pixel_color_intensities, 
            scribble_point_color_intensities,
            d_sigma,
            False
        )
        total_kernel_value[0] += spatial_kernel * chromo_kernel
    total_kernel_value[0] /= n_scribble_points
    return total_kernel_value.item()

@cuda.jit(device = True)
def pixel_factored_multivariate_gaussian_kernel(
        d_image_array, 
        x_coord, 
        y_coord,
        d_scribble_coordinates,
        d_scribble_color_intensity_values, 
        spatial_kernel_width,
        d_sigma,
    ) -> float:
    target_pixel_color_intensities = cuda.local.array(shape = 3, dtype = nb.uint8)
    target_pixel_coordinates = cuda.local.array(shape = 2, dtype = nb.uint8)
    for channel in range(3):
        target_pixel_color_intensities[channel] = d_image_array[channel, x_coord, y_coord]
    target_pixel_coordinates[0] = x_coord
    target_pixel_coordinates[1] = y_coord
    return pixel_factored_multivariate_gaussian_kernel_(
        d_image_array,
        target_pixel_color_intensities, 
        target_pixel_coordinates, 
        d_scribble_coordinates,
        d_scribble_color_intensity_values, 
        spatial_kernel_width,
        d_sigma
    )
