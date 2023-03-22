"""

``ParallelizationUtils.py``

Script containing the utils for parallelization.
These utils are basically cuda kernels that are omonimous to the class
methods they will be wrapped in.

It is necessary to follow this practice as there is not a full support for
compiling class methods using numba.cuda.jit().

However, it is possible to define such class member functions and use them 
as wrapper for the cuda kernel, which is what we do.

"""
from .CudaDeviceFunctions import *


@cuda.jit()
def get_class_factorised_kernel_cuda1_(
            d_image_array, 
            d_scribble_coordinates, 
            d_alpha, 
            d_sigma,
            d_scribble_color_intensity_values,
            d_output_array
        ) -> None:
        n_channels, image_width, image_height = d_image_array.shape
        x_coord, y_coord = cuda.grid(2)
        alpha = d_alpha.item()
        if x_coord < image_width and y_coord < image_height:
            spatial_kernel_width = find_scribble_point_with_minimum_distance(
                x_coord,
                y_coord, 
                d_scribble_coordinates
            )
            spatial_kernel_width = alpha * spatial_kernel_width
            
            d_output_array[x_coord, y_coord] = pixel_factored_multivariate_gaussian_kernel(
                d_image_array, 
                x_coord, 
                y_coord,
                d_scribble_coordinates,
                d_scribble_color_intensity_values, 
                spatial_kernel_width,
                d_sigma,
            )
