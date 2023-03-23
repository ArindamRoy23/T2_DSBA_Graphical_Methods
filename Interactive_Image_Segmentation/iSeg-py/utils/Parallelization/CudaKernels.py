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
def get_class_factorised_kernel_cuda_(
        d_image_array: Final[DeviceNDArray], 
        d_scribble_coordinates: Final[DeviceNDArray], 
        d_alpha: Final[DeviceNDArray], 
        d_sigma: Final[DeviceNDArray],
        d_scribble_color_intensity_values: Final[DeviceNDArray],
        d_output_array: DeviceNDArray,
        debug: int64
    ) -> None:
    """
    get_class_factorised_kernel_cuda1_(
        d_image_array: DeviceNDArray, -> device array containing the target image
        d_scribble_coordinates: DeviceNDArray, -> device array containing the scribble coordinates
        d_alpha: DeviceNDArray, -> device array containing the alpha (scaling) parameter for the spatial kernel width
        d_sigma: DeviceNDArray, -> device array containing the sigma (kernel width) paramter for the chromatic kernel
        d_scribble_color_intensity_values: DeviceNDArray, -> device array containing the intensity values of the scribble pixels ### ???
        d_output_array: DeviceNDArray -> device array on which to write the results
    )

    Kernel for performing pixelwise (gaussian) kernel density estimation for the likelihood of a given class  
    """
    debug = debug.item()
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
            debug
        )
