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
from utils.Parallelization.CudaDeviceFunctions import *

@cuda.jit()
def get_class_factorised_kernel_cuda_(
        image_array: np.ndarray, 
        scribble_coordinates: np.ndarray,
        alpha: float,
        sigma: float,
        scribble_color_intensity_values: np.ndarray,
        spatial_coord: np.ndarray, 
        chromatic_value: np.ndarray,
        spatial_kernel_argument: np.ndarray,
        chromo_kernel_argument: np.ndarray,
        spatial_covariance_matrix: np.ndarray, 
        chromo_covariance_matrix: np.ndarray, 
        spatial_inv_covariance_matrix: np.ndarray, 
        chromo_inv_covariance_matrix: np.ndarray,   
        spatial_kernel: np.ndarray, 
        chromo_kernel: np.ndarray,  
        output_array: np.ndarray,
    ) -> None:
    """
    __get_class_spatial_kernel_cuda_(
            target_image, 
            scribble_coordinates, 
            pixelwise_distance_map, 
            output_array
        ):
        cuda kernel for computing the spatial kernel density estimator 
        it will be called inside LikelihoodEstimator::__get_class_spatial_kernel_cuda_, 
        which will be its wrapper inside the LikelihoodEstimator class
    """
    n_channels, image_width, image_height = image_array.shape
    x_coord, y_coord = cuda.grid(2)
    target_size = (image_width, image_height)
    n_scribble_points = scribble_coordinates.shape[0]
    #alpha = np.float64(alpha)
    alpha = alpha.astype(np.float64).item()
    __find_scribble_pixel_color_intensity_values(
        image_array, 
        scribble_coordinates, 
        scribble_color_intensity_values
    )
    if x_coord < image_width and y_coord < image_height:
        spatial_kernel_width = __find_scribble_point_with_minimum_distance(
            x_coord, 
            y_coord, 
            scribble_coordinates
        )
        spatial_kernel_width = alpha * spatial_kernel_width
        spatial_coord[0] = x_coord; spatial_coord[1] = y_coord
        for channel in range(n_channels):
            chromatic_value[channel] = image_array[channel, x_coord, y_coord]
        __pixel_multivariate_gaussian_kernel(
            spatial_coord, 
            scribble_coordinates, 
            spatial_kernel_width,
            spatial_kernel_argument,
            spatial_covariance_matrix, 
            spatial_inv_covariance_matrix, 
            spatial_kernel
        )
        __pixel_multivariate_gaussian_kernel(
            chromatic_value,
            scribble_color_intensity_values, 
            sigma,
            chromo_kernel_argument,
            chromo_covariance_matrix, 
            chromo_inv_covariance_matrix, 
            chromo_kernel
        )
        factorised_kernel = 0.0
        for idx in range(n_scribble_points):
            factorised_kernel += spatial_kernel[idx] * chromo_kernel[idx]
        output_array[x_coord, y_coord] = factorised_kernel / n_scribble_points