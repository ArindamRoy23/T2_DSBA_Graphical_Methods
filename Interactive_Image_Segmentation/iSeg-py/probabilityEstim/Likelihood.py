import numpy as np 
import cupy as cp
import math

from numba import cuda
from utils.FileHandling.FileHandlingInterface import *
from utils.ProbaUtils import *
from utils.Parallelization.CudaKernels import get_class_factorised_kernel_cuda_

class Likelihood(object):
    def __init__(
            self, 
            n_classes: int, 
            alpha: float = 13e-1, # width of spatial kernel (as in the paper)
            sigma: float = 18e-1, # width of chromatic kernel (as in the paper)
            on_gpu: bool = False # whether to use gpu for the estimation
        ) -> None:
        """"
        __init__(self, n_classes, alpha, sigma, on_gpu = True):
            initializes the LikelihoodEstimator with the given
            number of classes on which to perform segmentation,
            the alpha (hyperparameter for spatial kernel) and 
            sigma (width of chromatic kernel) values and the 
            device on which to compute the estimation (CPU or GPU)
        """
        self.n_classes = n_classes
        self.on_gpu = on_gpu
        self.alpha = alpha
        self.sigma = sigma
        
    def __find_scribble_point_with_minimum_distance(
            self,
            x_coord: int, 
            y_coord: int,
            scribble_coordinates: np.ndarray
        ) -> float:
        """"
        __find_scribble_point_with_minimum_distance(
                self, 
                x_coord, 
                y_coord, 
                scribble_coordinates
            ):
            Given a pixel's coordinates and the scribble_coordinates array
            finds the l2 distance to the closest scribble point
        
        min_distance = __find_scribble_point_with_minimum_distance(
            x_coord,
            y_coord, 
            scribble_coordinates
        )
        return min_distance
        """
        l2_distance = lambda x1, x2, y1, y2: ((x1 - x2)**2 + (y1 - y2)**2)**(1/2) 
        min_distance = float("inf")
        n_scribble_pixels = scribble_coordinates.shape[0] # flat vector, only one element !!!! NO LONGER THE CASE
        for idx in range(n_scribble_pixels): # Change back to range(0, n_scribble_pixels - 1, 2) in case of flat vector
            x_coord_scribble, y_coord_scribble= scribble_coordinates[idx]
            # l2 distance
            distance = l2_distance(
                x_coord, 
                x_coord_scribble, 
                y_coord, 
                y_coord_scribble
            )
            if distance < min_distance:
                min_distance = distance
        return min_distance

    def __multivariate_gaussian_kernel(
            self,
            x: np.ndarray, 
            mu: np.ndarray, 
            sigma: np.ndarray # width of the kernel (Covariance matrix of gaussian)
        ) -> np.ndarray:
        """
        __pixel_multivariate_gaussian_kernel(
                self, 
                x, 
                mu,
                sigma
            ):
            computes the multivariate gaussian kernel given the taget array, 
            the mean array and the sigma array (for the kernel width).
            This computes the kernel function for a given pixel and a given class.
            Method should be called in a loop over each image pixel and each class.

            for each pixel x in Omega and each pixel x_ij in the scribble for class i it computes
                k(x - x_ij), where k is the kernel function. 
                The mean will be given by the value of the pixel (i.e.: kernel is centered at pixel x)
        
        kernel_val = __multivariate_gaussian_kernel(
            x, 
            mu, 
            sigma, 
            self.on_gpu
        )
        return kernel_val
        """
        n_dimensions = x.shape[0] # either 2 for spatial kernels or 3 for chromo ones
        covariance_matrix = cp.identity(n_dimensions) if self.on_gpu \
            else np.identity(n_dimensions)
        covariance_matrix = sigma * covariance_matrix
        det_covariance = cp.linalg.det(covariance_matrix) if self.on_gpu \
            else np.linalg.det(covariance_matrix)
        inv_covariance = cp.linalg.inv(covariance_matrix) if self.on_gpu \
            else np.linalg.inv(covariance_matrix)
        exponent_offset = x - mu
        exponent = cp.dot(exponent_offset.T, inv_covariance) if self.on_gpu \
            else np.dot(exponent_offset.T, inv_covariance)
        exponent = cp.dot(exponent, exponent_offset) if self.on_gpu \
            else np.dot(exponent, exponent_offset)
        exponent = -0.5 * exponent
        norm_denominator = cp.sqrt(det_covariance) * (2 * cp.pi)**(n_dimensions / 2) if self.on_gpu \
            else np.sqrt(det_covariance) * (2 * np.pi)**(n_dimensions / 2)
        norm = 1/norm_denominator
        kernel_val = norm * cp.exp(exponent) if self.on_gpu \
            else norm * np.exp(exponent) 
        return kernel_val
    
    def __pixel_multivariate_gaussian_kernel(
            self,  
            x: np.ndarray, # target pixel information: either of shape (n_channels, ) for chromatic k or (2, ) for the spatial one
            scribble_coordinates: np.ndarray, # coordinates of the scribble points
            kernel_width: float
        ) -> np.ndarray: # output shape: (1, n_scribble_points)
        """
        __pixel_multivariate_gaussian_kernel(
                self, 
                x, 
                scribble_coord inates, 
                pixelwise_kernel_width,
                spatial
                *args
            ):
            Computes the multivariate gaussian kernel for a given class and a given pixel.
            Basically, at the given pixel, computes a kernel for each of the scribble pixels in scribble_coordinates
            The output shape should be of (1, n_scribble_points)
        """
        n_scribble_points = scribble_coordinates.shape[0]
        gaussian_kernels = cp.empty(n_scribble_points) if self.on_gpu \
            else np.empty(n_scribble_points)
        for idx in range(n_scribble_points):
            x_scribble = scribble_coordinates[idx]
            ## TODO: COMPUTE KERNEL VALUE
            kernel_argument = x - x_scribble
            kernel_val = self.__multivariate_gaussian_kernel(
                kernel_argument, 
                x, 
                kernel_width
            )
            ##
            gaussian_kernels[idx] = kernel_val
        return gaussian_kernels

    def __find_scribble_pixel_color_intensity_values(
            self, 
            target_image: TargetImage,
            scribble_coordinates: np.ndarray
        ) -> np.ndarray:
        """
        __find_scribble_pixel_color_intensity_values(
                self,
                target_image, 
                scribble_coordinates
            ) -> np.ndarray:
            gets the color intensity values of the target image at the scribble points
            Along with the scribble locations, thiese are needed to compute the I_ij in the set S_i of equation (5)
            Basically, this is for computing the chromatic kernel of equation (9)

            output shape: n_scibble_pixels, n_channels
        """
        n_channels = target_image.get_image_channels()
        n_scribble_pixels = scribble_coordinates.shape[0] # flat vector, only one element !!!! NO LONGER THE CASE
        image_array = target_image.get_image_array()
        target_shape = (n_scribble_pixels, n_channels)
        scribble_color_intensity_values = np.empty(target_shape)
        for idx in range(n_scribble_pixels): # Change back to range(0, n_scribble_pixels - 1, 2) in case of flat vector
            x_coord,  y_coord = scribble_coordinates[idx]
            pixel_color_intensity = image_array[:, x_coord, y_coord]
            scribble_color_intensity_values[idx, : ] = pixel_color_intensity
        return scribble_color_intensity_values

    def __get_class_factorised_kernel(
            self, 
            target_image: TargetImage, 
            scribble_coordinates: np.ndarray
        ) -> np.ndarray:
        """
        __get_class_factorised_kernel(
                target_image, 
                scribble_coordinates
            ):
            gets the factorised likelihood for a given class
            output shape: (image_width, image_height)
        """
        image_array = target_image.get_image_array()
        image_width, image_height = target_image.get_image_shape()
        target_size = (image_width, image_height)
        n_scribble_points = scribble_coordinates.shape[0]
        scribble_color_intensity_values = self.__find_scribble_pixel_color_intensity_values(
            target_image, 
            scribble_coordinates
        )
        factorised_kernel_map = np.empty(target_size)
        for x_coord in range(image_width):
            for y_coord in range(image_height):
                spatial_kernel_width = self.__find_scribble_point_with_minimum_distance(
                    x_coord, 
                    y_coord, 
                    scribble_coordinates
                )
                spatial_coord = (x_coord, y_coord)
                chromatic_value = image_array[:, x_coord, y_coord]
                spatial_kernel = self.__pixel_multivariate_gaussian_kernel(
                    spatial_coord, 
                    scribble_coordinates,
                    self.alpha * spatial_kernel_width
                )   
                chromo_kernel = self.__pixel_multivariate_gaussian_kernel(
                    chromatic_value, 
                    scribble_color_intensity_values,
                    self.sigma
                )
                factorised_kernel = np.dot(spatial_kernel.T, chromo_kernel)
                factorised_kernel_map[x_coord, y_coord] = factorised_kernel / n_scribble_points
        return factorised_kernel_map

    def __get_class_factorised_kernel_cuda_(
            self, 
            target_image: TargetImage, 
            scribble_coordinates: np.ndarray
        ) -> None:

        image_array = target_image.get_image_array()
        n_scribble_points = scribble_coordinates.shape[0]
        n_channels, image_width, image_height = image_array.shape


        alpha = np.float64(self.alpha)
        sigma = np.float64(self.sigma)

        scribble_color_intensity_values = np.empty((n_scribble_points, n_channels))
        spatial_coord = np.empty((2, ))
        chromatic_value = np.empty((n_channels, )) # FIX
        spatial_kernel_argument = np.empty((2, ))
        chromo_kernel_argument = np.empty((n_channels, ))
        spatial_kernel = np.empty((n_scribble_points, ))
        chromo_kernel = np.empty((n_scribble_points, ))
        output_array = np.empty((image_width, image_height))
        #spatial_covariance_matrix = np.empty((2, 2))
        #spatial_inv_covariance_matrix = np.empty((2, 2))
        #chromo_covariance_matrix = np.empty((n_channels, n_channels))
        #chromo_inv_covariance_matrix = np.empty((n_channels, n_channels))
        spatial_kernel_exponent_offset = np.empty((2, ))
        chromo_kernel_exponent_offset = np.empty((n_channels))
        spatial_kernel_exponent = np.empty((2, ))
        chromo_kernel_exponent = np.empty((n_channels))


        d_alpha = cuda.to_device(alpha)
        d_sigma = cuda.to_device(sigma)
        d_image_array = cuda.to_device(image_array)
        d_scribble_coordinates = cuda.to_device(scribble_coordinates)
        d_scribble_color_intensity_values = cuda.to_device(scribble_color_intensity_values)
        d_spatial_coord = cuda.to_device(spatial_coord)
        d_chromatic_value = cuda.to_device(chromatic_value)
        d_spatial_kernel_argument = cuda.to_device(spatial_kernel_argument)
        d_chromo_kernel_argument = cuda.to_device(chromo_kernel_argument)
        #d_spatial_covariance_matrix = cuda.to_device(spatial_covariance_matrix)
        #d_chromo_covariance_matrix = cuda.to_device(chromo_covariance_matrix)
        #d_spatial_inv_covariance_matrix = cuda.to_device(spatial_inv_covariance_matrix)
        #d_chromo_inv_covariance_matrix = cuda.to_device(chromo_inv_covariance_matrix)
        d_spatial_kernel = cuda.to_device(spatial_kernel)
        d_chromo_kernel = cuda.to_device(chromo_kernel)
        d_spatial_kernel_exponent = cuda.to_device(spatial_kernel_exponent)
        d_chromo_kernel_exponent = cuda.to_device(chromo_kernel_exponent)
        d_spatial_kernel_exponent_offset = cuda.to_device(spatial_kernel_exponent_offset)
        d_chromo_kernel_exponent_offset = cuda.to_device(chromo_kernel_exponent_offset)
        d_output_array = cuda.to_device(output_array)


        threads_per_block = (16, 16)
        blocks_per_grid_x = math.ceil(image_width / threads_per_block[0])
        blocks_per_grid_y = math.ceil(image_height / threads_per_block[1])
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        cuda.synchronize()
        get_class_factorised_kernel_cuda_[blocks_per_grid, threads_per_block](
            d_image_array, 
            d_scribble_coordinates, 
            d_alpha, 
            d_sigma,
            d_scribble_color_intensity_values,
            d_spatial_coord, 
            d_chromatic_value,
            d_spatial_kernel_argument, 
            d_chromo_kernel_argument,    
            #d_spatial_covariance_matrix, 
            #d_chromo_covariance_matrix, 
            #d_spatial_inv_covariance_matrix, 
            #d_chromo_inv_covariance_matrix,
            d_spatial_kernel, 
            d_chromo_kernel,
            d_spatial_kernel_exponent_offset, 
            d_chromo_kernel_exponent_offset,
            d_spatial_kernel_exponent, 
            d_chromo_kernel_exponent,  
            d_output_array
        )
        cuda.synchronize()
        #output_array = cuda.to_host(d_output_array)
        return d_output_array # change back to return host array

    def __fit(
            self,  
            target_image: TargetImage, 
            encoded_scribble: EncodedScribble
        ) -> np.ndarray:
        """
        __fit_likelihood(target_imaghe, encoded_scribble):
            fits the likelihood to the given image and scribble set
        """
        encoded_scribble = encoded_scribble.get_encoded_scribble()
        image_size = target_image.get_image_shape()
        target_size = (self.n_classes, ) + image_size 
        kde_likelihood_map = np.empty(target_size) 
        for idx, class_scribble_coordinates in enumerate(encoded_scribble):
            kde_likelihood = self.__get_class_factorised_kernel_cuda_(
                target_image, 
                class_scribble_coordinates    
            ) if self.on_gpu else\
            self.__get_class_factorised_kernel(
                target_image, 
                scribble_coordinates
            )    
            kde_likelihood_map[idx, :, :] = kde_likelihood
        return kde_likelihood_map

    def fit(
            self, 
            target_image: TargetImage, 
            encoded_scribble: EncodedScribble
        ) -> np.ndarray:
        return self.__fit(target_image, encoded_scribble)