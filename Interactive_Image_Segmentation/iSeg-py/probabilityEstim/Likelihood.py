import numpy as np 
import cupy as cp

from numba import cuda
from ..utils.FileHandling.FileHandlingInterface import *
from ..utils.ProbaUtils import __find_scribble_point_with_minimum_distance 


class Likelihood(object):
    def __init__(
            self, 
            int: n_classes, 
            float: alpha, # width of spatial kernel 
            float: sigma, # width of chromatic kernel
            bool: on_gpu = True # whether to use gpu for the estimation
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
            int: x_coord, 
            int: y_coord,
            ndarray: scribble_coordinates
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
        """
        min_distance = __find_scribble_point_with_minimum_distance(
            x_coord,
            y_coord, 
            scribble_coordinates
        )
        return min_distance

    def __multivariate_gaussian_kernel(
            self,
            ndarray: x, 
            ndarray: mu, 
            ndarray: sigma # width of the kernel (Covariance matrix of gaussian)
        ) -> ndarray:
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
        """
        kernel_val = __multivariate_gaussian_kernel(
            x, 
            mu, 
            sigma, 
            self.on_gpu
        )
        return kernel_val
    
    def __pixel_multivariate_gaussian_kernel(
            self,  
            ndarray: x, # target pixel information: either of shape (n_channels, ) for chromatic k or (2, ) for the spatial one
            ndarray: scribble_coordinates, # coordinates of the scribble points
            float: kernel_width
        ) -> ndarray: # output shape: (1, n_scribble_points)
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
            kernel_val = __multivariate_gaussian_kernel(
                kernel_argument, 
                x, 
                kernel_width
            )
            ##
            gaussian_kernels[idx] = kernel_val
        return gaussian_kernels

    def __find_scribble_pixel_color_intensity_values(
            self, 
            TargetImage: target_image,
            ndarray: scribble_coordinates
        ) -> ndarray:
        """
        __find_scribble_pixel_color_intensity_values(
                self,
                target_image, 
                scribble_coordinates
            ) -> ndarray:
            gets the color intensity values of the target image at the scribble points
            Along with the scribble locations, thiese are needed to compute the I_ij in the set S_i of equation (5)
            Basically, this is for computing the chromatic kernel of equation (9)

            output shape: n_scibble_pixels, n_channels
        """
        n_channels = target_image.get_image_channels()
        image_array = target_image.get_image_array()
        n_scribble_pixels = scribble_coordinates.shape[0]//2
        target_shape = (n_scribble_pixes, n_channels)
        # should have shape of (n_scribble_pixels, n_channels)
        scribble_color_intensity_values = np.empty(target_shape)
        for idx in range(0, scribble_coordinates.shape[0], 2):
            x_coord = scribble_coordinates[idx]
            y_coord = scribble_coordinates[idx + 1]
            pixel_color_intensity = image_array[:, x_coord, y_coord]
            scribble_color_intensity_values[idx//2, : ] = pixel_color_intensity
        return scribble_color_intensity_values


    def __get_class_factorised_kernel(
            self, 
            TargetImage: target_image, 
            ndarray: scribble_coordinates
        ) -> ndarray:
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
                    spatial_kernel_width
                )          
                chromo_kernel = self.__pixel_multivariate_gaussian_kernel(
                    chromatic_value, 
                    scribble_color_intensity_values,
                    self.sigma
                )
                factorised_kernel = np.dot(spatial_kernel.T, chromo_kernel)
                factorised_kernel_map[x_coord, y_coord] = factorised_kernel / n_scribble_points
        return factorised_kernel_map

    def __fit_likelihood(
            TargetImage: target_image, 
            EncodedScribble: encoded_scribble
        ) -> ndarray:
        """
        __fit_likelihood(target_imaghe, encoded_scribble):
            fits the likelihood to the given image and scribble set
        """
        encoded_scribble = encoded_scribble.get_encoded_scribble()
        image_size = target_image.get_image_shape()
        target_size = self.n_classes + imaghe_size 
        kde_likelihood_map = np.empty(target_size) 
        for idx, class_scribble_coordinates in enumerate(encoded_scribble):
            kde_likelihood = self.__get_class_factorised_kernel(
                target_image, 
                class_scribble_coordinates    
            )
            kde_likelihood_map[target_size, :, :] = kde_likelihood
        return kde_likelihood_map