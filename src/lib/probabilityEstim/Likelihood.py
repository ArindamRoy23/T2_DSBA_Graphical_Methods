import numpy as np 
import cupy as cp
import math

from numba import cuda
from ..utils.FileHandling.FileHandlingInterface import *
from ..utils.Parallelization.CudaKernels import *

class Likelihood(object):
    def __init__(
            self, 
            n_classes: int, 
            alpha: float = 13e-1, # width of spatial kernel (as in the paper)
            sigma: float = 18e-1, # width of chromatic kernel (as in the paper)
            return_: int = 0
        ) -> None:
        """"
        __init__(self, n_classes, alpha, sigma, on_gpu = True):
            initializes the LikelihoodEstimator with the given
            number of classes on which to perform segmentation,
            the alpha (hyperparameter for spatial kernel) and 
            sigma (width of chromatic kernel) values and the 
            device on which to compute the estimation (CPU or GPU)

        return_ values: 
            0 -> No return_ging
            1 -> Spatial Kernel
            2 -> Chromatic Kernel
            3 -> Spatial Kernel Exponent Argument
            4 -> Chromatic Kernel Exponent Argument
            5 -> Spatial Kernel Normalization Term 
            7 -> Spatial Kernel Width
        """
        self.n_classes = n_classes
        self.alpha = alpha
        self.sigma = sigma
        self.return_ = return_
    

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
        n_scribble_pixels = scribble_coordinates.shape[0]
        image_array = target_image.get_image_array()
        target_shape = (n_scribble_pixels, n_channels)
        scribble_color_intensity_values = np.empty(target_shape)
        for idx in range(n_scribble_pixels):
            x_coord,  y_coord = scribble_coordinates[idx]
            pixel_color_intensity = image_array[:, x_coord, y_coord]
            scribble_color_intensity_values[idx, : ] = pixel_color_intensity
        return scribble_color_intensity_values


    def __get_class_factorised_kernel_cuda_(
            self, 
            target_image: TargetImage, 
            scribble_coordinates: np.ndarray
        ) -> None:
        """
        __get_class_factorised_kernel_cuda(
                target_image, 
                scribble_coordinates
            ):
            gets the factorised likelihood for a given class

            Performs parallel computations on GPU

            Returns: np.ndarray of shape (image_width, image_height)
             
        """


        image_array = target_image.get_image_array()
        n_scribble_points = scribble_coordinates.shape[0]
        n_channels, image_width, image_height = image_array.shape

        alpha = np.float64(self.alpha)
        sigma = np.float64(self.sigma)

        scribble_color_intensity_values = self.__find_scribble_pixel_color_intensity_values(
            target_image,
            scribble_coordinates
        )
    
        output_array = np.empty((image_width, image_height), dtype = np.float64)

        d_return_ = cuda.to_device(self.return_)


        d_alpha = cuda.to_device(alpha)
        d_sigma = cuda.to_device(sigma)
        d_image_array = cuda.to_device(image_array)
        d_scribble_coordinates = cuda.to_device(scribble_coordinates)
        d_scribble_color_intensity_values = cuda.to_device(scribble_color_intensity_values)

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
            d_output_array,
            d_return_
        )
        cuda.synchronize()
        return d_output_array

    def __fit(
            self,  
            target_image: TargetImage, 
            encoded_scribble: EncodedScribble,
            normalize: bool = False,
            smoothing: float = 1e-5
        ) -> np.ndarray:
        """
        __fit(
                self, 
                target_image: TargetImage, -> image to be segmented encoded in a TargetImage object
                scribble_coordinates: EncodedScribble -> scribble to base the segmentation on encoded in an EncodedScribble object
            ) -> np.ndarray
        
        fits the factorised Kernel Density Estimation given an image and a set of user scribbles

        It will either perform the computation in parallel on the gpu if self.on_gpu, otherwise on the cpu

        Returns: np.ndarray of shape (n_classes, image_width, image_height)
        """
        encoded_scribble = encoded_scribble.get_encoded_scribble()
        image_size = target_image.get_image_shape()
        target_size = (self.n_classes, ) + image_size 
        kde_likelihood_map = np.empty(target_size) 
        for idx, class_scribble_coordinates in enumerate(encoded_scribble):
            kde_likelihood = self.__get_class_factorised_kernel_cuda_(
                target_image, 
                class_scribble_coordinates    
            )
            kde_likelihood_map[idx, :, :] = kde_likelihood
        if normalize:
            kde_likelihood_map /= np.sum(kde_likelihood_map, axis = 0) # normalize to sum to one over each class
            ## should do this within cuda we can create another method for doing so
        
        kde_likelihood_map = -1 * np.log((kde_likelihood_map + smoothing))
        return kde_likelihood_map

    def fit(
            self, 
            target_image: TargetImage, 
            encoded_scribble: EncodedScribble,
            normalize: bool = False
        ) -> np.ndarray:
        """
        fit(
                self, 
                target_image: TargetImage, -> image to be segmented encoded in a TargetImage object
                scribble_coordinates: EncodedScribble -> scribble to base the segmentation on encoded in an EncodedScribble object
            ) -> np.ndarray
        
        fits the factorised Kernel Density Estimation given an image and a set of user scribbles

        
        It will either perform the computation in parallel on the gpu if self.on_gpu, otherwise on the cpu

        Returns: np.ndarray of shape (n_classes, image_width, image_height)
        """
        return self.__fit(
            target_image, 
            encoded_scribble,
            normalize = normalize
        )