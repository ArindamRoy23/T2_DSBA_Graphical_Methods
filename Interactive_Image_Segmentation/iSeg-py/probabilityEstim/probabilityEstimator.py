import numpy as np 
import numpy

from numba import cuda
from torch import Tensor

from ..utils.FileHandlingInterface import TargetImage, EncodedScribble

class ProbabilityDensityEstimator(object):
    """
    Most abstract class for representing probability density estimators
    It will be the parent of the LikelihoodEstimator and PriorEstimator classes
    """
    def __init__(
            self,
            int: n_classes,
            bool: on_gpu = True
        ) -> None:
        """
        __init__(self):
            initializes the object given the number of classes considered 
            for the segmentation task
        """
        self.n_classes = n_classes
        self.on_gpu = on_gpu

    def _fit(
            self,
            TargetImage: target_image,
            EncodedScribble: encoded_scribble
        ) -> ndarray | Tensor:
        """"
        _fit(self, target_image, encoded_scribble):
            fits the probability distribution to the given
            pair target_image, encoded_scribble
        """
        pass
    
class LikelihoodEstimator(ProbabilityDensityEstimator):
    """
    LikelihoodEstimator(ProbabilityDensityEstimator):
        children class of ProbabilityDensityEstimator used 
        for computing the Likelihood distribution
    """
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
        super(LikelihoodEstimator, self).__init__(n_classes, on_gpu = on_gpu)
        self.alpha = alpha
        self.sigma = sigma
        self.on_gpu = on_gpu

    def __find_scribble_point_with_minimum_distance(
            self,
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
        lp_distance = lambda x1, x2, y1, y2, p: ((x1 - x2)**p + (y1 - y2)**p)**(1/p) 
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
                y_coord_scribble,
                2
            )
            if distance < min_distance:
                min_distance = distance
        return min_distance

    @cuda.jit # not so sure that we can actually compile a class method though
    def __find_class_pixelwise_closest_scribble_point_cuda(
            self, 
            TargetImage: target_image,
            ndarray: scribble_coordinates
        ) -> ndarray | Tensor : # shape of output should be (1, image_width, image_height)  
        """
        __find_class_pixelwise_closest_scibble_point_cuda(self, TargetImage, scribble_coordinates):
            finds the closest point amongst the scribble point coordinates provided as argument.
            This will be used over each class.

            This method is to be used for computations on the gpu

            In the Kernel Density Estimation part this is necessary 
            for computing the width of the spatial kernel (equation (14 of the paper))   

            Output shape: (1, image_width, image_heigth)
        """
        image_size = target_image.get_image_size()
        pixel_map_size = (2, ) + image_size # tuple of (x, y) coordinate for each pixel 
        closest_pixel_map = cp.zeros(pixel_map_size)
        # TODO continue computations in parallel on GPU
        # (i.e.: implement same __find_class_pixelwise_closest_scribble_point with cuda)

    def __find_class_pixelwise_closest_scribble_point(
            self, 
            TargetImage: target_image, 
            ndarray: scribble_coordinates
        ) -> ndarray | Tensor:# shape of output should be (1, image_width, image_height) 
        """
         __find_class_pixelwise_closest_scibble_point(self, TargetImage, scribble_coordinates):
            finds the closest point amongst the scribble point coordinates provided as argument.
            This will be used over each class.

            Computations on the CPU

            In the Kernel Density Estimation part this is necessary 
            for computing the width of the spatial kernel (equation (14 of the paper))

            
            Output shape: (image_width, image_heigth)
        """
        image_size = target_image.get_image_size()
        pixel_map_size = image_size # tuple of (x, y) coordinate for each pixel 
        closest_pixel_map = cp.zeros(pixel_map_size)
        image_width, image_height = image_size
        for x_coord in range(image_width):
            for y_coord in range(image_height):
                # finding the distance
                distance = self.__find_scribble_point_with_minimum_distance(
                    x_coord, 
                    y_coord, 
                    scribble_coordinates
                )
                closest_pixel_map[x_coord, y_coord] = distance
        return closest_pixel_map
        
    def __find_pixelwise_closest_scribble_point(
            self, 
            TargetImage: target_image,
            EncodedScribble: encoded_scribble,
        ) -> ndarray | Tensor: # shape of output should be (self.n_classes, image_width, image_height)
        """
        __find_class_pixelwise_closest_scibble_point(self, TargetImage, EncodedScribble):
            finds the closest point amongst the scribble point coordinates provided as argument
            This will do the same job of __find_class_pixelwise_closest_scribble_point but for
            all the classes.

            In the Kernel Density Estimation part this is necessary 
            for computing the width of the spatial kernel (equation (14 of the paper))
            
            Output shape: (n_classes, image_width, image_heigth)
        """
        image_size = target_image.get_image_size()
        pixel_map_size = (self.n_classes, 2, ) + image_size # tuple of (x, y) coordinate for each pixel 
        closest_pixel_map = cp.zeros(pixel_map_size) if self.on_gpu else np.zeros(pixel_map_size)
        for class_label, class_pixels in enumerate(encoded_scribble):
            # compute the class_pixel_map either on gpu or on cpu
            class_closest_pixel_map = self.__find_class_pixelwise_closest_scribble_point_cuda( \
                    self, 
                    target_image, 
                    class_pixels
                ) if self.on_gpu else self.__find_class_pixelwise_closest_scribble_point(
                    self, 
                    target_image, 
                    class_pixels
                )
            # save the pixel_map
            closest_pixel_map[class_label] = class_closest_pixel_map
        return closest_pixel_map

    def __pixel_multivariate_gaussian_kernel(
            self,
            ndarray: x, 
            ndarray: mu, 
            ndarray: sigma # width of the kernel
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
        """
        pass
    
    @cuda.jit()
    def __get_class_spatial_kernel_cuda(
            self, 
            TargetImage: target_image,
            ndarray: scribble_coordinates
        ) -> ndarray:
        """
        __get_class_spatial_kernel_cuda(
                self, 
                target_image,
                scribble_coordinates
            ):
            Computes the pixelwise spatial kernel over an image for a given class on GPU

            output shape: (image_width, image_height)

            needs to call:
                self.__find_pixelwise_closest_scribble_point for computing kernel width at each point               in 
        """ 

    
    
    def __get_class_spatial_kernel(
            self, 
            TargetImage: target_image,
            ndarray: scribble_coordinates
        ) -> ndarray:
        """
        __get_class_spatial_kernel_cuda(
                self, 
                target_image,
                scribble_coordinates
            ):
            Computes the pixelwise spatial kernel over an image for a given class on CPU

            output shape: (image_width, image_height)

            methods to called inside here:
                self.__find_pixelwise_closest_scribble_point for computing kernel width at each point

        """ 

