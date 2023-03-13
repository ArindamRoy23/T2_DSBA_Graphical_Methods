from .probabilityDensityEstimator import *
from ..utils.ProbaUtils import * 

class LikelihoodEstimator(ProbabilityDensityEstimator):
    """
    LikelihoodEstimator(ProbabilityDensityEstimator):
        children class of ProbabilityDensityEstimator used 
        for computing the Likelihood distribution

    Methods-list:
        __init__(self, n_classes, alpha, sigma, on_gpu)
        __find_scribble_point_with_minimum_distance(self, x_coord, y_coord, scribble_coordinates)
        __find_class_pixelwise_closest_scribble_point_cuda_(target_image, scribble_cooridnates, output_array)
        __find_class_pixelwise_closest_scribble_point_cuda(target_image, scribble_coordinates)
        __find_class_pixelwise_closest_scribble_point(target_image, scribble_coordinates)
        __find_pixelwise_closest_scribble_point(self, target_image, encoded_scribble)
        __find_scribble_pixel_color_intensity_values(self, target_image, scribble_coordinates)
        __pixel_multivariate_gaussian_kernel(self, x, mu, sigma)
        __get_class_spatial_kernel_cuda(self, target_image, scribble_coordinates)
        __get_class_spatial_kernel(self, target_image, scribble_coordinates)
        __get_class_chromatic_kernel_cuda(self, target_image, scribble_coordinates)
        __get_class_chromatic_kernel(self, target_image, scribble_coordinates)
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


    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    
    ################### METHODS FOR COMPUTING THE SPATIAL KERNEL WIDTH

    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    
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

    def __find_class_pixelwise_closest_scribble_point_cuda_(
            self,
            TargetImage: target_image,
            ndarray: scribble_coordinates,
            ndarray: output_array
        ) -> None: # cuda kernels cannot return anything, they just write the results to an array passes as argument
        """
        __find_class_pixelwise_closest_scibble_point_cuda_(self, TargetImage, scribble_coordinates):
            finds the closest point amongst the scribble point coordinates provided as argument.
            This will be used over each class.

            This method is to be used for computations on the gpu

            In the Kernel Density Estimation part this is necessary 
            for computing the width of the spatial kernel (equation (14 of the paper))   

            (i.e.: implement same __find_class_pixelwise_closest_scribble_point with cuda)            
            Output shape: (1, image_width, image_heigth)
        """
        __find_class_pixelwise_closest_scribble_point_cuda_(
            target_image, 
            scribble_coordinates, 
            output_array
        )
    
    def __find_class_pixelwise_closest_scribble_point_cuda(
            self, 
            TargetImage: target_image, 
            ndarray: scribble_coordinates,
        ) -> ndarray: # shape of output should be (1, image_width, image_height)
        """
        __find_class_pixelwise_closest_scribble_cuda(
                target_image, 
                scribble_coordinate 
            ) -> ndarray:
            invokes the cuda kernel for computing the pixelswise distance map for a given class
        """ 
        pixel_map_size = target_image.get_image_shape() # tuple of (x, y) coordinate for each pixel 
        closest_pixel_map = cp.zeros(pixel_map_size)
        self.__find_class_pixelwise_closest_scribble_point_cuda_(
            target_image, 
            scribble_coordinates, 
            closest_pixel_maps
        )
        return closest_pixel_map

    def __find_class_pixelwise_closest_scribble_point(
            self, 
            TargetImage: target_image, 
            ndarray: scribble_coordinates
        ) -> ndarray:# shape of output should be (1, image_width, image_height) 
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
        closest_pixel_map = cp.empty(pixel_map_size)
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
        
    ##########################################################################################
    ####### FINALLY COMPUTING SPATIAL KERNEL WIDTH FOR ALL CLASSES AND PIXELS
    ##########################################################################################
    def __find_pixelwise_closest_scribble_point(
            self, 
            TargetImage: target_image,
            EncodedScribble: encoded_scribble,
        ) -> ndarray: # shape of output should be (self.n_classes, image_width, image_height)
        """
        __find_class_pixelwise_closest_scibble_point(self, TargetImage, EncodedScribble):
            finds the closest point amongst the scribble point coordinates provided as argument
            This will do the same job of __find_class_pixelwise_closest_scribble_point but for
            all the classes.

            In the Kernel Density Estimation part this is necessary 
            for computing the width of the spatial kernel (equation (14 of the paper))
            
            Output shape: (n_classes, image_width, image_heigth)
                A given entry (c, x, y) of the array will represent the spatial kernel width
                for class c computed at pixel x, y.
        """
        image_size = target_image.get_image_size()
        pixel_map_size = (self.n_classes, ) + image_size # tuple of (x, y) coordinate for each pixel 
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

    ##########################################################################################
    ##########################################################################################
    ##########################################################################################

    ################  METHODS FOR COMPUTING THE SPATUIAL AND CHROMATIC KERNELS

    ##########################################################################################
    ##########################################################################################
    ##########################################################################################

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
            bool: spatial = True, # If true, computes the spatial kernel else the chromatic one,
            *args, # optional argument, must be passed if not spatial. It would contain the chromatic value of the scribble pixels
            **kwargs
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
        gaussian_kernel = __pixelwise_multivariate_gaussian_kernel(
            x, 
            scribble_coordinates, 
            self.on_gpu,
            spatial = spatial,
            *args, # probably passing only kwargs is the best choice
            **kwargs
        )
        return gaussian_kernel

    def __get_class_spatial_kernel_cuda_(
            self, 
            TargetImage: target_image, 
            ndarray: scribble_coordinates, 
            ndarray: pixelwise_distance_map, 
            ndarray: output_array
        ) -> ndarray: # output should be of size (image_width, image_height)
        """"
        __get_class_spatial_kernel_cuda_(
                target_image, 
                scribble_coordinates, 
                pixelwise_distance_map,
                output_array
            ) -> ndarray:
        simple wrapper method for the cuda kernel
        """
        __get_class_spatial_kernel_cuda_(
            target_image, 
            scribble_cooridnates, 
            output_array,
            pixelwise_distance_map = pixelwise_distance_map, 
        )

    def __get_class_spatial_kernel_cuda(
            self, 
            TargetImage: target_image,
            ndarray: scribble_coordinates,
            ndarray: pixelwise_distance_map   
        ) -> ndarray:
        """
        __get_class_spatial_kernel_cuda(
                self, 
                target_image,
                scribble_coordinates,
                pixelwise_distance_map
            ):
            Computes the pixelwise spatial kernel over an image for a given class on GPU.
            Needs to be passed the distance map for each pixel as an argument 

            output shape: (n_scribble_points, image_width, image_height)

            needs to call:
                self.__find_pixelwise_closest_scribble_point for computing kernel width at each point 
        """
        target_size = target_immage.get_image_shape()
        output_array = cp.empty(target_size)
        self.__get_class_spatial_kernel_cuda_(
            target_image, 
            scribble_coordinates, 
            pixelwise_distance_map, 
            output_array
        )
        return output_array

    def __get_class_spatial_kernel(
            self, 
            TargetImage: target_image,
            ndarray: scribble_coordinates,
            ndarray: pixelwise_distance_map # widths of the kernels
        ) -> ndarray:
        """
        __get_class_spatial_kernel_cuda(
                self, 
                target_image,
                scribble_coordinates,
                pixelwise_distance_map
            ):
            Computes the pixelwise spatial kernel over an image for a given class on CPU
            Needs to be passed the distance map for each pixel as an argument 

            output shape: (n_scribble_points, image_width, image_height)

            methods to be called here:
                self.__find_pixelwise_closest_scribble_point for computing kernel width at each point
        """ 
        image_width, image_height = tearget_image.get_image_shape()
        image_array = target_image.get_image_array()
        n_scribble_points = scribble_coordinates.shape[0]
        image_spatial_kernel_map = np.empty((n_scribble_points, image_width, image_height))
        # For spatial kernel, scaling the distances using the alpha paramter
        pixelwise_distance_map = self.alpha * pixelwise_distance_map # check this is doable 
        for x_coord in range(image_width):
            for y_coord in range(image_height):
                coord = (x_coord, y_coord)
                spatial_kernel = self.__pixel_multivariate_gaussian_kernel(
                    coord, 
                    scribble_coordinates,
                    pixelwise_distance_map = pixelwise_distance_map
                )
                image_spatial_kernel_map[:, x_coord, y_coord] = spatial_kernel
        return image_spatial_kernel_map

    def __get_class_chromatic_kernel_cuda(
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

            output shape: (n_scribble_points, image_width, image_height)
        """
        pass 

    def __get_class_chromatic_kernel(
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

            output shape: (n_scribble_points, image_width, image_height)
        """
        image_width, image_height = tearget_image.get_image_shape()
        image_array = target_image.get_image_array()
        n_scribble_points = scribble_coordinates.shape[0]
        image_chromo_kernel_map = np.empty((n_scribble_points, image_width, image_height))
        # instead of the scribble coordinates we should use their chromatic values
        chromatic_values = self.__find_scribble_pixel_color_intensity_values(
            target_image, 
            scribble_coordinates
        )
        for x_coord in range(image_width):
            for y_coord in range(image_height):
                x_val = image_array[:, x_coord, y_coord]
                spatial_kernel = self.__pixel_multivariate_gaussian_kernel(
                    x_val, 
                    chromatic_values,
                    spatial = False,
                    sigma = self.sigma
                )
                image_chromo_kernel_map[:, x_coord, y_coord] = spatial_kernel
        return image_chromo_kernel_map
        
    def __get_class_factorised_kernel(
            self, 
            TargetImage: target_image, 
            ndarray: scribble_coordinates,
            ndarray: pixelwise_distance_map 
        ) -> ndarray:
        """
        """
        image_width, image_height = target_image.get_image_shape()
        target_size = (image_width, image_height)
        kernel_map = np.empty(target_size)
        class_chromatic_kernel = self.__get_class_chromatic_kernel(
            target_image, 
            scribble_coordinates
        )
        class_spatial_kernel = self.__get_class_spatial_kernel(
            target_image, 
            scribble_coordinates, 
            pixelwise_distance_map
        )
        for x_coord in range(image_width):
            for y_coord in range(image_height):
                pixel_spatial_kernel = class_spatial_kernel[:, x_coord, y_coord]
                pixel_chromatic_kernel = class_chromatic_kernel[:, x_coord, y_coord]
                assert pixel_chromatic_kernel.shape[0] == pixel_spatial_kernel.shape[0]
                n_scribble_points = pixel_chromatic_kernel.shape[0]
                pixel_factorised_kernel_value = np.dot(pixel_chromatic_kernel.T, pixel_spatial_kernel)
                kernel_map[x_coord, y_coord] = pixel_factorised_kernel_value / n_scribble_points
        return kernel_map