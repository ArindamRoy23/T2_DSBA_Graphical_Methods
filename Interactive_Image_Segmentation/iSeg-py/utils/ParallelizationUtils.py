"""

``ParallelizationUtils.py``

Script containing the utils for parallelization.
These utils are basically cuda kernels that are omonimous  to the class
methods they will be wrapped in.

It is necessary to follow this practice as there is not a full support for
compiling class methods using numba.cuda.jit().

However, it is possible to define such class member functions and use them 
as wrapper for the cuda kernel, which is what we do.

"""

from numba import cuda
from .ProbaUtils import __find_scribble_point_with_minimum_distance


import numpy as np
import cupy as cp

@cuda.jit # not so sure that we can actually compile a class method though
def __find_class_pixelwise_closest_scribble_point_cuda_(
        TargetImage: target_image,
        ndarray: scribble_coordinates,
        ndarray: output_array
    ) -> None : # cuda kernels cannot return anything, they just write the results to an array passes as argument
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
    image_size = target_image.get_image_size()
    # TODO continue computations in parallel on GPU
    x_coord, y_coord = cuda.grid(2) # only need to iterate over the pixels, no channels involved
    if x_coord < image_size[0] and y_coord < image_size[1]: # iterating over each pixel
        # computing the distance to the closest scribble point of the class
        distance = __find_scribble_point_with_minimum_distance(
            x_coord, 
            y_coord,
            scribble_coordinates 
        )
        # writing distance to output_array
        output_array[x_coord, y_coord] = distance


