import numpy as np 
import cupy as cp

from numba import cuda
from torch import Tensor

from ..utils.FileHandlin.FileHandlingInterface import TargetImage, EncodedScribble
from ..utils.Parallelization.CudaKernels import __find_class_pixelwise_closest_scribble_point_cuda_

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
 