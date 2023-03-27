"""

"""
from ..utils.FileHandling.FileHandlingInterface import *
from ..utils.SVCDUtils.SVCDUtils import *

import numpy as np

class Prior(object):
    def __init__(
            self, 
            gamma: float,
            debug: int = 0,
            return_: int = 0
        ) -> None:
        """
        :param img: image of dimensions c x h x w
        """
        self.gamma = gamma
        self.debug = debug
        self.fitted_half_g_ = False
        self.fitted_half_g = None
        self.return_ = return_
        self.utils = SVCDUtils()
        
    
    def __prior_energy(
            self, 
            img: np.ndarray, 
            theta: np.ndarray
        ) -> float:
        """
        Computes the prior energy (eq. 21)

        :param img: image to segment of dimensions c x h x w
        :param self.gamma: float
        :param theta: segmentation of the image of dimensions n x h x w
        
        """
        d_Theta = self.utils.gradient_I(theta) # 2 x n_classes x h x w
        norm_d_Theta = np.sqrt(d_Theta[0]**2 + d_Theta[1]**2)# n_classes x h x w ##np.abs(d_Theta) # 2 x c x h x w
        halfg = self.utils.half_g(img)
        prod = halfg * norm_d_Theta # n_classes x h x w
        return np.sum(prod) # scalar

    def fit(
            self, 
            target_image: TargetImage, 
            theta: np.ndarray
        ) -> float:
        if self.debug > 1:
            print(f"Segmenting target_image of type {type(target_image)}")
        return self.__prior_energy(target_image, theta)