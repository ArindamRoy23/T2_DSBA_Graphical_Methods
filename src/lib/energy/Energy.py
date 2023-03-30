"""


"""

import numpy as np 
import cupy as cp
import math

from numba import cuda
from scipy import sparse

from ..utils.FileHandling.FileHandlingInterface import *
from ..utils.SVCDUtils.SVCDUtils import *
from ..probabilityEstim.Likelihood import Likelihood
from ..probabilityEstim.Prior import Prior

class Energy(object):
    """
    
    """
    def __init__(
            self,
            n_classes: int, 
            lambda_: float = 8e-4, 
            gamma: float = 5e-0, 
            alpha: float = 13e-1, # width of spatial kernel (as in the paper)
            sigma: float = 18e-1, # width of chromatic kernel (as in the paper)
            debug: int = 0,
            return_: int = 0,
            transpose: bool = False
        ) -> None:
        """
        
        """
        self.n_classes = n_classes
        self.gamma = gamma            
        self.alpha = alpha
        self.sigma = sigma
        self.debug = debug
        self.lambda_ = lambda_
        self.return_ = return_
        self.transpose = transpose
        self.likelihood = Likelihood(
            self.n_classes,
            alpha = self.alpha,
            sigma = self.sigma,
            transpose = self.transpose
        )
        self.prior = Prior(
            gamma = self.gamma,
            debug = self.debug,
            return_ = self.return_
        )
        self.utils = SVCDUtils()


    def __energy(
            self, 
            theta: np.ndarray, 
            fitted_likelihood: np.ndarray,
            halfg: np.ndarray
        ) -> float:
        """
        
        """
        d_xi = self.utils.derivative(theta)
        norm_grad_xi = np.sqrt(d_xi[0]**2 + d_xi[1]**2)
        energy_reg = np.sum(halfg * norm_grad_xi)
        energy_dat = np.sum(theta * fitted_likelihood)
        energy = energy_reg + energy_dat
        return energy
    

    def __primal_energy(
            self, 
            theta: np.ndarray, 
            fitted_likelihood: np.ndarray,
            half_g: np.ndarray
        ) -> float:
        """
        
        """
        # uncomment
        theta = self.utils.get_argmax_matrix(theta)
        dtheta = self.utils.derivative(theta)
        part1 = self.lambda_ * np.sum(theta * fitted_likelihood)
        part2 = np.sum(half_g * np.sqrt(dtheta[0]**2 + dtheta[1]**2))
        return part1 + part2


    def __dual_energy(
            self, 
            xi: np.ndarray, 
            fitted_likelihood: np.ndarray
        ) -> float:
        """
        
        """
        arg = np.min(self.lambda_ * fitted_likelihood - self.utils.divergence(xi), axis=0)
        return np.sum(arg)


    def energy(
            self, 
            theta: np.ndarray,
            half_g: np.ndarray, 
            target_image: TargetImage, 
            encoded_scribble: EncodedScribble, 
            normalize: bool = True,
        ) -> float:
        """
        
        """
        fitted_likelihood = self.likelihood.fit(
            target_image, 
            encoded_scribble, 
            normalize = normalize
        )
        return self.__energy(
            theta,
            fitted_likelihood, 
            half_g
        )

    def primal_energy(
            self,
            theta: np.ndarray,
            half_g: np.ndarray, 
            target_image: TargetImage, 
            encoded_scribble: EncodedScribble,
            normalize: bool = False,
        ) -> float:
        """
        
        """
        fitted_likelihood = self.likelihood.fit(
            target_image, 
            encoded_scribble, 
            normalize = normalize
        )
        return self.__primal_energy(
            theta, 
            fitted_likelihood, 
            half_g
        )

    def dual_energy(
            self,
            xi: np.ndarray,
            target_image: TargetImage, 
            encoded_scribble: EncodedScribble,
            normalize: bool = True 
        ) -> float:
        """
        
        """
        fitted_likelihood = self.likelihood.fit(
            target_image, 
            encoded_scribble, 
            normalize = normalize
        )
        return self.__dual_energy(
            xi, 
            fitted_likelihood
        )