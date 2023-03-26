"""


"""

import numpy as np 
import cupy as cp
import math

from numba import cuda
from scipy import sparse

from ..utils.FileHandling.FileHandlingInterface import *
from ..utils.ProbabilityUtils.Divergence import *
from ..probabilityEstim.Likelihood import Likelihood
from ..probabilityEstim.Prior import Prior

class Energy(object):
    """
    
    """
    def __init__(
            self,
            n_classes: int, 
            lambda_: float = 8e-4, 
            gamma: float = 1e-0, 
            alpha: float = 13e-1, # width of spatial kernel (as in the paper)
            sigma: float = 18e-1, # width of chromatic kernel (as in the paper)
            debug: int = 0
        ) -> None:
        """
        
        """
        self.n_classes = n_classes
        self.gamma = gamma            
        self.alpha = alpha
        self.sigma = sigma
        self.debug = debug
        self.lambda_ = lambda_
        self.likelihood = Likelihood(
            self.n_classes,
            alpha = self.alpha,
            sigma = self.sigma,
        )
        self.prior = Prior(
            gamma = self.gamma,
            debug = self.debug
        )
        self.fitted_likelihood_ = False
        self.fitted_likelihood = None ## compute only once for each energy
    
    def __make_derivative_matrix(
            self, 
            width: int, 
            height: int
        ) -> np.ndarray:
        """
        
        """
        return make_derivative_matrix(
            width, 
            height
        )

    def __fit_likelihood(
            self, 
            target_image: TargetImage, 
            encoded_scribble: EncodedScribble,
            normalize: bool = True
        ) -> np.ndarray:
        """
        
        """
        if not self.fitted_likelihood_:
            self.fitted_likelihood = self.likelihood.fit(
                target_image, 
                encoded_scribble, 
                normalize = normalize
            )
            self.fitted_likelihood_ = True
        return self.fitted_likelihood
    
    def fit_likelihood(
            self,
            target_image: TargetImage, 
            encoded_scribble: EncodedScribble,
            normalize: bool = True
        ) -> np.ndarray:
        """
        
        """
        return self.__fit_likelihood(
            target_image, 
            encoded_scribble, 
            normalize = normalize
        )

    def __fit_prior(
            self, 
            target_image: TargetImage, 
            theta: np.ndarray
        ) -> float:
        """
        
        """
        fitted_prior = self.prior.fit(
            target_image, 
            theta
        )
        return fitted_prior

    def fit_prior(
            self,
            target_image: TargetImage, 
            theta: np.ndarray
        ) -> float:
        """
        
        """
        return self.__fit_prior(
            target_image, 
            theta
        )

    def __derivative(
            self, 
            array: np.ndarray
        ) -> np.ndarray:
        """
        
        """
        return derivative(
            array
        )

    def __divergence(
            self, 
            array: np.ndarray
        ):
        """
        
        """
        return divergence(
            array
        )

    def __energy(
            self, 
            theta: np.ndarray, 
            fitted_likelihood: np.ndarray,
            halfg: float
        ) -> float:
        """
        
        """
        d_xi = self.__derivative(theta)
        norm_grad_xi = np.sqrt(d_xi[0]**2 + d_xi[1]**2)
        #energy_reg = np.sum(halfg * norm_grad_xi)
        energy_dat = np.sum(theta * fitted_likelihood)
        #energy = energy_reg + energy_dat
        energy = halfg + energy_dat
        return energy
    
    def energy(
            self, 
            theta: np.ndarray,
            target_image: TargetImage, 
            encoded_scribble: EncodedScribble, 
            normalize: bool = False,
        ) -> float:
        """
        
        """
        fitted_prior = self.__fit_prior(
            target_image, 
            theta
        )
        fitted_likelihood = self.__fit_likelihood(
            target_image, 
            encoded_scribble, 
            normalize = True
        )
        return self.__energy(
            theta,
            fitted_likelihood, 
            fitted_prior
        )

    def __primal_energy(
            self, 
            theta: np.ndarray, 
            fitted_likelihood: np.ndarray,
            fitted_prior: float
        ) -> float:
        """
        
        """
        dtheta = self.__derivative(theta)
        part1 = self.lambda_ * np.sum(theta * fitted_likelihood)
        part2 = np.sum(fitted_prior * np.sqrt(dtheta[0]**2 + dtheta[1]**2))
        return part1 + part2
    
    def primal_energy(
            self,
            theta: np.ndarray,
            target_image: TargetImage, 
            encoded_scribble: EncodedScribble,
            normalize: bool = False,
        ) -> float:
        """
        
        """
        fitted_likelihood = self.__fit_likelihood(
            target_image, 
            encoded_scribble, 
            normalize = normalize
        )
        fitted_prior = self.__fit_prior(
            target_image, 
            theta
        )
        return self.__primal_energy(
            theta, 
            fitted_likelihood, 
            fitted_prior
        )

    def __dual_energy(
            self, 
            xi: np.ndarray, 
            fitted_likelihood: np.ndarray
        ) -> float:
        """
        
        """
        arg = np.min(self.lambda_ * fitted_likelihood - self.__divergence(xi), axis=0)
        return np.sum(arg)
    
    def dual_energy(
            self,
            xi: np.ndarray,
            target_image: TargetImage, 
            encoded_scribble: EncodedScribble,
            normalize: bool = True 
        ) -> float:
        """
        
        """
        fitted_likelihood = self.__fit_likelihood(
            target_image, 
            encoded_scribble, 
            normalize = normalize
        )
        return self.__dual_energy(
            xi, 
            fitted_likelihood
        )