"""

"""
from ..FileHandling.FileHandlingInterface import *


import numpy as np 
import cupy as cp
import math

from numba import cuda
from scipy import sparse
from scipy.sparse._coo import coo_matrix


class SVCDUtils():
    """
    Helper Functions for Segmentation

    """
    def __init__(
            self,
            debug: bool = False
        ) -> None:  
        """
        
        """      
        self.derivative_matrix = None
        self.debug = debug

    def init_halfg(
            self, 
            image: np.ndarray | TargetImage, 
            gamma: float = 5, 
            return_: int = 0
        ) -> np.ndarray:
        """
        Creates 1/2*g(x) (eq.16)

        Measures perimeter of each set.

        Args:
            image: the RGB image with the dimensions [channel, height, width].

        Returns:
            1/2*g(x)
        """
        if "TargetImage" in str(type(image)):
            image = image.get_image_array()
        image = image / 255
        grayscale_img = np.mean(image, axis=0)[None]
        deriv_img = self.derivative(grayscale_img)
        deriv_img = np.sum(np.abs(deriv_img), axis=0)
        #if not return_:
        return np.exp(- gamma * deriv_img)/2
        #halfg = np.exp(-self.gamma * deriv_img)/2
        #if return_ == 1:
        #    return (- gamma * deriv_img)#/2
        #if return_ == 2:
        #    return deriv_img


    def derivative(
            self, 
            array: np.ndarray
        ) -> np.ndarray:
        """
        Creates the derivative of an input array.

        Args: 
            array: input with dimension: [class, height, width]

        Returns:
            Derivative [2, class, height, width] of the input array.
        """
        c, h, w = array.shape
        if self.debug:
            print(f"derivative matrix shape {self.derivative_matrix.shape}\nright matrix shape {np.transpose(array, (2, 1, 0)).reshape(-1,c).shape}")
        dxy = self.derivative_matrix @ np.transpose(array, (2, 1, 0)).reshape(-1,c)
        dxy = np.transpose(dxy.reshape(2, w, h, c), (0, 3, 2, 1))
        return dxy


    def make_derivative_matrix(
            self, 
            w: int, 
            h: int
        ) -> coo_matrix:
        """
        Creates a matrixform of the first order differencing.

        Args:
            w: width of the image to be differentiated
            h: height of the image to be differentiated

        Returns:
            First order differencing in matrix form.
        """
        def generate_D_matrix(n):
            e = np.ones([2,n])
            e[1,:] = -e[1,:] 
            return sparse.spdiags(e, [0,1], n, n)

        Dy = sparse.kron(sparse.eye(w), generate_D_matrix(h)) 
        Dx = sparse.kron(generate_D_matrix(w),sparse.eye(h)) 

        D = sparse.vstack([Dy, Dx])
        self.derivative_matrix = D
        return D


    def divergence(
            self, 
            array: np.ndarray
        ) -> np.ndarray:
        """
        Creates the divergence of an input array.
        Args: 
            array: input with dimension: [2, class, height, width]
        Returns:
            Divergence [2, class, height, width] of the input array.
        """
        n_classes, height, width = array.shape[1:]
        dxy = self.derivative_matrix.T @ np.transpose(array, (0, 3, 2, 1)).reshape(-1, n_classes)
        dxy = np.transpose(dxy.reshape(width, height, n_classes), (2, 1, 0))
        return -dxy


    @staticmethod
    def projection_kappa(
            xi:  np.ndarray, 
            half_g: np.ndarray,
            smoothing: float = 1e-5
        ) -> np.ndarray:
        """
        Projection |xi_i|<=g/2, Eq.(23). 
        Args:
            xi: input of dimension [2, class, height, width]
            halfg: 1/2 g, initialized by init_halfg(...)
        Returns:
            Projected input xi onto |xi_i|<=g/2.
        """
        norm_xi = np.sqrt(xi[0]**2 + xi[1]**2) / (half_g + smoothing) 
        const = norm_xi > 1.0
        xi[0][const] = xi[0][const] / norm_xi[const] # x
        xi[1][const] = xi[1][const] / norm_xi[const] # y
        return xi

    
    @staticmethod
    def projection_simplex(
            v: np.ndarray,
            smoothing: float = 0e-5
        ) -> np.ndarray:
        """
        Projection onto a simplex.
        
        As described in Algorithm 1 of
        https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
        min_w 0.5||w-v||² st sum_i w_i = z, w_i >= 0
        Args: 
            v: input array of dimension [class, height, width]
        Returns:
            Projection of the input v onto a simplex.
        """
        nc, height, width= v.shape
        # sort v into mu: mu_1 >= mu_2 >= ... mu_p
        v2d = v.reshape(nc, -1)
        mu = np.sort(v2d, axis = 0)[::-1]
        # Find p
        A = np.ones([nc,nc])
        z = 1
        sum_vecs = (np.tril(A) @ mu) - z
        c_vec = np.arange(nc)+1.
        c_vec=np.expand_dims(c_vec, axis=0).T
        cond = (mu - 1/c_vec * sum_vecs) > 0
        cond_ind = c_vec * cond
        p = np.max(cond_ind, axis=0)
        pn =np.expand_dims(p.astype(int)-1,0)
        # Calculate Theta by selecting p-entry from sum_vecs
        theta = 1 / p * np.take_along_axis(sum_vecs, indices=pn, axis=0)
        # Calculate w
        w = v2d-theta
        w[w<0] = 0
        w = w.reshape([nc,height,width])
        tmp = np.clip(v,0.000001,1)
        tmp = tmp / np.sum(tmp, axis=0, keepdims=True)
        return w


    @staticmethod
    def gradient_I(
            img: np.ndarray
        ) -> np.ndarray:
        """
        Computes the gradient of an image. Dims: 2 x c x h x w

        :param img: image of dimensions c x h x w
        """
        diffs_x = np.diff(img, axis = 2)
        diffs_y = np.diff(img, axis = 1)
        
        last_cols_reshaped = np.expand_dims(img[:,:,-1], 2)
        last_rows_reshaped = np.expand_dims(img[:,-1,:], 1)

        dx = np.concatenate((diffs_x, last_cols_reshaped), axis = 2)
        dy = np.concatenate((diffs_y, last_rows_reshaped), axis = 1)

        return np.array([dx, dy])
    
    @staticmethod
    def get_argmax_matrix(
            theta: np.ndarray
        ) -> np.ndarray:
        """
        
        """
        n_classes, height, width = theta.shape
        binary_theta = np.zeros_like(theta)
        # get the index of the dimension with the largest value along the n_class axis
        max_idx = np.argmax(theta , axis=0)
        # set the corresponding value to 1 in the binary array
        y_idx, x_idx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        binary_theta[max_idx, y_idx, x_idx] = 1
        return binary_theta