import numpy as np

class Prior(object):
    def __init__(
            self, 
            gamma: float
        ) -> None:
        """
        :param img: image of dimensions c x h x w
        """
        self.gamma = gamma
        
    def _gradient_I(
            self, 
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

    def _gradient(
            self, 
            img -> np.ndarray
        ) -> np.ndarray:
        """
        Computes the term g(x) (eq. 16).

        :param img: image to segment of dimensions c x h x w
        :param self.gamma: float
        """
        grayscale_img = np.mean(img, axis=0)[None]
        abs_gradient_img = np.abs(self._gradient_I(grayscale_img))
        return np.exp(-abs_gradient_img*self.gamma)
    
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
        d_Theta = self._gradient_I(theta) # 2 x c x h x w
        abs_d_Theta = np.abs(d_Theta) # 2 x c x h x w
        g = self._gradient(img, self.gamma) # 2 x h x w
        prod = g*abs_d_Theta # 2 x c x h x w
        return 0.5*np.sum(prod) # scalar
    
    def fit(
            self, 
            img: np.ndarray, 
            theta: np.ndarray
        ) -> float:
        return self._prior_energy(img, theta)