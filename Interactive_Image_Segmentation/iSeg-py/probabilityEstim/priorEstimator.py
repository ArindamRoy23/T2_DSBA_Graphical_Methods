import numpy as np

class PriorEstimator:

    def __init__(self) -> None:
        pass
    
    def _gradient_I(self, img):
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


    def _g(self, img, gamma):
        """
        Computes the term g(x) (eq. 16).

        :param img: image to segment of dimensions c x h x w
        :param gamma: float
        """
        grayscale_img = np.mean(img, axis=0)[None]
        abs_gradient_img = np.abs(self.gradient(grayscale_img))
        return np.exp(-abs_gradient_img*gamma)
    
    def _prior_energy(self, img, gamma, theta):
        """
        Computes the prior energy (eq. 21)

        :param img: image to segment of dimensions c x h x w
        :param gamma: float
        :param theta: segmentation of the image of dimensions n x h x w
        """
        d_Theta = self._gradient_I(theta)
        abs_d_Theta = np.abs(d_Theta)
        g = self._g(img, gamma)
        return 0.5*np.sum(g*abs_d_Theta)