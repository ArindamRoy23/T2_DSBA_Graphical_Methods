import numpy as np

class PriorEstimator:

    def __init__(self) -> None:
        """ 
        self.derivative_mat # 2, c, h, w
        self.Theta # c, h, w
        self.g
            self.gamma # scalar
            self.derivative_mat # 2, c, h, w
         """
        self.img = np.array([[[1, 4, 2], [5, 3, 2]], [[4, 1, 6], [2, 1, 1]]])

        self.gradient = self._gradient_I(self.img)


    def _gradient_I(self, img):
        diffs_x = np.diff(img, axis = 2)
        diffs_y = np.diff(img, axis = 1)
        
        last_cols_reshaped = np.expand_dims(img[:,:,-1], 2)
        last_rows_reshaped = np.expand_dims(img[:,-1,:], 1)

        dx = np.concatenate((diffs_x, last_cols_reshaped), axis = 2)
        dy = np.concatenate((diffs_y, last_rows_reshaped), axis = 1)

        return np.array([dx, dy])


    def _g(self, img, gamma):
        grayscale_img = np.mean(img, axis=0)[None]
        abs_gradient_img = np.abs(self.gradient(grayscale_img))
        return np.exp(-abs_gradient_img*gamma)
    
    def _prior_energy(self, img, gamma, theta):
        d_Theta = self._gradient_I(theta)
        abs_d_Theta = np.abs(d_Theta)
        g = self._g(img, gamma)
        return 0.5*np.sum(g*abs_d_Theta)