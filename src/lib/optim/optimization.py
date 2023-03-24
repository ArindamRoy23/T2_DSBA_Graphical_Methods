from ..energy.Energy import *
from ..utils.ProbabilityUtils.Divergence import *


class Optimization:

    def __init__(
            self,
            energy: 
        ) -> None:
        """
        
        """
        self.energy = None

    def __derivative(
            self, 
            array: np.ndarray
        ) -> np.ndarray:
        """
        
        """
        return derivative(
            array
        )

    def __projection_kappa(
            self, 
            xi:  np.ndarray, 
            fitted_prior: float
        ) -> np.ndarray:
        """
        Projection |xi_i|<=g/2, Eq.(23). 
        Args: 
            xi: input of dimension [2, class, height, width]
            halfg: 1/2 g, initialized by init_halfg(...)
        Returns:
            Projected input xi onto |xi_i|<=g/2.
        """
        norm_xi = np.sqrt(xi[0]**2 + xi[1]**2) / fitted_prior 
        const = norm_xi>1.0
        xi[0][const] = xi[0][const] / norm_xi[const] # x
        xi[1][const] = xi[1][const] / norm_xi[const] # y
        return xi

     def __projection_simplex(
            self, 
            v: np.ndarray
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
        tmp = np.clip(v, 0.000001, 1)
        tmp = tmp / np.sum(tmp, axis=0, keepdims=True)
        return w

    def _primal_energy():
        """
        Primal Energy (eq 29)
        """
        pass

    def _dual_energy():
        """
        Dual Energy (eq 31).
        """
        pass

    def _primal_update():
        """
        Primal Update Step (eq.28).
        """
        pass

    
    def __dual_update(
            self, 
            xi_old: np.ndarray, 
            theta_bar: np.ndarray, 
            tau_dual: float, 
            halfg: float
        ):
        """Dual Update Step (eq.28).
        Args: 
            xi_old: dual variable
            theta_bar: "Primal Dual Hybrid Gradient" variable based on the
                       current segmentation
            tau_dual: dual stepsize
            halfg: 1/2 g, initialized by init_halfg(...)
        Returns:
            The dual update.
        """
        xi = xi_old + tau_dual * self.derivative(theta_bar)
        return self.projection_kappa(
            xi, 
            halfg
        )


    def run(self):
        """
        Computes the optimal segmentation (theta) iteratively.
        """
        energies = []
        primal_energies = []
        dual_energies = []
        for iteration in num_iter:
            # Store old values
            pass
            # Do updates -> call update-functions
            
            # Save energies
            
        return optimal_theta