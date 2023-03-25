"""

"""
from ..energy.Energy import *
from ..utils.ProbabilityUtils.Divergence import *

from tqdm import tqdm

class SVCDSeg(object):
    """
    
    """
    def __init__(
            self,
            n_classes: int, 
            lambda_: float = 8e-4, 
            gamma: float = 1e-0, 
            alpha: float = 13e-1, # width of spatial kernel (as in the paper)
            sigma: float = 18e-1, # width of chromatic kernel (as in the paper)
            tau_primal: float = 25e-2, 
            tau_dual: float = 5e-1,
            max_iter: int = 500,
            early_stop: bool = False,
            tolerance: float = 1e-5,
            use_tqdm: bool = True,  
            debug: int = 0 
        ) -> None:
        """
        
        """
        self.n_classes = n_classes
        self.lambda_ = lambda_
        self.gamma = gamma            
        self.alpha = alpha
        self.sigma = sigma
        self.debug = debug
        self.tau_primal = tau_primal
        self.tau_dual = tau_dual
        self.max_iter = max_iter
        self.use_tqdm = use_tqdm
        self.tolerance = tolerance
        self.early_stop = early_stop

        self.energy = Energy(
            n_classes = self.n_classes,
            lambda_ = self.lambda_,
            alpha = self.alpha, 
            gamma = self.gamma, 
            sigma = self.sigma,
            debug = self.debug
        )
        # saving fitted likelihood as attribute
        self.fitted_likelihood = None
        # saving fitted prior as an attribute
        self.fitted_prior = None
        # Primal Variables of the optimization 
        self.theta = None
        # Auxiliary Variables of the optimization (same shape as theta)
        self.theta_bar = None
        # Dual Variables of the optimization (How to determine the dimensions idk)
        self.xi = None
        self.prior_history = []
        self.theta_history = []
        self.xi_history = []
        self.theta_bar_history = []
        self.primal_energy_history = [float("inf")]
        self.dual_energy_history = [-float("inf")]
        self.energy_history = [float("inf")]
        # defining iterator for optimization
        self.iterator = range(self.max_iter)

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
        ) -> np.ndarray:
        return divergence(
            array
        )

    def __projection_kappa(
            self, 
            xi:  np.ndarray, 
            fitted_prior: float,
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
        norm_xi = np.sqrt(xi[0]**2 + xi[1]**2) / (fitted_prior + smoothing) 
        const = norm_xi>1.0
        xi[0][const] = xi[0][const] / norm_xi[const] # x
        xi[1][const] = xi[1][const] / norm_xi[const] # y
        return xi

    def __projection_simplex(
            self, 
            v: np.ndarray,
            smoothing: float = 1e-4
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
        pn = np.expand_dims(p.astype(int)-1,0)
        # Calculate Theta by selecting p-entry from sum_vecs
        if self.debug > 1:
            print(f"\ncond_ind {cond_ind}\n")
            print(f"\nsum_vecs {sum_vecs}\n")
            print(f"\npn {pn}\n")
            print(f"\np {p}\n np.take_along_axis(sum_vecs, indices=pn, axis=0)  {np.take_along_axis(sum_vecs, indices=pn, axis=0)}")
            print(f"denominator {p * np.take_along_axis(sum_vecs, indices=pn, axis=0)}")
        theta = 1 / (p * np.take_along_axis(sum_vecs, indices=pn, axis=0) + smoothing)
        # Calculate w
        w = v2d-theta
        w[w<0] = 0
        w = w.reshape([nc,height,width])
        tmp = np.clip(v, 0.000001, 1)
        tmp = tmp / np.sum(tmp, axis=0, keepdims=True)
        return w

    def __set_variables_shape(
            self, 
            target_image: TargetImage 
        ) -> None:
        """

        """
        width, height = target_image.get_image_shape()
        self.theta = np.zeros((self.n_classes, width, height))
        self.theta_bar = np.zeros((self.n_classes, width, height))
        self.xi = np.zeros((2, self.n_classes, width, height))


    def __primal_update(
            self, 
            fitted_likelihood: np.ndarray
        ) -> None:#np.ndarray:
        """
        Primal Update Step (eq.28).
        Args: 
            theta_old: the current segmentation
            xi: dual update result
            tau_primal: primal stepsize
            f: the dataterm create by init_dataterm(...)
        Returns:
            The primal update.
        """
        self.theta_history.append(self.theta)
        self.theta = self.theta + self.tau_primal * (
            self.__divergence(self.xi) - self.lambda_ * fitted_likelihood
            )
        self.theta = self.__projection_simplex(self.theta)
        #return self.theta

    def __dual_update(
            self,
            fitted_prior: float,
        ) -> None:#np.ndarray:
        """
        Dual Update Step (eq.28).
        Args: 
            xi_old: dual variable
            theta_bar: "Primal Dual Hybrid Gradient" variable based on the
                       current segmentation
            tau_dual: dual stepsize
            halfg: 1/2 g, initialized by init_halfg(...)
        Returns:
            The dual update.
        """
        self.xi_history.append(self.xi)
        self.xi = self.xi + self.tau_dual * self.__derivative(
            self.theta_bar
        )
        self.xi = self.__projection_kappa(
            self.xi, 
            fitted_prior
        )
        #return self.xi

    def __auxiliary_update(
            self
        ) -> None:#np.ndarray:
        """
        To be called after the update of primal theta
        """
        self.theta_bar_history.append(self.theta_bar)
        last_theta = self.theta_history[-1]
        self.theta_bar = 2 * self.theta - last_theta
        #return self.theta_bar

    def __init_iterator(
            self
        ) -> None:
        """
        
        """
        if self.use_tqdm and not tqdm:
            raise RuntimeWarning(
                "tqdm logging requested, but tqdm could not be imported."
            )
        elif self.use_tqdm:
            self.iterator = tqdm(self.iterator)
    
    def __update_energy_history(
            self,
            energy: float, 
            primal_energy: float, 
            dual_energy: float
        ) -> None:
        # save energy
        self.energy_history.append(
            energy
        )
        # save primal energy
        self.primal_energy_history.append(
            primal_energy
        )
        # save dual energy
        self.dual_energy_history.append(
            dual_energy
        )

    def __check_tolerance(
            self,
            absolute: bool = True
        ) -> bool:
        energy = self.energy_history[-1]
        last_energy = self.energy_history[-2]
        diff = last_energy - energy
        return diff < 0 or diff <= self.tolerance

    def __fit(
            self,
            target_image: TargetImage, 
            encoded_scribble: EncodedScribble
        ) -> np.ndarray:
        """
        Computes the optimal segmentation (theta) iteratively.
        """
        # initializing the iterator
        self.__init_iterator(

        )
        # setting the shape of the optimization variables
        self.__set_variables_shape(
            target_image
        )

        # compute likelihood 
        self.fitted_likelihood = self.energy.fit_likelihood(
            target_image, 
            encoded_scribble
        )
        # compute prior
        self.fitted_prior = self.energy.fit_prior(
            target_image, 
            self.theta
        )
        if self.debug:
            print(f"\nself.fitted_prior {self.fitted_prior}")
            
        # starting the optimization loop
        for iteration in self.iterator:
            # do dual update
            self.__dual_update(
                self.fitted_prior
            )
            # do primal update
            self.__primal_update(
                self.fitted_likelihood
            )
            # do auxiliary update
            self.__auxiliary_update(

            )
            # compute energy
            energy = self.energy.energy(
                self.theta, 
                target_image, 
                encoded_scribble
            )
            # compute primal energy
            primal_energy = self.energy.primal_energy(
                self.theta, 
                target_image, 
                encoded_scribble
            )
            
            # compute dual energy
            dual_energy = self.energy.dual_energy(
                self.xi,
                target_image, 
                encoded_scribble
            )
            # update energy history
            self.__update_energy_history(
                energy, 
                primal_energy, 
                dual_energy
            )
            self.prior_history.append(self.fitted_prior)
            # update prior 
            self.fitted_prior = self.energy.fit_prior(
                target_image, 
                self.theta
            )
            # check tolerance
            if self.early_stop and self.__check_tolerance():
                raise RuntimeWarning(
                    f"The execution ended after {iteration} iterations as energy was not decreasing enough, possible divergence"
                )
                break
        return self.theta

    def fit(
            self,
            target_image: TargetImage, 
            encoded_scribble: EncodedScribble
        ) -> np.ndarray:
        if self.debug:
            print(f"Segmenting target_image of type {type(target_image)} with scribbles of type {type(encoded_scribble)}")
        return self.__fit(
            target_image,
            encoded_scribble
        )