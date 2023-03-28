"""

"""
from ..energy.Energy import *
from ..utils.SVCDUtils.SVCDUtils import *

from tqdm import tqdm

class SVCDSeg(object):
    """
    
    """
    def __init__(
            self,
            n_classes: int, 
            lambda_: float = 8e-4, 
            gamma: float = 5e-0, 
            alpha: float = 18e-1, # width of spatial kernel (as in the paper)
            sigma: float = 13e-1, # width of chromatic kernel (as in the paper)
            tau_primal: float = 25e-2, 
            tau_dual: float = 5e-1,
            max_iter: int = 5000,
            early_stop: bool = False,
            tolerance: float = 1e-5,
            use_tqdm: bool = True,  
            debug: int = 0,
            return_: int = 0,
            transpose: bool = False 
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
        self.return_ = return_
        self.transpose = transpose
        self.energy = Energy(
            n_classes = self.n_classes,
            lambda_ = self.lambda_,
            alpha = self.alpha, 
            gamma = self.gamma, 
            sigma = self.sigma,
            debug = self.debug,
            return_ = self.return_,
            transpose = self.transpose
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
        # halfg
        self.halfg = None
        self.prior_history = []
        self.theta_history = []
        self.xi_history = []
        self.theta_bar_history = []
        self.primal_energy_history = [float("inf")]
        self.dual_energy_history = [-float("inf")]
        self.energy_history = [float("inf")]
        # defining iterator for optimization
        self.iterator = range(self.max_iter)
        # initializing utils object
        self.utils = SVCDUtils()


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


    def __check_tolerance(
            self,
            absolute: bool = True
        ) -> bool:
        """
        
        """
        energy = self.energy_history[-1]
        last_energy = self.energy_history[-2]
        diff = last_energy - energy
        return diff < 0 or diff <= self.tolerance


    def __primal_update(
            self,
        ) -> None:
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
            self.utils.divergence(self.xi) - self.lambda_ * self.fitted_likelihood
            )
        self.theta = self.utils.projection_simplex(self.theta)
        assert np.all(np.sum(self.theta, axis = 0) - 1 < 1e-5), "Not all elements of the array are 1"
        


    def __dual_update(
            self
        ) -> None:
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
        self.xi = self.xi + self.tau_dual * self.utils.derivative(
            self.theta_bar
        )
        self.xi = self.utils.projection_kappa(
            self.xi, 
            self.halfg
        )

    def __auxiliary_update(
            self
        ) -> None:
        """

        """
        self.theta_bar_history.append(self.theta_bar)
        current_theta = self.theta
        last_theta = self.theta_history[-1]
        self.theta_bar = 2 * current_theta - last_theta


    def __update_energy_history(
            self,
            energy: float, 
            primal_energy: float, 
            dual_energy: float
        ) -> None:
        """
        
        """
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


    def __fit_preproc(
            self, 
            target_image: TargetImage, 
            encoded_scribble: EncodedScribble
        ) -> None:
        """
        
        """
        # get image size
        image_shape = target_image.get_image_shape()
        # initialize matrix for utils derivation
        self.utils.derivative_matrix = self.utils.make_derivative_matrix(
            *image_shape[1: :-1]
        )
        # initialize matrix for energy.utils derivation
        self.energy.utils.derivative_matrix = self.utils.make_derivative_matrix(
            *image_shape[1: :-1]
        )
        # compute halfg
        self.halfg = self.utils.init_halfg(
            target_image,
            gamma = self.gamma
        )
        # initializing the iterator
        self.__init_iterator(

        )
        # setting the shape of the optimization variables
        self.__set_variables_shape(
            target_image
        )
        # compute likelihood 
        self.fitted_likelihood = self.energy.likelihood.fit(
            target_image, 
            encoded_scribble,        
            normalize = True,
            neg_log = True
        )
        if self.debug > 1:
            print(f"\nself.fitted_prior {self.fitted_prior}")
        

    def __fit_step(
            self,
            target_image: TargetImage, 
            encoded_scribble: EncodedScribble
        ) -> np.ndarray:
        """
        
        """
        if self.debug > 1:
            print(f"__fit_step")
        # do dual update
        self.__dual_update(
        
        )
        if self.debug > 1:
            print(f"dual update done")
        # do primal update
        self.__primal_update(
        
        )
        if self.debug > 1:
            print(f"primal update done")
        # do auxiliary update
        self.__auxiliary_update(
            
        )
        if self.debug > 1:
            print(f"auxiliary update done")
        # compute energy
        energy = self.energy.energy(
            self.theta, 
            self.halfg,
            target_image, 
            encoded_scribble
        )
        if self.debug > 1:
            print(f"energy computed")
        # compute primal energy
        primal_energy = self.energy.primal_energy(
            self.theta,
            self.halfg, 
            target_image, 
            encoded_scribble
        )
        if self.debug > 1:
            print(f"primal energy computed")
        # compute dual energy
        dual_energy = self.energy.dual_energy(
            self.xi,
            target_image, 
            encoded_scribble
        )
        if self.debug > 1:
            print(f"dual energy computed")
        # update energy history
        self.__update_energy_history(
            energy, 
            primal_energy, 
            dual_energy
        )
        # check tolerance
        if self.early_stop and self.__check_tolerance():
            raise RuntimeWarning(
                f"The execution ended after {iteration} iterations as energy was not decreasing enough, possible divergence"
            )
            return 1


    def __fit(
            self,
            target_image: TargetImage, 
            encoded_scribble: EncodedScribble
        ) -> np.ndarray:
        """
        
        """
        # initializing object for fit
        self.__fit_preproc(
            target_image, 
            encoded_scribble
        )
        # starting the optimization loop
        for iteration in self.iterator:
            stop = self.__fit_step(
                target_image, 
                encoded_scribble
            )
            if stop:
                break
        return self.theta


    def fit(
            self,
            target_image: TargetImage, 
            encoded_scribble: EncodedScribble
        ) -> np.ndarray:
        """
        
        """
        if self.debug > 1:
            print(f"Segmenting target_image of type {type(target_image)} with scribbles of type {type(encoded_scribble)}")
        return self.__fit(
            target_image,
            encoded_scribble
        )