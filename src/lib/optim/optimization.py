from probabilityEstim.priorEstimator import PriorEstimator
from probabilityEstim.probabilityEstimator import ProbabilityDensityEstimator
from probabilityEstim.estimator import Estimator # This is what we need, where we put the likelihood and prior together

class Optimization:

    def __init__(self) -> None:
        self.energy = None

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

    def _dual_update():
        """
        Dual Update Step (eq.28).
        """
        pass


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