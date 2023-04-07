"""

"""

from src.parser.SVCDArgParser import *
from src.experiments.Experiments import *

def main():
    """
    
    """
    SVCDParser = SVCDArgParser()
    parsed_arguments = SVCDParser()
    experiment = Experiment()


    target_image_path = parsed_arguments.target_image
    scribble_path = parsed_arguments.scribble
    lambda_ = parsed_arguments.l
    gamma = parsed_arguments.gamma
    alpha = parsed_arguments.alpha
    sigma = parsed_arguments.sigma
    tau_primal = parsed_arguments.tau_primal
    tau_dual = parsed_arguments.tau_dual
    max_iter = parsed_arguments.max_iter
    early_stop = parsed_arguments.early_stop
    tolerance = parsed_arguments.tolerance
    use_tqdm = parsed_arguments.use_tqdm
    

    segmenter = experiment.run_segmentation(
        target_image_path, 
        scribble_path, 
        lambda_ = lambda_,
        gamma = gamma,
        alpha = alpha, 
        sigma = sigma, 
        tau_primal = tau_primal, 
        tau_dual = tau_dual, 
        max_iter = max_iter, 
        early_stop = early_stop, 
        tolerance = tolerance, 
        use_tqdm = use_tqdm
    )

    final_segmentation = np.argmax(segmenter.theta_t1, axis = 0).transpose(1, 0)

    image_id = experiment.find_image_id(target_image_path)
    
    experiment.save_segmentation_mask(segmentation_mask, image_id)