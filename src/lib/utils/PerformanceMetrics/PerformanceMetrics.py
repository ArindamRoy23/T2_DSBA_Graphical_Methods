import numpy as np
import pandas as pd

class PerformanceMetrics(object):
    def __init__(
            self
        ):
        """
        
        """
        ...

    def dice_score(
            self, 
            true_image, 
            generated_image
        ):
        true_image = true_image.flatten()
        generated_image = generated_image.flatten()
        if len(true_image) != len(generated_image):
            raise ValueError('Error in input and output image')
        intersection = 0
        for ind in range(len(true_image)):
            if true_image[ind] == generated_image[ind]:
                intersection = intersection + 1
        return (2 * intersection / (2 * len(true_image)))