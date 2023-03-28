import numpy as np
import pandas as pd


class performance_metrics:
    def __init__(self):
        pass

    def dice(self, true_image, generated_image):
        true_image = true_image.flatten()
        generated_image = generated_image.flatten()
        if len(true_image) != len(generated_image):
            print('Error in input and output image')
        else:
            intersection = 0
            for ind in range(len(true_image)):
                if true_image[ind] == generated_image[ind]:
                    intersection = intersection + 1
            return (2 * intersection / (2 * len(true_image)))


if __name__ == '__main__':
    from PIL import Image

    y_pred = np.asarray(Image.open('Sample_Image.png'))
    y_true = np.asarray(Image.open('Sample_Image_mod.png'))
    print(y_pred)
    performance_metrics = performance_metrics()
    print(performance_metrics.dice(y_pred,y_pred))
