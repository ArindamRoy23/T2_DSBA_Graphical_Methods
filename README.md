# T2_DSBA_Graphical_Methods

This repository contains the code implementation of the proposed method for interactive multi-label segmentation, which explicitly takes into account the spatial variation of color distributions. This is built as a part of the graphical methods project for master 2 in Data Science and Business Analytics. 

It requires a machine endowed with a cuda-capable GPU. This cli tool allows users to generate a segmentation mask for a given image using the algorithm derived in the mentioned paper, while also computing the dice score of the segmentation.

## Abstract
Interactive image segmentation involves the process of manually specifying the regions of interest in an image. In this project, the proposed method estimates a joint distribution over color and spatial location using a generalized Parzen density estimator applied to each user scribble. This likelihood is then incorporated in a Bayesian MAP estimation approach to multi-region segmentation which in turn is optimized using recently developed convex relaxation techniques.


## Usage
To use the materials in this repository, you can clone the repository to your local machine using Git:
git clone https://github.com/ArindamRoy23/T2_DSBA_Graphical_Methods.git

To run the segmentation on an image, please tune and run main.py


### Arguments: 
Arguments

The following arguments can be passed to the program:

*    ```-i```, ```--target-image```: Input image file (required)
*    ```-s```, ```--scribble```: Scribble file (required)
*    ```-o```, ```--output-file```: Output file (required)
*    ```-l```, ```--lambda```: Lambda parameter for the model (default=8e-4)
*    ```-g```, ```--gamma```: Gamma parameter for the model (default=5e-0)
*    ```-a```, ```--alpha```: Alpha parameter for the model (default=18e-1)
*    ```-s```, ```--sigma```: Sigma parameter for the model (default=13e-1)
*    ```-tp```, ```--tau-primal```: Tau Primal parameter for the model (default=25e-2)
*    ```-td```, ```--tau-dual```: Tau Dual parameter for the model (default=5e-1)
*    ```-m```, ```--max-iter```: Maximum number of iterations for the model (default=1500)
*    ```-es```, ```--early-stop```: Whether to use early stopping for the model (default=False)
*    ```-tl```, ```--tolerance```: Improvement tolerance for the model (default=1e-5)
*    ```-ut```, ```--use-tqdm```: Whether to use tqdm for the model (default=True)

### Packages required 

Following packages are required for running this code. Anaconda environment is recommended.:
* matplotlib==3.5.1
* scikit-learn==1.0.2
* numba==0.54.1
* cupy-cuda113==9.5.0
* pandas==1.4.0
* numpy==1.22.2

## Likelihood function 
This function aims to assign each pixel in an image to a specific class based on its color and location. The likelihood function, which estimates the joint probability of observing a color and location given a class, is formulated using a Gaussian kernel estimator. The prior function is chosen to favor segmentation regions with shorter boundaries. Together, these functions are used to estimate the posterior probability for a given mapping. To find the optimal mapping, the problem is formulated in its variational formulation. 

The python implementation of the likelihood function can be found in lib/probabilityEstim/Likelihood.py
The python implementation of the energies can be found in lib/energy/energy.py

## Performance metric 
The Dice score, also known as the Sørensen–Dice coefficient, is a similarity metric used to evaluate the performance of binary image segmentation tasks. It measures the overlap between the predicted and ground truth segmentations, ranging from 0 to 1, where 0 indicates no overlap and 1 indicates a perfect overlap. This project was evaluated on dice scores. 

Reference: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
The python implementation of this function can be found in Scoring.ipynb

## Results
### Dataset
PASCAL dataset (Pattern Analysis, Statistical Modelling and Computational Learning) is a well-known image dataset used for object recognition tasks in computer vision research. The dataset contains a set of natural images with pixel-level object annotations for a wide range of object categories. The dataset was created as part of the PASCAL Visual Object Classes challenge, which was a yearly competition aimed at advancing the state-of-the-art in object recognition.

The PASCAL dataset consists of more than 11,000 images, each of which has been manually labeled with object annotations. The images are drawn from a variety of sources, including web searches, Flickr, and personal collections. The object annotations include the location and size of each object in the image, as well as a label indicating the object category. The dataset includes a total of 20 object categories, including animals, vehicles, and household objects.

The PASCAL dataset has been widely used in computer vision research, and has served as the basis for many advances in object recognition, including the development of new algorithms for object detection, segmentation, and classification. In addition, the dataset has been used as a benchmark for evaluating the performance of different computer vision systems.

Overall, the PASCAL dataset has played a significant role in advancing the field of computer vision, and continues to be an important resource for researchers working on object recognition and related tasks.

The results are stored in the plots folder, stored for a few sample images with different values of lambda.  

# Contributors
* Lorenzo Consoli
* Oskar Girardin
* Arindam Roy

