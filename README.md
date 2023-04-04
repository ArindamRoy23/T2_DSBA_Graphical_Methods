# T2_DSBA_Graphical_Methods

This is a repo to complete Grahical Methods project work in the masters of DSBA. 

## Datadets

* ScribbleSup: Scribble-Supervised Convolutional Networks for Semantic Segmentation (https://jifengdai.org/downloads/scribble_sup/)
Scribbles provided as XML files

## SVCDSegmenter

The program implements the "Spatially Varying Color Distribution for Interactive, Multilabel Image Segmentation" paper by AUTHOR1 and AUTHOR2. It requires a machine endowed with a cuda-capable GPU. This cli tool allows users to generate a segmentation mask for a given image using the algorithm derived in the mentioned paper, while also computing the dice score of the segmentation. It saves the segmentation mask to the specified folder, with the same name as the target image.
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

