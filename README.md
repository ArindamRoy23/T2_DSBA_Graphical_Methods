# T2_DSBA_Graphical_Methods

This is a repo to complete Grahical Methods project work in the masters of DSBA. 

Attached work on brain image segmentation 
https://github.com/BWGZK/CycleMix


## Datadets

* ScribbleSup: Scribble-Supervised Convolutional Networks for Semantic Segmentation (https://jifengdai.org/downloads/scribble_sup/)
Scribbles provided as XML files

* CycleMix: A Holistic Strategy for Medical Image Segmentation from Scribble Supervision (https://github.com/BWGZK/CycleMix)
Scribbles provided as medical images NII files

## CPP implementation
### Compiler 

* dev-c++ (Windows)
* Gcc (Unix-like)

### Libraries

* OpenCV (https://opencv.org/)
* Armadillo (https://arma.sourceforge.net/)
* ITK (https://itk.org/)


``````
src
    utils
        FileHandling -> scribbles, image reading
        Parallelization -> cuda kernels
    
    Probability (folder)
        class: ProbabilityDensityEstimator
            class: Likelihood
            class: Prior
                class (Likelihood, Prior): Energy
                    class: Primal
                    class: Dual

    Optimization (folder)                  
        class: AlgoOptimization
            class: PrimalDualOptimization
    
    Visualization (folder)
        class VizSegmentation

    main.py
        - loads image, scribbles
        - model
        - computes segmentation



