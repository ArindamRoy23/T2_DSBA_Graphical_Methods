/*

Header defining the class headers for a probabilityEstimator
This class should be used for handling the computation relative to 
the kernel density estimation of the joint probability over each image
and should then be used by the model class in order to learn and make inference

For how to encode the scribbles, I think std::map<int, std::vector<int>> is the most convenient representation 
Considering that we need to compute the set S_i for each class i (according to the paper).
It will be based on this set that we will estimate the Joint Conditional Probability P(I, x | u).
This will be used in the computation of the loss function for the primal-dual formulation of the 
optimization problem presented in the paper (the term f_i, primal part).

Important note (from ChatGPT):
By defining a pixel-level function/method, you can abstract away the details of the computation 
and make it easier to parallelize. In the CPU implementation, you would call this function 
for each pixel in a nested loop. In the GPU implementation, you would call this function for each pixel in the kernel

*/
#include "rapidxml.hpp"
#include "rapidxml_utils.hpp"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <thread>
#include <string>
#include <vector>
#include <filesystem>
#include <armadillo> // namespace arma
#include <opencv2/opencv.hpp> // namespace cv
#include <itkImage.h> // namespace itk

#ifndef ProbEstimator_H
#define ProbEstimator_H

template <typename... Args>

// Basic definition, still need a lot but more or less this is what it should do 
class KernelProbabilityEstimator{
    private:
        // Number of classes to define the distribution over 
        int nClasses; // number of classes to estimate the density over
        // Kernel function to be used (can take as many arguments of any type)
        std::vector<float> (*kernelFunction)(Args...); 
    public:
        // Constructor
        KernelProbabilityEstimator(
                int nClasses, 
                std::vector<float> (*kernelFunction)(Args...)
            ){
            this -> nClasses = nClasses;
            this -> kernelFunction = kernelFunction;
        }
        // Getter of nClasses
        int getNClasses(){
            return this -> nClasses;
        }
        // Getter of Kernel Function
        std::vector<float> (*getKernelFunction())(
                Args...
            ) const {
            return this -> kernelFunction;
        }
        // Applies kernel to pixel on a pixel of a cv::Mat object 
        // output should be of size 1*1*nClasses (we should have a channel per class)
        cv::Mat applyPixelwiseKernel(
            int, // xCoord
            int, // yCoord
            float, // width of the kernel (to be computed before calling the method)
            cv::Vec3b, // the pixel which to compute the kernel on
            std::map<int, std::vector<int>> // the map of the scribble pixel (to be already provided)
            );
        // Estimates joint density P(I, x | u) from image and scribble map on CPU
        cv::Mat estimateJointDensity(
                cv::Mat, 
                std::map<int, std::vector<int>>,
                float, // sigma (for computing chromo kernel)
                float // alpha (for computing the distance kernel)
            ); 
        // Estimates joint density P(I, x | u) from image and scribble map on GPU
        cv::cuda::GpuMat estimateJointDensity(
                cv::cuda::Mat, 
                std::map<int, std::vector<int>>,
                float, // sigma (for computing chromo kernel)
                float // alpha (for computing the distance kernel)
            );
        // Estimates joint density P(I, x | u) from image and scribble image on CPU 
        cv::Mat estimateJointDensity(
                cv::Mat, 
                cv::Mat, 
                float, // sigma (for computing chromo kernel)
                float // alpha (for computing the distance kernel)
            );
        // Estimates joint density P(I, x | u) from image and scribble image on GPU
        cv::cuda::GpuMat estimateJointDensity(
                cv::cuda::GpuMat, 
                cv::cuda::GpuMat,
                float, // sigma (for computing chromo kernel)
                float // alpha (for computing the distance kernel)
            );
};

#endif // ProbEstimator_H