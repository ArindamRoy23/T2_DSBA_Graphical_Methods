#include "./probaEst.hpp"
#include "../utils/probaUtils.hpp"
#include "../utils/utils.hpp"


cv::Mat KernelProbabilityEstimator::applyPixelwiseKernel(
        int xCoord, // xCoord
        int yCoord, // yCoord
        float kernelWidth, // width of the kernel (to be computed before calling the method)
        cv::Vec3b pixel, // the pixel which to compute the kernel on
        std::map<int, std::vector<int>> scribbleMap // the map of the scribble pixel (to be already provided)
    ){
    // write function here (also we need to define the function for computing the gaussian kernel)
    // We also need to write the code to map each scribble element to the color intensities (for each channel)

};
cv::Mat KernelProbabilityEstimator::estimateJointDensity(
        cv::Mat image, 
        std::map<int, std::vector<int>> scribbleMap,
        float sigma,
        float alpha
    ){
    /*
    Dependencies of this function (for debugging):
        ProbaUtils.findClosestScribblePixelPerClass
        ProbaFunctions.findDistanceToClosestScribble
        this -> applyPixelwiseKernel (CPU)
    */
    int width = image.cols; 
    int height = image.rows;
    for(int xPixel = 0; xPixel < width; xPixel++){
        for(int yPixel = 0; yPixel < height; yPixel++){
            // retrieving each pixel
            cv::Vec3b pixel = image.at<cv::Vec3b>(xPixel, yPixel)
            // find the closest scribble point (for each class) to the current pixel
            std::map<int, std::vector<int>> closestPixelPerClass = ProbaFunctions.findClosestScribblePixelPerClass(xPixel, yPixel, scribbleMap);
            // find the distance to the closes scribble point (for each class) to the current pixel (for computing the width of the spatial kernel)
            std::map<int, float> distanceToClosestPixelPerClass = ProbaFunctions.findDistanceToClosestScribble(xPixel, yPixel, closestPixelPerClass);
            // continue here and finish following steps:
            // 1: compute spatial kernel for each class and each scribble pixel (KernelProbabilityEstimator::applyPixelwiseKernel)
            // 2: compute chromo kernel for each class and each scribble pixel (KernelProbabilityEstimator::applyPixelwiseKernel)
            // 3: multiply the two for each class and each scribble pixel
            // 4: sum the result of point 3 over each scribble pixel 
            // 5: average the result over each class to obtain the estimate of P(I, x | u)
            // For doing these passages We also need to write the code to map each 
            // scribble element to their chromatic intensities (for each channel)
            // This is needed to compute the chromatic kernel at point 2 (actually this is the first thing we need to do)

        }
    } 
};

cv::cuda::GpuMat KernelProbabilityEstimator::estimateJointDensity(
        cv::cuda::GpuMat image, 
        std::map<int, std::vector<int>> scribbleMap,
        float sigma,
        float alpha
    ){
    
};

cv::Mat KernelProbabilityEstimator::estimateJointDensity(
        cv::Mat image, 
        cv::Mat scribbleImage,
        float sigma,
        float alpha

    ){
    int width = image.cols; 
    int height = image.rows;
};

cv::cuda::GpuMat KernelProbabilityEstimator::estimateJointDensity(
        cv::cuda::GpuMat image, 
        cv::cuda::GpuMat scribbleImage,
        float sigma,
        float alpha
    ){
    
        
};