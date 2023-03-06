/*

Definition of the utils for reading Images
using the OpenCV library. 
Images are provided either directly in a Jpeg format 
and then read directly in openCV as cv::Mat or 
cv::cuda::GpuMat objects or first read through the 
ITK (scientific/medical image processing) library and then converted to
cv::Mat objects.

*/


#include "./utils.hpp"

#ifndef ImageIOFunctions_H
#define ImageIOFunctions_H

struct{
    // Loads Jpeg Image on CPU memory as cv::Mat object
    cv::Mat loadJpegImageCPU(
            const std::string& pathToImage
        ){
        cv::Mat loadedImage = cv::imread(pathToImage);
        return loadedImage;
    };

    // Loads Jpeg Image on GPU memory as cv::cuda::GpuMat object
    cv::cuda::GpuMat loadJpegImageGPU(
            const std::string& pathToImage
        ){
        cv::Mat loadedImage = this -> loadJpegImageCPU(pathToImage);
        cv::cuda::GpuMat loadedImageGPU(loadedImage);
        return loadedImageGPU;
    };

    // Loads NIfTI image on CPU memory as cv::Mat object
    cv::Mat loadNIfTIImageCPU(
            const std::string& pathToImage
        ){
        typedef itk::Image<float, 3> ImageType;
        typedef itk::ImageFileReader<ImageType> ReaderType;
        ReaderType::Pointer reader = ReaderType::New();
        reader->SetFileName(pathToImage);
        reader->Update();
        cv::Mat loadedImage(
            reader -> GetOutput() -> GetRequestedRegion().GetSize()[1],
            reader -> GetOutput() -> GetRequestedRegion().GetSize()[0],
            CV_32F,
            reader -> GetOutput() -> GetBufferPointer(),
            reader -> GetOutput() -> GetStride()[1] * sizeof(float)
        )
        return loadedImage;
    };

    // Loads NIgfTI image on GPU memory as cv::cuda::GpuMat object
    cv::cuda::GpuMat loadNIfTIImageGPU(
            const std::string& pathToImage
        ){
        cv::Mat loadedImage = this -> loadNIfTIImageCPU(pathToImage);
        cv::gpu::GpuMat loadedImageGPU(loadedImage);
        return loadedImageGPU;
    }

} ImageFunctions;

#endif // ImagIOFunctions_H

