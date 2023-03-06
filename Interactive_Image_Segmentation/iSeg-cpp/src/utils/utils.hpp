/*

Definition of function and utilities for 
reading and handling images, scribbles and classes

struct and member functions overview:

struct ImageFunctions (provides functions for reading images):
    loadJpegImageCPU
    loadJpegImageGPU
    loadNIfTIImageCPU
    loadNIfTIImageGPU

struct ScribbleFunctions (provides function for reading scribbles):

struct XLMParser (provides functions for parsing XLM files):
    readXLMToString
    getTagValuesFromXLMString
    encodeClassesToIntegers

*/
#include "tinyxml2.h"
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


#ifndef Functions_H
#define Functions_H

#ifndef DirectoryItertools_H
#define DirectoryItertools_H

struct{

    // Gets the vector of files inside a directory
    std::vector<std::string> getFilesPath(
            const std::string& pathToFolder
        ){
        std::vector<std::string> filesPath;
        for (const auto& entry : std::filesystem::directory_iterator(pathToFolder)) {
            if (entry.is_regular_file()) {
            filesPath.push_back(entry.path().string());
            }
        }
        return filesPath;
    };

} DirTools;

#endif // DirectoryItertools_H

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

#ifndef XLMParser_H
#define XLMParser_H

struct{

    // Reads XLM to string
    std::string readXLMToString(
            const std::string& XLMPath
        ){
        std::ifstream file(XLMPath);
        std::string xml;
        file.seekg(0, std::ios::end);
        xml.reserve(file.tellg());
        file.seekg(0, std::ios::beg);
        xml.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        return xlm;
    };

    // Gets a vector containing the Label present in a XLM file
    std::vector<std::string> getTagValuesFromXLMString(
            const std::string& XLMPath, 
            const std::string& classTagName
        ){
        std::string xlm = this -> readXLMToString(XLMPath);
        rapidxml::xml_document<> doc;
        doc.parse<0>(&xml[0]);
        rapidxml::xml_node<>* root = doc.first_node();
        for (rapidxml::xml_node<>* node = root->first_node(classTagName); node; node = node->next_sibling(classTagName)) {
            labels.push_back(node->value());
        }
        return labels; 
    };

    // Finds all distinct classes in a directory of XLM files and maps them to a placeholder
    std::map<std::string, int> getDistinctClasses(
            const std::string& pathToFolder, 
            const std::string& classTagName
        ){
        std::vector<std::string> XLMFilePaths = DirTools.getFilesPath(pathToFolder);
        std::map<std::string, int> classNameMap;
        int placeholder = 0;
        for(auto ptXLMFilePath = XLMFilePaths.begin(); ptXLMFilePath != XLMFilePaths.end(); ++ptXLMFilePath){
            std::string XLMFilePath = *ptXLMFilePath;
            std::vector<std::string> tagValues = getTagValuesFromXLMString(XLMFilePath, classTagName);
            for(auto ptTagValue = tagValues.begin(); ptTagValue !+ tagValues.end(); ++ptTagValue){
                std::string tagValue = *ptTagValue;
                classNameMap[tagValue] = placeholder;
            }
        }
        return classNameMap;
    };

    // Maps each distinct class to an integer
    std::map<std::string, int> mapClassesToInt(
            const std::map<std::string, int>& distinctClasses
        ){
        std::map<std::string, int> classMap = createIndexMap(distinctClasses);
        return classMap;
    };

    // Maps each class to a set of pixels (scribble annotation) from the XLM file 
    std::map<int, std::vector<int>> mapPixelsToClass(
            const std::string XLMFilePath, 
            std::map<std::string, int>& classMap,
            const std::string& classTagName,
            const std::string& pointTagName,
        ){
        std::string xlmString = readXLMToString(XLMFilePath);

    };

} XLMParser,

#endif // XLMParser_H

#ifndef ScribbleIOFunctions_H
#define ScribbleIOFunctions_H

struct{



} ScribbleFunctions;

#endif // ScribbleIOFunctions_H

#ifndef IterTools_H
#define IterTools_H

struct{

    std::map<std::string, int> createIndexMap(
            const std::map<std::string, int>& classNameMap
        ){
        std::map<std::string, int> indexMap;
        int index = 0;
        for (const auto& [key, value] : classNameMap) {
            indexMap[key] = index;
            index++;
        }
        return indexMap;
    };

} IterTools;

#endif // Itertools_H
#endif // Functions_H