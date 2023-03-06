/*

Definition of include statements and of utilities
for Iteration, File Listing 

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


#ifndef Functions_H
#define Functions_H

#ifndef DirectoryItertools_H
#define DirectoryItertools_H

struct{

    // Returns a vector containing all the files in the folder passed as argument
    std::vector<std::string> getFilesPath(
            const std::string& pathToFolder
        ){
        std::vector<std::string> filesPath;
        for (const auto& entry : std::filesystem::directory_iterator(pathToFolder)) {
            if (entry.is_regular_file()){
                filesPath.push_back(entry.path().string());
            }
        }
        return filesPath;
    };

} DirTools;

#endif // DirectoryItertools_H

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

    // Maps each distinct class to an integer
    std::map<std::string, int> mapClassesToInt(
            const std::map<std::string, int>& distinctClasses
        ){
        std::map<std::string, int> classMap = IterTools.createIndexMap(distinctClasses);
        return classMap;
    };


} IterTools;

#endif // Itertools_H

#ifndef VariousTools_H
#define VariousTools_H

struct{

    float computePixelsDistance(
            int xCoord1, 
            int xCoord2,
            int yCoord1, 
            int yCoord2 
        ){
            int xDist = xCoord1 - xCoord2;
            int yDist = yCoord1 - yCoord2;
            int totDist = xDist*xDist + yDist*yDist;
            float dist = sqrt(static_cast<float>(totDist));
            return dist;
        }

} VariousTools;

#endif // VariousTools_h

#endif // Functions_H