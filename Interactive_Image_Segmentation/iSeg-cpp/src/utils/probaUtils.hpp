#include "./utils.hpp"

#ifndef ProbaUtils_H
#define ProbaUtils_H

struct{

    // Function for getting the closest scribble point for a given pixel at certain coordinates passed as argument
    // (like this, the function cannot be parallelized, but we could do better)
    // even though the last observation is trivial (cause we cannot parallelize a function that is called over a single pixel)
    // it is still possible to define a function that takes a cv::cuda::GpuMat object as input instead of a given pixel
    // I still need to study better how to parallelize the computations on openCV
    std::map<int, std::vector<int>> findClosestScribblePixelPerClass(
            int xCoord, // x coordinate of the given pixel
            int yCoord, // y coordinate of the given pixel
            std::map<int, std::vector<int>> scribbleMap // scribble map
        ){
        std::map<int, std::vector<int>> closestPoints;
        for (const auto& [class, coordVector] : classNameMap) {
            float minDistance = std::numeric_limits<float>::infinity()
            int xCoordClosest;
            int yCoordClosest;
            for(auto ptCoordVector = coordVector.begin(); ptCoordVector != coordVector.end(); ptCoordVector = ptCoordVector + 2){
                int xCoordScribble = *ptCoordVector;
                int yCoordScribble = *(ptCoordVector + 1);
                float distance = VariousTools.computePixelsDistance(xCoord, xCoordScribble, yCoord, yCoordScribble);
                if(distance < minDIstance){
                    xCoordClosest = xCoordScribble;
                    yCoordClosest = yCoordScribble;
                    minDistance = distance;
                }
            }
            std::vector<int> closestPoint = {xCoordClosest, yCoordClosest};
            closestPoints[class] = closestPoint;
        }
        return closestPoints;
    };
    std::map<int, float> findDistanceToClosestScribble(
            int xCoord, 
            int yCoord, 
            std::map<int, std::vector<int>> closestPixelPerClass
        ){
        std::map<int, float> distanceToClosestPixelPerClass;
        for(const auto& [class, closestPixel]: closestPixelPerClass){
            int xClosest = closestPixel[0];
            int yClosest = closestPixel[1];
            float distance = VariousTools.computePixelsDistance(xCoord, xClosest, yCoord, yClosest);
            distanceToClosestPixelPerClass[class] = distance;
        }
        return distanceToClosestPixelPerClass;
    };

} ProbaFunctions

#endif // ProbaUtils_H