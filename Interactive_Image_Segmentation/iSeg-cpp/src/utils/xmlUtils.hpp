/*

Definition of utilities for parsing the XML document
in which the scribbles are encoded for each image in 
the Pascal dataset

Structure of the XML document:
    root tag: Annotation.
        first layer child: polygon (annotationTagName)
            second layer childer: tag (classTagName)
            second layer childer: point
                third layer childer: X (X coordinate of the point)
                third layer childer: Y (Y coordinate of the point)
        first layer child: polygon
            etc etc 
*/

#include "./utils.hpp"

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
            const std::string& annotationTagName, // "polygon" is the tag name for the annotation
            const std::string& classTagName // "tag" is the tag name for the class
        ){
        /*
        Dependencies of this function (for debugging):
            this -> readXLMToString
        */
        std::string xlm = this -> readXLMToString(XLMPath);
        rapidxml::xml_document<> doc;
        doc.parse<0>(&xml[0]);
        rapidxml::xml_node<>* root = doc.first_node(); // annotation node (root)
        for (rapidxml::xml_node<>* annotationNode = root->first_node(annotationTagName); annotationNode; annotationNode = annotationNode->next_sibling(annotationTagName)){ // polygon node (children)
            // each polygon will only have one "tag" childer node, no need for looping here
            rapidxml::xml_node<>* tagNode = node -> first_node(classTagName);
            std::string tagValue = tagNode->value();
            labels.push_back(tagValue);
        }
        return labels; 
    };

    // Finds all distinct classes in a directory of XLM files and maps them to a placeholder
    std::map<std::string, int> getDistinctClasses(
            const std::string& pathToFolder,
            const std::string& annotationTagName, // "polygon" is the tag name for the annotation
            const std::string& classTagName // "tag" is the tag name for the class 
        ){
        /*
        Dependencies of this function (for debugging):
            this -> getTagValuesFromXLMString
        */
        std::vector<std::string> XLMFilePaths = DirTools.getFilesPath(pathToFolder);
        std::map<std::string, int> classNameMap;
        int placeholder = 0;
        for(auto ptXLMFilePath = XLMFilePaths.begin(); ptXLMFilePath != XLMFilePaths.end(); ++ptXLMFilePath){
            std::string XLMFilePath = *ptXLMFilePath;
            std::vector<std::string> tagValues = this -> getTagValuesFromXLMString(XLMFilePath, annotationTagName, classTagName);
            for(auto ptTagValue = tagValues.begin(); ptTagValue != tagValues.end(); ++ptTagValue){
                std::string tagValue = *ptTagValue;
                classNameMap[tagValue] = placeholder;
            }
        }
        return classNameMap;
    };

    // Maps each class to a set of pixels (scribble annotation) from the XLM file
    // The coordinate of each set of pixels will be given as a continuous vector
    // i.e.: [x1, y1, x2, y2, ...] 
    std::map<int, std::vector<int>> mapPixelsToClass(
            const std::string XLMFilePath, 
            std::map<std::string, int>& classMap, // map to be created with IterTools.mapClassesToInt utility
            const std::string& annotationTagName, // "polygon" is the tag name for the annotation
            const std::string& classTagName // "tag" is the tag name for the class 
            const std::string& pointTagName,
        ){
        /*
        Dependencies of this function (for debugging):
            this -> readXLMToString
            iterTools.mapClassesToInt (not necessary a dependence)
        */
        std::map<int, std::vector<int>> pixelMap;
        std::string xlmString = this -> readXLMToString(XLMFilePath);
        rapidxml::xml_document<> doc;
        doc.parse<0>(&xml[0]);
        rapidxml::xml_node<>* root = doc.first_node();
        for(rapidxml::xml_node<>* annotationNode = root -> first_node(annotationTagName); annotationNode; annotationNode = annotationNode -> next_sibling(annotationTagName)){
            rapidxml::xml_node<>* tagNode = annotationNode -> first_node(classTagName);
            std::string tagName = tagNode -> value();
            int labelId = classMap[tagName];
            std::vector<int> labelPoints = pixelMap[labelId];    
            for(rapidxml::xml_node<>* pointNode = tagNode -> first_node(pointTagName); pointNode; pointNode = pointNode -> next_sibling(pointTagName)){
                rapidxml::xml_node<>* xNode = pointNode -> first_node("X");
                int xCoord = xNode -> value();
                rapidxml::xml_node<>* yNode = pointNode -> first_node("Y");
                int yCoord = yNode -> value();
                labelPoints -> push_back(xCoord);
                labelPoints -> push_back(y_coord);
            }
        }
        return pixelMap;
    };

} XLMParser;

#endif // XLMParser_H