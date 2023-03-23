""""

Interface for accessing the Image and the scribble to do the segmentation with.

These only two classes that should be imported outside the ./ folder

They will be respectively be initialized as follows:

    Pascal Dataset (Jpeg):
        image = TargetImage(image_path)
        scribble = EncodedScribble(scribble_path)
    CycleMix Dataset (NIfTI):
        image = TargetImage(image_path, is_jpeg = False)
        scribble = EncodedScribble(scribble_path, is_xml = False)

In both cases, it is possible to get the actual content of the objects as follows:
    
    image_tensor = image.get_image_tensor()
    scribble_dictionary = get_encoded_scribble()

"""

from .FileHandlingProtected import *

class TargetImage(NIfTIImageReader, JpegImageReader):
    """
    TargetImage(NIfTIImageReader, JpegImageReader):
        Initializes target image, whether it is a Jpeg or a NIfTI file 
        (as specified by the is_jpeg attribute)

    This is the ultimate object with which we will interact later
    """
    def __init__(
            self,
            path_to_file: str,
            is_jpeg: bool = True,
        ) -> None:
        """
        __init__(self, path_to_file):
            calls parent class initializer
        """
        self.is_jpeg = is_jpeg
        super(TargetImage, self).__init__(path_to_file)
    
    def get_image_array(
            self
        ) -> np.ndarray:
        """
        get_image_array(self):
            returns the underlying image as a np.ndarray object
        """
        image_array = JpegImageReader.get_image_array(self) if self.is_jpeg\
                else NIfTIImageReader.get_image_array(self)
        return image_array
    
    def get_image_shape(
            self
        ) -> tuple:
        """
        get_image_shape(self):
            gets the shape of the underlying image array.
        """
        image_array = self.get_image_array()
        return tuple(image_array.shape[1:])
    
    def get_image_channels(
            self
        ) -> tuple:
        """
        get_image_channels(self):
            gets number of image channels
        """
        image_array = self.get_image_array()
        return image_array.shape[0]
     
class EncodedScribble(XMLScribbleReader, NIfTIScribbleReader):
    """
    EncodedScribble(XMLScribbleReader, NIfTIScribbleReader):
        Initializes encoded scribble, whether it is a XML or a NIfTI file 
        (as specified by the is_jpeg attribute)
        
    This is the ultimate object with which we will interact later
    """
    def __init__(
            self,
            path_to_file: str,
            is_xml: bool = True,
        ) -> None:
        """
        __init__(self, path_to_file):
            calls parent class initializer
        """
        self.is_xml = is_xml
        super(EncodedScribble, self).__init__(path_to_file)

    def __get_encoded_scribble(
            self
        ) -> dict:
        """
        __get_encoded_scribble(self):
            gets the underlying scribble, encoded in a dictionary
        """
        encoded_scribble = XMLScribbleReader.encode_scribble(self) if self.is_xml\
                else NIfTIScribbleReader.encode_scribble(self)
        return encoded_scribble

    def get_encoded_scribble(
            self
        ) -> list[np.ndarray]:
        """"
        get_encoded_scribble(self):
            gets the underlying scribble, encoded in a list of np.arrays as follows:

            1) each element of the output_list will correspond to a class 
                (i. e.: ```class_i = output_list[i]``` will correspond to the i-th class)
            2) each element of the array corresponding to class i 
                (i. e.: output_list[i]) will contain the coordinates of the scribble points
                associated to that class as a flat array (i.e.: [x1, y1, x2, y2, ...])
            
        """
        encoded_scribbles = self.__get_encoded_scribble()
        n_classes = len(encoded_scribbles)
        output_list = [np.empty(0, )]*n_classes
        class_id_map = {key: idx for idx, key in enumerate(list(encoded_scribbles))}
        for class_name, pixel_list in encoded_scribbles.items():
            class_id = class_id_map[class_name]
            output_list[class_id] = np.array(pixel_list)
        return output_list 

    def get_class_names(
            self
        ) -> list[str]:
        """
        get_class_names(self):
            gets the name of each distinct classes encoded in the scribble
        
        Returns: Names of classes in a list of strings
        """
        distinct_classes = self.__get_encoded_scribble().keys()
        return list(distinct_classes)

    def get_n_classes(
            self
        ) -> int:
        """
        get_class_names(self):
            gets the number of distinct classes encoded in the scribble
        
        Returns: number of distinct classes as an int
        """
        class_names = self.get_class_names()
        return len(class_names)

    def get_scribble_dictionary(
            self
        ) -> dict:
        return self.__get_encoded_scribble()