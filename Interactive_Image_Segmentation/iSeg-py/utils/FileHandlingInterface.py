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

from fileHandling import *

class TargetImage(NIfTIImageReader, JpegImageReader):
    """
    TargetImage(NIfTIImageReader, JpegImageReader):
        Initializes target image, whether it is a Jpeg or a NIfTI file 
        (as specified by the is_jpeg attribute)

    This is the ultimate object with which we will interact later
    """
    def __init__(
            self,
            str: path_to_file,
            bool: is_jpeg = True,
        ) -> None:
        """
        __init__(self, path_to_file):
            calls parent class initializer
        """
        self.is_jpeg = is_jpeg
        super(TargetImage, self).__init__(path_to_file)
    
    def get_image_tensor(
            self
        ) -> Tensor:
        """
        get_image_tensor(self):
            gets the underlying image as a torch.Tensor object
        """
        image = JpegImageReader.get_image_tensor(self) if self.is_jpeg\
                else NIfTIImageReader.get_image_tensor(self)
        return image
    
    def get_image_array(
            self
        ) -> np.ndarray:
        """
        get_image_array(self):
            returns the underlying image as a np.ndarray object
        """
        image = JpegImageReader.get_image_array(self) if self.is_jpeg\
                else NIfTIImageReader.get_image_array(self)
        return image_array
    
    def get_image_shape(
            self
        ) -> tuple:
        """
        gets_image_shape(self):
            gets the shape of the underlying image array.
        """
        image_array = self.get_image_array()
        return tuple(image_array.shape[1:])
     
class EncodedScribble(XMLScribbleReader, NIfTIScribbleReader):
    """
    EncodedScribble(XMLScribbleReader, NIfTIScribbleReader):
        Initializes encoded scribble, whether it is a XML or a NIfTI file 
        (as specified by the is_jpeg attribute)
        
    This is the ultimate object with which we will interact later
    """
    def __init__(
            self,
            str: path_to_file,
            bool: is_xml = True,
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
            self,
            dict: class_id_map 
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
        n_classes = len(class_map)
        output_list = [np.empty(0, )]*n_classes
        encoded_scribbles = self.__get_encoded_scribble()
        for class_name, pixel_list in encoded_scribbles.items():
            class_id = class_id_map[class_name]
            output_list[class_id] = np.array(pixel_list)
        return output_list 
