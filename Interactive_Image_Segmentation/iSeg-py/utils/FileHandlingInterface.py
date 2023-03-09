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

    def get_encoded_scribble(
            self
        ) -> dict:
        """
        get_encoded_scribble(NIfTIImageReader, JpegImageReader):
            gets the underlying scribble, encoded in a dictionary
        """
        encoded_scribble = XMLScribbleReader.encode_scribble(self) if self.is_xml\
                else NIfTIScribbleReader.encode_scribble(self)
        return encoded_scribble