"""
Script providing classes for handling the files
In our assignment we are considering two datasets:
1) Pascal
2) CycleMix

For dataset 1) the file encoding is the following:
    1.Images) Jpeg File
    2.Scribbles) XML File

For dataset 2) the file encoding is the following 
    2.Images) NIfTI File
    2.Scribbles) NIfTI File

The different file formats calls for the need to handle
differently each type of file. 

However, for the future image processing we'll have to perform, 
we can assume that it is convenient to have 
the images and the scribbles encoded in the following way:
    Images) torch.Tensor
    Scribbles) dictionary (or anything you guys think could be better)

This script does basically this. It provides the classes for 
reading the images and the scribbles in the desired format, as 
mentioned just above.

The structure of the script, and all the inheritances are these:

--FileReader(object)
    |
    |--JpegImageReader(FileReader)
    |   |
    |   |--TargetImage(JpegImageReader, NIfTIImageReader)
    |   |
    |--NIfTIImageReader(FileReader)
    |   |
    |   |--NIfTIScribbleReader(NIfTIImageReader)
    |           |
    |           |--EncodedScribble(NIfTIScribbleReader, XMLScribbleReader)
    |           |
    |--XMLScribbleReader(FileReader)

The only two classes that should be imported outside this script are
TargetImage and EncodedScribble. 

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

import torchvision.transforms as transforms
import torch.Tensor as Tensor

from collections import defaultdict
from bs4 import BeautifulSoup
from pathlib import Path
from PIL import Image
from  nibabel.nifti1 import Nifti1Image as Nifti1Image
from nib.loadsave import load as nib_load

class FileReader(object):
    """
    FileReader: 
        Most abstract class for handling files.
    """
    def __init__(
            self, 
            str | os.path.Path: path_to_file
        ):
        """"
        __init__(self, _path_to_file):
            initialize the FileReader with the given path
        """
        self.path_to_file = path_to_file

    def get_file_content(
            self
        ):
        """
        open_file(self): 
            base interface for getting file content
        """
        pass
        
class JpegImageReader(FileReader):
    """
    JpegReader(FileReader):
        Class for reading Jpeg image files, Children of FileReader class.
    """
    def __init__(
            self, 
            str | os.path.Path: path_to_file
        ) -> None:
        """
        __init__(self, path_to_file):
            calls parent class initializer
        """
        super(JpegImageReader, self).__init__(path_to_file)
    
    def __get_file_content(
            self
        ) -> Image:
        """
        __get_file_content(self):
            returns the underlying image as a Image image object   
        """
        path = self.path_to_file
        with Image.open(self.path_to_file) as image:
            return image

    def get_image_tensor(
            self
        ) -> Tensor:
        """
        get_image_tensor(self):
            returns the underlying image as a Tensor object
        """
        transform = transforms.ToTensor()
        image = self.__get_file_content()
        image_tensor = transform(image)
        return image_tensor
    
class NIfTIImageReader(FileReader):
    """
    NIfTIImageReader(FileReader):
        Class for reading NIfTI image files, childer of FileReader class
    """
    def __init__(
            self, 
            str | os.path.Path: path_to_file
        ) -> None:
        """
        __init__(self, path_to_file):
            calls parent class initializer
        """
        super(NIfTIImageReader, self).__init__(path_to_file)

    def __get_file_content(
            self
        ) -> Nifti1Image:
        """
        __get_file_content(self):
            returns the underlying image as a Nifti1Image image object
        """
        with nib_load(self.path_to_file) as image:
            return image

    def get_image_tensor(
            self
        ) -> Tensor:
        """
        get_image_tensor(self):
            returns the underlying image as a Tensor object
        """
        image = self.__get_file_content()
        data = image.get_fdata()
        image_tensor = torch.from_numpy(data)
        return image_tensor

class NIfTIScribbleReader(NIfTIImageReader):
    """
    NIfTIScribbleReader(NIfTIImageReader):
        class for reading the scribble from a Nifti1 Image
    """
    def __init__(
            self,
            str | os.path.Path: path_to_file,
        ) -> None:
        """
        __init__(self, path_to_file):
            calls parent class initializer
        """
        super(NIfTIScribbleReader, self).__init__(path_to_file)
    
    def encode_scribble(
            self
        ) -> dict:
        """
        TODO
        encode_scribble(self):
            encodes the scribble in the desired format (dict I thought) from the image_tensor
        """
        image_tensor = self.get_image_tensor()
        # continue
        # still I need to look at how are these scribble like
        # so i wouldn't know how to define it
   
class XMLScribbleReader(FileReader):
    """
    XMLScribbleReader(FileReader):
        Object for reading Xml Scribbles
    """
    def __init__(
            self,
            str | os.path.Path: path_to_file
        ) -> None:
        """
        __init__(self, path_to_file):
            calls parent class initializer
        """
        super(XMLScribbleReader, self).__init__(path_to_file)

    def __get_file_content(
            self
        ) -> BeautifulSoup:
        """
        __get_file_content(self):
            returns the underlying xml as a BeautifulSoup object 
        """
        with open(self.path_to_file, 'r') as xml_file:
            xml_text = xml_file.read()
        soup = BeautifulSoup(xml_text, 'xml')
        return soup

    def encode_scribble(
            self,
            str: annotation_tag = "polygon",
            str: class_tag = "tag",
            str: point_tag = "point"
        ) -> dict:
        """        
        encode_scribble(self):
            encodes the scribble in the desired format (dict I thought) from the BeautifulSoup object
        """
        encoded_scribbles = defaultdict(list)
        soup = self.__get_file_content()
        annotations = soup.find_all(annotation_tag)
        for annotation in annotations:
            class_name = annotation.find(class_tag).xml_text
            points = annotation.find_all(point_tag)
            for point in points:
                x = point.find("X").xml_text
                y = point.find("Y").xml_text
                encoded_scribbles[class_name].append([x, y])
        return dict(encode_scribbles)


class TargetImage(NIfTIImageReader, JpegImageReader):
    """
    TargetImage(NIfTIImageReader, JpegImageReader):
        Initializes target image, whether it is a Jpeg or a NIfTI file 
        (as specified by the is_jpeg attribute)

    This is the ultimate object with which we will interact later
    """
    def __init__(
            self,
            str | os.path.Path: path_to_file
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
     
class EncodedScribble(XMLScribbleReader, NIfTIScribbleReader):
    """
    EncodedScribble(XMLScribbleReader, NIfTIScribbleReader):
        Initializes encoded scribble, whether it is a XML or a NIfTI file 
        (as specified by the is_jpeg attribute)
        
    This is the ultimate object with which we will interact later
    """
    def __init__(
            self,
            str | os.path.Path: path_to_file
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