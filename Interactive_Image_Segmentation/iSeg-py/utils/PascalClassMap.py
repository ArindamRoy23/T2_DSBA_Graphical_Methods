"""

PascalClassMap.py

    Script for creating the mapping from class names to integers in the 
    Pascal dataset. The resulting class map will be saved in a Json File
    as a dictionary.
    Basically, it should iterate over the folder containing the scribbles,
    for each scribble xml, it should get the present class names, 
    collect all the class names of all the files in the given folder
    and finally map them to integers

"""

import json
import os
import bs4 as BeautifulSoup

def create_clas_int_map(
        folder_name: str,
        dump_file: str,
        annotation_tag: str = "polygon",
        class_tag: str = "tag"
    ) ->None:
    file_list = os.listdir(folder_name)
    class_dict = dict()
    for idx, xml_file in enumerate(file_list):
        xml_file_oath = os.path.join(folder_name, xml_file)        
        with open(self.path_to_file, 'r') as xml_file:
            xml_text = xml_file.read()
        soup = BeautifulSoup(xml_text, 'xml')
        annotations = soup.find_all(annotation_tag)
        for annotation in annotations:
            class_name = annotation.find(class_tag).contents[0]
            class_dict[class_name] = 0
        print(f"{idx / len(file_list)} files read\n")
    class_dict = {key: idx for idx, key in enumerate(list(class_dict.keys()))}
    print(f"All files read, dumping dictionary to {dump_file}")
    with open(dump_file, "w") as file_:
        json.dump(class_dict, file_)
    print("dictionary dumped")