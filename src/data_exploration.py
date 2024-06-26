import cv2
import os
from PIL import Image


def get_avgrage_width_and_lenght(root_directory, class_names):
    dimentions = []
    for class_name in class_names:
        images_names = os.listdir(root_directory+'/'+class_name)
        for image_name in images_names:
            image_path = root_directory+'/'+class_name+'/'+image_name
            with Image.open(image_path) as image:
                dimentions.append(image.size)
                
    avrage_lenght = sum([d[0] for d in dimentions]) / len(dimentions)   
    avrage_width = sum(d[1] for d in dimentions) / len(dimentions)
    return avrage_lenght, avrage_width


