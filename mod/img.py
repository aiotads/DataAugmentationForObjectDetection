import glob
import os
import numpy as np

def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """

    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        raise ValueError('Please give the image directory')
    elif input_path_extension == "txt":
        raise ValueError('Please give the image directory')
    else:
        return glob.glob(os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))

def get_img_size(img):
    img_shape = img.shape[:2] # (height, width)
    size = (img_shape[1], img_shape[0]) # (width, height)
    return size

def convert_bboxes(labeling_data, img_size):
    
    _bboxes = list()
    
    
    for i, label in enumerate(labeling_data):
        left, top, width, height = label[1:]

        x =  img_size[0] * left       
        y =  img_size[1] * top        
        w =  (img_size[0] * width) / 2      
        h =  (img_size[1] * height) / 2

        xmin = x - w
        ymin = y - h
        xmax = x + w
        ymax = y + h 

        list_bbox = [xmin, ymin, xmax, ymax, label[0]]
        
        _bboxes.append(list_bbox)
        _bboxes_array = np.array(_bboxes)
    
    return _bboxes_array
