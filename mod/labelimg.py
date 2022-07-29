import cv2

def convert_bboxes(labeling_data, img_size):
    
    _bboxes = list()
    left, top, width, height = labeling_data[1:]

    x =  img_size[0] * left       
    y =  img_size[1] * top        
    w =  (img_size[0] * width) / 2      
    h =  (img_size[1] * height) / 2

    xmin = x - w
    ymin = y - h
    xmax = x + w
    ymax = y + h 

    list_bbox = [xmin, ymin, xmax, ymax, labeling_data[0]]
    
    _bboxes.append(list_bbox)
    return _bboxes

def get_imginfo(img):
    img_shape = img.shape[:2] # (height, width)
    size = (img_shape[1], img_shape[0]) # (width, height)
    return size