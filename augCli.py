# Copyright (c) 2022 Innodisk Crop.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import cv2
import argparse
import glob
import os
import time

from data_aug.data_aug import *
from data_aug.bbox_util import *
import pickle as pkl
import numpy as np 
import matplotlib.pyplot as plt
import datetime

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

def check_image_exist(image_list):
    if image_list:
        pass
        print("get image path in list: ", image_list)
    else:
        raise ValueError('Please give the image directory which have .png .jpg or .jpeg inside.')

def output_save_image(image, path):
    output_time ="{}-{}".format(datetime.datetime.now().date(),datetime.datetime.now().time())
    print(path)
    print(output_time)
    if path != None:
        if not os.path.exists(path):
            os.makedirs(path)
            time.sleep(1)
        cv2.imwrite('./{}/{}.png'.format(path, output_time), image)


def output_time():
    loc_dt = datetime.datetime.today() 
    time_del = datetime.timedelta(hours=0)
    new_dt = loc_dt + time_del 
    datetime_format = new_dt.strftime("%Y-%m-%d_%H-%M-%S")
    return datetime_format


def flip_image(img, bboxes, label, time, path, flip):
    if flip == "1":
        img_ver, bboxes_ = RandomVerticalFlip(1)(img.copy(), bboxes.copy())
        plotted_img = draw_rect(path, time, label, img_ver, bboxes_)
     
    
    elif flip == "2":
        img_ver, bboxes_ = RandomDiagonalFlip(1)(img.copy(), bboxes.copy())
        plotted_img = draw_rect(path, time, label, img_ver, bboxes_)

    elif flip == "3":
        img_ver, bboxes_ = RandomHorizontalFlip(1)(img.copy(), bboxes.copy())
        plotted_img = draw_rect(path, time, label, img_ver, bboxes_)
    
    cv2.imwrite('{}/{}.png'.format(path, time), img_ver)
    
    return plotted_img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p',    '--img_path',   type=str           , help='Please give dataset directory')
    ap.add_argument('-op',   '--output_path',type=str           , help='Please give aug dataset output directory')
    ap.add_argument('-flip', '--flip',       type=str           , help='flip image, 1=Vertical, 2=Diagonal, 3=Horizontal')
    ap.add_argument('-s',    '--show',       action='store_true', help='show result')

    args = ap.parse_args()
    print(args.img_path)
    # test path: /home/hueiru/Desktop/test_python/sample_D/NG
    list_image = load_images(args.img_path)
   
    index = 0
    for i in list_image:
        print("image path: ", i)
        img = cv2.imread(i)
        img_shape = img.shape[:2] # (height, width)
        size = (img_shape[1], img_shape[0]) # (width, height)

        f = open("{}.txt".format((i.split('.')[:-1])[0]), 'r')
        _bboxes = list()
        for line in f.readlines():
            print(line)
            labeling_data = list(map(float, line.split(" "))) # change list str() to int()

            # left, top, width, height = [0.591406,0.730469,0.079687,0.080078]
            left, top, width, height = labeling_data[1:]


            x =  size[0] * left       
            y =  size[1] * top        
            w =  (size[0] * width) / 2      
            h =  (size[1] * height) / 2

            xmin = x - w
            ymin = y - h
            xmax = x + w
            ymax = y + h 

            list_bbox = [xmin, ymin, xmax, ymax, labeling_data[0]]
            
            _bboxes.append(list_bbox)
            
        bboxes = np.array(_bboxes) 
        f.close

        plotted_img = flip_image(img, bboxes, labeling_data[0], output_time()+str(index), args.output_path, args.flip)
        index = index+1
        
        if args.show: 
            cv2.imshow("result", plotted_img)
            cv2.waitKey(0)
            cv2.destroyWindow('result')    
 
  


if __name__ == '__main__':

    main()
