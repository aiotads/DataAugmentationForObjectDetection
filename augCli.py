# Copyright (c) 2022 Innodisk Crop.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import cv2
import argparse
import numpy as np
import time

from data_aug.data_aug import *
from data_aug.bbox_util import * 
from mod.file import *
from mod.classes import *
from mod.img import *
from mod.util import *

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p',    '--img_path',       type=str,   help='Please give class name and dataset directory')
    ap.add_argument('-op',   '--output_path',    type=str,              help='Please give aug dataset output directory')
    ap.add_argument('-gen',  '--gen',            action='store_true',   help='gen empty label txt')
    ap.add_argument('-f',    '--flip',           type=str,              help='flip image, 1=Vertical, 2=Diagonal, 3=Horizontal')
    ap.add_argument('-hsv',  '--hsv',            type=str,              help='random image hsv')
    ap.add_argument('-v',  '--value',            type=int,              help='value')
    ap.add_argument('-b',    '--brightness',     type=str,   help='Usage pyth')
    # ap.add_argument('-s',    '--show',           action='store_true',   help='show result')
    # ap.add_argument('-sp',   '--split',          type=float,            help='split ratio')

    args = ap.parse_args()


    print(args.img_path)
    
    list_image = load_images(args.img_path)

    for i in list_image:
        print("image path: {}".format(i))
        filename, ext = os.path.splitext(i)
        img = cv2.imread(i)

        '''正方形'''
        size = get_img_size(img)
    
        data = np.loadtxt("{}.txt".format(filename))
        if len(np.shape(data)) == 1:
            data = np.array(data).reshape(1, -1)
            
        data = convert_bboxes(data, img_size=size)
    
        if args.flip:
            print('Start fliping.')
            flip_image(img=img, bboxes=data, time=time.time(), path=args.output_path, flip=args.flip)

        if args.brightness:
            print('Start duplicating.')
     
            for value in random.sample(range(0, 80), 10):
                range_brightness_image(img=img, bboxes=data, time=time.time(), path=args.output_path, value=value)
           
            
if __name__ == '__main__':
    main()