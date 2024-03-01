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
    ap.add_argument('-p',    '--img_path',    type=str,              help='Please give the dataset directory')
    ap.add_argument('-o',    '--output_path', type=str,              help='Please give dataset output directory')
    ap.add_argument('-f',    '--flip',        type=str,              help='flip image, V(Vertical), D(Diagonal), H(Horizontal)')
    ap.add_argument('-m',    '--mosaic',      action='store_true',              help='mosaic')
    ap.add_argument('-b',    '--blur',        type=str,              help='mosaic')
    # ap.add_argument('-gen',  '--gen',         action='store_true',   help='gen empty label txt')
    ap.add_argument('-hsv',  '--hsv',         type=str, nargs='+',   help='random image hsv')
    ap.add_argument('-br',    '--brightness',  type=str,   help='Value random brightness+-20, 5 pics')
    ap.add_argument('-d',    '--dimming',     type=str,   help='Value random dimming+-20, 5 pics')
    ap.add_argument('-sc',    '--scale',      action='store_true',   help='random Scale')
    ap.add_argument('-mv_dn', '--move_down',  action='store_true',   help='random move down')
    ap.add_argument('-s',    '--show',        action='store_true',   help='show result')

    args = ap.parse_args()

    print(args.img_path)
    
    list_image = load_images(args.img_path)

    for i in list_image:
        print("image path: {}".format(i))
        filename, ext = os.path.splitext(i)
        img = cv2.imread(i)
        name = parser_image_name(i)
        size = get_img_size(img)
    
        data = np.loadtxt("{}.txt".format(filename))
        if len(np.shape(data)) == 1:
            data = np.array(data).reshape(1, -1)
        
        data = convert_bboxes(data, img_size=size)
    
        if args.flip:
            print('Starting flip.')
            flip_image(img=img, bboxes=data, time=time.time(), path=args.output_path, name=name, flip=args.flip)

        if args.brightness:
            print('Start brightness.')
            for value in random.sample(range(int(args.brightness)-20, int(args.brightness)+20), 5):
                range_brightness_image(img=img, bboxes=data, time=time.time(), path=args.output_path, name=name,value=value)         
        
        if args.dimming:
            print('Start dimming.')
            for value in random.sample(range(int(args.dimming)-20, int(args.dimming)+20), 5):
                range_dimming_image(img=img, bboxes=data, time=time.time(), path=args.output_path, name=name,value=value)         
            
        if args.hsv:
            print("Start hsv")
            for value in range(5):
                hsv_image(img=img, bboxes=data, time=time.time(), path=args.output_path, name=name, hsv=args.hsv)

        if args.blur:
            print("start blur")
            if int(args.blur) < 0 or int(args.blur) > 6:
                print("please between 0~5")
                return
            blur(img=img, bboxes=data, time=time.time(), path=args.output_path, name=name, mosaic_level=args.blur)

        if args.scale:
            print("start scale")
            scale_image(img=img, bboxes=data, time=time.time(), path=args.output_path, name=name)
        
        if args.move_down:
            print("start move down")
            move_down_image(img=img, bboxes=data, time=time.time(), path=args.output_path, name=name)
        
    if args.mosaic:
        print("mosaic")
        mosaic(list_image, path=args.output_path, name=name, value=(480,640))


if __name__ == '__main__':
    main()