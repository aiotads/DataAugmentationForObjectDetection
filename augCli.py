# Copyright (c) 2022 Innodisk Crop.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import cv2
import argparse
import numpy as np

from data_aug.data_aug import *
from data_aug.bbox_util import * 
from mod.file import *
from mod.classes import *
from mod.img import *
from mod.util import *

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p',    '--img_path',       type=str, nargs='+',   help='Please give class name and dataset directory')
    ap.add_argument('-op',   '--output_path',    type=str,              help='Please give aug dataset output directory')
    ap.add_argument('-gen',  '--gen',            action='store_true',   help='gen empty label txt')
    ap.add_argument('-f',    '--flip',           type=str,              help='flip image, 1=Vertical, 2=Diagonal, 3=Horizontal')
    ap.add_argument('-hsv',  '--hsv',            type=str,              help='random image hsv')
    ap.add_argument('-b',    '--brightness',     type=int, nargs='+',   help='Usage python3 augCli.py -b <demand of train amount(each class)> <brightness min value> <brightness max value>')
    ap.add_argument('-s',    '--show',           action='store_true',   help='show result')
    ap.add_argument('-sp',   '--split',          type=float,            help='split ratio')

    args = ap.parse_args()
    check_folder_exist(args.output_path)
    
    input_path = 0
    classes = {}
    class_index = 0
    infos = args.img_path
    while input_path < len(infos):
        classes[infos[input_path]] = infos[input_path+1]
        input_path += 2
    classes_info = get_classesinfo(classes)

    for img_path in classes_info['path']:
        
        index = 0
        spilt_list_data = []
        list_image = load_images(img_path)
        print(class_index, classes_info['name'][class_index])

        for i in list_image:
            print("image path: ", i)

            img = cv2.imread(i)
            size = get_imginfo(img)
            f = check_txt_exist("{}.txt".format((i.split('.')[:-1])[0]))
            
            lines = f.readlines()
            if not lines == []:
                for line in lines:
                    labeling_data = list(map(float, line.split(" "))) # change list str() to int()
                bboxes = np.array(convert_bboxes(labeling_data, size))
                f.close
            else:
                bboxes = []
                labeling_data = [0]
            
            if args.flip:
                print('Start fliping.')
                plotted_img = flip_image(img, bboxes, labeling_data[0], classes_info['name'][class_index]+output_time()+str(index), args.output_path, args.flip)
                spilt_list_data.append("{}/{}.png".format(args.output_path, classes_info['name'][class_index]+output_time()+str(index)))
                index = index+1

            if args.brightness:
                print('Start duplicating.')
                check_duplicate = cal_duplicate(args.brightness[0], args.split, classes_info['amount'][class_index],args.brightness[1], args.brightness[2])
                for value in check_duplicate:
                    plotted_img = range_brightness_image(img, bboxes, labeling_data[0], classes_info['name'][class_index]+output_time()+str(index), args.output_path, value)
                    spilt_list_data.append("{}/{}.png".format(args.output_path, classes_info['name'][class_index]+output_time()+str(index)))
                    index = index+1

            if args.show: 
                cv2.imshow("result", plotted_img)
                cv2.waitKey(0)
                cv2.destroyWindow('result')

        splitdata(spilt_list_data, "{}/test".format(args.output_path), "{}/val".format(args.output_path), args.split)
        index += 1
        class_index += 1                
            
if __name__ == '__main__':
    main()