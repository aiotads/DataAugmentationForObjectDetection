# Copyright (c) 2022 Innodisk Crop.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import cv2
import argparse
import glob
import os
import time
import random
import shutil

from data_aug.data_aug import *
from data_aug.bbox_util import *
import pickle as pkl
import numpy as np 
import matplotlib.pyplot as plt
import datetime
from mod.labelimg import *
from mod.classes import *

#img.py
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
#??
def check_image_exist(image_list):
    if image_list:
        pass
        print("get image path in list: ", image_list)
    else:
        raise ValueError('Please give the image directory which have .png .jpg or .jpeg inside.')

#??
def output_save_image(image, path):
    output_time ="{}-{}".format(datetime.datetime.now().date(),datetime.datetime.now().time())
    print(path)
    print(output_time)
    if path != None:
        if not os.path.exists(path):
            os.makedirs(path)
            time.sleep(1)
        cv2.imwrite('./{}/{}.png'.format(path, output_time), image)

#img.py
def output_time():
    loc_dt = datetime.datetime.today() 
    time_del = datetime.timedelta(hours=0)
    new_dt = loc_dt + time_del 
    datetime_format = new_dt.strftime("%Y-%m-%d_%H-%M-%S")
    return datetime_format

#util.py
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
    
    #TO-DO: add option--select all
    # elif flip == "0":
    #     img_ver, bboxes_ = RandomVerticalFlip(1)(img.copy(), bboxes.copy())
    #     plotted_img = draw_rect(path, time, label, img_ver, bboxes_)
    #     img_ver, bboxes_ = RandomDiagonalFlip(1)(img.copy(), bboxes.copy())
    #     plotted_img = draw_rect(path, time, label, img_ver, bboxes_)
    #     img_ver, bboxes_ = RandomHorizontalFlip(1)(img.copy(), bboxes.copy())
    #     plotted_img = draw_rect(path, time, label, img_ver, bboxes_)

#??
def hsv_image(img, bboxes, label, time, path, hsv):
    # hsv = (h,s,v)
    img_ver, bboxes_ = RandomHSV(hsv)(img.copy(), bboxes.copy())
    plotted_img = draw_rect(path, time, label, img_ver, bboxes_)
    cv2.imwrite('{}/{}.png'.format(path, time), img_ver)
    return plotted_img

#util.py
def range_brightness_image(img, bboxes, label, time, path, value):
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img_ver = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    plotted_img = draw_rect(path, time, label, img_ver, bboxes)
    cv2.imwrite('{}/{}.png'.format(path, time), img_ver)
    return plotted_img

#??
def empty_label_txt(path, time):
    open("{}/{}.txt".format(path, time), 'a').close()

#util.py
def splitdata(files, test_data_path, val_data_path, ratio=0.8):
    # for file in files:
        # print(os.path.abspath(image_path))
        # files = load_images(os.path.abspath(image_path))
        
    random.shuffle(files)

    cut = int(len(files)*round(ratio, 1))
    arr1 = files[:cut]
    arr2 = files[cut:]

    val_cut = int(len(arr2)*0.5)
    val_arr = arr2[:val_cut]
    test_arr = arr2[val_cut:]

    for j in test_arr:
        shutil.move(j, test_data_path)
        shutil.move(j.split('.')[:-1][0]+".txt", test_data_path)

    for j in val_arr:
        shutil.move(j, val_data_path)
        shutil.move(j.split('.')[:-1][0]+".txt", val_data_path)
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p',    '--img_path',       type=str, nargs='+',   help='Please give class name and dataset directory')
    ap.add_argument('-op',   '--output_path',    type=str,              help='Please give aug dataset output directory')
    ap.add_argument('-gen',  '--gen',            action='store_true',   help='gen empty label txt')
    ap.add_argument('-f',    '--flip',           type=str,              help='flip image, 1=Vertical, 2=Diagonal, 3=Horizontal')
    ap.add_argument('-hsv',  '--hsv',            type=str,              help='random image hsv')
    ap.add_argument('-b',    '--brightness',     type=int, nargs='+',   help='Usage python3 augCli.py -b <demand of train amount(each class)> <brightness min value> <brightness max value>')
    ap.add_argument('-s',    '--show',           action='store_true',   help='show result')
    ap.add_argument('-sp',   '--split',         type=float,            help='split ratio')

    args = ap.parse_args()

    os.makedirs("{}/val".format(args.output_path))
    os.makedirs("{}/test".format(args.output_path))
    
    i = 0
    classes = {}
    infos = args.img_path
    while i < len(infos):
        classes[infos[i]] = infos[i+1]
        i += 2

    classes_info = get_classesinfo(classes)
    for img_path in classes_info['path']:
        list_image = load_images(image_path)
        index = 0
        spilt_list_data = []
        break
        for i in list_image:
            print("image path: ", i)

            img = cv2.imread(i)
            size = get_imginfo(img)

            f = open("{}.txt".format((i.split('.')[:-1])[0]), 'a')
            if os.path.isfile("{}.txt".format((i.split('.')[:-1])[0])):
                f = open("{}.txt".format((i.split('.')[:-1])[0]), 'r')

            _bboxes = list()
            lines = f.readlines()
            if not lines == []:
                for line in lines:
                    print('There is labeling: ', line)
                    labeling_data = list(map(float, line.split(" "))) # change list str() to int()
                bboxes = np.array(convert_bboxes(labeling_data, size))
                f.close
                print(bboxes)
            else:
                print('.txt is NULL')
                bboxes = []
                labeling_data = [0]
            
            if args.flip:
                print('Start fliping.')
                # empty_label_txt(args.output_path, output_time()+str(index))
                plotted_img = flip_image(img, bboxes, labeling_data[0], class_name+output_time()+str(index), args.output_path, args.flip)
                spilt_list_data.append("{}/{}.png".format(args.output_path, class_name+output_time()+str(index)))
                index = index+1

            if args.brightness:
                print('Start duplicating.')
                total_demand = int(args.brightness[0]/args.split)
                check_duplicate = []
                duplicate_amount = int((total_demand - class_amount)/class_amount)
                if not (args.brightness[1] + args.brightness[2]) > duplicate_amount*2:
                    raise ValueError('Please give larger range or decrease total amount.')
                
                while len(check_duplicate) != duplicate_amount:
                    _value = random.randrange(args.brightness[1], args.brightness[2], 2)
                    if _value not in check_duplicate:
                        check_duplicate.append(_value)

                for value in check_duplicate:
                    # empty_label_txt(args.output_path, output_time()+str(index))
                    spilt_list_data.append("{}/{}.png".format(args.output_path, class_name+output_time()+str(index)))
                    plotted_img = range_brightness_image(img, bboxes, labeling_data[0], class_name+output_time()+str(index), args.output_path, value)
                    index = index+1
        # break       
        splitdata(spilt_list_data, "{}/test".format(args.output_path), "{}/val".format(args.output_path), args.split)
        index = index+1                
            
        # if args.show: 
        #     cv2.imshow("result", plotted_img)
        #     cv2.waitKey(0)
        #     cv2.destroyWindow('result')
 
if __name__ == '__main__':

    main()