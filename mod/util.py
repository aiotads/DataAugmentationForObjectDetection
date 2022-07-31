import shutil
import random
import cv2
import matplotlib.pyplot as plt
from data_aug.data_aug import *
from data_aug.bbox_util import *

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

def splitdata(files, test_data_path, val_data_path, ratio=0.8):
        
    random.shuffle(files)

    cut = int(len(files)*round(ratio, 1))
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

def hsv_image(img, bboxes, label, time, path, hsv):
    # hsv = (h,s,v)
    img_ver, bboxes_ = RandomHSV(hsv)(img.copy(), bboxes.copy())
    plotted_img = draw_rect(path, time, label, img_ver, bboxes_)
    cv2.imwrite('{}/{}.png'.format(path, time), img_ver)
    return plotted_img

def cal_duplicate(train_demand, split_ratio, current_amount, brightness_min, brightness_max):
    check_duplicate = []
    total_demand = int(train_demand/split_ratio)
    duplicate_amount = int((total_demand - current_amount)/current_amount)
    if not (brightness_min + brightness_max) > duplicate_amount*2:
        raise ValueError('Please give larger range or decrease total amount.')     
    print(duplicate_amount)
    while len(check_duplicate) != duplicate_amount:
        _value = random.randrange(brightness_min, brightness_max, 2)
        if _value not in check_duplicate:
            check_duplicate.append(_value)
    return check_duplicate

