import shutil
import random
import cv2
import matplotlib.pyplot as plt
from data_aug.data_aug import *
from data_aug.bbox_util import *
from PIL import Image, ImageDraw
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import time

# def flip_image(img, bboxes, label, time, path, flip):
def flip_image(img, bboxes, time, path, name, flip):
    if flip == "V":
        mode = "flip_v"
        img, _bboxes = RandomVerticalFlip(1)(img.copy(), bboxes.copy())
        plotted_img, is_obj_exist = draw_rect(path, time, img, mode, name, _bboxes)
     
    elif flip == "D":
        mode = "flip_d"
        img, _bboxes = RandomDiagonalFlip(2)(img.copy(), bboxes.copy())
        plotted_img, is_obj_exist = draw_rect(path, time, img, mode, name, _bboxes)

    elif flip == "H":
        mode = "flip_h"
        img, _bboxes = RandomHorizontalFlip(3)(img.copy(), bboxes.copy())
        plotted_img, is_obj_exist = draw_rect(path, time, img, mode, name, _bboxes)
    
    if is_obj_exist:
        cv2.imwrite('{}/{}_{}_{}.png'.format(path, mode, name, time), img)   

def range_brightness_image(img, bboxes, time, path, name, value):
    mode = "br"
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = np.full(hsv[:, :, 2].shape, value, dtype=np.uint8)
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], mask, dtype=cv2.CV_8U)

    img_ver = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    plotted_img, is_obj_exist = draw_rect(path, time, img_ver, mode, name, bboxes)
    if is_obj_exist:
        cv2.imwrite('{}/{}_{}_{}.png'.format(path, mode, name, time), img_ver)
    return plotted_img

def range_dimming_image(img, bboxes, time, path, name, value):
    mode = "dim"
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = np.full(hsv[:, :, 2].shape, value, dtype=np.uint8)
    hsv[:, :, 2] = cv2.subtract(hsv[:, :, 2], mask)

    img_ver = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    plotted_img, is_obj_exist = draw_rect(path, time, img_ver, mode, name, bboxes)
    if is_obj_exist:
        cv2.imwrite('{}/{}_{}_{}.png'.format(path, mode, name, time), img_ver)
    return plotted_img

def range_rotate_image(img, bboxes, time, path, name, value):
    mode = "rotate"
    img_ver, _bboxes = RandomRotate(int(value))(img.copy(), bboxes.copy())
    plotted_img, is_obj_exist = draw_rect(path, time, img_ver, mode, name, _bboxes)
    if is_obj_exist:
        cv2.imwrite('{}/{}_{}_{}.png'.format(path, mode, name, time), img_ver)
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

def hsv_image(img, bboxes, time, path, name, hsv):
    hue, saturation, brightness = hsv
    mode = "hsv" 
    img_ver, bboxes = RandomHSV(int(hue), int(saturation), int(brightness))(img, bboxes)
    plotted_img, is_obj_exist = draw_rect(path, time, img_ver, mode, name, bboxes)
    if is_obj_exist:
        cv2.imwrite('{}/{}_{}_{}.png'.format(path, mode, name, time), img_ver)
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

def blur(img, bboxes, time, path, name, mosaic_level):
    mode = "blur"
    height, width = img.shape[:2]

    block_size = 20
    num_blocks_x = width // block_size
    num_blocks_y = height // block_size

    for y in range(num_blocks_y):
        for x in range(num_blocks_x):
            top_left = (x * block_size, y * block_size)
            bottom_right = ((x + 1) * block_size, (y + 1) * block_size)

            block = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            block = cv2.resize(block, (block_size, block_size), interpolation=cv2.INTER_NEAREST)
            img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = block

    img = cv2.resize(img, (width // int(mosaic_level), height // int(mosaic_level)), interpolation=cv2.INTER_NEAREST)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)

    plotted_img, is_obj_exist = draw_rect(path, time, img, mode, name, bboxes)
    if is_obj_exist:
        cv2.imwrite('{}/{}_{}_{}.png'.format(path, mode, name, time), img)

def scale_image(img, bboxes, time, path, name, value=0.2):
    mode = "scale"
    img_ver, bboxes = RandomScale(value)(img, bboxes)
    plotted_img, is_obj_exist = draw_rect(path, time, img_ver, mode, name, bboxes)
    if is_obj_exist:
        cv2.imwrite('{}/{}_{}_{}.png'.format(path, mode, name, time), img_ver)
    return plotted_img

def move_down_image(img, bboxes, time, path, name, value=0.2):
    mode = "mv_dn"
    img_ver, bboxes = RandomTranslate(value)(img, bboxes)
    plotted_img, is_obj_exist = draw_rect(path, time, img_ver, mode, name, bboxes)
    if is_obj_exist:
        cv2.imwrite('{}/{}_{}_{}.png'.format(path, mode, name, time), img_ver)
    return plotted_img

def parser_image_name(path):
    basename = os.path.basename(path)
    file_name, _ = basename.rsplit('.', 1)
    
    return file_name

def mosaic(list_image, path, value):
    mode = "mosaic"
    input_size = value
    combined_list = []

    while len(list_image) % 4 != 0:
        print("mosaic pairing pic")
        list_image += random.sample(list_image, 4 - len(list_image) % 4)
    
    for idx, l in enumerate(list_image):
        base_img_path, _ = os.path.splitext(l)
        label_path = base_img_path + '.txt'
        image = cv2.imread(l)
        print("{} images, read image {} path: {}".format(len(list_image),idx, l))
        h,w,_ = image.shape
        with open(label_path, 'r') as file:
            labels_str = ''
            for line in file:
                class_id, x_center, y_center, width, height = map(float, line.split())
                x_min = int((x_center - width / 2) * w)
                y_min = int((y_center - height / 2) * h)
                x_max = int((x_center + width / 2) * w)
                y_max = int((y_center + height / 2) * h)
                labels_str += f"{x_min},{y_min},{x_max},{y_max},{int(class_id)} "
            
            new_item = f"{l} {labels_str.strip()}\n"
            combined_list.append(new_item)
    grouped_list = [combined_list[i:i + 4] for i in range(0, len(combined_list), 4)]
    for g_idx, group in enumerate(grouped_list):
        print("save mosaic image NO.{} ".format(g_idx))
        _time=time.time()
        image_data, box_data = get_random_data(group, [input_size[0], input_size[1]])
        if box_data:
            image_data = Image.fromarray((image_data * 255).astype(np.uint8))
            image_data.save('{}/{}_{}_{}.png'.format(path, mode, str(g_idx), _time))
        
            with open("{}/{}_{}_{}.txt".format(path, mode, str(g_idx), _time), 'w') as f:
                for item in box_data:
                    x_min, y_min, x_max, y_max, label = item
                    x_center = ((x_min + x_max) / 2) / input_size[1]
                    y_center = ((y_min + y_max) / 2) / input_size[0]
                
                    width = (x_max - x_min) / input_size[1]
                    height = (y_max - y_min) / input_size[0] 
                    f.write(f"{int(label)} {x_center} {y_center} {width} {height}\n")

def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a

def merge_bboxes(bboxes, cutx, cuty):
    merge_bbox = []
    for i in range(len(bboxes)):
        for box in bboxes[i]:
            tmp_box = []
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            if i == 0:
                if y1 > cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue
            if i == 1:
                if y2 < cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue
            if i == 2:
                if y2 < cuty or x2 < cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue
            if i == 3:
                if y1 > cuty or x2 < cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue
            tmp_box.append(x1)
            tmp_box.append(y1)
            tmp_box.append(x2)
            tmp_box.append(y2)
            tmp_box.append(box[-1])
            merge_bbox.append(tmp_box)
    return merge_bbox

def get_random_data(annotation_line, input_shape, random=True, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    h, w = input_shape
    min_offset_x = 0.4
    min_offset_y = 0.4
    scale_low = 1 - min(min_offset_x, min_offset_y)
    scale_high = scale_low + 0.2
    image_datas = []
    box_datas = []
    index = 0
    place_x = [0, 0, int(w * min_offset_x), int(w * min_offset_x)]
    place_y = [0, int(h * min_offset_y), int(w * min_offset_y), 0]
    for line in annotation_line:
        # 每一行进行分割
        line_content = line.split()
        # 打开图片
        image = Image.open(line_content[0])
        image = image.convert("RGB")
        # 图片的大小
        iw, ih = image.size
        # 保存框的位置
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])

        # image.save(str(index)+".jpg")
        # 是否翻转图片
        flip = rand() < .5
        if flip and len(box) > 0:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            box[:, [0, 2]] = iw - box[:, [2, 0]]

        # 对输入进来的图片进行缩放
        new_ar = w / h
        scale = rand(scale_low, scale_high)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 进行色域变换
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = rgb_to_hsv(np.array(image) / 255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        image = hsv_to_rgb(x)

        image = Image.fromarray((image * 255).astype(np.uint8))
        # 将图片进行放置，分别对应四张分割图片的位置
        dx = place_x[index]
        dy = place_y[index]
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image) / 255
        # Image.fromarray((image_data*255).astype(np.uint8)).save(str(index)+"distort.jpg")
        index = index + 1
        box_data = []
        # 对box进行重新处理
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box

        image_datas.append(image_data)
        box_datas.append(box_data)

        img = Image.fromarray((image_data * 255).astype(np.uint8))
        for j in range(len(box_data)):
            thickness = 3
            left, top, right, bottom = box_data[j][0:4]
            draw = ImageDraw.Draw(img)
            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=(255, 255, 255))
        # img.show()

    # 将图片分割，放在一起
    cutx = np.random.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)))
    cuty = np.random.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)))

    new_image = np.zeros([h, w, 3])
    new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
    new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
    new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
    new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

    # 对框进行进一步的处理
    new_boxes = merge_bboxes(box_datas, cutx, cuty)

    return new_image, new_boxes