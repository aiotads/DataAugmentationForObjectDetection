

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