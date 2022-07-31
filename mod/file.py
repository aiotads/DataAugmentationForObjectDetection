import datetime
import os

def output_time():
    loc_dt = datetime.datetime.today() 
    time_del = datetime.timedelta(hours=0)
    new_dt = loc_dt + time_del 
    datetime_format = new_dt.strftime("%Y-%m-%d_%H-%M")
    return datetime_format

def check_txt_exist(txt_path):
    if os.path.isfile(txt_path):
        f = open(txt_path,'r')
    else:
        print('generate emty ', txt_path)
        f = open(txt_path,'a')
    return f

def check_folder_exist(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs("{}/val".format(folder_path))
        os.makedirs("{}/test".format(folder_path))
   
#NOT USE
def check_image_exist(image_list):
    if image_list:
        pass
        print("get image path in list: ", image_list)
    else:
        raise ValueError('Please give the image directory which have .png .jpg or .jpeg inside.')

#NOT USE
def output_save_image(image, path):
    output_time ="{}-{}".format(datetime.datetime.now().date(),datetime.datetime.now().time())
    print(path)
    print(output_time)
    if path != None:
        if not os.path.exists(path):
            os.makedirs(path)
            time.sleep(1)
        cv2.imwrite('./{}/{}.png'.format(path, output_time), image)

#NOT USE
def empty_label_txt(path, time):
    open("{}/{}.txt".format(path, time), 'a').close()