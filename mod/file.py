import datetime
import os

def output_time():
    loc_dt = datetime.datetime.today() 
    time_del = datetime.timedelta(hours=0)
    new_dt = loc_dt + time_del 
    datetime_format = new_dt.strftime("%Y-%m-%d_%H-%M")
    return datetime_format

def open_file(txt_path):
    if os.path.isfile(txt_path):
        f = open(txt_path,'r')
    else:
        print('generate emty ', txt_path)
        f = open(txt_path,'a')
    return f
