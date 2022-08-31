from mod.img import load_images

def get_classesinfo(classes):
    classes_info = {'name':list(classes.keys()), 'path':list(classes.values()), 'amount':[], 'diff':[]}

    amount_max = 0
    class_amount = 0
    for path in classes_info['path']:
        classes_info['amount'].extend([len(load_images(path))])
        if class_amount > amount_max:
            amount_max = class_amount
    return classes_info