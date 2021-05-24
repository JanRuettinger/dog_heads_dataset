# Make initial crop 
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import os

def bounding_box(root):
    objects = root.findall('object')
    for o in objects:
        bndbox = o.find('headbndbox') # reading bound box
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
    return (xmin,ymin,xmax,ymax)


def process_files(file_list, source_path, destination_path, annotation_path):
    
    for i,f in enumerate(file_list):
        try:
            img_path = f
            a = annotation_path + img_path + ".xml"
            tree = ET.parse(a)
            root = tree.getroot()
            bbox=bounding_box(root)
            im=Image.open(os.path.join(source_path,img_path))
            im=im.crop(bbox)
            dest_im_path = destination_path + str(i) + ".jpg"
            im.save(dest_im_path)
        except Exception as e: 
            print(e)
            
        if (i % 5000) == 0:
            print(i)
            
# train
train_file_list_path = "/scratch/shared/beegfs/janhr/data/unsup3d_extended/tsinghua_dogs_high_res/TrainAndValList/train.lst"
source_path = "/scratch/shared/beegfs/janhr/data/unsup3d_extended/tsinghua_dogs_high_res/high-resolution/"
destination_path = "/scratch/shared/beegfs/janhr/data/unsup3d_extended/tsinghua_dogs_high_res_cropped/train/"
annotation_path = "/scratch/shared/beegfs/janhr/data/unsup3d_extended/tsinghua_dogs_high_res/High-Annotations/"

with open('/scratch/shared/beegfs/janhr/data/unsup3d_extended/tsinghua_dogs_high_res/TrainAndValList/train.lst') as file:
    file_list = list(file)

train_file_list = []
for i in file_list:
    s = i.split(".//")[1]
    t = s.split("\n")[0]
    train_file_list.append(t)

process_files(train_file_list, source_path, destination_path, annotation_path)


# validation
val_file_list_path = "/scratch/shared/beegfs/janhr/data/unsup3d_extended/tsinghua_dogs_high_res/TrainAndValList/validation.lst"
source_path = "/scratch/shared/beegfs/janhr/data/unsup3d_extended/tsinghua_dogs_high_res/high-resolution/"
destination_path = "/scratch/shared/beegfs/janhr/data/unsup3d_extended/tsinghua_dogs_high_res_cropped/val/"
annotation_path = "/scratch/shared/beegfs/janhr/data/unsup3d_extended/tsinghua_dogs_high_res/High-Annotations/"

with open('/scratch/shared/beegfs/janhr/data/unsup3d_extended/tsinghua_dogs_high_res/TrainAndValList/train.lst') as file:
    file_list = list(file)

train_file_list = []
for i in file_list:
    s = i.split(".//")[1]
    t = s.split("\n")[0]
    train_file_list.append(t)

process_files(val_file_list, source_path, destination_path, annotation_path)