# Make initial crop 
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import os
from pathlib import Path

train_file_list_path = Path("/scratch/shared/beegfs/janhr/data/unsup3d_extended/tsinghua_dogs_high_res/TrainAndValList/train.lst")
val_file_list_path = Path("/scratch/shared/beegfs/janhr/data/unsup3d_extended/tsinghua_dogs_high_res/TrainAndValList/validation.lst")
annotation_path = Path("/scratch/shared/beegfs/janhr/data/unsup3d_extended/tsinghua_dogs_high_res/High-Annotations/")
source_path = Path("/scratch/shared/beegfs/janhr/data/unsup3d_extended/tsinghua_dogs_high_res/high-resolution/")
destination_path = Path("/scratch/shared/beegfs/janhr/data/unsup3d_extended/tsinghua_dogs_high_res_cropped/all")


destination_path.mkdir(parents=True, exist_ok=True)

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
            img_name = f.name
            parent_folder_name = f.parent.relative_to(f.parent.parent)
            a = annotation_path / parent_folder_name / (img_name + ".xml")
            tree = ET.parse(a)
            root = tree.getroot()
            bbox=bounding_box(root)
            im=Image.open(f)
            rgb_im = im.convert('RGB')
            rgb_im = rgb_im.crop(bbox)
            dest_im_path = destination_path / img_name
            rgb_im.save(dest_im_path)
        except Exception as e: 
            print(e)
            
        print(f"{i}/{len(file_list)}", end="\r")
        
with open(train_file_list_path) as file:
    train_file_list = list(file)

train_files = []
for i in train_file_list:
    s = i.split(".//")[1]
    t = s.split("\n")[0]
    t = source_path / t
    train_files.append(t)
    
print(f"Found {len(train_files)} images.")
    
process_files(train_files, source_path, destination_path, annotation_path)


with open(val_file_list_path) as file:
    val_file_list = list(file)

val_files = []
for i in val_file_list:
    s = i.split(".//")[1]
    t = s.split("\n")[0]
    t = Path(t)
    t = source_path / t
    val_files.append(t)
    
print(f"Found {len(val_files)} images.")
    
process_files(val_files, source_path, destination_path, annotation_path)