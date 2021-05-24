import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from pathlib import Path
import random
from shutil import copyfile

cropped_imgs_path = "/scratch/local/ssd/janhr/data/dogs_cropped/all/"
train_dir_path = "/scratch/local/ssd/janhr/data/dogs_cropped/train"
val_dir_path = "/scratch/local/ssd/janhr/data/dogs_cropped/val"
train_validation_ratio = 0.1

img_names = os.listdir(cropped_imgs_path)
print(f"{len(img_names)} images were found!")

# Create train and val dir
Path(train_dir_path).mkdir(parents=True, exist_ok=True)
Path(val_dir_path).mkdir(parents=True, exist_ok=True)

validation_samples = random.sample(range(1, len(img_names)), int(train_validation_ratio*len(img_names)))

for i,v in enumerate(img_names):
    src = cropped_imgs_path + v
    if i in validation_samples:
        dst = val_dir_path + v
        copyfile(src, dst)
        # move to val folder
    else:
        # copy to train folder
        dst = train_dir_path + v
        copyfile(src, dst)