import os, random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from shutil import copyfile

cropped_imgs_path = Path("/scratch/shared/beegfs/janhr/data/unsup3d_extended/dogs_cropped/all")
train_dir_path = Path("/scratch/shared/beegfs/janhr/data/unsup3d_extended/dogs_cropped/train")
val_dir_path = Path("/scratch/shared/beegfs/janhr/data/unsup3d_extended/dogs_cropped/val")
train_validation_ratio = 0.1

# Create train and val dir
Path(train_dir_path).mkdir(parents=True, exist_ok=True)
Path(val_dir_path).mkdir(parents=True, exist_ok=True)

img_names = os.listdir(str(cropped_imgs_path))
print(f"{len(img_names)} images were found!")

validation_samples = random.sample(range(1, len(img_names)), int(train_validation_ratio*len(img_names)))

for i,v in enumerate(img_names):
    src = str(cropped_imgs_path / v)
    if i in validation_samples:
        dst = str(val_dir_path / v)
        copyfile(src, dst)
    else:
        dst = str(train_dir_path / v)
        copyfile(src, dst)