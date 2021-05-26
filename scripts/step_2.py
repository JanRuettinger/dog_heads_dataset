import cv2, os
from ast import literal_eval
from imutils import face_utils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

df_all = pd.read_csv("../csv_files/preprocessing_img_data.csv", index_col = 0)

print(f"A bounding box could only be detected for {len(df_all[df_all['bb'] == True])} images.")

num_imgs_after_filter_1 = 0
num_imgs_after_filter_2 = 0
num_imgs_after_filter_3 = 0
valid_imgs = 0

image_output_dir = Path("/scratch/shared/beegfs/janhr/data/unsup3d_extended/dogs_cropped/all")
image_input_dir = Path("/scratch/shared/beegfs/janhr/data/unsup3d_extended/tsinghua_dogs_high_res_cropped/all")
image_output_dir.mkdir(parents=True, exist_ok=True)

for index, row in df_all.iterrows():
    img_name = row["img_name"]
    img_path = image_input_dir / img_name
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. filter: pixel size
    num_pixels = row["num_pixels_initial_crop"]
    if num_pixels < 256*256:
        continue
    num_imgs_after_filter_1 += 1
        

    # 2. filter check if head and landmarks were detected
    if row["bb"] != True or row["lm"] != True:
        continue
    num_imgs_after_filter_2 += 1

    # 3. noise right or left of eye -> side view -> filter out
    right_eye = literal_eval(row["lm_02"])
    left_eye = literal_eval(row["lm_05"])
    noise = literal_eval(row["lm_03"])

    if noise[0] > right_eye[0] or noise[0] < left_eye[0]:
        continue
    num_imgs_after_filter_3 += 1
        
    
    x1 = int(row["bb_x1"])
    y1 = int(row["bb_y1"])
    x2 = int(row["bb_x2"])
    y2 = int(row["bb_y2"])
    
    img_width = img.shape[1]
    img_height = img.shape[0]

    x1 = max(0,x1)
    x2 = min(img_width,x2)
    y1 = max(0,y1)
    y2 = min(img_height,y2)
    
    # 5. Square images
    width = x2-x1
    height = y2-y1

    
    # make crop a square
    crop = None
    if (height > width):
        ratio = height/width
        border_x1 = int((ratio-1)*width/2)
        border_x2 = int((ratio-1)*width/2)
        border_x1_copy = 0
        border_x2_copy = 0
        # make crop wider: change x1 and x2 or copy border
        # extend x1 as best as possible
        new_x1 = x1 - border_x1
        if new_x1 < 0: 
            # extend x1 as fast as possible and extend border after crop
            border_x1_copy = border_x1 - x1
#             print(f"old x1: {x1} new_x1: {new_x1}, broder_x1_copy: {border_x1_copy}")
            new_x1 = 0
        x1 = new_x1
        
        new_x2 = x2 + border_x2
        if new_x2 > img_width: 
            # extend x1 as fast as possible and extend border after crop
            border_x2_copy = new_x2 - img_width
#             print(f"old x2: {x2} new_x2: {new_x2}, broder_x2_copy: {border_x2_copy}")
            new_x2 = img_width
        x2 = new_x2
             
        crop = img[y1:y2, x1:x2]
#         print(f"border_x1_copy: {border_x1_copy}, border_x2_copy: {border_x2_copy}")
        crop = cv2.copyMakeBorder(crop,0,0,border_x1_copy,border_x2_copy,cv2.BORDER_REPLICATE)
    
    elif(width > height):
        ratio = width/height
        border_y1 = int((ratio-1)*height/2)
        border_y2 = int((ratio-1)*height/2)
        border_y1_copy = 0
        border_y2_copy = 0
        # make crop wider: change x1 and x2 or copy border
        # extend x1 as best as possible
        new_y1 = y1 - border_y1
        if new_y1 < 0: 
            # extend x1 as fast as possible and extend border after crop
            border_y1_copy = border_y1 - y1
#             print(f"old y1: {y1} new_y1: {new_y1}, broder_y1_copy: {border_y1_copy}")
            new_y1 = 0
        y1 = new_y1
        
        new_y2 = y2 + border_y2
        if new_y2 > img_height: 
            # extend x1 as fast as possible and extend border after crop
            border_y2_copy = new_y2 - img_height
#             print(f"old y2: {y2} new_y2: {new_y2}, broder_y2_copy: {border_y2_copy}")
            new_y2 = img_height
        y2 = new_y2

        crop = img[y1:y2, x1:x2]
#         print(f"border_y1_copy: {border_y1_copy}, border_y2_copy: {border_y2_copy}")
        crop = cv2.copyMakeBorder(crop,border_y1_copy,border_y2_copy,0,0,cv2.BORDER_REPLICATE)
    else:
        crop = img[y1:y2, x1:x2]
    
    
# ONLY FOR DEBUG
#     fig = plt.figure(figsize=(16,16))
#     x1 = int(row["bb_x1"])
#     y1 = int(row["bb_y1"])
#     x2 = int(row["bb_x2"])
#     y2 = int(row["bb_y2"])
#     img_org = img.copy()
#     cv2.rectangle(img_org, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=(255,0,0), lineType=cv2.LINE_AA)

#     shape = np.array([literal_eval(row["lm_00"]),literal_eval(row["lm_01"]),literal_eval(row["lm_02"]),literal_eval(row["lm_03"]),literal_eval(row["lm_04"]), literal_eval(row["lm_05"])])
#     for i, p in enumerate(shape):
#         cv2.circle(img_org, center=tuple(p), radius=3, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
#         cv2.putText(img_org, str(i), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

#     ax1 = fig.add_subplot(221)
#     ax1.set_title("Original image with bounding box + key points")
#     ax1.xaxis.tick_top()
#     ax1.imshow(img_org)
    
#     ax1 = fig.add_subplot(222)
#     ax1.set_title("Final cropped img")
#     ax1.xaxis.tick_top()
#     ax1.imshow(crop)
#     break

    # resize and save image
    resized_img = cv2.resize(crop, (256,256))
    img_out = cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR)
    img_output_path = str(image_output_dir / img_name)
    cv2.imwrite(img_output_path, img_out)

    # bring image to same size
    valid_imgs += 1
    print(f"{index}/{len(df_all)}", end="\r")
    
total_num_imgs = len(df_all)

print(f"Total number of images: {total_num_imgs}")
print(f"Number of images which pass filter 1: {num_imgs_after_filter_1}")
print(f"Number of images which pass filter 1 & 2: {num_imgs_after_filter_2}")
print(f"Number of images which pass filter 1,2 & 3: {num_imgs_after_filter_3}")
print(f"Total number of valid images: {valid_imgs}")
