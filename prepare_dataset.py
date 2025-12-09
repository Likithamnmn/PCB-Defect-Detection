import os
import random
import shutil
from glob import glob


img_root = "images"
label_root = "labels"


train_img = "images/train"
val_img = "images/val"
train_lbl = "labels/train"
val_lbl = "labels/val"


for path in [train_img, val_img, train_lbl, val_lbl]:
    os.makedirs(path, exist_ok=True)


image_files = []
for cls_folder in os.listdir(img_root):
    folder_path = os.path.join(img_root, cls_folder)
    if os.path.isdir(folder_path):
        image_files.extend(glob(os.path.join(folder_path, "*.jpg")))


random.shuffle(image_files)


split_ratio = 0.8
split_index = int(len(image_files) * split_ratio)

train_files = image_files[:split_index]
val_files = image_files[split_index:]

def move_files(files, dest_img, dest_lbl):
    for img_path in files:
        img_name = os.path.basename(img_path)
        lbl_name = img_name.replace(".jpg", ".txt")
        lbl_path = os.path.join(label_root, lbl_name)

        
        shutil.move(img_path, os.path.join(dest_img, img_name))

        
        if os.path.exists(lbl_path):
            shutil.move(lbl_path, os.path.join(dest_lbl, lbl_name))


move_files(train_files, train_img, train_lbl)
move_files(val_files, val_img, val_lbl)

print("Dataset prepared successfully!")
print(f"Train images: {len(train_files)}, Val images: {len(val_files)}")
