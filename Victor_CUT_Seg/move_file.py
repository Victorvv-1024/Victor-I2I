from PIL import Image
import os
from glob import glob
import numpy as np
import shutil
import re


if __name__ == '__main__':
    img_path = 'datasets/TLA4AE/white_TLA4_manual'
    dest_path = 'datasets/Victor_dataset/white'
    src_img = []
    for ext in ('*.jpg', '*.png', '*.JPG', '*.PNG'):
        src_img.extend(glob(os.path.join(img_path, ext)))

    for img_name in src_img:
        # print(img_name)
        img = Image.open(img_name)
        img = np.array(img)
        if img.shape[-1] > 3: continue
        else:
            name = re.split('[/]', img_name)[-1]
            # print(name)
            shutil.copyfile(os.path.join(img_name), os.path.join(dest_path, name))