import numpy as np
import os
import shutil
from glob import glob

if __name__ == '__main__':
    path = 'datasets/victor_dataset/white'
    files_all = []
    for ext in ('*.jpg', '*.png', '*.JPG', '*.PNG'):
        files_all.extend(glob(os.path.join(path, ext)))
    print(len(files_all))

    # # clean duplicates
    # # 1. lower all filenames
    # files_all = [x.lower() for x in files_all]
    # # 2. convert to set
    # files_all_set = set(files_all)
    # # 3. convert it back to list
    # files_all_clean = list(files_all_set)
    # print(len(files_all_clean))


    files_train = np.random.choice(files_all, int(len(files_all)*.8), replace=False)
    print(len(files_train))
    files_test = [file for file in files_all if file not in files_train]

    output_train = os.path.join(path, 'train')
    output_test = os.path.join(path, 'test')

    print('start')

    for file in files_train:
        # try:
        #     shutil.move(file, output_train)
        # except FileNotFoundError:
        #     file = file[:-3] + file[-3:].upper()
        #     shutil.move(file, output_train)
        shutil.move(file, output_train)

    for file in files_test:
        # try:
        #     shutil.move(file, output_test)
        # except FileNotFoundError:
        #     file = file[:-3] + file[-3:].upper()
        #     shutil.move(file, output_test)
        shutil.move(file, output_test)