"""
This package includes all the modules related to data loading and preprocessing
We returns the mask and the masked out real image for training
For testing, we return the mask and the NOT masked out real image
"""
import torch.utils.data as data
import os
from glob import glob
import numpy as np
from torch.autograd import Variable
from PIL import Image, UnidentifiedImageError
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from utils.util import img2tensor
import random
import matplotlib.pyplot as plt



class PairedEczemaDataset(data.Dataset):
    def __init__(self, src_img_path):
        super().__init__()
        
        self.src_img_path = []
        for ext in ('*.jpg', '*.png', '*.JPG', '*.PNG'):
            self.src_img_path.extend(glob(os.path.join(src_img_path, ext)))
        
    def __len__(self):
        return len(self.src_img_path)
    
    
    def transform(self, input_img, real_img):
        # Resize
        resize = T.Resize(size=(286,286), interpolation=T.InterpolationMode.NEAREST)
        input_img, real_img = resize(input_img), resize(real_img)

        # Random crop
        i, j, h, w = T.RandomCrop.get_params(
            real_img, output_size=(256, 256))
        real_img = TF.crop(real_img, i, j, h, w)
        input_img = TF.crop(input_img, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            real_img = TF.hflip(real_img)
            input_img = TF.hflip(input_img)

        return input_img, real_img
    
    def normalize(self, input_img, real_img):
        normalize_input_img = input_img.squeeze(0)
        normalize_input_img = torch.clamp(normalize_input_img, min = 0.0, max=1.0)

        normalize_real_img = real_img.squeeze(0)
        normalize_real_img = (normalize_real_img - 127.5) / 127.5

        return normalize_input_img, normalize_real_img
        
    
    def __getitem__(self, index):
        img_path = self.src_img_path[index]
        img_name = img_path.split('/')[-1]
        name, ext = img_name.split('.')
        mask_path = os.path.join('datasets/victor_dataset/white_mask/mask', name + '_mask.' + ext)
        
        input_img = np.asarray(plt.imread(mask_path))
        input_img = input_img[:,:,1]
        input_img = input_img[:, :, np.newaxis]
        real_img = np.asarray(Image.open(img_path))[:,:,:3]
           
        input_img = Variable(torch.from_numpy(input_img.astype(np.float32)))
        input_img = input_img.unsqueeze(0)
        input_img = input_img.permute(0,3,1,2)

        real_img = img2tensor(real_img)

        # transform the imgs
        t_input_img, t_real_img = self.transform(input_img, real_img)

        # normalize the imgs
        n_input_img, n_real_img = self.normalize(t_input_img, t_real_img)
        
        return n_input_img, n_real_img
    
class UnpairedEczemaDataset(data.Dataset):
    def __init__(self, src_img_path):
        super().__init__()
        
        self.src_img_path = []
        for ext in ('*.jpg', '*.png', '*.JPG', '*.PNG'):
            self.src_img_path.extend(glob(os.path.join(src_img_path, ext)))

        
    def __len__(self):
        return len(self.src_img_path)
    
    
    def transform(self, real_img):
        # Resize
        resize = T.Resize(size=(286,286), interpolation=T.InterpolationMode.NEAREST)
        real_img = resize(real_img)

        # Random crop
        i, j, h, w = T.RandomCrop.get_params(
            real_img, output_size=(256, 256))
        real_img = TF.crop(real_img, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            real_img = TF.hflip(real_img)

        return real_img
    
    def normalize(self, real_img):
        normalize_real_img = real_img.squeeze(0)
        normalize_real_img = (normalize_real_img - 127.5) / 127.5
        
        return normalize_real_img
        
    
    def __getitem__(self, index):
        try:
            real_img = np.asarray(Image.open(self.src_img_path[index]))[:,:,:3]
        except UnidentifiedImageError as e:
            real_img =  np.asarray(Image.open(self.src_img_path[np.random.randint(0,len(self.src_img_path))]))[:,:,:3]
           
        real_img = img2tensor(real_img)
        # transform the imgs
        t_real_img = self.transform(real_img)

        # normalize the imgs
        n_real_img = self.normalize(t_real_img)
        
        return n_real_img
    

class TestDataset(data.Dataset):
    def __init__(self, src_img_path):
        super().__init__()
        
        self.src_img_path = []
        for ext in ('*.jpg', '*.png', '*.JPG', '*.PNG'):
            self.src_img_path.extend(glob(os.path.join(src_img_path, ext)))

        
    def __len__(self):
        return len(self.src_img_path)
    
    
    def transform(self, real_img):
        # Resize
        resize = T.Resize(size=(256,256), interpolation=T.InterpolationMode.NEAREST)
        real_img = resize(real_img)

        return real_img
    
    def normalize(self, real_img):
        normalize_real_img = real_img.squeeze(0)
        normalize_real_img = (normalize_real_img - 127.5) / 127.5
        
        return normalize_real_img
        
    
    def __getitem__(self, index):
        try:
            real_img = np.asarray(Image.open(self.src_img_path[index]))[:,:,:3]
        except UnidentifiedImageError as e:
            real_img =  np.asarray(Image.open(self.src_img_path[np.random.randint(0,len(self.src_img_path))]))[:,:,:3]
           
        real_img = img2tensor(real_img)
        # transform the imgs
        t_real_img = self.transform(real_img)

        # normalize the imgs
        n_real_img = self.normalize(t_real_img)
        
        return n_real_img