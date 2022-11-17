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
from PIL import Image
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from utils.util import img2tensor



class EczemaDataset(data.Dataset):
    
    def __init__(self, src_img_path):
        super().__init__()
        
        self.src_img_path = []
        for ext in ('*.jpg', '*.png', '*.JPG', '*.PNG'):
            self.src_img_path.extend(glob(os.path.join(src_img_path, ext)))

        
    def __len__(self):
        return len(self.src_img_path)
    
    
    def transform(self):
        transforms = T.Compose(
            [T.Resize(size=(286,286), interpolation=T.InterpolationMode.NEAREST),
             T.RandomCrop(size=(256,256)),
             T.RandomHorizontalFlip(p=0.5)
             ]
        )
        return transforms
    
    def normalize(self, input_img, real_img):
        normalize_input_img = input_img.squeeze(0)
        normalize_input_img = normalize_input_img[0,:,:]
        normalize_input_img = torch.clamp(normalize_input_img, max=1.0)
        normalize_input_img = F.one_hot(normalize_input_img.to(torch.int64), 2)
        normalize_input_img = normalize_input_img.permute(2,0,1)
        
        normalize = T.Normalize(mean=(127.5,127.5,127.5), std=(127.5,127.5,127.5))
        normalize_real_img = normalize(real_img)
        normalize_real_img = normalize_real_img.squeeze(0)
        
        return normalize_input_img, normalize_real_img
        
    
    def __getitem__(self, index):
        src_img = np.asarray(Image.open(self.src_img_path[index]))
        w = src_img.shape[1] // 2
        input_img, real_img = src_img[:,:w,:], src_img[:,w:,:]
           
        input_img, real_img = img2tensor(input_img), img2tensor(real_img)
        # transform the imgs
        transform = self.transform()
        t_input_img, t_real_img = transform(input_img), transform(real_img)
        # normalize the imgs
        n_input_img, n_real_img = self.normalize(t_input_img, t_real_img)
        
        return n_input_img, n_real_img
        
        