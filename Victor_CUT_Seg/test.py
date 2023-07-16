from options.train_options import ArgParse
from models import create_model
from utils.create_dataset import TestDataset
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from torch.nn.functional import softmax
from torch.autograd import Variable
from torchvision.utils import save_image
from utils import util
import pickle
import os
from utils import image_generation


if __name__ == '__main__':
    opt = ArgParse().parse_args()

    #setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else: 
        device = torch.device('cpu')

    # create dataset
    test_src = TestDataset(opt.test_src_dir)
    test_tar = TestDataset(opt.test_tar_dir)
    test_src_dataloader = DataLoader(test_src, batch_size=1, shuffle=False)
    test_tar_dataloader = DataLoader(test_tar, batch_size=1, shuffle=False)

    # create model
    model = create_model(opt)
    # load netG
    netG = getattr(model, 'netG')
    if isinstance(netG, torch.nn.DataParallel):
        netG = netG.module
    load_path = opt.load_path
    state_dict = torch.load(load_path, map_location=str(device))
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    netG.load_state_dict(state_dict)
    
    # create image
    image_generation.ag_generate_image(test_src_dataloader, netG, opt=opt, device=device, translated_only=True)