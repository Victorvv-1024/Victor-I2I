"""
v8 uses smp segmentor and dice score to train the whole network
v7 uses the smp segmentor, the loss function is normal BCE, the whole network is trained at the same time.
v5 trains the resnet segmentor first for 50 epochs, then train the whole composite network
v4 trains the unet segmentor first, then the whole composite network
"""
from options.train_options import ArgParse
from models.segmentor import Segmentor
from models.cut_seg import CUT_SEG_model
from models.cycleGAN import CycleGAN
from models.dcl_model import DCLModel
from models.distance_model import DistanceGAN
from utils.create_dataset import EczemaDataset
import torch
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import random
import numpy as np
from torch.nn.functional import softmax
from torch.autograd import Variable
from utils import util
import pickle
import os


def on_epoch_end(generator, source_dataset, target_dataset, save_dir, epoch, num_img=2):
    
    titles = ['Source', "Translated", "Target", "Identity"]
    _, ax = plt.subplots(num_img, len(titles), figsize=(20, 10))
    if ax.ndim == 1:
        [ax[i].set_title(title) for i, title in enumerate(titles)]
    elif ax.ndim > 1:
        [ax[0, i].set_title(title) for i, title in enumerate(titles)]
    
    # randomly pick the tensor from the test dataset
    source_random_idx = []
    target_random_idx = []
    for _ in range(num_img):
        source_random_idx.append(random.randint(0, len(source_dataset)-1))
        target_random_idx.append(random.randint(0, len(target_dataset)-1))
        
    # generate imgs from the random picked dataset
    for i, idx in enumerate(zip(source_random_idx,target_random_idx)):
        source_tensor, target_tensor = source_dataset[idx[0]], target_dataset[idx[1]]

        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else: device = torch.device('cpu')

        source_tensor = source_tensor
        target_tensor = target_tensor

        src_input, src_real = source_tensor
        tar_input, tar_real = target_tensor
        src_input = src_input.to(device)
        src_real = src_real.to(device)
        tar_input = tar_input.to(device)
        tar_real = tar_real.to(device)

        # source img
        real_img = util.tensor2img(src_real)
        source_img = (real_img * 127.5 + 127.5).astype(np.uint8)

        # generate the translated img
        translated = generator(src_real)
        translated = util.tensor2img(translated)
        translated = (translated * 127.5 + 127.5).astype(np.uint8)

        # target img
        target = util.tensor2img(tar_real)
        target = (target * 127.5 + 127.5).astype(np.uint8)
        
        # generate identity image
        idt = generator(tar_real)
        idt = util.tensor2img(idt)
        idt = (idt * 127.5 + 127.5).astype(np.uint8)
        
        if ax.ndim == 1:
            [ax[j].imshow(img) for j, img in enumerate([source_img, translated, target, idt])]
            [ax[j].axis("off") for j in range(len(titles))]
        elif ax.ndim > 1:
            [ax[i, j].imshow(img) for j, img in enumerate([source_img, translated, target, idt])]
            [ax[i, j].axis("off") for j in range(len(titles))]
    
    save_dir = os.path.join(save_dir, 'img')
    # save the images
    plt.savefig(f'{save_dir}/epoch={epoch + 1}.png')
    plt.close()
        

if __name__ == '__main__':
    opt = ArgParse().parse_args()
    
    # generate the train dataset
    train_src = EczemaDataset(opt.train_src_dir)
    train_tar = EczemaDataset(opt.train_tar_dir)
    # generate the test dataset
    test_src = EczemaDataset(opt.test_src_dir)
    test_tar = EczemaDataset(opt.test_tar_dir)
    
    # create the dataloaders
    BATCH_SIZE = opt.batchSize
    DROP_LAST = True
    SHUFFLE = True
    src_dataloader = DataLoader(train_src, batch_size=BATCH_SIZE, drop_last=DROP_LAST, shuffle=SHUFFLE)
    tar_dataloader = DataLoader(train_tar, batch_size=BATCH_SIZE, drop_last=DROP_LAST, shuffle=SHUFFLE)
    
    # create gan model
    if opt.model == 'cyclegan':
        model = CycleGAN(opt)
    elif opt.model == 'cut_seg':
        model = CUT_SEG_model(opt)
    elif opt.model == 'dcl':
        model = DCLModel(opt)
    elif opt.model == 'distance':
        model = DistanceGAN(opt, src_dataloader, tar_dataloader)

    loss_hist = []

    if opt.load:
        model.load_networks(opt.load_epoch)
        start_epoch = opt.load_epoch
    else: start_epoch = opt.epoch_count
    
    # start training
    for epoch in range(start_epoch, opt.n_epochs + opt.n_epochs_decay + 1): # outer loop for different epochs; we save the model by <epoch_count>
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        epoch_start_time = time.time()  # timer for entire epoch
        print(f'training at epoch: {epoch}')

        for idx, (src_img, tar_img) in enumerate(zip(src_dataloader, tar_dataloader)):  # inner loop within one epoch
            data = (src_img, tar_img)
            
            if epoch == opt.epoch_count and idx == 0:
                print(f'Initialization starts here at epoch and idx: {epoch, idx}, pretrain the netF')
                model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()

            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
        
        # end of one epoch
        if epoch % opt.save_epoch_freq == 0: # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d' % (epoch))
            model.save_networks(epoch)
            print(f'model saved successfully')
        
            """save the training loss and the generate images after each epoch"""
            print('saving the loss at the end of epoch %d' % (epoch))
            loss_hist.append(model.get_current_losses())
            pickle.dump(loss_hist, open(os.path.join('./output', opt.out_dir, 'loss', 'loss_'+str(epoch)), 'wb'))
            print(f'training loss is saved successfully')

            # generate and save images after each epoch end
            print(f'save generated images')
            if opt.model == 'cyclegan' or opt.model == 'dcl' or opt.model == 'distance':
                on_epoch_end(model.netG_A, test_src, test_tar, os.path.join('./output', opt.out_dir), epoch, num_img=2)
            elif opt.model == 'cut_seg':
                on_epoch_end(model.netG, test_src, test_tar, os.path.join('./output', opt.out_dir), epoch, num_img=2)
            print(f'images are generated successfully')

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
