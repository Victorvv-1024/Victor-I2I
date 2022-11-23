"""v3 has the same network structure and loss fuction as Kexins'
v4 trains the segmentor first for 100 epochs, then train the whole composite network 
"""
from options.train_options import ArgParse
from models.cut_seg import CUT_SEG_model
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


def on_epoch_end(generator, segmenter, source_dataset, target_dataset, save_dir, epoch, num_img=2):
    
    titles = ['Source', "Translated", "Target", "Identity", 'MASK','Predicted mask', 'Pr Masked out']
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
        src_input, src_real = source_tensor
        tar_input, tar_real = target_tensor
        
        if torch.backends.mps.is_available():
            src_input = src_input.to(torch.device('mps'))
            src_real = src_real.to(torch.device('mps'))
            tar_real = tar_real.to(torch.device('mps'))
        
        # generate the predicted mask for the src img
        pr_src_mask = segmenter(src_real)
        pr_src_mask = softmax(pr_src_mask, dim=-1)
        # pr_src_mask = pr_src_mask.squeeze(0).permute(1,2,0).cpu().detach().numpy() # as numpy array, shape (H x W x C)
        pr_src_mask = util.tensor2img(pr_src_mask)
        pr_src_mask = np.argmax(pr_src_mask,axis=-1)
        
        # use the predicted mask to mask out the src img
        real_img = util.tensor2img(src_real)
        masked_real_img = util.mask_image(pr_src_mask, real_img)
        
        # cast the masked real img into tensor
        masked_real_img = util.img2tensor(masked_real_img)
        
        if torch.backends.mps.is_available(): masked_real_img = masked_real_img.to(torch.device('mps'))
        
        # generate the translated img
        translated = generator(masked_real_img)
        translated = util.tensor2img(translated)
        translated = (translated * 127.5 + 127.5).astype(np.uint8)
        
        # source img
        source_img = (real_img * 127.5 + 127.5).astype(np.uint8)
        
        # generate identity image
        idt = generator(tar_real)
        idt = util.tensor2img(idt)
        idt = (idt * 127.5 + 127.5).astype(np.uint8)
        
        # original mask
        ori_mask = util.tensor2img(src_input, isMask=True)
        
        if ax.ndim == 1:
            [ax[j].imshow(img) for j, img in enumerate([source_img, translated, util.tensor2img(tar_real), idt, ori_mask, pr_src_mask, util.tensor2img(masked_real_img)])]
            [ax[j].axis("off") for j in range(6)]
        elif ax.ndim > 1:
            [ax[i, j].imshow(img) for j, img in enumerate([source_img, translated, util.tensor2img(tar_real), idt, ori_mask, pr_src_mask, util.tensor2img(masked_real_img)])]
            [ax[i, j].axis("off") for j in range(6)]
    
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
    BATCH_SIZE = 1
    DROP_LAST = True
    SHUFFLE = True
    src_dataloader = DataLoader(train_src, batch_size=BATCH_SIZE, drop_last=DROP_LAST, shuffle=SHUFFLE)
    tar_dataloader = DataLoader(train_tar, batch_size=BATCH_SIZE, drop_last=DROP_LAST, shuffle=SHUFFLE)
    
    # create the model
    model = CUT_SEG_model(opt)
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

            if epoch <= 50:
                # print(f'epoch = {epoch}, train segmentor only')
                model.set_input(data)
                model.optimize_segmentor() # optimize the segmentor first
            else:
                # print(f'epoch = {epoch}, train the composite network')
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
        pickle.dump(loss_hist, open(os.path.join(opt.out_dir, 'loss', 'loss_'+str(epoch)), 'wb'))
        print(f'training loss is saved successfully')

        # generate and save images after each epoch end
        print(f'save generated images')
        on_epoch_end(model.netG, model.netS, test_src, test_tar, opt.out_dir, epoch, num_img=1)
        print(f'images are generated successfully')

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
