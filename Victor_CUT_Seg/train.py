"""
v8 uses smp segmentor and dice score to train the whole network
v7 uses the smp segmentor, the loss function is normal BCE, the whole network is trained at the same time.
v5 trains the resnet segmentor first for 50 epochs, then train the whole composite network
v4 trains the unet segmentor first, then the whole composite network
"""
from options.train_options import ArgParse
from models.segmentor import Segmentor
from models import create_model
from utils.create_dataset import PairedEczemaDataset, UnpairedEczemaDataset
import torch
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import random
import numpy as np
from torch.nn.functional import softmax
from torch.autograd import Variable
from torchvision.utils import save_image
from utils import util
import pickle
import os


def on_epoch_end(generator, source_dataset, target_dataset, save_dir, epoch, num_img=2, ag=True):
    if ag:
        titles = ['Source', "Translated", "Mask", "Content", "Target", "Identity", "Mask", "Content"]
    else:
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

        src_input, src_real = source_tensor
        # tar_input, tar_real = target_tensor
        src_real = source_tensor
        tar_real = target_tensor
        # src_input = src_input.to(device)
        src_real = src_real.to(device)
        # tar_input = tar_input.to(device)
        tar_real = tar_real.to(device)


        # generate the translated img
        if ag:
            translated, translated_mask, translated_content = generator(src_real.unsqueeze(0))

            translated = util.tensor2img(translated)
            translated = (translated * 127.5 + 127.5).astype(np.uint8)
            # translated = (translated * 0.5 + 0.5).astype(np.uint8)

            translated_mask = util.tensor2img(translated_mask, isMask=True)

            translated_content = util.tensor2img(translated_content)
            translated_content = (translated_content * 127.5 + 127.5).astype(np.uint8)
            # translated_content = (translated_content * 0.5 + 0.5).astype(np.uint8)

        else:
            translated = generator(src_real)
            translated = util.tensor2img(translated)
            translated = (translated * 127.5 + 127.5).astype(np.uint8)

        # generate identity image
        if ag:
            idt, idt_mask, idt_content = generator(tar_real.unsqueeze(0))

            idt = util.tensor2img(idt)
            idt = (idt * 127.5 + 127.5).astype(np.uint8)
            # idt = (idt * 0.5 + 0.5).astype(np.uint8)

            idt_mask = util.tensor2img(idt_mask.data, isMask=True)

            idt_content = util.tensor2img(idt_content)
            idt_content = (idt_content * 127.5 + 127.5).astype(np.uint8)
            # idt_content = (idt_content * 0.5 + 0.5).astype(np.uint8)
        else:
            idt = generator(tar_real)
            idt = util.tensor2img(idt)
            idt = (idt * 127.5 + 127.5).astype(np.uint8)
        
        # source img
        real_img = util.tensor2img(src_real)
        source_img = (real_img * 127.5 + 127.5).astype(np.uint8)
        # source_img = (real_img * 0.5 + 0.5).astype(np.uint8)


        # target img
        target = util.tensor2img(tar_real)
        target = (target * 127.5 + 127.5).astype(np.uint8)
        # target = (target * 0.5 + 0.5).astype(np.uint8)


        if ax.ndim == 1:
            if ag:
                [ax[j].imshow(img) for j, img in enumerate([source_img, translated, translated_mask, translated_content, target, idt, idt_mask, idt_content])]
                [ax[j].axis("off") for j in range(len(titles))]
            else:
                [ax[j].imshow(img) for j, img in enumerate([source_img, translated, target, idt])]
                [ax[j].axis("off") for j in range(len(titles))]
        elif ax.ndim > 1:
            if ag:
                [ax[i, j].imshow(img) for j, img in enumerate([source_img, translated, translated_mask, translated_content, target, idt, idt_mask, idt_content])]
                [ax[i, j].axis("off") for j in range(len(titles))]
            else:
                [ax[i, j].imshow(img) for j, img in enumerate([source_img, translated, target, idt])]
                [ax[i, j].axis("off") for j in range(len(titles))]
    
    save_dir = os.path.join(save_dir, 'img')
    # save the images
    plt.savefig(f'{save_dir}/epoch={epoch + 1}.png')
    plt.close()

def on_epoch_end_seg(generator, segmentor, source_dataset, target_dataset, save_dir, epoch, num_img=2):
    titles = ['source content', 'translated sc', "source background", 'translated']
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

        # attach
        mask_A, real_A = source_tensor
        mask_A = mask_A.to(device)
        real_A = real_A.to(device)

        # predict the mask
        pr_mask = segmentor(real_A.unsqueeze(0))
        # content_mask_A = softmax(pr_mask, dim=-1)
        # pr_mask = pr_mask[0]
        # content_mask_A = np.zeros([pr_mask.shape[1], pr_mask.shape[2]])
        # for r in range(pr_mask.shape[1]):
        #     for c in range(pr_mask.shape[2]):
        #         if 0 < pr_mask[0][r][c]:
        #             content_mask_A[r][c] = 1
        # # print(content_mask_A)
        # content_mask_A = Variable(torch.from_numpy(content_mask_A.astype(np.float32)))
        # content_mask_A = content_mask_A.unsqueeze(0).unsqueeze(0)

        # get content mask
        content_mask_A = pr_mask.repeat(1,3,1,1)
        content_mask_A = content_mask_A.to(device)

        # get the content from image
        content_A = real_A*content_mask_A

        # generator translate content_A
        fake_B_content = generator(content_A)

        # global picture
        background_A = real_A*(1-content_mask_A)
        fake_B_global = fake_B_content + background_A

        # cast tensor to img
        real_content = util.tensor2img(content_A)
        translated_content = util.tensor2img(fake_B_content)
        real_background = util.tensor2img(background_A)
        translated = util.tensor2img(fake_B_global)

        # normalize
        real_content = (real_content * 127.5 + 127.5).astype(np.uint8)
        translated_content = (translated_content * 127.5 + 127.5).astype(np.uint8)
        real_background = (real_background * 127.5 + 127.5).astype(np.uint8)
        translated = (translated * 127.5 + 127.5).astype(np.uint8)



        [ax[i, j].imshow(img) for j, img in enumerate([real_content, translated_content, real_background, translated])]
        [ax[i, j].axis("off") for j in range(len(titles))]
    
    save_dir = os.path.join(save_dir, 'img')
    # save the images
    plt.savefig(f'{save_dir}/epoch={epoch + 1}.png')
    plt.close()        

if __name__ == '__main__':
    opt = ArgParse().parse_args()
    
    # generate the train dataset
    train_src = PairedEczemaDataset(opt.train_src_dir)
    if opt.train_tar_dir == 'datasets/datasets_paired/train/pairedB':
        train_tar = PairedEczemaDataset(opt.train_tar_dir)
    else:
        train_tar = UnpairedEczemaDataset(opt.train_tar_dir)
    # generate the test dataset
    test_src = PairedEczemaDataset(opt.test_src_dir)
    test_tar = UnpairedEczemaDataset(opt.test_tar_dir)
    
    # create the dataloaders
    BATCH_SIZE = opt.batchSize
    DROP_LAST = True
    SHUFFLE = True
    src_dataloader = DataLoader(train_src, batch_size=BATCH_SIZE, drop_last=DROP_LAST, shuffle=SHUFFLE)
    tar_dataloader = DataLoader(train_tar, batch_size=BATCH_SIZE, drop_last=DROP_LAST, shuffle=SHUFFLE)
    
    # create gan model
    model = create_model(opt)

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

            if tar_img is None:
                continue
            
            data = (src_img, tar_img)
            
            if epoch == opt.epoch_count and idx == 0:
                print(f'Initialization starts here at epoch and idx: {epoch, idx}, pretrain the netF')
                model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()

            model.set_input(data)  # unpack data from dataset and apply preprocessing
            if opt.model[:2] == 'ag':
                model.optimize_parameters(epoch=epoch)
            else:
                model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
        
        if epoch % opt.save_epoch_freq == 0: # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d' % (epoch))
            model.save_networks(epoch)
            print(f'model saved successfully')
    
            # generate and save images after each epoch end
            print(f'save generated images')
            if opt.model == 'cyclegan' or opt.model == 'dcl' or opt.model == 'distance' or opt.model == 'ag_cycle':
                if opt.model == 'ag_cycle': 
                    on_epoch_end(model.netG_A, test_src, test_tar, os.path.join('./output', opt.out_dir), epoch-1, num_img=2, ag=True)
                else:
                    on_epoch_end(model.netG_A, test_src, test_tar, os.path.join('./output', opt.out_dir), epoch-1, num_img=2)
            else:
                if opt.model == 'ag_cut':
                    on_epoch_end(model.netG, test_src, test_tar, os.path.join('./output', opt.out_dir), epoch-1, num_img=2, ag=True)
                if opt.model[-3:] == 'seg':
                    on_epoch_end_seg(model.netG, model.netS, test_src, test_tar, os.path.join('./output', opt.out_dir), epoch-1, num_img=2)
                else:
                    on_epoch_end(model.netG, test_src, test_tar, os.path.join('./output', opt.out_dir), epoch-1, num_img=2)
            print(f'images are generated successfully')
        
        """save the training loss and the generate images after each epoch"""
        print('saving the loss at the end of epoch %d' % (epoch))
        loss_hist.append(model.get_current_losses())
        pickle.dump(loss_hist, open(os.path.join('./output', opt.out_dir, 'loss', 'loss_'+str(epoch)), 'wb'))
        print(f'training loss is saved successfully')

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
