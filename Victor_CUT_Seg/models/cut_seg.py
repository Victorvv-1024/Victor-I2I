""" Create a CUT/FastCUT model, described in the paper
Contrastive Learning for Unpaired Image-to-Image Translation
Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
ECCV, 2020 (https://arxiv.org/abs/2007.15651).
"""

import numpy as np
import os
import torch
import torch.nn as nn
from collections import OrderedDict

import utils.util as util
from .networks import define_G, define_F, define_S, define_D, get_scheduler
from .losses import GANLoss, SEGLoss, PatchNCELoss

class CUT_SEG_model(nn.Module):
    def __init__(self, opt):
        super(CUT_SEG_model, self).__init__()
        self.opt = opt
        self.optimizers = []
        # specify the training losses you want to print out.
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'SEG']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else: self.device = torch.device('cpu')

        print(f'using device: {self.device}')
        
        if self.opt.isTrain:
            self.model_names = ['G', 'F', 'D', 'S']
        else:  # during test time, only load G and S
            self.model_names = ['G', 'S']

        if self.opt.nce_idt and self.opt.isTrain:
            self.loss_names += ['NCE_Y']
            
        # define the generator, G
        # print(opt.input_nc, opt.output_nc, opt.ngf, opt.netG)
        self.netG = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.antialias, opt.antialias_up, opt)
        # define the sampler, F
        # print(opt.input_nc, opt.output_nc, opt.ngf, opt.netF)
        self.netF = define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.antialias, opt)
        # define the segmentor, S
        # print(opt.input_nc, opt.output_nc, opt.ngf, opt.netS)
        self.netS = define_S(opt.input_nc, opt.num_class, opt.ngf, opt.netS, opt.normS, not opt.no_dropout, opt.init_type, opt.init_gain, opt.antialias, opt.antialias_up, opt)
        
        if self.opt.isTrain:
            # print(opt.output_nc, opt.ndf, opt.netD)
            self.netD = define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.antialias, opt)
            
            # define loss functions
            self.criterionGAN = GANLoss().to(self.device)
            self.criterionNCE = []
            self.criterionSEG = SEGLoss(seg_lambda=opt.netS_lambda).to(self.device)
            
            for _ in self.nce_layers:
                nceLoss = PatchNCELoss(opt).to(self.device)
                self.criterionNCE.append(nceLoss)
                
            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            # define the optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_S = torch.optim.Adam(self.netS.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_S)
        
        # other utilities
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
            
    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        self.forward() # compute segmentation and fake image
        if self.opt.isTrain:
            self.compute_D_loss().backward() # calculate gradients for D
            self.compute_S_loss().backward() # calculate gradients for S
            self.compute_G_loss().backward() # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)
    
    """optimize the segmentor for the first 100 epoch, to generate good masks first"""
    def optimize_segmentor(self):
        # forward
        self.forward()

        # update S
        self.set_requires_grad(self.netS, True)
        self.optimizer_S.zero_grad()
        self.loss_S = self.compute_S_loss()
        self.loss_S.backward()
        self.optimizer_S.step()
    
    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update S
        self.set_requires_grad(self.netS, True)
        self.optimizer_S.zero_grad()
        self.loss_S = self.compute_S_loss()
        self.loss_S.backward()
        self.optimizer_S.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netS, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()
            
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (): include the data itself and its metadata information.
        """
        # A is the source and B is the target
        A, B = input
        self.mask_A, self.real_A = A
        self.mask_B, self.real_B = B
        """attach to the device"""
        self.mask_A = self.mask_A.to(self.device)
        self.real_A = self.real_A.to(self.device)
        self.mask_B = self.mask_B.to(self.device)
        self.real_B = self.real_B.to(self.device)
        # cat them
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        self.mask = torch.cat((self.mask_A, self.mask_B), dim=0)

    def mask_realImage(self):
        """mask out real image using the ground truth mask
        """
        mask_A_img = util.tensor2img(self.mask_A, isMask=True)
        real_A_img = util.tensor2img(self.real_A)
        mask_B_img = util.tensor2img(self.mask_B, isMask=True)
        real_B_img = util.tensor2img(self.real_B)
        masked_real_A_img = util.mask_image(mask_A_img, real_A_img)
        masked_real_B_img = util.mask_image(mask_B_img, real_B_img)
        self.masked_real_A = util.img2tensor(masked_real_A_img).to(self.device)
        self.masked_real_B = util.img2tensor(masked_real_B_img).to(self.device)
        self.masked_real = torch.cat((self.masked_real_A, self.masked_real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.masked_real_A
        
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        # mask out the input image using the ground truth mask and generate fake image use the masked real image only if it is training
        if self.opt.isTrain:
            self.mask_realImage()
            self.fake = self.netG(self.masked_real)
        else: self.fake = self.netG(self.real)

        self.fake_B = self.fake[:self.real_A.size(0)] # G_enc(X) -> Y
        if self.opt.nce_idt: self.idt_B = self.fake[self.real_A.size(0):] # G_enc(Y)
    
    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        self.loss_D_real = self.criterionGAN(self.pred_real, True).mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D
    
    def compute_S_loss(self):
        """Calculate SEG loss for the segmentor"""
        fake_mask = self.netS(self.real)
        self.loss_S_real = self.criterionSEG(fake_mask, self.mask).mean()

        fake_B = self.fake_B.detach()
        fake_B_mask = self.netS(fake_B)
        self.loss_S_fake = self.criterionSEG(fake_B_mask, self.mask_A).mean()

        self.loss_SEG = (self.loss_S_real + self.loss_S_fake) * 0.5

        self.loss_S = self.loss_SEG 
        # loss_S is used to optimize SEG 
        return self.loss_S
        
    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE
        
        if self.opt.netS_lambda > 0:
            fake_B = self.fake_B
            fake_B_mask = self.netS(fake_B)
            loss_fake_SEG = self.criterionSEG(fake_B_mask, self.mask_A).mean()
        else: loss_fake_SEG = 0.0

        self.loss_G = self.loss_G_GAN + loss_NCE_both + loss_fake_SEG

        return self.loss_G
    
    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
    
    
    """Auxilary functions"""
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda()
                else:
                    torch.save(net.cpu().state_dict(), save_path)
    
    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                if self.opt.isTrain and self.opt.pretrained_name is not None:
                    load_dir = os.path.join(self.opt.checkpoints_dir, self.opt.pretrained_name)
                else:
                    load_dir = self.save_dir

                load_path = os.path.join(load_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                #    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)
                
    def print_networks(self):
        """Print the total number of parameters in the network and (if verbose) network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')
    
    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.opt.isTrain:
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.opt.isTrain or opt.continue_train:
            load_suffix = opt.epoch
            self.load_networks(load_suffix)

        # self.print_networks()
        
    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret
    
    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
    
    def parallelize(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                setattr(self, 'net' + name, torch.nn.DataParallel(net, device_ids=[0]))
        
    # For TESTING
    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        """
        with torch.no_grad():
            self.forward()