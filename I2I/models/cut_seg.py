import numpy as np
import os
import torch
import itertools

import utils.util as util
from .base_model import BaseModel
from .networks import define_G, define_F,define_D, define_S
from .losses import GANLoss, SEGLoss, PatchNCELoss, DiceLoss

class CUT_SEG_model(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020
    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    def __init__(self, opt):
        """we pass the segmentor as an argument to this model"""
        BaseModel.__init__(self, opt)
        self.opt = opt
        # specify the training losses you want to print out.
        self.loss_names = ['G_GAN', 'D', 'G', 'NCE', 'S']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        print(f'using device: {self.device}')
        
        if self.opt.isTrain:
            self.model_names = ['G', 'F', 'D_global','S', 'D_local']
        else:  # during test time, only load G and S
            self.model_names = ['G','S']

        if self.opt.nce_idt and self.opt.isTrain:
            self.loss_names += ['NCE_Y']
            
        # define the generator, G
        self.netG = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.antialias, opt.antialias_up, opt)
        # define the sampler, F
        self.netF = define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.antialias, opt)
        # define the segmentor, S
        self.netS = define_S(opt.input_nc, opt.num_class, opt.ngf, opt.netS, opt.normS, not opt.no_dropout, opt.init_type, opt.init_gain, opt.antialias, opt.antialias_up, opt)
        # self.load_netS(path=opt.load_seg_path, epoch=opt.load_seg_epoch)
        
        if self.opt.isTrain:
            # print(opt.output_nc, opt.ndf, opt.netD)
            # global discriminator
            self.netD_global = define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.antialias, opt)
            # local discriminator
            self.netD_local = define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.antialias, opt)
            # define loss functions
            self.criterionGAN = GANLoss().to(self.device)
            self.criterionNCE = []
            if opt.netS_Loss == 'bce' or opt.netS_Loss == 'BCE':
                self.criterionSEG = SEGLoss(seg_lambda=opt.netS_lambda).to(self.device)
            elif opt.netS_Loss == 'dice' or opt.netS_Loss == 'DICE':
                self.criterionSEG = DiceLoss().to(self.device)
            else: 
                raise NotImplementedError('segmentation loss function is not implemented')
            
            for _ in self.nce_layers:
                nceLoss = PatchNCELoss(opt).to(self.device)
                self.criterionNCE.append(nceLoss)
                
            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            # define the optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            # self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_global.parameters(), self.netD_local.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_S = torch.optim.Adam(self.netS.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_S)


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
    
    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad([self.netD_global, self.netD_local], True)
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
        self.set_requires_grad([self.netD_global, self.netD_local], False)
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
        self.real_B = B
        """attach to the device"""
        self.mask_A = self.mask_A.to(self.device)
        self.real_A = self.real_A.to(self.device)
        # self.mask_B = self.mask_B.to(self.device)
        self.real_B = self.real_B.to(self.device)
        # cat them
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        # self.mask = torch.cat((self.mask_A, self.mask_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.mask_A

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # get the content mask
        self.content_mask_A = self.mask_A
        self.content_mask_A = self.content_mask_A.repeat(1,3,1,1)
        # self.content_mask_A = self.content_mask[:self.real_A.size(0)]
        self.content_mask_B = self.netS(self.real_B).detach()
        self.content_mask_B = self.content_mask_B.repeat(1,3,1,1)
        self.content_mask = torch.cat((self.content_mask_A, self.content_mask_B), dim=0)
        # if self.opt.nce_idt: self.content_mask_B = self.content_mask[self.real_A.size(0):]

        # get the content from the image
        # self.content_A = self.real*self.content_mask
        self.content_A = self.real_A * self.content_mask_A
        self.content_B = self.real_B * self.content_mask_B
        self.content = torch.cat((self.content_A, self.content_B), dim=0)
        # self.content_A = self.content[:self.real_A.size(0)] # X_content
        # self.content_B = self.content[self.real_A.size(0):] # Y_content

        # flip both real and content
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])
                self.content = torch.flip(self.content, [3])
                self.content_mask = torch.flip(self.mask, [3])
        
        # generator only learns to generate content
        self.fake_content = self.netG(self.content)
        self.fake_B_content = self.fake_content[:self.real_A.size(0)] #G(X_content)
        if self.opt.nce_idt: self.idt_B_content = self.fake_content[self.real_A.size(0):] # G(Y_content)

        # global picture
        self.background = self.real*(1-self.content_mask)
        self.background_A = self.background[:self.real_A.size(0)] # X_background
        self.fake_B_global = self.fake_B_content + self.background_A # G(X) = G(X_content) + X_background

    
    def compute_D_loss(self):
        """Calculate GAN loss for both discriminator"""
        # local discriminator focus on the content
        fake_B_content = self.fake_B_content.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake_B_content = self.netD_local(fake_B_content)
        self.loss_D_content_fake = self.criterionGAN(pred_fake_B_content, False).mean()
        # Real content from distribution B
        self.pred_real_content = self.netD_local(self.content_B)
        self.loss_D_real_content = self.criterionGAN(self.pred_real_content, True).mean()

        # global discriminator focus on the whole image
        fake_B_global = self.fake_B_global.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake_global = self.netD_global(fake_B_global)
        self.loss_D_global_fake = self.criterionGAN(pred_fake_global, False).mean()
        # Real gloval image from distribution B
        self.pred_real_global= self.netD_global(self.real_B)
        self.loss_D_real_gloobal = self.criterionGAN(self.pred_real_global, True).mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_content_fake + self.loss_D_real_content + self.loss_D_global_fake + self.loss_D_real_gloobal)*0.25

        return self.loss_D
    
    def compute_S_loss(self):
        """Calculate SEG loss for the segmentor"""
        # compute mask for real
        fake_mask_A = self.netS(self.real_A)
        self.loss_S_real = self.criterionSEG(fake_mask_A, self.mask_A).mean()


        fake_B_global = self.fake_B_global.detach()
        fake_B_mask = self.netS(fake_B_global)
        self.loss_S_fake = self.criterionSEG(fake_B_mask, self.mask_A).mean()
        self.loss_S = (self.loss_S_real + self.loss_S_fake) * 0.5

        return self.loss_S
        
    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake_local = self.fake_B_content
        fake_global = self.fake_B_global
        # First, G(A) should fake both discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake_local = self.netD_local(fake_local)
            pred_fake_global = self.netD_global(fake_global)
            self.loss_G_GAN = self.criterionGAN(pred_fake_local, True).mean() * self.opt.lambda_GAN
            self.loss_G_GAN += self.criterionGAN(pred_fake_global, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0: # compare X_content and G(X_content)
            self.loss_NCE = self.calculate_NCE_loss(self.content_A, self.fake_B_content)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0: # compare Y_content and G(Y_content)
            self.loss_NCE_Y = self.calculate_NCE_loss(self.content_B, self.idt_B_content)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE
    

        self.loss_G = self.loss_G_GAN + loss_NCE_both

        return self.loss_G
    
    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [-3,2]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers