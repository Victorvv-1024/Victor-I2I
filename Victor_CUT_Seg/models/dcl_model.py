import numpy as np
import os
import torch
import itertools

import utils.util as util
from .base_model import BaseModel
from .networks import define_G, define_F,define_D, define_S
from .losses import GANLoss, SEGLoss, PatchNCELoss, DiceLoss
from utils.imagepool import ImagePool

class DCLModel(BaseModel):
    """ This class implements DCLGAN model.
    This code is inspired by CUT and CycleGAN.
    """
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'NCE1', 'D_B', 'G_B', 'NCE2', 'G']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['idt_B', 'idt_A']

        if self.isTrain:
            self.model_names = ['G_A', 'F1', 'D_A', 'G_B', 'F2', 'D_B']
        else:  # during test time, only load G
            self.model_names = ['G_A', 'G_B']

        # define networks (both generator and discriminator)
        self.netG_A = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.antialias, opt.antialias_up, opt)
        self.netG_B = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.antialias, opt.antialias_up, opt)
        self.netF1 = define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.antialias, opt)
        self.netF2 = define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.antialias, opt)
        # define the segmentor, S
        # self.netS = define_S(opt.input_nc, opt.num_class, opt.ngf, opt.netS, opt.normS, not opt.no_dropout, opt.init_type, opt.init_gain, opt.antialias, opt.antialias_up, opt)
        # self.load_netS(path=opt.load_seg_path, epoch=opt.load_seg_epoch)

        if self.isTrain:
            self.netD_A = define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.antialias, opt)
            self.netD_B = define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.antialias, opt)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = GANLoss().to(self.device)
            self.criterionNCE = []
            if opt.netS_Loss == 'bce' or opt.netS_Loss == 'BCE':
                self.criterionSEG = SEGLoss(seg_lambda=opt.netS_lambda).to(self.device)
            elif opt.netS_Loss == 'dice' or opt.netS_Loss == 'DICE':
                self.criterionSEG = DiceLoss().to(self.device)
            else: 
                raise NotImplementedError('segmentation loss function is not implemented')

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.criterionSim = torch.nn.L1Loss('sum').to(self.device)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
    
    # def load_netS(self, path, epoch):
    #     """Load all the networks from the disk.

    #     Parameters:
    #         epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
    #     """
    #     load_filename = '%s_net_%s.pth' % (epoch, 'S')
    #     load_path = os.path.join(path, load_filename)
    #     netS = getattr(self, 'netS')
    #     if isinstance(netS, torch.nn.DataParallel):
    #         netS = netS.module
    #     state_dict = torch.load(load_path, map_location=str(self.device))
    #     if hasattr(state_dict, '_metadata'):
    #         del state_dict._metadata
    #     netS.load_state_dict(state_dict)
        
    #     self.netS = netS

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        self.forward() 
        if self.opt.isTrain:
            self.compute_G_loss().backward()  # calculate graidents for G
            self.backward_D_A()  # calculate gradients for D_A
            self.backward_D_B()  # calculate graidents for D_B
            self.optimizer_F = torch.optim.Adam(itertools.chain(self.netF1.parameters(), self.netF2.parameters()))
            self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()

        # update G
        self.set_requires_grad([self.netD_A, self.netD_B], False)
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
        if self.isTrain:
            self.mask_realImage()
            self.fake_B = self.netG_A(self.masked_real_A)
            self.fake_A = self.netG_B(self.masked_real_B)
            if self.opt.nce_idt:
                self.idt_A = self.netG_A(self.masked_real_B)
                self.idt_B = self.netG_B(self.masked_real_A)
        else: 
            self.fake_B = self.netG_A(self.real_A)
            self.fake_A = self.netG_B(self.real_B)
            if self.opt.nce_idt:
                self.idt_A = self.netG_A(self.real_B)
                self.idt_B = self.netG_B(self.real_A)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B) * self.opt.lambda_GAN

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A) * self.opt.lambda_GAN

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fakeB = self.fake_B
        fakeA = self.fake_A

        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fakeB = self.netD_A(fakeB)
            pred_fakeA = self.netD_B(fakeA)
            self.loss_G_A = self.criterionGAN(pred_fakeB, True).mean() * self.opt.lambda_GAN
            self.loss_G_B = self.criterionGAN(pred_fakeA, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_A = 0.0
            self.loss_G_B = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE1 = self.calculate_NCE_loss1(self.real_A, self.fake_B) * self.opt.lambda_NCE
            self.loss_NCE2 = self.calculate_NCE_loss2(self.real_B, self.fake_A) * self.opt.lambda_NCE
        else:
            self.loss_NCE1, self.loss_NCE_bd, self.loss_NCE2 = 0.0, 0.0, 0.0
            
        if self.opt.lambda_NCE > 0.0:

            # L1 IDENTICAL Loss
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * self.opt.lambda_identity
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * self.opt.lambda_identity
            loss_NCE_both = (self.loss_NCE1 + self.loss_NCE2) * 0.5 + (self.loss_idt_A + self.loss_idt_B) * 0.5
        else:
            loss_NCE_both = (self.loss_NCE1 + self.loss_NCE2) * 0.5
        
        if self.opt.netS_lambda > 0:
            fake_B_mask = self.netS(fakeB)
            loss_fake_SEG_B = self.criterionSEG(fake_B_mask, self.mask_B).mean()
            
            fake_A_mask = self.netS(fakeA)
            loss_fake_SEG_A = self.criterionSEG(fake_A_mask, self.mask_A).mean()
            loss_SEG_both = (loss_fake_SEG_A + loss_fake_SEG_B) * 0.5
        else:
            loss_SEG_both = 0

        self.loss_G = (self.loss_G_A + self.loss_G_B) * 0.5 + loss_NCE_both + loss_SEG_both
        return self.loss_G

    def calculate_NCE_loss1(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG_B(tgt, self.nce_layers, encode_only=True)
        feat_k = self.netG_A(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF1(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF2(feat_q, self.opt.num_patches, sample_ids)
        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / n_layers

    def calculate_NCE_loss2(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG_A(tgt, self.nce_layers, encode_only=True)
        feat_k = self.netG_B(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF2(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF1(feat_q, self.opt.num_patches, sample_ids)
        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / n_layers