from collections import OrderedDict
import numpy as np
import os
import torch
import torch.nn as nn
import itertools

from .networks import define_G, define_D, define_S
from .losses import GANLoss, SEGLoss, DiceLoss
from utils.imagepool import ImagePool
from utils import util
from .networks import get_scheduler


class CycleGAN(nn.Module):
    def __init__(self, opt):
        super(CycleGAN, self).__init__()
        self.opt = opt
        self.optimizers = []
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'S']

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else: self.device = torch.device('cpu')

        if self.opt.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        
        self.netG_A = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.antialias, opt.antialias_up, opt)
        self.netG_B = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.antialias, opt.antialias_up, opt)
        self.netS_A = define_S(opt.input_nc, opt.num_class, opt.ngf, opt.netS, opt.normS, not opt.no_dropout, opt.init_type, opt.init_gain, opt.antialias, opt.antialias_up, opt)
        self.netS_B = define_S(opt.input_nc, opt.num_class, opt.ngf, opt.netS, opt.normS, not opt.no_dropout, opt.init_type, opt.init_gain, opt.antialias, opt.antialias_up, opt)

        if self.opt.isTrain:  # define discriminators
            self.netD_A = define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.antialias, opt)
            self.netD_B = define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.antialias, opt)
        if self.opt.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = GANLoss().to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            if opt.netS_Loss == 'bce' or opt.netS_Loss == 'BCE':
                self.criterionSEG = SEGLoss(seg_lambda=opt.netS_lambda).to(self.device)
            elif opt.netS_Loss == 'dice' or opt.netS_Loss == 'DICE':
                self.criterionSEG = DiceLoss().to(self.device)
            else: 
                raise NotImplementedError('segmentation loss function is not implemented')
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_S = torch.optim.Adam(itertools.chain(self.netS_A.parameters(), self.netS_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_S)
        
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
        # self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        # self.mask = torch.cat((self.mask_A, self.mask_B), dim=0)

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

        # self.masked_real = torch.cat((self.masked_real_A, self.masked_real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.masked_real_A    

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        # mask out the input image using the ground truth mask and generate fake image use the masked real image only if it is training
        if self.opt.isTrain:
            self.mask_realImage()
            self.fake_B = self.netG_A(self.masked_real_A)
            self.rec_A = self.netG_B(self.fake_B)

            self.fake_A = self.netG_B(self.masked_real_B)
            self.rec_B = self.netG_A(self.fake_A)
        else: 
            self.fake_B = self.netG_A(self.real_A)
            self.rec_A = self.netG_B(self.fake_B)

            self.fake_A = self.netG_B(self.real_B)
            self.rec_B = self.netG_A(self.fake_A)

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
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
    
    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
    
    def backward_S(self):
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        
        self.loss_S_A = self.criterionSEG(self.netS_A(self.real_A), self.mask_A).mean()
        self.loss_S_B = self.criterionSEG(self.netS_B(self.real_B), self.mask_B).mean()

        fake_A = self.fake_A.detach()
        fake_A_mask = self.netS_A(fake_A)
        self.loss_S_A_fake = self.criterionSEG(fake_A_mask, self.mask_A).mean()

        fake_B = self.fake_B.detach()
        fake_B_mask = self.netS_B(fake_B)
        self.loss_S_B_fake = self.criterionSEG(fake_B_mask, self.mask_B).mean()

        rec_A = self.rec_A.detach()
        rec_A_mask = self.netS_A(rec_A)
        self.loss_S_recA = self.criterionSEG(rec_A_mask, self.mask_A).mean() * lambda_A

        rec_B = self.rec_B.detach()
        rec_B_mask = self.netS_B(rec_B)
        self.loss_S_recB = self.criterionSEG(rec_B_mask, self.mask_B).mean() * lambda_B

        self.loss_S = self.loss_S_A + self.loss_S_B + self.loss_S_A_fake + self.loss_S_B_fake + self.loss_S_recA + self.loss_S_recB
        self.loss_S.backward()
    
    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        
        if self.opt.netS_lambda > 0:
            fake_B = self.fake_B
            fake_B_mask = self.netS_B(fake_B)
            loss_fake_SEG_B = self.criterionSEG(fake_B_mask, self.mask_B).mean()

            fake_A = self.fake_A
            fake_A_mask = self.netS_A(fake_A)
            loss_fake_SEG_A = self.criterionSEG(fake_A_mask, self.mask_A).mean()
        else: 
            loss_fake_SEG_B = 0.0
            loss_fake_SEG_A = 0.0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + loss_fake_SEG_A + loss_fake_SEG_B
        self.loss_G.backward()
    
    def data_dependent_initialize(self, data):
        return

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # S_A and S_B
        self.set_requires_grad([self.netS_A, self.netS_B], True)
        self.optimizer_S.zero_grad()
        self.backward_S()
        self.optimizer_S.step()
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

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

        self.print_networks()
        
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