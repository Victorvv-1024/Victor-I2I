import numpy as np
import torch
import itertools

from .base_model import BaseModel
from .networks import define_G, define_F,define_D
from .losses import GANLoss, PatchNCELoss

class AG_Cut(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020
    The code borrows heavily from the PyTorch implementation of CycleGAN and Attention-Guided GAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    https://github.com/Ha0Tang/AttentionGAN/tree/master/AttentionGAN-v1
    """
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        # specify the training losses you want to print out.
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'reg']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        
        if self.opt.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G and S
            self.model_names = ['G']

        if self.opt.nce_idt and self.opt.isTrain:
            self.loss_names += ['NCE_Y']
            
        # define the generator, G
        self.netG = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.antialias, opt.antialias_up, opt)
        # define the sampler, F
        self.netF = define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.antialias, opt)
        
        if self.opt.isTrain:
            self.netD = define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.antialias, opt)
            
            # define loss functions
            self.criterionGAN = GANLoss().to(self.device)
            self.criterionPix = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.criterionNCE = []
            
            for _ in self.nce_layers:
                nceLoss = PatchNCELoss(opt).to(self.device)
                self.criterionNCE.append(nceLoss)
            
            # define the optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.scheduler_G = torch.optim.lr_scheduler.CyclicLR(self.optimizer_G, base_lr=opt.base_lr, max_lr=opt.lr, cycle_momentum=False)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.scheduler_D = torch.optim.lr_scheduler.CyclicLR(self.optimizer_D, base_lr=opt.base_lr, max_lr=opt.lr, cycle_momentum=False)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


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
            self.compute_D_loss().backward() # calculate gradients for D
            self.compute_G_loss(epoch=1).backward() # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.scheduler_F = torch.optim.lr_scheduler.CyclicLR(self.optimizer_F, base_lr=self.opt.base_lr, max_lr=self.opt.lr, cycle_momentum=False)
                self.optimizers.append(self.optimizer_F)
    
    def optimize_parameters(self, epoch):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        if self.opt.cycliclr:

            self.scheduler_D.step()
        else:

            self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        
        self.loss_G = self.compute_G_loss(epoch)
        self.loss_G.backward()
        if self.opt.cycliclr:

            self.scheduler_G.step()
        else:

            self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            if self.opt.cycliclr:

                self.scheduler_F.step()
            else:

                self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (): include the data itself and its metadata information.
        """
         # A is the source and B is the target
        A, B = input

        self.true_mask_A, self.real_A = A
        self.true_mask_A = self.true_mask_A.to(self.device)
        self.real_A = self.real_A.to(self.device)

        if isinstance(B, list):
            _, self.real_B = B
        else: self.real_B = B
        self.real_B = self.real_B.to(self.device)
        
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # print(self.real_A.shape)
        # print(self.real_B.shape)
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A

        # self.true_mask_A = self.true_mask_A[:,[1]]
        self.true_mask_A = self.true_mask_A.repeat(1,3,1,1)

        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])
                
        self.fake, self.mask, self.temp = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)] # G_enc(X) -> Y
        self.mask_B = self.mask[:self.real_A.size(0)] # attention mask
        self.temp_B = self.temp[:self.real_A.size(0)]

        self.mask_A = self.mask[self.real_A.size(0):]
        if self.opt.nce_idt: 
            self.idt_B = self.fake[self.real_A.size(0):] # G_enc(Y)
    
    # gradient penalty
    def gradient_penalty(self, netD, real, fake):
        # [b, 1]
        t = torch.rand(1, 1).cuda()
        # # [b, 1] => [b, 2]  broadcasting so t is the same for x1 and x2
        t = t.expand_as(real)
        # interpolation
        mid = t * real + (1 - t) * fake
        # set it to require grad info
        mid.requires_grad = True
        pred = netD(mid)
        grads = torch.autograd.grad(outputs=pred, inputs=mid,
                            grad_outputs=torch.ones_like(pred),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        gp = torch.pow(grads.norm(2, dim=1) - 1, 2).mean()
        return gp
    
    def compute_D_loss(self):
        gp = 0
        if self.opt.netD == 'probag':
            # Real
            self.pred_real = self.netD(self.real_B)
            self.loss_D_real = -self.pred_real.mean()
            # Fake
            fake = self.fake_B.detach()
            pred_fake = self.netD(fake)
            self.loss_D_fake = pred_fake.mean()
            # gradient penalty
            gp = self.gradient_penalty(self.netD, self.real_B, fake)
            
            # # attended fake
            # self.attended_fake = self.fake_B.detach()*self.mask_B.detach()
            # # Fake, stop backprop to the generator
            # pred_fake_attended = self.netD_att(self.attended_fake)
            # self.loss_D_fake_att = pred_fake_attended.mean()
            # # attended real
            # self.attended_real = self.real_B * self.true_mask_B
            # pred_real_attended = self.netD_att(self.attended_real)
            # self.loss_D_real_att = -pred_real_attended.mean()
            # # gradient penalty
            # gp_attended = self.gradient_penalty(self.netD_att, self.attended_real, self.attended_fake)
        
        else:
            """Calculate GAN loss for the discriminator"""
            fake = self.fake_B.detach()
            # Fake; stop backprop to the generator by detaching fake_B
            pred_fake = self.netD(fake)
            self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
            # Real
            pred_real = self.netD(self.real_B)
            self.loss_D_real= self.criterionGAN(pred_real, True).mean()

            # attended region
            # 3d attention mask
            # attention_mask_B = self.mask_B.repeat(1, 3, 1, 1)
            # self.attended_fake = self.fake_B.detach()*self.mask_B.detach()
            # # Fake, stop backprop to the generator
            # pred_fake_attended = self.netD_attended(self.attended_fake)
            # self.loss_D_fake_attended = self.criterionGAN(pred_fake_attended, False).mean()
            # # 3d attention mask for real img
            # # attention_mask_A = self.mask_A.repeat(1,3,1,1)
            # self.attended_real = self.real_B*self.mask_A.detach()
            # # Real
            # self.pred_real_attended = self.netD_attended(self.attended_real)
            # self.loss_D_real_attended = self.criterionGAN(self.pred_real_attended, True).mean()

            
        
        # self.loss_D_real = (self.loss_D_real_att + self.loss_D_real_global) * 0.5
        # self.loss_D_fake = (self.loss_D_fake_att + self.loss_D_fake_global) * 0.5
        # loss_D_global = self.loss_D_real_global + self.loss_D_fake_global + 10 * gp_global
        # loss_D_attended = self.loss_D_real_attended + self.loss_D_fake_attended + 10 * gp_attended
        # combine loss and calculate gradients
        self.loss_D = self.loss_D_real + self.loss_D_fake + 10 * gp
        
        return self.loss_D
        
    def compute_G_loss(self, epoch):
        """Calculate GAN and NCE loss for the generator"""
        lambda_pixel = self.opt.lambda_pixel
        lambda_reg = self.opt.lambda_reg

        fake = self.fake_B
        # fake_attended = self.attended_fake
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            if self.opt.netG == 'probag':
                pred_fake = self.netD(fake)
                self.loss_G_GAN = -pred_fake.mean() * self.opt.lambda_GAN
            else:
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
        
        # pixel loss
        self.loss_pix = self.criterionPix(self.mask_B, self.true_mask_A).mean() * lambda_pixel

        # attention loss
        self.loss_reg = lambda_reg * (
                torch.sum(torch.abs(self.mask_B[:, :, :, :-1] - self.mask_B[:, :, :, 1:])) +
                torch.sum(torch.abs(self.mask_B[:, :, :-1, :] - self.mask_B[:, :, 1:, :])))
        
        
        # Total loss
        if epoch < self.opt.gan_curriculum:
            rate = self.opt.starting_rate
        else:
            rate = self.opt.default_rate
        
        
        self.loss_G = (self.loss_G_GAN + self.loss_reg)*(1.-rate) + (self.loss_pix + loss_NCE_both)* rate
        if self.opt.strongG:
            self.loss_G = self.loss_G_GAN + self.loss_reg + self.loss_pix + loss_NCE_both
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