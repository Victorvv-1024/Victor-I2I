import torch
import os

from .networks import define_S
from .base_model import BaseModel
from .losses import SEGLoss, DiceLoss


class Segmentor(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        # specify the training losses you want to print out.
        self.loss_names = ['S_A', 'S_B', 'S']

        print(f'using device: {self.device}')
        
        self.model_names = ['S']
            
        # define the segmentor, S
        self.netS = define_S(opt.input_nc, opt.num_class, opt.ngf, opt.netS, opt.normS, not opt.no_dropout, opt.init_type, opt.init_gain, opt.antialias, opt.antialias_up, opt)
        
        if self.opt.isTrain:
            
            if opt.netS_Loss == 'bce' or opt.netS_Loss == 'BCE':
                self.criterionSEG = SEGLoss(seg_lambda=opt.netS_lambda).to(self.device)
            elif opt.netS_Loss == 'dice' or opt.netS_Loss == 'DICE':
                self.criterionSEG = DiceLoss().to(self.device)
            else: 
                raise NotImplementedError('segmentation loss function is not implemented')
            
            self.optimizer_S = torch.optim.Adam(self.netS.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_S)

    def data_dependent_initialize(self, data):
        return

    def optimize_parameters(self):
        # forward
        self.forward()

        # update S
        self.set_requires_grad(self.netS, True)
        self.optimizer_S.zero_grad()
        self.loss_S = self.compute_S_loss()
        self.loss_S.backward()
        self.optimizer_S.step()
    
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
        
    
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_mask_A = self.netS(self.real_A)
        self.fake_mask_B = self.netS(self.real_B)

        
    def compute_S_loss(self):
        """Calculate SEG loss for the segmentor"""
        self.loss_S_A = self.criterionSEG(self.fake_mask_A, self.mask_A).mean()

        self.loss_S_B = self.criterionSEG(self.fake_mask_B, self.mask_B).mean()

        self.loss_S = (self.loss_S_A + self.loss_S_B) * 0.5

        # loss_S is used to optimize SEG 
        return self.loss_S
    
    def load(self, path, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(path, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                net.load_state_dict(state_dict)
        