"""
This file is for parsing the cmd arguments for user's training
"""
import argparse
import os
from utils import util

def ArgParse():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='For model training')
    # basic parameters
    parser.add_argument('--model', type=str, default='cut_seg', help='chooses which model to use.', choices=['cut_seg', 'cyclegan', 'dcl', 'distance', 'kx_cut', 'ag_cut', 'ag_cycle'])
    parser.add_argument('--train_src_dir', help='Train-source dataset folder', type=str, default='datasets/victor_dataset/white/train')
    parser.add_argument('--train_tar_dir', help='Train-target dataset folder', type=str, default='datasets/victor_dataset/non_white/train')
    parser.add_argument('--test_src_dir', help='Test-source dataset folder', type=str, default='datasets/victor_dataset/white/test')
    parser.add_argument('--test_tar_dir', help='Test-target dataset folder', type=str, default='datasets/victor_dataset/non_white/test')
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--name', type=str, default='demo_v4', help='name of the experiment. It decides where to store samples and models')
    # parser.add_argument('--easy_label', type=str, default='demo_v4', help='Interpretable name')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/victor_dataset', help='models are saved here')
    parser.add_argument('--load', help='if to load the network', action='store_true')
    parser.add_argument('--load_path', help='where to load the network')
    parser.add_argument('--load_epoch', help='which checkpoint to load', type=int, default=5)
    # the output dir is set for demo
    parser.add_argument('--out_dir', help='Outputs folder', type=str)
    
    """GAN parameters"""
    parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
    parser.add_argument('--netD', type=str, default='basic', choices=['basic', 'n_layers', 'ag', 'probag'], help='specify discriminator architecture. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
    parser.add_argument('--netG', type=str, default='resnet_9blocks', choices=['resnet_9blocks', 'resnet_6blocks', 'ag', 'probag'], help='specify generator architecture')
    parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
    parser.add_argument('--normG', type=str, default='instance', choices=['instance', 'batch', 'none'], help='instance normalization or batch normalization for G')
    parser.add_argument('--normD', type=str, default='instance', choices=['instance', 'batch', 'none'], help='instance normalization or batch normalization for D')
    parser.add_argument('--init_type', type=str, default='xavier', choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='network initialization')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--no_dropout', type=util.str2bool, nargs='?', const=True, default=True,
                        help='no dropout for the generator')
    parser.add_argument('--antialias', action='store_true', help='if specified, use antialiased-downsampling')
    parser.add_argument('--antialias_up', action='store_true', help='if specified, use [upconv(learned filter)]')
    
    """netS parameters"""
    parser.add_argument('--netS_lambda', type=int, default=10, help='lambda for SEG loss')
    parser.add_argument('--netS_Loss', type=str, help='semantic segmentation loss function', choices=['dice', 'bce', 'DICE', 'BCE'], default='bce')
    parser.add_argument('--netS', type=str, default='smp', choices=['resnet', 'unet_256', 'smp'], help='how to segment the input image')
    parser.add_argument('--smp_arch', type=str, default='Unet', help='the segmentor architectur')
    parser.add_argument('--smp_encoder', type=str, default='efficientnet-b3', help='the encoder name')
    parser.add_argument('--normS', type=str, default='instance', choices=['instance', 'batch', 'none'], help='instance normalization or batch normalization for S')
    parser.add_argument('--num_class', type=int, default=1, help='# of output image channels for segmented mask')

    # training parameters
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=200, help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--cycliclr', action='store_true', help='to use cyclic learning rate')
    parser.add_argument('--base_lr', type=float, default=0.00001, help='base lr for cyclic')
    parser.add_argument('--isTrain', type=util.str2bool, default=True, help='select to train the model')
    parser.add_argument('--strongG', action='store_true', help='stronger loss G')
    
    # network saving and loading parameters
    parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--evaluation_freq', type=int, default=5000, help='evaluation freq')
    parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    parser.add_argument('--pretrained_name', type=str, default=None, help='resume training from another checkpoint')

    """cyclegan parameters"""
    parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
    parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
    parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss.\
                        For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
    parser.add_argument('--pool_size', type=int, default=100, help='the size of image pool')

    """cut parameters"""
    parser.add_argument('--CUT_mode', type=str, default="CUT", choices=['CUT', 'cut', 'FastCUT', 'fastcut'], help='')
    parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
    
    
    """netF paramters"""
    parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
    parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=True, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
    parser.add_argument('--nce_layers', type=str, default='0,3,5,7,11', help='compute NCE loss on which layers')
    parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
    parser.add_argument('--netF_nc', type=int, default=256)
    parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
    parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
    parser.add_argument('--flip_equivariance',
                    type=bool, nargs='?', default=False,
                    help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
    
    """dcl paramters"""
    parser.add_argument('--DCL_mode', type=str, default="DCL", choices='DCL')
    
    """distance parameters"""
    parser.add_argument('--lambda_distance_A', type=float, default=1.0, help='weight for distance loss (A -> B)')
    parser.add_argument('--lambda_distance_B', type=float, default=1.0, help='weight for distance loss (B -> A)')

    """attention gan parameters"""
    parser.add_argument('--lambda_pixel', type=float, default=1)
    parser.add_argument('--lambda_reg', type=float, default=1e-6)
    parser.add_argument('--lambda_content', type=float, default=1e-6)

    parser.add_argument('--gan_curriculum', type=int, default=10, help='Strong GAN loss for certain period at the beginning')
    parser.add_argument('--starting_rate', type=float, default=0.01, help='Set the lambda weight between GAN loss and Recon loss during curriculum period at the beginning. We used the 0.01 weight.')
    parser.add_argument('--default_rate', type=float, default=0.5, help='Set the lambda weight between GAN loss and Recon loss after curriculum period. We used the 0.5 weight.')
    
    opt, _ = parser.parse_known_args()

    # Set default parameters for CUT and FastCUT
    if opt.CUT_mode.lower() == "cut" and (opt.model.lower() == 'cut_seg' or opt.model.lower() == 'kx_cut'):
        parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        parser.set_defaults(pool_size=0)  # no image pooling

    elif opt.CUT_mode.lower() == "fastcut" and (opt.model.lower() == 'cut_seg' or opt.model.lower() == 'kx_cut'):
        parser.set_defaults(
            nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
            n_epochs=150, n_epochs_decay=50
        )
        parser.set_defaults(pool_size=0)  # no image pooling

    elif opt.model.lower() == 'cyclegan':
        parser.set_defaults(
            lambda_identity=0.5, pool_size = 50, batchSize=1
        )
    elif opt.model.lower() == 'dcl' and opt.DCL_mode.lower() == 'dcl':
         parser.set_defaults(
            nce_idt=True, lambda_identity=1.0, 
            lambda_NCE = 2.0, nce_layers = '4,8,12,16'
        )
         parser.set_defaults(pool_size=0)  # no image pooling
    elif opt.model.lower() == 'distance':
        parser.set_defaults(
            lambda_identity=0.5, pool_size = 50, batchSize=2
        )
    elif opt.model.lower() == 'ag_cycle':
        parser.set_defaults(
            lambda_identity=0,
            netD='ag', netG='ag', pool_size=50,
            output_nc=4
        )
    elif opt.CUT_mode.lower() == "cut" and opt.model.lower() == 'ag_cut':
        parser.set_defaults(
            lambda_GAN=1.0, lambda_NCE=1.0,
            netD='ag', netG='ag', pool_size=0,
            output_nc=4, nce_idt=True
        )
    elif opt.CUT_mode.lower() == "fastcut" and opt.model.lower() == 'ag_cut':
        parser.set_defaults(
            lambda_identity=0, lambda_NCE=10.0,
            netD='ag', netG='ag', pool_size=0,
            output_nc=4, nce_idt=False,flip_equivariance=True,
            n_epochs=150, n_epochs_decay=50
        )
    else:
        raise ValueError(opt.CUT_mode)

    return parser

if __name__ == '__main__':
    parser = ArgParse().parse_known_args()
    