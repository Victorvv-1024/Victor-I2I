""" Implement the following layers that used in CUT/FastCUT model.
Padding2D
InstanceNorm
AntialiasSampling
ConvBlock
ConvTransposeBlock
ResBlock
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np


"""
Auxilary functions for the networks
1. get padding layer
2. get normalization layer
3. get filter for upsampling and downsampling
"""
# Padding
def get_pad_layer(pad_type):
    """
    Return a 2D padding layer.
    Padding is used to ensure the size of feature remain unchanged, by wrapping the outcome in some 'frame'.
    
    Args:
        pad_type (string): the name of the padding layer: reflect | replicate | zero
    """
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        raise ValueError('Pad type [%s] not recognized' % pad_type)
    return PadLayer

# Norm Layer
class Identity(nn.Module):
    """Return the input without any mappinp"""
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none (identity)

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise ValueError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

# Up/Downsampling
def get_filter(k_size):
    """Return the filter for up/downsampling

    Args:
        k_size (int): the size of the kernel/filter
    """
    if(k_size == 1):
        kernel = np.array([1., ])
    elif(k_size == 2):
        kernel = np.array([1., 1.])
    elif(k_size == 3):
        kernel = np.array([1., 2., 1.])
    elif(k_size == 4):
        kernel = np.array([1., 3., 3., 1.])
    elif(k_size == 5):
        kernel = np.array([1., 4., 6., 4., 1.])
    elif(k_size == 6):
        kernel = np.array([1., 5., 10., 10., 5., 1.])
    elif(k_size == 7):
        kernel = np.array([1., 6., 15., 20., 15., 6., 1.])

    filter = torch.Tensor(kernel[:, None] * kernel[None, :])
    filter = filter / torch.sum(filter)

    return filter


"""
Classes of Layers for the network.
"""

class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', k_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.k_size = k_size
        self.pad_off = pad_off
        self.k_sizes = [int(1. * (k_size - 1) / 2), int(np.ceil(1. * (k_size - 1) / 2)), int(1. * (k_size - 1) / 2), int(np.ceil(1. * (k_size - 1) / 2))]
        self.k_sizes = [size + pad_off for size in self.k_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filter = get_filter(k_size=self.k_size)
        self.register_buffer('filt', filter[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.k_sizes)

    def forward(self, inp):
        if(self.k_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])
        
class Upsample(nn.Module):
    def __init__(self, channels, pad_type='replicate', k_size=4, stride=2):
        super(Upsample, self).__init__()
        self.k_size = k_size
        self.filt_odd = np.mod(k_size, 2) == 1
        self.k_sizes = int((k_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filter = get_filter(self.k_size) * (stride**2)
        self.register_buffer('filt', filter[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.k_sizes, groups=inp.shape[1])[:, :, 1:, 1:]
        if(self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]
        
# Basics Blocks
class Conv2dBlock(nn.Module):
    """ ConBlock layer consists of Conv2D + Normalization + Activation.
    """
    def __init__(self, input_dim, output_dim, kernel_size, stride=(1,1),
                 padding='valid', padding_const=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        self.padding = padding
        if pad_type == 'reflect':
            self.pad = get_pad_layer(pad_type)(padding_const)
        elif pad_type == 'zero':
            self.pad = get_pad_layer(pad_type)(padding_const)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'batch':
            self.norm = get_norm_layer(norm_type=norm)(norm_dim)
        elif norm == 'inst':
            self.norm = get_norm_layer(norm_type=norm)(norm_dim)
        elif norm == 'none':
            self.norm = get_norm_layer(norm_type=norm)
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        if self.padding != 'valid': x = self.pad(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

    
class ResBlock(nn.Module):
    """ 
    ResBlock is a ConvBlock with skip connections.
    Original Resnet paper (https://arxiv.org/pdf/1512.03385.pdf).
    """
    def __init__(self, dim, norm='inst', activation='relu', pad_type='reflect', nz=0):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim + nz, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim + nz, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out
    
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='inst', activation='relu', pad_type='reflect', nz=0):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type, nz=nz)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class ConvTransposeBlock(nn.Module):
    """ ConvTransposeBlock layer consists of Conv2DTranspose + Normalization + Activation.
    """
    def __init__(self, input_dim, output_dim, kernel_size, stride=(1,1),
                 padding='valid', padding_const=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        self.padding = padding
        if pad_type == 'reflect':
            self.pad = get_pad_layer(pad_type)(padding_const)
        elif pad_type == 'zero':
            self.pad = get_pad_layer(pad_type)(padding_const)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
            
        # initialize normalization
        norm_dim = output_dim
        if norm == 'batch':
            self.norm = get_norm_layer(norm_type=norm)(norm_dim)
        elif norm == 'inst':
            self.norm = get_norm_layer(norm_type=norm)(norm_dim)
        elif norm == 'none':
            self.norm = get_norm_layer(norm_type=norm)
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)
            
        # initialize convolution
        self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)


    def call(self, x):
        if self.padding != 'valid': x = self.pad(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
    