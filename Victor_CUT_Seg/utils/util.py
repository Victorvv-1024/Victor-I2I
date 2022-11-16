"""Utility Functions"""

import argparse
from argparse import Namespace
import numpy as np
from torch.autograd import Variable
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def tensor2img(tensor, isMask=False):
    """This function translates a tensor into a image format

    Args:
        tensor (torch.Tensor): of shape B x C x H x W
        isMask (bool, optional): indicates if the input tensor is a mask tensor or not. Defaults to False.

    Returns:
        ndarray: image of shape H x W x C
    """
    tensor = tensor.squeeze(0).permute(1,2,0) # H x W x C
    if isMask:
        tensor = tensor[:,:,1]
    # cast to numpy ndarray
    image = tensor.cpu().detach().numpy()
    return image

def mask_image(mask, image):
    """This function masks out the image using the given mask

    Args:
        mask (ndarray): the semantic mask of the image
        image (ndarray): the image to be masked out

    Returns:
        ndarray: the masked out image, NOT tensor form
    """
    mask_3d = np.stack((mask, mask, mask), axis=-1)
    masked_image = image * mask_3d
    masked_image = np.where(mask_3d == 1, image, mask_3d)

    return masked_image

def img2tensor(img):
    """This function translates an img to tensor form

    Args:
        img (ndarray): of shape H x W x C

    Returns:
        torch.Tensor: of shape B x C x H x W
    """
    tensor = Variable(torch.from_numpy(img.astype(np.float32)))
    tensor = tensor.unsqueeze(0)
    tensor = tensor.permute(0,3,1,2)

    return tensor