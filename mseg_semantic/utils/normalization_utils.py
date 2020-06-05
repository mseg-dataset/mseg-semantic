#!/usr/bin/python3

import numpy as np
import torch

from typing import Optional, Tuple


def get_imagenet_mean_std() -> Tuple[Tuple[float,float,float], Tuple[float,float,float]]:
    """ See use here in Pytorch ImageNet script: 
        https://github.com/pytorch/examples/blob/master/imagenet/main.py#L197

        Returns:
        -   mean: Tuple[float,float,float], 
        -   std: Tuple[float,float,float] = None
    """
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    return mean, std
            

def normalize_img(  input: torch.Tensor, 
                    mean: Tuple[float,float,float], 
                    std: Optional[Tuple[float,float,float]] = None):
    """ Pass in by reference Torch tensor, and normalize its values.

        Args:
        -   input: Torch tensor of shape (3,M,N), must be in this order, and
                of type float (necessary).
        -   mean: mean values for each RGB channel
        -   std: standard deviation values for each RGB channel

        Returns:
        -   None
    """
    if std is None:
        for t, m in zip(input, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(input, mean, std):
            t.sub_(m).div_(s)

