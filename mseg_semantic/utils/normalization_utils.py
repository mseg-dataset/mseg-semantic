#!/usr/bin/python3

from typing import Optional, Tuple

import numpy as np
import torch


def get_imagenet_mean_std() -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """See use here in Pytorch ImageNet script:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py#L197

    Returns:
        mean: average R,G,B values in [0,255] range
        std: standard deviation of R,G,B values in [0,255] range
    """
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    return mean, std


def normalize_img(
    input: torch.Tensor, mean: Tuple[float, float, float], std: Optional[Tuple[float, float, float]] = None
) -> None:
    """Pass in by reference Torch tensor, and normalize its values.

    Args:
        input: Torch tensor of shape (3,M,N), must be in this order, and
            of type float (necessary).
        mean: mean values for each RGB channel
        std: standard deviation values for each RGB channel
    """
    if std is None:
        for t, m in zip(input, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(input, mean, std):
            t.sub_(m).div_(s)
