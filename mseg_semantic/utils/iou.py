#!/usr/bin/python3

from typing import Tuple

import numpy as np
import torch
from torch import nn


"""
Utilies to compute quantities required to compute IoU per class
on a single image, using Numpy or Pytorch.
"""


def intersectionAndUnion(
    output: np.ndarray, target: np.ndarray, K: int, ignore_index: int = 255
) -> Tuple[np.array, np.array, np.array]:
    """Compute IoU on Numpy arrays on CPU.

    We will be reasoning about each matrix cell individually, so we can reshape (flatten)
    these arrays into column vectors and the evaluation result won’t change. Compare
    horizontally-corresponding cells. Wherever ground truth (target)
    pixels should be ignored, set prediction also to the ignore label.
    `intersection` represents values (class indices) in cells where
    output and target are identical. We bin such correct class indices.

    Note output and target sizes are N or N * L or N * H * W

    Args:
        output: Numpy array represeting predicted label map,
            each value in range 0 to K - 1.
        target: Numpy array representing ground truth label map,
            each value in range 0 to K - 1.
        K: integer number of possible classes
        ignore_index: integer representing class index to ignore

    Returns:
        area_intersection: 1d Numpy array of length (K,) with counts
            for each of K classes, where pred & target matched
        area_union: 1d Numpy array of length (K,) with counts
        area_target: 1d Numpy array of length (K,) with bin counts
            for each of K classes, present in this GT label map.
    """
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    # flatten the tensors to 1d arrays
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = 255
    intersection = output[np.where(output == target)[0]]
    # contains the number of samples in each bin.
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(
    output: torch.Tensor, target: torch.Tensor, K: int, ignore_index: int = 255, cuda_available: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute IoU on the GPU.

    Note output and target sizes are N or N * L or N * H * W

    Args:
        output: Pytorch tensor represeting predicted label map,
            each value in range 0 to K - 1.
        target: Pytorch tensor representing ground truth label map,
            each value in range 0 to K - 1.
        K: integer number of possible classes
        ignore_index: integer representing class index to ignore
        cuda_available: CUDA is available to Pytorch to use

    Returns:
        area_intersection: 1d Pytorch tensor of length (K,) with counts
            for each of K classes, where pred & target matched
        area_union: 1d Pytorch tensor of length (K,) with counts
        area_target: 1d Pytorch tensor of length (K,) with bin counts
            for each of K classes, present in this GT label map.
    """
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K - 1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K - 1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    if cuda_available:
        return area_intersection.cuda(), area_union.cuda(), area_target.cuda()
    else:
        return area_intersection, area_union, area_target
