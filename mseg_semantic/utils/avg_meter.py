#!/usr/bin/python3

import numpy as np
import torch
import torch.distributed as dist
from typing import List

import mseg_semantic.utils.iou as iou_utils


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SegmentationAverageMeter(AverageMeter):
    """An AverageMeter designed specifically for evaluating segmentation results."""

    def __init__(self) -> None:
        """Initialize object."""
        self.intersection_meter = AverageMeter()
        self.union_meter = AverageMeter()
        self.target_meter = AverageMeter()
        self.accuracy = 0

    def update_metrics_cpu(self, pred: np.ndarray, target: np.ndarray, num_classes: int) -> None:
        """
        Args:
            pred
            target
            classes
        """
        intersection, union, target = iou_utils.intersectionAndUnion(pred, target, num_classes)
        self.intersection_meter.update(intersection)
        self.union_meter.update(union)
        self.target_meter.update(target)
        self.accuracy = sum(self.intersection_meter.val) / (sum(self.target_meter.val) + 1e-10)
        self.intersection = 0.0

    def update_metrics_gpu(
        self, pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_idx: int, is_distributed: bool
    ) -> None:
        """
        Args:
            pred:
            target:
            num_classes:
            ignore_idx:
        """
        intersection, union, target = iou_utils.intersectionAndUnionGPU(pred, target, num_classes, ignore_idx)
        if is_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        self.intersection = intersection.cpu().numpy()
        union, target = union.cpu().numpy(), target.cpu().numpy()

        self.intersection_meter.update(self.intersection)
        self.union_meter.update(union)
        self.target_meter.update(target)
        self.accuracy = sum(self.intersection_meter.val) / (sum(self.target_meter.val) + 1e-10)

    def get_metrics(self, exclude: bool = False, exclude_ids: List[int] = None):
        """
        Args:
            exclude:
            exclude_ids:

        Returns:
            iou_class: Array
            accuracy_class: Array
            mIoU: float
            mAcc: float
            allAcc: float
        """
        iou_class = self.intersection_meter.sum / (self.union_meter.sum + 1e-10)
        accuracy_class = self.intersection_meter.sum / (self.target_meter.sum + 1e-10)

        if exclude:
            mIoU = np.mean(exclusion(iou_class, exclude_ids))
            mAcc = np.mean(exclusion(accuracy_class, exclude_ids))

            # print('original miou is:, ', np.mean(iou_class))
        else:
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
        allAcc = sum(self.intersection_meter.sum) / (sum(self.target_meter.sum) + 1e-10)
        return iou_class, accuracy_class, mIoU, mAcc, allAcc


def exclusion(array: np.ndarray, excluded_ids: List[int]) -> np.ndarray:
    """take in array of IoU/Acc., return non-excluded IoU/acc values"""
    all_ids = np.arange(array.size)
    # valid indices --> take complement of set intersection
    relevant_array = array[~np.in1d(all_ids, excluded_ids)]
    return relevant_array
