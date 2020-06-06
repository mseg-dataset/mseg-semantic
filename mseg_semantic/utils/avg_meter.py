#!/usr/bin/python3

import numpy as np
import torch.distributed as dist

from mseg_semantic.utils.iou import intersectionAndUnion, intersectionAndUnionGPU


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class SegmentationAverageMeter(AverageMeter):
    """ 
    An AverageMeter designed specifically for evaluating segmentation results.
    """
    def __init__(self):
        """ Initialize object. """
        self.intersection_meter = AverageMeter()
        self.union_meter = AverageMeter()
        self.target_meter = AverageMeter()
        self.accuracy = 0

    def update_metrics_cpu(self, pred, target, num_classes) -> None:
        """
            Args:
            -   pred
            -   target
            -   classes

            Returns:
            -   None
        """
        intersection, union, target = intersectionAndUnion(pred, target, num_classes)
        self.intersection_meter.update(intersection)
        self.union_meter.update(union)
        self.target_meter.update(target)
        self.accuracy = sum(self.intersection_meter.val) / (sum(self.target_meter.val) + 1e-10)
        self.intersection = 0.

    def update_metrics_gpu(self, pred, target, num_classes, ignore_idx, is_distributed):
        """ 
            Args:
            -    pred
            -   target
            -   num_classes
            -   ignore_idx

            Returns:
            -   None
        """
        intersection, union, target = intersectionAndUnionGPU(pred, target, num_classes, ignore_idx)
        if is_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        self.intersection = intersection.cpu().numpy()
        union, target = union.cpu().numpy(), target.cpu().numpy()
        
        self.intersection_meter.update(self.intersection)
        self.union_meter.update(union)
        self.target_meter.update(target)
        self.accuracy = sum(self.intersection_meter.val) / (sum(self.target_meter.val) + 1e-10)

    def get_metrics(self, exclude=False, exclude_ids=None):
        """
            Args:
            -   None

            Returns:
            -   iou_class: Array
            -   accuracy_class: Array
            -   mIoU: float
            -   mAcc: float
            -   allAcc: float
        """
        iou_class = self.intersection_meter.sum / (self.union_meter.sum + 1e-10)
        accuracy_class = self.intersection_meter.sum / (self.target_meter.sum + 1e-10)

        if exclude:
            mIoU = np.mean(self.exclusion(iou_class, exclude_ids))
            mAcc = np.mean(self.exclusion(accuracy_class, exclude_ids))

            print('original miou is:, ', np.mean(iou_class))
        else:
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
        allAcc = sum(self.intersection_meter.sum) / (sum(self.target_meter.sum) + 1e-10)
        return iou_class, accuracy_class, mIoU, mAcc, allAcc

    def exclusion(self, array, ids): # id is excluded id
        ''' take in array of iou/acc., return mean excluded values'''

        total = len(array)

        count = []
        for i in range(total):
            if i not in ids:
                count += [i]

        return array[count]

