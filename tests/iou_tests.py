#!/usr/bin/python3

"""Unit tests on IoU computation."""

import numpy as np
import torch

import mseg_semantic.utils.iou as iou_utils


def test_intersectionAndUnion_2classes() -> None:
    """
    No way to compute union of two sets, without understanding where they intersect.
    """
    pred = np.array([[0, 0], [1, 0]])
    target = np.array([[0, 0], [1, 1]])
    num_classes = 2

    # contain the number of samples in each bin.
    area_intersection, area_union, area_target = iou_utils.intersectionAndUnion(
        pred, target, K=num_classes, ignore_index=255
    )
    assert area_intersection.shape == (2,)
    assert area_union.shape == (2,)
    assert area_target.shape == (2,)
    assert np.allclose(area_intersection, np.array([2, 1]))
    assert np.allclose(area_target, np.array([2, 2]))
    assert np.allclose(area_union, np.array([3, 2]))


def test_intersectionAndUnion_3classes() -> None:
    """
    (0,0) are matched once. (1,1) are matched once. (2,2) are matched once,
    giving us intersection [1,1,1] for those three classes.

    No way to compute union of two sets, without understanding where they intersect.
    Union of sets
    {0} union {0} -> {0}
    {0} union {1} -> {0,1}
    {2} union {2} -> {2}
    {1} union {1} -> {1}
    yields class counts [2,2,1]
    """
    pred = np.array([[2, 0], [1, 0]])
    target = np.array([[2, 0], [1, 1]])
    num_classes = 3

    # contain the number of samples in each bin.
    area_intersection, area_union, area_target = iou_utils.intersectionAndUnion(
        pred, target, K=num_classes, ignore_index=255
    )
    assert area_intersection.shape == (3,)
    assert area_union.shape == (3,)
    assert area_target.shape == (3,)
    assert np.allclose(area_intersection, np.array([1, 1, 1]))
    assert np.allclose(area_target, np.array([1, 2, 1]))
    assert np.allclose(area_union, np.array([2, 2, 1]))


def test_mIoU():
    """ """
    intersection = np.array([1, 1, 1])
    union = np.array([2, 1, 1])
    miou = np.mean(intersection / (union + 1e-10))


def test_intersectionAndUnionGPU_3classes() -> None:
    """
    (0,0) are matched once. (1,1) are matched once. (2,2) are matched once,
    giving us intersection [1,1,1] for those three classes.

    No way to compute union of two sets, without understanding where they intersect.
    Union of sets
    {0} union {0} -> {0}
    {0} union {1} -> {0,1}
    {2} union {2} -> {2}
    {1} union {1} -> {1}
    yields class counts [2,2,1]
    """
    pred = torch.tensor([[2, 0], [1, 0]])
    target = torch.tensor([[2, 0], [1, 1]])
    num_classes = 3

    # contain the number of samples in each bin.
    area_intersection, area_union, area_target = iou_utils.intersectionAndUnionGPU(
        pred, target, K=num_classes, ignore_index=255, cuda_available=False
    )
    assert area_intersection.shape == (3,)
    assert area_union.shape == (3,)
    assert area_target.shape == (3,)
    assert torch.allclose(area_intersection, torch.tensor([1, 1, 1]).float())
    assert torch.allclose(area_target, torch.tensor([1, 2, 1]).float())
    assert torch.allclose(area_union, torch.tensor([2, 2, 1]).float())


def test_intersectionAndUnion_ignore_label() -> None:
    """
    Handle the ignore case. Since 255 lies outside of the histogram bins,
    it will be ignored.
    """
    pred = np.array([[1, 0], [1, 0]])
    target = np.array([[255, 0], [255, 1]])
    num_classes = 2

    # contain the number of samples in each bin.
    area_intersection, area_union, area_target = iou_utils.intersectionAndUnion(
        pred, target, K=num_classes, ignore_index=255
    )
    assert area_intersection.shape == (2,)
    assert area_union.shape == (2,)
    assert area_target.shape == (2,)
    assert np.allclose(area_intersection, np.array([1, 0]))
    assert np.allclose(area_target, np.array([1, 1]))
    assert np.allclose(area_union, np.array([2, 1]))


def test_intersectionAndUnionGPU_ignore_label() -> None:
    """
    Handle the ignore case. Since 255 lies outside of the histogram bins,
    it will be ignored.
    """
    pred = torch.tensor([[1, 0], [1, 0]])
    target = torch.tensor([[255, 0], [255, 1]])
    num_classes = 2

    # contain the number of samples in each bin.
    area_intersection, area_union, area_target = iou_utils.intersectionAndUnionGPU(
        pred, target, K=num_classes, ignore_index=255, cuda_available=False
    )
    assert area_intersection.shape == (2,)
    assert area_union.shape == (2,)
    assert area_target.shape == (2,)
    assert torch.allclose(area_intersection, torch.tensor([1, 0]).float())
    assert torch.allclose(area_target, torch.tensor([1, 1]).float())
    assert torch.allclose(area_union, torch.tensor([2, 1]).float())


if __name__ == "__main__":
    # test_intersectionAndUnion_2classes()
    test_intersectionAndUnion_3classes()
