#!/usr/bin/python3

"""
Utilities for evaluating models trained on relabeled data.

As we loop through two dataloaders (unrelabeled and relabeled ground truth),
we feed each pair of label maps to this function.

If the labeling was completely incorrect, like ade20k table instead of
nightstand, then we have to go with the penalty route though

Another option -- relax such parent-child relationships since
predicting finest-granularity is not too fair

Penalizing in the person/motorcyclist case seems unfair to the relabeled
model, since it gets at least as good as the unrelabeled model
lose-lose unless we specify the full hierarchy and employ it
"""
from typing import Tuple

import numpy as np
import torch

from mseg_semantic.utils.transform import ToUniversalLabel


def convert_label_to_pred_taxonomy(target_img: np.ndarray, to_universal_transform) -> np.ndarray:
    """
    Args:
        label map in `semseg`-format taxonomy, e.g. `coco-panoptic-133`
            or `coco-panoptic-133-relabeled`

    Returns:
        label map in universal taxonomy
    """
    target_img = torch.from_numpy(target_img).type(torch.LongTensor)
    _, target_img = to_universal_transform(target_img, target_img)
    return target_img.type(torch.uint8).numpy()


def eval_rel_model_pred_on_unrel_data(
    pred_rel: np.ndarray,
    target_img: np.ndarray,
    target_img_relabeled: np.ndarray,
    orig_to_u_transform,
    relabeled_to_u_transform,
    ignore_idx: int = 255,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Rather than eval unrelabeled model on the univ. relabeled data, we instead map correctness
    of relabeled model on relabeled univ. data, back to the unrelabeled univ. space.
    Could go either way since it is a 1:1 mapping per pixel. Operate on universal taxonomy.

    If in universal space, any pixel differs, if in coco-relabeled-universal prediction is correct,
    we map it to the coco-unrelabeled-universal label and thus count it as correct.

    Look at diff on disk to see where ground truth was changed. Relabeled data is the ``oracle''.
    coco-unrel->coco-unrel-universal, then coco-unrel-universal is already in universal on disk.

    Args:
        pred_rel: Numpy array of shape (H,W) of dtype int64 representing prediction,
            of a relabeled model, predictions are already in universal taxonomy.
        target_img: Numpy array of shape (H,W) of dtype int64 representing unrelabeled ground truth,
            in original `semseg` format taxonomy, e.g. `coco-panoptic-133`
        target_img_relabeled:  Numpy array of shape (H,W) of dtype int64 representing relabeled ground truth,
            in relabeled taxonomy, e.g. `coco-panoptic-133-relabeled`
        orig_to_u_transform: original dataset tax. -> universal tax. transformation
        relabeled_to_u_transform: relabeled dataset tax. -> universal tax. transformation

    Returns:
        pred_unrel: rewarded & penalized predictions, in universal taxonomy
            as if was made by an `unrelabeled` model
        target_img: Numpy array representing unrelabeled ground truth
            in universal taxonomy
        acc_diff: difference in accuracy.
    """
    target_img = convert_label_to_pred_taxonomy(target_img, orig_to_u_transform)
    target_img_relabeled = convert_label_to_pred_taxonomy(target_img_relabeled, relabeled_to_u_transform)
    # construct a "correct" target image here: if pixel A is relabeled as pixel B, and prediction is B, then map prediction B back to A

    relabeled_pixels = target_img_relabeled != target_img
    correct_pixels = pred_rel == target_img_relabeled
    incorrect_pixels = pred_rel != target_img_relabeled

    correct_relabeled_pixels = relabeled_pixels * correct_pixels
    incorrect_relabeled_pixels = relabeled_pixels * incorrect_pixels

    # Apply Reward -> if you chose the oracle's relabeled class, then
    # reset your predicted class index to not be what network said,
    # but what unrelabeled ground truth was.
    # np.where() sets where True, yield x, otherwise yield y.
    pred_unrel = np.where(correct_relabeled_pixels, target_img, pred_rel)

    # Apply Penalty -> if the model predicted the un-relabeled class, we
    # must penalize it for not choosing the `truth` from our oracle
    # the `ignore_idx` will penalize a prediction in mIoU calculation (but not penalize GT)
    guaranteed_wrong_pred = np.ones_like(pred_rel) * ignore_idx
    pred_unrel = np.where(incorrect_relabeled_pixels, guaranteed_wrong_pred, pred_unrel)

    num_px = target_img.size  # number of pixels in the image
    accuracy_before = get_px_accuracy(pred_rel, target_img)

    # anywhere GT was relabeled as `unlabeled`, immediately transfer to `target_img`
    # otherwise, target img will have poorly-labeled class, vs. prediction's `255`
    target_img[np.where(target_img_relabeled == ignore_idx)] = ignore_idx

    accuracy_after = get_px_accuracy(pred_unrel, target_img)
    acc_diff = accuracy_after - accuracy_before

    if verbose:
        print("Pct of img relabeled: ", np.sum(relabeled_pixels) / num_px * 100)
        print(f"Acc before: {accuracy_before}, Acc after: {accuracy_after}")
    return pred_unrel, target_img, acc_diff


def get_px_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """Quick and dirty method for analysis, not an officially correct implementation"""
    num_px = target.size
    is_correct = np.logical_or.reduce([pred == target, target == 255])
    return is_correct.sum() / num_px * 100
