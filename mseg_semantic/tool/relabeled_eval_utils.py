#!/usr/bin/python3

import numpy as np
import pdb
import torch

from mseg_semantic.utils.transform import ToUniversalLabel
from mseg.utils.names_utils import (
    load_class_names,
    get_universal_class_names,
)

"""
For evaluating models trained on relabeled data.

As we loop through two dataloaders (unrelabeled and relabeled ground truth),
we feed each pair of label maps to this function.

I guess if the labeling was completely incorrect, like ade20k table instead of
nightstand, then we have to go with the penalty route though
so you are thinking maybe we relax such parent-child relationships since
predicting finest-granularity is not too fair
penalizing in the person/motorcyclist case seems unfair to the relabeled
model, since it gets at least as good as the unrelabeled model
lose-lose unless we specify the full hierarchy and employ it

"""

def convert_label_to_pred_taxonomy(
    target_img: torch.Tensor,
    to_universal_transform
    ) -> np.ndarray:
    """
        Args:
        -   label map in `semseg`-format taxonomy, e.g. `coco-panoptic-133`
            or `coco-panoptic-133-relabeled`

        Returns:
        -   label map in universal taxonomy
    """
    _, target_img = to_universal_transform(target_img, target_img)
    return target_img.type(torch.uint8).numpy()


def eval_relabeled_pair(
    pred: np.ndarray,
    target_img: torch.Tensor,
    target_img_relabeled: torch.Tensor,
    orig_to_u_transform,
    relabeled_to_u_transform,
    ignore_idx: int = 255
    ):
    """
        Rather than unrelabeled model on the univ. relabeled data, we instead map correctness
        of relabeled model on relabeled univ. data, back to the unrelabeled univ. space. 
        Could go either way since it is a 1:1 mapping per pixel. Operate on universal taxonomy.

        If in universal space, any pixel differs, if in coco-relabeled-universal prediction is correct,
        we map it to the coco-unrelabeled-universal label and thus count it as correct.

        Look at diff on disk to see where ground truth was changed. Relabeled data is the ``oracle''.
        coco-unrel->coco-unrel-universal, then coco-unrel-universal is already in universal on disk.

        Args:
        -   pred: torch.Tensor of shape (H,W) of dtype int64 representing prediction,
                predictions are already in universal taxonomy.
        -   target_img: torch.Tensor of shape (H,W) of dtype int64 representing unrelabeled ground truth,
                in original `semseg` format taxonomy, e.g. `coco-panoptic-133`
        -   target_img_relabeled:  Numpy array of shape (H,W) of dtype int64 representing relabeled ground truth,
                in relabeled taxonomy, e.g. `coco-panoptic-133-relabeled`
        -   orig_to_u_transform:
        -   relabeled_to_u_transform:

        Returns:
        -   
    """
    target_img = convert_label_to_pred_taxonomy(target_img, orig_to_u_transform)
    target_img_relabeled = convert_label_to_pred_taxonomy(target_img_relabeled, relabeled_to_u_transform)
    # construct a "correct" target image here: if pixel A is relabeled as pixel B, and prediction is B, then map prediction B back to A

    pdb.set_trace()
    relabeled_pixels = (target_img_relabeled != target_img)
    correct_pixels = (pred == target_img_relabeled)
    incorrect_pixels = (pred != target_img_relabeled)
    
    correct_relabeled_pixels = relabeled_pixels * correct_pixels
    incorrect_relabeled_pixels = relabeled_pixels * incorrect_pixels

    # Apply Reward -> set prediction's class index to not be what network said, 
    # but what original ground truth was.
    # np.where() sets where True, yield x, otherwise yield y.
    pred_final = np.where(correct_relabeled_pixels, target_img, pred)

    # Apply Penalty -> if the model predicted the un-relabeled class, we
    # must penalize it for not choosing the `truth` from our oracle
    # the `ignore_idx` will penalize a prediction in mIoU calculation (but not penalize GT)
    guaranteed_wrong_pred = np.ones_like(pred)*ignore_idx
    pred_final = np.where(incorrect_relabeled_pixels, guaranteed_wrong_pred, pred_final)

    accuracy_before = (pred == target_img).sum()/target_img.size
    accuracy_after = (pred_final == target_img).sum()/target_img.size
    print('Pct of img relabeled: ', np.sum(target_img_relabeled == target_img)/target_img.size)

    print('Acc before: ', accuracy_before, ' Acc after: ', accuracy_after)
    return pred_final, target_img


def test_eval_relabeled_pair1():
    """
    Person vs. Motorcyclist in center
    Relabeled model correctly predicts `motorcylist`. for `motorcylist`.
    
            [0,0,0,0],
            [0,1,1,0],
            [0,1,1,0],
            [0,1,1,0]
    """
    orig_dname = 'coco-panoptic-133'
    relabeled_dname = 'coco-panoptic-133-relabeled'
    original_names = load_class_names(orig_dname)
    relabeled_names = load_class_names(relabeled_dname)
    u_names = get_universal_class_names()

    pred = np.ones((4,4), dtype=np.uint8) * u_names.index('sky')
    pred[1:,1:3] = u_names.index('motorcyclist')

    # original COCO image, in coco-panoptic-133
    target_img = torch.ones(4,4) * original_names.index('sky-other-merged')
    target_img = target_img.type(torch.LongTensor)
    target_img[1:,1:3] = original_names.index('person')
    #target_img = target_img.reshape(1,4,4)

    # relabeled COCO image, in coco-panoptic-133-relabeled
    target_img_relabeled = torch.ones(4,4) * relabeled_names.index('sky')
    target_img_relabeled = target_img_relabeled.type(torch.LongTensor)
    target_img_relabeled[1:,1:3] = relabeled_names.index('motorcyclist')
    #target_img_relabeled = target_img_relabeled.reshape(1,4,4)

    orig_to_u_transform = ToUniversalLabel(orig_dname)
    relabeled_to_u_transform = ToUniversalLabel(relabeled_dname)
    pred_final, target_img = eval_relabeled_pair(
        pred,
        target_img,
        target_img_relabeled,
        orig_to_u_transform,
        relabeled_to_u_transform
    )
    # treated as 100% accuracy
    assert np.allclose(pred_final, target_img)


def test_eval_relabeled_pair2():
    """
    Person vs. Motorcyclist in center.
    Relabeled model incorrectly predicts `person` for `motorcylist`.
    
            [0,0,0,0],
            [0,1,1,0],
            [0,1,1,0],
            [0,1,1,0]
    """
    orig_dname = 'coco-panoptic-133'
    relabeled_dname = 'coco-panoptic-133-relabeled'
    original_names = load_class_names(orig_dname)
    relabeled_names = load_class_names(relabeled_dname)
    u_names = get_universal_class_names()

    pred = np.ones((4,4), dtype=np.uint8) * u_names.index('sky')
    pred[1:,1:3] = u_names.index('person')

    # original COCO image, in coco-panoptic-133
    target_img = torch.ones(4,4) * original_names.index('sky-other-merged')
    target_img = target_img.type(torch.LongTensor)
    target_img[1:,1:3] = original_names.index('person')

    # relabeled COCO image, in coco-panoptic-133-relabeled
    target_img_relabeled = torch.ones(4,4) * relabeled_names.index('sky')
    target_img_relabeled = target_img_relabeled.type(torch.LongTensor)
    target_img_relabeled[1:,1:3] = relabeled_names.index('motorcyclist')

    orig_to_u_transform = ToUniversalLabel(orig_dname)
    relabeled_to_u_transform = ToUniversalLabel(relabeled_dname)
    pred_final, target_img = eval_relabeled_pair(
        pred,
        target_img,
        target_img_relabeled,
        orig_to_u_transform,
        relabeled_to_u_transform
    )
    # treated as 0% accuracy
    pdb.set_trace()
    #assert np.allclose(pred_final, target_img)

    # (array([[142, 142, 142, 142],
    #        [142, 255, 255, 142],
    #        [142, 255, 255, 142],
    #        [142, 255, 255, 142]], dtype=uint8), array([[142, 142, 142, 142],
    #        [142, 125, 125, 142],
    #        [142, 125, 125, 142],
    #        [142, 125, 125, 142]], dtype=uint8))



if __name__ == '__main__':
    """ """
    #test_eval_relabeled_pair1()
    test_eval_relabeled_pair2()



