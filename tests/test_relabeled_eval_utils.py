#!/usr/bin/python3

"""Unit tests to ensure that label maps are properly updated from relabeled masks."""

from pathlib import Path

import imageio
import mseg.utils.names_utils as names_utils
import numpy as np
import torch

import mseg_semantic.tool.relabeled_eval_utils as relabeled_eval_utils
from mseg_semantic.utils.transform import ToUniversalLabel


ROOT_ = Path(__file__).resolve().parent
TEST_DATA_ROOT_ = ROOT_ / "test_data"


def test_eval_relabeled_pair1() -> None:
    """
    Person vs. Motorcyclist in center
    Relabeled model correctly predicts `motorcylist`. for `motorcylist`.

    Motorcyclist silhouette pattern:
            [0,0,0,0],
            [0,1,1,0],
            [0,1,1,0],
            [0,1,1,0]
    """
    orig_dname = "coco-panoptic-133"
    relabeled_dname = "coco-panoptic-133-relabeled"
    original_names = names_utils.load_class_names(orig_dname)
    relabeled_names = names_utils.load_class_names(relabeled_dname)
    u_names = names_utils.get_universal_class_names()

    # prediction in universal taxonomy
    pred_rel = np.ones((4, 4), dtype=np.uint8) * u_names.index("sky")
    pred_rel[1:, 1:3] = u_names.index("motorcyclist")

    # original COCO image, in coco-panoptic-133
    target_img = np.ones((4, 4)) * original_names.index("sky-other-merged")
    target_img[1:, 1:3] = original_names.index("person")
    # target_img = target_img.reshape(1,4,4)

    # relabeled COCO image, in coco-panoptic-133-relabeled
    target_img_relabeled = np.ones((4, 4)) * relabeled_names.index("sky")
    target_img_relabeled[1:, 1:3] = relabeled_names.index("motorcyclist")
    # target_img_relabeled = target_img_relabeled.reshape(1,4,4)

    orig_to_u_transform = ToUniversalLabel(orig_dname)
    relabeled_to_u_transform = ToUniversalLabel(relabeled_dname)
    pred_unrel, target_img, _ = relabeled_eval_utils.eval_rel_model_pred_on_unrel_data(
        pred_rel, target_img, target_img_relabeled, orig_to_u_transform, relabeled_to_u_transform
    )
    # treated as 100% accuracy
    assert np.allclose(pred_unrel, target_img)


def test_eval_relabeled_pair2() -> None:
    """
    Person vs. Motorcyclist in center.
    Relabeled model incorrectly predicts `person` instead of `motorcylist`.

            [0,0,0,0],
            [0,1,1,0],
            [0,1,1,0],
            [0,1,1,0]
    """
    orig_dname = "coco-panoptic-133"
    relabeled_dname = "coco-panoptic-133-relabeled"
    original_names = names_utils.load_class_names(orig_dname)
    relabeled_names = names_utils.load_class_names(relabeled_dname)
    u_names = names_utils.get_universal_class_names()

    pred_rel = np.ones((4, 4), dtype=np.uint8) * u_names.index("sky")
    pred_rel[1:, 1:3] = u_names.index("person")

    # original COCO image, in coco-panoptic-133
    target_img = np.ones((4, 4)) * original_names.index("sky-other-merged")
    target_img[1:, 1:3] = original_names.index("person")

    # relabeled COCO image, in coco-panoptic-133-relabeled
    target_img_relabeled = np.ones((4, 4)) * relabeled_names.index("sky")
    target_img_relabeled[1:, 1:3] = relabeled_names.index("motorcyclist")

    orig_to_u_transform = ToUniversalLabel(orig_dname)
    relabeled_to_u_transform = ToUniversalLabel(relabeled_dname)
    pred_unrel, target_gt_univ, _ = relabeled_eval_utils.eval_rel_model_pred_on_unrel_data(
        pred_rel, target_img, target_img_relabeled, orig_to_u_transform, relabeled_to_u_transform
    )
    # treated as 0% accuracy for person's silhouette and interior

    target_gt = np.ones((4, 4), dtype=np.uint8) * u_names.index("sky")
    target_gt[1:, 1:3] = u_names.index("person")
    assert np.allclose(target_gt_univ, target_gt)

    IGNORE_IDX = 255  # represents unlabeled
    gt_pred_unrel = np.ones((4, 4), dtype=np.uint8) * u_names.index("sky")
    gt_pred_unrel[1:, 1:3] = IGNORE_IDX
    assert np.allclose(pred_unrel, gt_pred_unrel)


def test_eval_relabeled_pair_annotated_as_unlabel() -> None:
    """
    When labels were inaccurate, we often marked them as `unlabeled`,
    e.g. COCO cabinets included `counter` pixels.
    """
    orig_dname = "coco-panoptic-133"
    relabeled_dname = "coco-panoptic-133-relabeled"
    original_names = names_utils.load_class_names(orig_dname)
    relabeled_names = names_utils.load_class_names(relabeled_dname)
    u_names = names_utils.get_universal_class_names()

    wall = u_names.index("wall")
    counter = u_names.index("counter_other")
    cabinet = u_names.index("cabinet")
    pred_rel = np.array(
        [
            [wall, wall, wall, wall],
            [counter, counter, counter, counter],
            [cabinet, cabinet, cabinet, cabinet],
            [cabinet, cabinet, cabinet, cabinet],
        ]
    ).astype(np.uint8)

    # original COCO image, in coco-panoptic-133
    wall = original_names.index("wall-wood")
    cabinet = original_names.index("cabinet-merged")
    target_img = np.array(
        [
            [wall, wall, wall, wall],
            [cabinet, cabinet, cabinet, cabinet],
            [cabinet, cabinet, cabinet, cabinet],
            [cabinet, cabinet, cabinet, cabinet],
        ]
    ).astype(np.uint8)

    # relabeled COCO image, in coco-panoptic-133-relabeled
    # since the counter & cabinet could not be separated w/o
    # drawing new boundary, mark both as `unlabeled`, i.e. 255
    wall = relabeled_names.index("wall")
    # fmt: off
    target_img_relabeled = np.array(
        [
            [wall, wall, wall, wall],
            [255, 255, 255, 255],
            [255, 255, 255, 255],
            [255, 255, 255, 255]
        ]
    ).astype(np.uint8)
    # fmt: on

    orig_to_u_transform = ToUniversalLabel(orig_dname)
    relabeled_to_u_transform = ToUniversalLabel(relabeled_dname)
    pred_unrel, target_gt_univ, acc_diff = relabeled_eval_utils.eval_rel_model_pred_on_unrel_data(
        pred_rel, target_img, target_img_relabeled, orig_to_u_transform, relabeled_to_u_transform
    )

    # goes from 75% to 100%
    assert acc_diff == 25

    wall = u_names.index("wall")
    # fmt: off
    gt_pred_unrel = np.array(
        [
            [wall, wall, wall, wall],
            [255, 255, 255, 255],
            [255, 255, 255, 255],
            [255, 255, 255, 255]
        ], dtype=np.uint8
    )
    # fmt: on
    assert np.allclose(pred_unrel, gt_pred_unrel)

    # fmt: off
    gt_target_gt_univ = np.array(
        [
            [wall, wall, wall, wall],
            [255, 255, 255, 255],
            [255, 255, 255, 255],
            [255, 255, 255, 255]
        ], dtype=np.uint8
    )
    # fmt: on
    assert np.allclose(target_gt_univ, gt_target_gt_univ)


def test_get_px_accuracy_perfect() -> None:
    """ """
    pred = np.ones((4, 4), dtype=np.uint8)
    target = np.ones((4, 4), dtype=np.uint8)
    assert 100 == relabeled_eval_utils.get_px_accuracy(pred, target)


def test_get_px_accuracy_all_wrong() -> None:
    """ """
    pred = np.ones((4, 4), dtype=np.uint8)
    target = np.zeros((4, 4), dtype=np.uint8)
    assert 0 == relabeled_eval_utils.get_px_accuracy(pred, target)


def test_get_px_accuracy_all_relabeled() -> None:
    """ """
    pred = np.ones((4, 4), dtype=np.uint8)
    target = np.ones((4, 4), dtype=np.uint8) * 255
    assert 100 == relabeled_eval_utils.get_px_accuracy(pred, target)


# def test_eval_relabeled_pair_coco_real_unlabel():
#     """ """
#     orig_dname = 'coco-panoptic-133'
#     relabeled_dname = 'coco-panoptic-133-relabeled'

#     pred_rel_fpath = f'{TEST_DATA_ROOT_}/relabeled_gt/000000025986-pred.png'
#     pred_rel = imageio.imread(pred_rel_fpath)

#     target_img_relabeled_fpath = f'{TEST_DATA_ROOT_}/relabeled_gt/000000025986-relabeled-gt.png'
#     target_img_relabeled = imageio.imread(target_img_relabeled_fpath)

#     target_img_fpath = f'{TEST_DATA_ROOT_}/relabeled_gt/000000025986-unrelabeled-gt.png'
#     target_img = imageio.imread(target_img_fpath)

#     orig_to_u_transform = ToUniversalLabel(orig_dname)
#     relabeled_to_u_transform = ToUniversalLabel(relabeled_dname)
#     pred_unrel, target_gt_univ, acc_diff = relabeled_eval_utils.eval_rel_model_pred_on_unrel_data(
#         pred_rel,
#         target_img,
#         target_img_relabeled,
#         orig_to_u_transform,
#         relabeled_to_u_transform
#     )
#     # accuracy should decrease (predicted table instead of counter-other)
#     pdb.set_trace()


if __name__ == "__main__":
    """ """
    test_eval_relabeled_pair1()
    test_eval_relabeled_pair2()
    test_eval_relabeled_pair_annotated_as_unlabel()

    # test_eval_relabeled_pair_coco_real_unlabel()

    test_get_px_accuracy_perfect()
    test_get_px_accuracy_all_wrong()
    test_get_px_accuracy_all_relabeled()
