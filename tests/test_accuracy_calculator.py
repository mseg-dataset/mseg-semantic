#!/usr/bin/python3

from pathlib import Path
from typing import List, Tuple

import imageio
import mseg.utils.dir_utils as dir_utils
import mseg.utils.names_utils as names_utils
import mseg.utils.txt_utils as txt_utils
import numpy as np

from mseg_semantic.tool.accuracy_calculator import AccuracyCalculator
from mseg_semantic.tool.test_universal_tax import get_excluded_class_ids
from mseg_semantic.utils.config import CfgNode

_ROOT = Path(__file__).resolve().parent


def get_dummy_datalist() -> List[Tuple[str,str]]:
    """
    Write dummy camvid data.
    Expect inter [4,2,1]
    Expect union [7,5,1]
    IoUs: 4/7, 2/5, 1/1
    """
    pred1 = np.array([[0, 0], [1, 0]]).astype(np.uint8)
    target1 = np.array([[0, 0], [1, 1]]).astype(np.uint8)
    # inter [2, 1, 0]
    # union [3, 2, 0]

    pred2 = np.array([[2, 0], [1, 0]]).astype(np.uint8)
    target2 = np.array([[2, 0], [1, 1]]).astype(np.uint8)
    num_classes = 3
    # intersection, [1,1,1]
    # union, [2,2,1]

    pred3 = np.array([[1, 0], [1, 0]]).astype(np.uint8)
    target3 = np.array([[255, 0], [255, 1]]).astype(np.uint8)
    # intersection, [1, 0, 0])
    # union, [2, 1, 0]

    dir_utils.check_mkdir(f"{_ROOT}/accuracy_calculator_data/ground_truth")
    gt_fpath1 = f"{_ROOT}/accuracy_calculator_data/ground_truth/img1.png"
    gt_fpath2 = f"{_ROOT}/accuracy_calculator_data/ground_truth/img2.png"
    gt_fpath3 = f"{_ROOT}/accuracy_calculator_data/ground_truth/img3.png"

    imageio.imwrite(gt_fpath1, target1)
    imageio.imwrite(gt_fpath2, target2)
    imageio.imwrite(gt_fpath3, target3)

    dir_utils.check_mkdir(f"{_ROOT}/accuracy_calculator_data/gray")
    imageio.imwrite(f"{_ROOT}/accuracy_calculator_data/gray/img1.png", pred1)
    imageio.imwrite(f"{_ROOT}/accuracy_calculator_data/gray/img2.png", pred2)
    imageio.imwrite(f"{_ROOT}/accuracy_calculator_data/gray/img3.png", pred3)

    # dummy RGB filepaths
    data_list = [
        (gt_fpath1, gt_fpath1),
        (gt_fpath2, gt_fpath2),
        (gt_fpath3, gt_fpath3),
    ]
    return data_list


def test_constructor() -> None:
    """ """
    args = CfgNode()
    args.img_name_unique = True
    args.taxonomy = "oracle"  # pretend testing a model training in own taxonomy (oracle),
    # not in unified universal taxonomy
    args.vis_freq = 1
    args.model_path = "/path/to/dummy/model"

    data_list = get_dummy_datalist()
    dataset_name = "camvid-11"
    class_names = names_utils.load_class_names(dataset_name)
    camvid_class_names = [
        "Building",
        "Tree",
        "Sky",
        "Car",
        "SignSymbol",
        "Road",
        "Pedestrian",
        "Fence",
        "Column_Pole",
        "Sidewalk",
        "Bicyclist",
    ]
    assert class_names == camvid_class_names

    save_folder = f"{_ROOT}/accuracy_calculator_data"
    num_eval_classes = 11
    ac = AccuracyCalculator(
        args=args,
        data_list=data_list,
        dataset_name=dataset_name,
        class_names=class_names,
        save_folder=save_folder,
        eval_taxonomy="test_dataset",
        num_eval_classes=num_eval_classes,
        excluded_ids=[],
    )


def test_execute() -> None:
    """ """
    args = CfgNode()
    args.img_name_unique = True
    args.taxonomy = "oracle"  # pretend testing a model training in own taxonomy (oracle),
    # not in unified universal taxonomy
    args.vis_freq = 1
    args.model_path = "/path/to/dummy/model"

    data_list = get_dummy_datalist()
    dataset_name = "camvid-11"
    class_names = names_utils.load_class_names(dataset_name)
    camvid_class_names = [
        "Building",
        "Tree",
        "Sky",
        "Car",
        "SignSymbol",
        "Road",
        "Pedestrian",
        "Fence",
        "Column_Pole",
        "Sidewalk",
        "Bicyclist",
    ]
    assert class_names == camvid_class_names

    save_folder = f"{_ROOT}/accuracy_calculator_data"
    num_eval_classes = 11

    ac = AccuracyCalculator(
        args=args,
        data_list=data_list,
        dataset_name=dataset_name,
        class_names=class_names,
        save_folder=save_folder,
        eval_taxonomy="test_dataset",
        num_eval_classes=num_eval_classes,
        excluded_ids=[],
    )
    ac.compute_metrics(save_vis=False)

    results_txt_fpath = f"{_ROOT}/accuracy_calculator_data/results.txt"
    lines = txt_utils.read_txt_file(results_txt_fpath, strip_newlines=False)

    assert "Eval result: mIoU/mAcc/allAcc 0.1792" in lines[0]
    assert "Class_00 result: iou/accuracy 0.5714/1.0000, name: Building." in lines[1]
    assert "Class_01 result: iou/accuracy 0.4000/0.4000, name: Tree." in lines[2]
    assert "Class_02 result: iou/accuracy 1.0000/1.0000, name: Sky." in lines[3]
    assert "Class_03 result: iou/accuracy 0.0000/0.0000, name: Car." in lines[4]
    assert "Class_04 result: iou/accuracy 0.0000/0.0000, name: SignSymbol." in lines[5]
    assert "Class_05 result: iou/accuracy 0.0000/0.0000, name: Road." in lines[6]
    assert "Class_06 result: iou/accuracy 0.0000/0.0000, name: Pedestrian." in lines[7]
    assert "Class_07 result: iou/accuracy 0.0000/0.0000, name: Fence." in lines[8]
    assert "Class_08 result: iou/accuracy 0.0000/0.0000, name: Column_Pole." in lines[9]
    assert "Class_09 result: iou/accuracy 0.0000/0.0000, name: Sidewalk." in lines[10]
    assert "Class_10 result: iou/accuracy 0.0000/0.0000, name: Bicyclist." in lines[11]


def test_relabeled_data_example() -> None:
    """ """
    pass
    # pdb.set_trace()
    # dataset_name = 'coco-panoptic-133'
    # excluded_ids = get_excluded_class_ids(dataset_name)
    # ac = AccuracyCalculator(
    # 	args=args,
    # 	data_list=test_data_list,
    # 	dataset_name=dataset_name,
    # 	class_names=class_names,
    # 	save_folder=args.save_folder,
    # 	eval_taxonomy=eval_taxonomy,
    # 	num_eval_classes=num_eval_classes,
    # 	excluded_ids=excluded_ids
    # )
    # ac.compute_metrics_relabeled_data(test_data_relabeled.data_list)


if __name__ == "__main__":
    test_constructor()
    test_execute()
    test_relabeled_data_example()
