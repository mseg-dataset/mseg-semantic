#!/usr/bin/python3

import argparse
import logging
import os
from pathlib import Path
from typing import List

import cv2
import mseg.utils.names_utils as names_utils
import torch
import torch.nn.functional as F
from mseg.utils.dataset_config import infos

import mseg_semantic.utils.logger_utils as logger_utils
from mseg_semantic.utils import config
from mseg_semantic.utils.config import CfgNode
# from mseg_semantic.utils.verification_utils import verify_architecture
from mseg_semantic.tool.accuracy_calculator import AccuracyCalculator
from mseg_semantic.tool.inference_task import InferenceTask
from mseg_semantic.tool.mseg_dataloaders import create_test_loader


cv2.ocl.setUseOpenCL(False)

"""
Test an `oracle` model -- trained and tested on the same taxonomy/dataset.
Thus, is designed for testing a single model's performance on a single-dataset.
"""

logger = logger_utils.get_logger()


def get_parser() -> CfgNode:
    """
    TODO: add to library to avoid replication.
    """
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation")
    parser.add_argument("--config", type=str, default="config/wilddash_18/wilddash_18_flat.yaml", help="config file")
    parser.add_argument(
        "--file_save", type=str, default="default", help="eval result to save, when lightweight option is on"
    )
    # parser.add_argument('--file_load', type=str, default='', help='possible additional config')
    # parser.add_argument('--checkpoint_load', type=str, default='', help='possible checkpoint loading directly specified in argument')
    parser.add_argument(
        "opts", help="see config/ade20k/ade20k_pspnet50.yaml for all options", default=None, nargs=argparse.REMAINDER
    )  # model path is passed in
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def test_oracle_taxonomy_model(args, use_gpu: bool = True) -> None:
    """
    Test a model that was trained in the exact same taxonomy we wish
    to evaluate in.

    Args:
        args:
        use_gpu: whether to use GPU for inference.
    """
    if "scannet" in args.dataset:
        args.img_name_unique = False
    else:
        args.img_name_unique = True

    # These are all `oracle` models
    model_taxonomy = "test_dataset"
    eval_taxonomy = "test_dataset"

    args.data_root = infos[args.dataset].dataroot
    dataset_name = args.dataset

    if len(args.scales) > 1:
        scale_type = "ms"  # multi-scale
    else:
        scale_type = "ss"  # single-scale

    model_results_root = f"{Path(args.model_path).parent}/{Path(args.model_path).stem}"
    args.save_folder = f"{model_results_root}/{args.dataset}/{args.base_size}/{scale_type}/"

    # args.save_folder = f'{Path(args.model_path).stem}/{args.dataset}/{args.base_size}/'

    class_names = names_utils.load_class_names(dataset_name)
    args.num_model_classes = len(class_names)
    num_eval_classes = args.num_model_classes

    # verify_architecture(args)

    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    logger.info(args)

    args.print_freq = 100
    args.test_list = infos[args.dataset].vallist

    if not args.has_prediction:
        itask = InferenceTask(
            args=args,
            base_size=args.base_size,
            crop_h=args.test_h,
            crop_w=args.test_w,
            input_file=None,
            model_taxonomy=model_taxonomy,
            eval_taxonomy=eval_taxonomy,
            scales=args.scales,
        )
        itask.execute()

    logger.info(">>>>>>>>> Calculating accuracy from cached results >>>>>>>>>>")
    if args.split == "test":
        logger.info("Ground truth labels are not known for test set, cannot compute its accuracy.")
        return

    excluded_ids = []  # no classes are excluded from evaluation of the test sets
    _, test_data_list = create_test_loader(args)
    ac = AccuracyCalculator(
        args=args,
        data_list=test_data_list,
        dataset_name=dataset_name,
        class_names=class_names,
        save_folder=args.save_folder,
        eval_taxonomy=eval_taxonomy,
        num_eval_classes=num_eval_classes,
        excluded_ids=excluded_ids,
    )
    ac.compute_metrics()

    logger.info(">>>>>>>>> Accuracy computation completed >>>>>>>>>>")


if __name__ == "__main__":
    """
    Usage:

    """
    use_gpu = True
    args = get_parser()

    logger.info(args)
    test_oracle_taxonomy_model(args, use_gpu)
