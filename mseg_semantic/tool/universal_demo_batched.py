#!/usr/bin/python3

import argparse
import math
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from mseg.utils.names_utils import load_class_names, get_universal_class_names

import mseg_semantic.utils.logger_utils as logger_utils
from mseg_semantic.utils import config
from mseg_semantic.utils.config import CfgNode
from mseg_semantic.tool.batched_inference_task import BatchedInferenceTask


_ROOT = Path(__file__).resolve().parent.parent

"""
Run single-scale inference over all images in a directory, over all frames of a video,
or over all images specified in a .txt file.

We assume a fixed image size for the whole dataset, and you must provide the
native image width and height at the command line. The "base size" will be
set automatically (config param will be ignored) such that the image is
scaled up to the largest possible size while fitting within the crop.
"""


logger = logger_utils.get_logger()

cv2.ocl.setUseOpenCL(False)


def determine_max_possible_base_size(h: int, w: int, crop_sz: int) -> int:
    """Given a crop size and original image dims for aspect ratio, determine
    the max base_size that will fit within the crop.
    """
    longer_size = max(h, w)
    if longer_size == h:
        scale = crop_sz / float(h)
        base_size = math.floor(w * scale)
    else:
        scale = crop_sz / float(w)
        base_size = math.floor(h * scale)

    return base_size


def run_universal_demo_batched(args, use_gpu: bool = True) -> None:
    """
    Args:
        args:
        use_gpu: whether to use GPU for inference.
    """
    if "scannet" in args.dataset:
        args.img_name_unique = False
    else:
        args.img_name_unique = True

    args.u_classes = get_universal_class_names()
    args.print_freq = 10

    args.split = "test"
    logger.info(args)
    logger.info("=> creating model ...")
    args.num_model_classes = len(args.u_classes)
    args.base_size = determine_max_possible_base_size(
        h=args.native_img_h, w=args.native_img_w, crop_sz=min(args.test_h, args.test_w)
    )

    itask = BatchedInferenceTask(
        args,
        base_size=args.base_size,
        crop_h=args.test_h,
        crop_w=args.test_w,
        input_file=args.input_file,
        model_taxonomy="universal",
        eval_taxonomy="universal",
        scales=args.scales,
    )
    itask.execute()


def get_parser() -> CfgNode:
    """ """
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation")
    parser.add_argument(
        "--config", type=str, default=f"{_ROOT}/config/test/default_config_360_ss.yaml", help="config file"
    )
    parser.add_argument(
        "--file_save", type=str, default="default", help="eval result to save, when lightweight option is on"
    )
    parser.add_argument(
        "opts", help="see config/ade20k/ade20k_pspnet50.yaml for all options", default=None, nargs=argparse.REMAINDER
    )  # model path is passed in
    args = parser.parse_args()
    print(args)
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


if __name__ == "__main__":
    """Example CLI usage:
    python mseg_semantic/tool/universal_demo_batched.py
      --config mseg_semantic/config/test/480/default_config_batched_ss.yaml
      native_img_h 1200
      native_img_w 1920
      model_name mseg-3m-480p
      model_path ./pretrained-semantic-models/mseg-3m-480p/mseg-3m-480p.pth
      input_file ~/argoverse/train1/273c1883-673a-36bf-b124-88311b1a80be/ring_front_center
    """
    use_gpu = True
    args = get_parser()

    assert isinstance(args.native_img_w, int)
    assert isinstance(args.native_img_h, int)
    assert isinstance(args.model_name, str)
    assert isinstance(args.model_path, str)
    assert isinstance(args.input_file, str)
    if not os.path.isdir(args.input_file):
        raise RuntimeError("Please provide a valid image directory using the input_file argument")

    if args.dataset == "default":
        args.dataset = Path(args.input_file).stem

    print(args)
    run_universal_demo_batched(args, use_gpu)
