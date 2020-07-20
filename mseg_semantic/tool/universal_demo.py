#!/usr/bin/python3

import argparse
import cv2
import logging
import time
import numpy as np
import os
from pathlib import Path
import pdb
import time

from typing import List, Optional, Tuple

from mseg.utils.names_utils import load_class_names, get_universal_class_names

from mseg_semantic.utils import config
from mseg_semantic.utils.config import CfgNode
from mseg_semantic.tool.inference_task import InferenceTask

_ROOT = Path(__file__).resolve().parent.parent

"""
Run inference over all images in a directory, over all frames of a video,
or over all images specified in a .txt file.
"""

def get_logger():
    """
    """
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    return logger

logger = get_logger()

cv2.ocl.setUseOpenCL(False)


def run_universal_demo(args, use_gpu: bool = True) -> None:
    """
        Args:
        -   args:
        -   use_gpu
    """
    if 'scannet' in args.dataset:
        args.img_name_unique = False
    else:
        args.img_name_unique = True

    args.u_classes = get_universal_class_names()
    args.print_freq = 10

    args.split = 'test'
    #os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    logger.info(args)
    logger.info("=> creating model ...")
    args.num_model_classes = len(args.u_classes)

    itask = InferenceTask(
        args,
        base_size = args.base_size,
        crop_h = args.test_h,
        crop_w = args.test_w,
        input_file=args.input_file,
        model_taxonomy='universal',
        eval_taxonomy='universal',
        scales = args.scales
    )
    itask.execute()


def get_parser() -> CfgNode:
    """ """
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, 
        default=f'{_ROOT}/config/final_test/1080/default_config_360.yaml', help='config file')
    parser.add_argument('--file_save', type=str, default='default', help='eval result to save, when lightweight option is on')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', 
        default=None, nargs=argparse.REMAINDER) # model path is passed in 
    args = parser.parse_args()
    print(args)
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


if __name__ == '__main__':
    """
    """
    use_gpu = True
    args = get_parser()

    assert isinstance(args.model_name, str)
    assert isinstance(args.model_path, str)

    if args.dataset == 'default':
        args.dataset = Path(args.input_file).stem

    print(args)
    run_universal_demo(args, use_gpu)

