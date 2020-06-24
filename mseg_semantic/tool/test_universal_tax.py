#!/usr/bin/python3

import argparse
import cv2
import logging
import numpy as np
import os
from pathlib import Path
import pdb
import torch
import torch.nn as nn
from typing import List, Optional, Tuple
import time

from mseg.utils.dataset_config import infos
from mseg.utils.dir_utils import create_leading_fpath_dirs
from mseg.utils.names_utils import load_class_names, get_universal_class_names

from mseg.taxonomy.taxonomy_converter import (
    TaxonomyConverter,
    DEFAULT_TRAIN_DATASETS,
    TEST_DATASETS
)
from mseg.taxonomy.naive_taxonomy_converter import NaiveTaxonomyConverter

from mseg_semantic.model.pspnet import PSPNet
from mseg_semantic.tool.inference_task import InferenceTask
from mseg_semantic.utils.transform import ToUniversalLabel
from mseg_semantic.utils import dataset, transform, config
from mseg_semantic.utils.config import CfgNode


"""
Provides functionality to test a non-`oracle` model -- 
i.e. a model trained in our universal/unified taxonomy,

If we wish to evaluate on a test dataset, we will remap
predictions via linear mapping to a new evaluation taxonomy.
Test labels are not remapped/modified in this case.

If we wish to evaluate on a training dataset, we remap
original labels from the val set to the universal taxonomy,
and then evaluate only classes jointly present in the 
training dataset taxonomy and universal taxonomy.
"""


cv2.ocl.setUseOpenCL(False)


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


def get_relabeled_dataset(dataset_name: str) -> str:
    """ """
    return dataset_name + '-relabeled'


def evaluate_universal_tax_model(args, use_gpu: bool = True) -> None:
    """
        Args:
        -   args:
        -   use_gpu

        Returns:
        -   None
    """
    if 'scannet' in args.dataset:
        args.img_name_unique = False
    else:
        args.img_name_unique = True

    model_taxonomy = 'universal'

    # automatically decide which evaluation taxonomy to use
    if args.dataset in DEFAULT_TRAIN_DATASETS:
        eval_taxonomy = 'universal'
    elif args.dataset in TEST_DATASETS:
        eval_taxonomy = 'test_dataset'
    else:
        logger.info("Unknown dataset, please check")

    pdb.set_trace()
    if eval_taxonomy == 'universal' \
        and 'mseg' in args.model_name \
        and ('unrelabeled' not in args.model_name):
        args.eval_relabeled = True
    else:
        args.eval_relabeled = False

    args.data_root = infos[args.dataset].dataroot
    dataset_name = args.dataset
    args.names_path = infos[args.dataset].names_path

    model_root = str(Path(args.model_path).parent) + str(Path(args.model_path).stem)
    if eval_taxonomy == 'universal':
        if args.eval_relabeled:
            args.save_folder = f'{model_root}/{args.dataset}_universal_relabeled/{args.base_size}/'
        else:
            args.save_folder = f'{model_root}/{args.dataset}_universal/{args.base_size}/'
    else:
        args.save_folder = f'{model_root}/{args.dataset}/{args.base_size}/'

    args.print_freq = 300

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    logger.info(args)

    # always evaluating on val split
    args.test_list = infos[args.dataset].vallist

    if args.split == 'test':
        args.vis_freq = 1

    args.num_model_classes = len(get_universal_class_names())

    if not args.has_prediction:
        itask = InferenceTask(
            args=args,
            base_size = args.base_size,
            crop_h = args.test_h,
            crop_w = args.test_w,
            input_file=None,
            model_taxonomy=model_taxonomy,
            eval_taxonomy=eval_taxonomy,
            scales = args.scales
        )
        itask.execute()

    # TODO: pass the excluded ids to the AccuracyCalculator
    tc = None
    if eval_taxonomy == 'universal' and (args.dataset in tc.train_datasets):
        # evaluating on training datasets, within a subset of the universal taxonomy
        excluded_ids = tc.exclude_universal_ids(dataset_name)

    if args.eval_taxonomy == 'universal':
        names = ''
    elif args.eval_taxonomy == 'test_dataset':
        names = [line.rstrip('\n') for line in open(args.names_path)]
    elif args.eval_taxonomy == 'naive':
        raise NotImplementedError

    if args.eval_relabeled:
        args.dataset_relabeled = get_relabeled_dataset(args.dataset)
        args.test_list_relabeled = infos[args.dataset_relabeled].vallist
        args.data_root_relabeled = infos[args.dataset_relabeled].dataroot
        test_data_relabeled = dataset.SemData(
            split=args.split,
            data_root=args.data_root_relabeled,
            data_list=args.test_list_relabeled,
            transform=test_transform
        )
        ac = AccuracyCalculator(args, test_data_list, dataset_name, class_names, save_folder)
    
    else:
        ac = AccuracyCalculator(args, test_data_list, dataset_name, class_names, save_folder)

    ac.execute()

    if args.split != 'test':
        if args.eval_relabeled:
            ac.cal_acc_for_relabeled_model(test_data.data_list, test_data_relabeled.data_list, gray_folder, names, demo=True)
        else:
            ac.cal_acc(test_data.data_list, gray_folder, names, demo=True)


def get_parser() -> CfgNode:
    """
    TODO: add to library to avoid replication.
    """
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/wilddash_18/wilddash_18_flat.yaml', help='config file')
    parser.add_argument('--file_save', type=str, default='default', help='eval result to save, when lightweight option is on')
    # parser.add_argument('--file_load', type=str, default='', help='possible additional config')
    # parser.add_argument('--checkpoint_load', type=str, default='', help='possible checkpoint loading directly specified in argument')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER) # model path is passed in 
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


if __name__ == '__main__':
    """

    python -u tool/test.py --config=${config}

    """
    use_gpu = True
    args = get_parser()

    assert isinstance(args.model_name, str)
    assert isinstance(args.model_path, str)
    assert args.dataset != 'default'

    logger.info(args)
    evaluate_universal_tax_model(args, use_gpu)

