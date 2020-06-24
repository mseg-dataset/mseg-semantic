#!/usr/bin/python3

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
from mseg_semantic.tool.mseg_dataloaders import get_test_loader
from mseg_semantic.transform import ToUniversalLabel
from mseg_semantic.utils import dataset, transform, config

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


def evaluate_universal_tax_model(use_gpu: bool = True) -> None:
    """
    """
    pdb.set_trace()
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
        print("Unknown dataset, please check")

    if args.universal and 'mseg' in args.model_path and ('unrelabeled' not in args.model_path):
        args.eval_relabeled = True
    else:
        args.eval_relabeled = False

    args.data_root = infos[args.dataset].dataroot
    self.dataset_name = args.dataset
    args.names_path = infos[args.dataset].names_path

    if args.universal:
        if args.eval_relabeled:
        # args.save_folder = create_leading_fpath_dirs(args.model_path, return_dir=True) + f'/{args.dataset}_universal/{args.base_size}/'
            args.save_folder = args.model_path[:-4] + f'/{args.dataset}_universal_relabel/{args.base_size}/'
        else:
            args.save_folder = args.model_path[:-4] + f'/{args.dataset}_universal/{args.base_size}/'

    else:
        # args.save_folder = create_leading_fpath_dirs(args.model_path, return_dir=True) + f'/{args.dataset}/{args.base_size}/'
        args.save_folder = args.model_path[:-4] + f'/{args.dataset}/{args.base_size}/'

    args.print_freq = 300

    self.args = args
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    test_loader, test_data_list = get_test_loader(args, split='val')

    args.vis_freq = len(test_data.data_list) // 10 + 1
    if args.split == 'test':
        args.vis_freq = 1

    if args.universal:
        names = ''
    else:
        names = [line.rstrip('\n') for line in open(args.names_path)]

    self.args.num_model_classes = len(get_universal_class_names())

    if not args.has_prediction:
        temp_classes = args.classes
        args.classes = args.u_classes
        print(args.classes)
        self.model = self.load_model(args) # model expects u_classes as logits size

        self.save_folder = args.save_folder
        self.u_classes = args.u_classes
        self.classes = args.classes

        if args.universal and (self.args.dataset in self.tc.train_datasets):
            self.excluded_ids = self.tc.exclude_universal_ids(self.dataset_name)
        
        itask = InferenceTask(
            args,
            base_size = args.base_size,
            crop_h = args.test_h,
            crop_w = args.test_w,
            data_list=test_data.data_list,
            gray_folder=gray_folder,
            model_taxonomy=model_taxonomy,
            eval_taxonomy=eval_taxonomy,
            scales = args.scales
        )
        itask.execute()

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
            self.cal_acc_for_relabeled_model(test_data.data_list, test_data_relabeled.data_list, gray_folder, names, demo=True)
        else:
            self.cal_acc(test_data.data_list, gray_folder, names, demo=True)


    def convert_label_to_pred_taxonomy(self, target): 
        """
        """

        if self.args.universal:
            _, target = ToUniversalLabel(self.tc, self.args.dataset)(target, target)
            return target.type(torch.uint8).numpy()
        else:
            return target

    def convert_pred_to_label_tax_and_softmax(self, output):
        """ """

        if not self.args.universal:
            output = self.tc.transform_predictions_test(output, self.args.dataset)
        else:
            output = self.tc.transform_predictions_universal(output, self.args.dataset)
        return output



if __name__ == '__main__':
    """

    python -u tool/test.py --config=${config}

    """
    use_gpu = True
    args = self.get_parser()

    assert isinstance(args.model_name, str)
    assert isinstance(args.model_path, str)
    assert args.dataset != 'default'

    print(args)
    evaluate_universal_tax_model(args, use_gpu)

