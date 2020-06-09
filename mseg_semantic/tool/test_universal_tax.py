#!/usr/bin/python3


import cv2
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pdb
import sys
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data
import torch.nn as nn
from typing import List, Optional, Tuple
import time

from model.pspnet import PSPNet
# from model.psanet import PSANet
from util import dataset, transform, config
# from util.dataset_config import infos, v2gids # to be tested
from mseg.utils.dataset_config import infos, v2gids
from util.normalization_utils import get_imagenet_mean_std
from util.verification_utils import verify_architecture
from taxonomy.utils_flat import TaxonomyConverter
from taxonomy.utils_baseline import StupidTaxonomyConverter

# from vis_utils.names_utils import load_class_names
from mseg.utils.names_utils import load_class_names
from test_runner import TestRunner

from util.dir_utils import create_leading_fpath_dirs

"""
Test a non-`oracle` model -- trained on universal taxonomy,
and remapped via linear mapping to a new evaluation taxonomy.
"""


class ToFlatLabel(object):
    def __init__(self, tc_init, dataset):
        self.dataset = dataset
        self.tc = tc_init

    def __call__(self, image, label):
        return image, self.tc.transform_label(label, self.dataset)

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


def get_universal_class_names(args):
    """
        Args:
        -   

        Returns:
        -   
    """
    tsv_fpath = f'taxonomy_{v2gids[args.version]}.tsv'
    tsv_data = pd.read_csv(tsv_fpath, sep='\t', keep_default_na=False)
    names = tsv_data['universal'].tolist()[:-1] # exclude unlabeled
    return names

class UniversalFlatTestRunner:
    """
    """
    def __init__(self, use_gpu = True):
        self.use_gpu = use_gpu
        args = self.get_parser()
        if 'scannet' in args.dataset:
            args.img_name_unique = False
        else:
            args.img_name_unique = True

        self.mean, self.std = get_imagenet_mean_std()
        if args.version == 0:
            tc = StupidTaxonomyConverter(version=0)
            args.universal = False
        else:
            tc = TaxonomyConverter(version=args.version)
        self.tc = tc

        # automatically decide whether to use args.universal
        if args.dataset in self.tc.train_datasets:
            args.universal = True # for now setting all to False

        elif args.dataset in self.tc.test_datasets:
            args.universal = False
        else:
            print("Unknown dataset, please check")


        if args.universal and 'mseg' in args.model_path and ('unrelabeled' not in args.model_path):
            args.relabel = True
        else:
            args.relabel = False

        args.data_root = infos[args.dataset].dataroot
        self.dataset_name = args.dataset
        args.u_classes = tc.classes
        args.classes = len(load_class_names(args.dataset))
        self.class_names = load_class_names(args.dataset)
        args.names_path = infos[args.dataset].names_path

        if args.dataset in tc.convs.keys() and self.use_gpu:
            tc.convs[args.dataset].cuda()
        tc.softmax.cuda()

        if args.universal:
            if args.relabel:
            # args.save_folder = create_leading_fpath_dirs(args.model_path, return_dir=True) + f'/{args.dataset}_universal/{args.base_size}/'
                args.save_folder = args.model_path[:-4] + f'/{args.dataset}_universal_relabel/{args.base_size}/'
            else:
                args.save_folder = args.model_path[:-4] + f'/{args.dataset}_universal/{args.base_size}/'

        else:
            # args.save_folder = create_leading_fpath_dirs(args.model_path, return_dir=True) + f'/{args.dataset}/{args.base_size}/'
            args.save_folder = args.model_path[:-4] + f'/{args.dataset}/{args.base_size}/'

        os.makedirs(args.save_folder, exist_ok=True)


        args.print_freq = 300

        self.args = args
        print(args)


        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))

        test_transform = transform.Compose([transform.ToTensor()])

        args.test_list = infos[args.dataset].vallist
        test_data = dataset.SemData(split=args.split, data_root=args.data_root, data_list=args.test_list, transform=test_transform)

        if args.relabel:
            args.dataset_relabeled = self.get_relabeled_dataset(args.dataset)
            args.test_list_relabeled = infos[args.dataset_relabeled].vallist
            args.data_root_relabeled = infos[args.dataset_relabeled].dataroot
            test_data_relabeled = dataset.SemData(split=args.split, data_root=args.data_root_relabeled, data_list=args.test_list_relabeled, transform=test_transform)

        args.vis_freq = len(test_data.data_list) // 10 + 1


        if args.split == 'test':
            args.vis_freq = 1

        index_start = args.index_start
        if args.index_step == 0:
            index_end = len(test_data.data_list)
        else:
            index_end = min(index_start + args.index_step, len(test_data.data_list))
        test_data.data_list = test_data.data_list[index_start:index_end]
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

        if args.universal:
            names = get_universal_class_names(args)
        else:
            names = [line.rstrip('\n') for line in open(args.names_path)]

        if not args.has_prediction:
            temp_classes = args.classes
            args.classes = args.u_classes
            print(args.classes)
            self.model = self.load_model(args) # model expects u_classes as logits size
            args.classes = temp_classes

            self.save_folder = args.save_folder
            self.u_classes = args.u_classes
            self.classes = args.classes

            if args.universal:
                self.pred_dim = self.u_classes
            else:
                self.pred_dim = self.classes

            gray_folder = os.path.join(args.save_folder, 'gray')
            color_folder = os.path.join(args.save_folder, 'color')
            if args.universal and (self.args.dataset in self.tc.train_datasets):
                self.excluded_ids = self.tc.exclude_universal_ids(self.dataset_name)
            
            
            #### --- JUST ADDED ---- ######
            itask = InferenceTask(
                args,
                base_size = args.base_size,
                crop_h = args.test_h,
                crop_w = args.test_w,
                data_list=test_data.data_list,
                gray_folder=gray_folder,
                output_taxonomy='universal',
                scales = args.scales
            )
            itask.execute(test_loader)
            #### --- JUST ADDED ---- ######



            if args.split != 'test':
                if args.relabel:
                    self.cal_acc_for_relabeled_model(test_data.data_list, test_data_relabeled.data_list, gray_folder, names, demo=True)
                else:
                    self.cal_acc(test_data.data_list, gray_folder, names, demo=True)



    def convert_label_to_pred_taxonomy(self, target): 
        """
        """

        if self.args.universal:
            _, target = ToFlatLabel(self.tc, self.args.dataset)(target, target)
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

    def get_relabeled_dataset(self, dataset_name):

        return dataset_name + '-relabeled'
        # if 'qvga'



if __name__ == '__main__':
    """

    python -u tool/test.py --config=${config}

    """
    use_gpu = True
    test_runner = UniversalFlatTestRunner(use_gpu)

