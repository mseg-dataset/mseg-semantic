#!/usr/bin/python3

import logging
import numpy as np
import os
from pathlib import Path
import torch
import torch.nn.functional as F

from mseg.utils.dataset_config import infos

from mseg_semantic.utils.config import CfgNode
from mseg_semantic.utils.verification_utils import verify_architecture
from mseg_semantic.tool.inference_task import InferenceTask

"""
Test an `oracle` model -- trained and tested on the same taxonomy/dataset.
Thus, is designed for testing a single model's performance on a single-dataset.
"""


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

logger = get_logger()


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


def test_oracle_taxonomy_model(use_gpu: bool = True):
    """
    Test a model that was trained in the exact same taxonomy we wish
    to evaluate in.
    """
    args = get_parser()
    logger.info(args)

    if 'scannet' in args.dataset:
        args.img_name_unique = False
    else:
        args.img_name_unique = True

    args.taxonomy = 'oracle'
    dataset = args.dataset
    args.names_path = infos[self.dataset].names_path
    args.save_folder = f'{Path(args.model_path).stem}/{args.dataset}/{args.base_size}/'
    os.makedirs(args.save_folder, exist_ok=True)
    args.data_root = infos[self.dataset].dataroot
    args.test_list = infos[self.dataset].vallist

    dataset_name = dataset
    class_names = list(np.genfromtxt(args.names_path, delimiter='\n', dtype=str))
    num_classes = len(class_names)
    pred_dim = num_classes

    # verify_architecture(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info(f"Classes: {args.classes}")
    gray_folder = os.path.join(args.save_folder, 'gray')

    relpath_list = infos[args.dataset].vallist

    if not args.has_prediction:
        args.print_freq = 100
        itask = InferenceTask(
            args,
            base_size=args.base_size,
            crop_h=args.test_h,
            crop_w=args.test_w,
            input_file=None,
            gray_folder=gray_folder,
            model_taxonomy='test_dataset', # i.e. is oracle
            eval_taxonomy='test_dataset',
            scales=args.scales
        )
        itask.execute(test_loader)

    if args.split != 'test':
        ac = AccuracyCalculator(args, test_data_list, dataset_name, class_names, save_folder)
        ac.execute()

    # def get_best_base_size(self): # currently only for models trained with qvga

    #     mapping_qvga = {
    #         'voc2012': 240, # in the supplementary material, wvga self training result is using xx, since 240 (480) is bad.
    #         # 'voc2012': 300, 
    #         'coco': 240,
    #         'ade20k': 240, 
    #         'map': 600,
    #         'bdd': 600,
    #         'idd': 600,
    #         'nyudepthv2-37': 360,
    #         'wilddash': 480,
    #         'camvid': 600,
    #         'cityscapes': 720,
    #         'scannet-20': 180,
    #     }

    #     for x, y in mapping_qvga.items():
    #         if self.dataset_name.startswith(x):
    #             return y * 2


if __name__ == '__main__':
    """ """
    use_gpu = True
    test_oracle_taxonomy_model(use_gpu)


