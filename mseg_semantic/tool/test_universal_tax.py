#!/usr/bin/python3

import argparse
import cv2
import logging
import os
from pathlib import Path
import torch
import torch.nn as nn
from types import SimpleNamespace
from typing import List, Optional, Tuple

from mseg.utils.dataset_config import infos
import mseg.utils.names_utils as names_utils
from mseg.taxonomy.taxonomy_converter import TaxonomyConverter, DEFAULT_TRAIN_DATASETS, TEST_DATASETS

import mseg_semantic.tool.mseg_dataloaders as dataloader_utils
import mseg_semantic.utils.logger_utils as logger_utils
from mseg_semantic.tool.accuracy_calculator import AccuracyCalculator
from mseg_semantic.tool.inference_task import InferenceTask
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

logger = logger_utils.get_logger()


def evaluate_universal_tax_model(args, use_gpu: bool = True) -> None:
    """
    Args:
        args:
        use_gpu
    """
    if "scannet" in args.dataset:
        args.img_name_unique = False
    else:
        args.img_name_unique = True

    model_taxonomy = "universal"

    # automatically decide which evaluation taxonomy to use
    if args.dataset in DEFAULT_TRAIN_DATASETS:
        eval_taxonomy = "universal"
    elif args.dataset in TEST_DATASETS:
        eval_taxonomy = "test_dataset"
    else:
        logger.info("Unknown dataset, please check")

    if eval_taxonomy == "universal" and "mseg" in args.model_name and ("unrelabeled" not in args.model_name):
        eval_relabeled = True
    else:
        eval_relabeled = False

    args.data_root = infos[args.dataset].dataroot
    dataset_name = args.dataset

    if len(args.scales) > 1:
        scale_type = "ms"  # multi-scale
    else:
        scale_type = "ss"  # single-scale

    model_results_root = f"{Path(args.model_path).parent}/{Path(args.model_path).stem}"
    if eval_taxonomy == "universal":
        if eval_relabeled:
            args.save_folder = f"{model_results_root}/{args.dataset}_universal_relabeled/{args.base_size}/{scale_type}/"
        else:
            args.save_folder = f"{model_results_root}/{args.dataset}_universal/{args.base_size}/{scale_type}/"
    else:
        args.save_folder = f"{model_results_root}/{args.dataset}/{args.base_size}/{scale_type}/"

    args.print_freq = 300

    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    logger.info(args)

    # always evaluating on val split
    args.test_list = infos[args.dataset].vallist

    if args.split == "test":
        args.vis_freq = 1

    args.num_model_classes = len(names_utils.get_universal_class_names())

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

    if args.split == "test":
        logger.info("Ground truth labels are not known for test set, cannot compute its accuracy.")
        return

    if eval_taxonomy == "universal" and (args.dataset in DEFAULT_TRAIN_DATASETS):
        # evaluating on training datasets, within a subset of the universal taxonomy
        excluded_ids = get_excluded_class_ids(dataset_name)
    else:
        excluded_ids = []

    if eval_taxonomy == "universal":
        class_names = names_utils.get_universal_class_names()
        num_eval_classes = len(class_names)
    elif eval_taxonomy == "test_dataset":
        class_names = names_utils.load_class_names(args.dataset)
        num_eval_classes = len(class_names)
    elif eval_taxonomy == "naive":
        # get from NaiveTaxonomyConverter class attributes
        raise NotImplementedError

    _, test_data_list = dataloader_utils.create_test_loader(args)
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

    if eval_relabeled:
        logger.info(">>>>>>>>> Calculating *relabeled* accuracy from cached results >>>>>>>>>>")
        args.dataset_relabeled = get_relabeled_dataset(args.dataset)
        relabeled_args = {
            "split": "val",
            "data_root": infos[args.dataset_relabeled].dataroot,
            "test_list": infos[args.dataset_relabeled].vallist,
            "index_start": args.index_start,
            "index_step": args.index_step,
            "workers": args.workers,
        }
        relabeled_args = SimpleNamespace(**relabeled_args)
        _, test_data_relabeled_list = dataloader_utils.create_test_loader(relabeled_args)
        # AccuracyCalculator is constructed for the unrelabeled dataset
        # we will pass relabeled dataset info as args later
        ac.compute_metrics_relabeled_data(test_data_relabeled_list)

    else:
        logger.info(">>>>>>>>> Calculating accuracy from cached results >>>>>>>>>>")
        ac.compute_metrics()

    logger.info(">>>>>>>>> Accuracy computation completed >>>>>>>>>>")


def get_relabeled_dataset(dataset_name: str) -> str:
    """ """
    return dataset_name + "-relabeled"


def get_excluded_class_ids(dataset: str) -> List[int]:
    """Find the classes to exclude when evaluating a "relabeled" MSeg model
    on the val split of a training dataset.

    We retrieve the dictionary `id_to_uid_maps` with (k,v) pairs where
    "k" is the original, unrelabeled training dataset ID, and "v" is
    the universal taxonomy ID.

    Args:
        dataset: name of a MSeg training dataset, e.g. 'coco-panoptic-133'

    Returns:
        zero_class_ids
    """
    tc = TaxonomyConverter()
    # from train to universal. do this zero out or not does not affect when training and testing on same dataset.
    id_maps = tc.id_to_uid_maps[dataset]  
    nonzero_class_ids = set(id_maps.values())
    zero_class_ids = [x for x in range(tc.num_uclasses) if x not in nonzero_class_ids]
    return zero_class_ids


def get_parser() -> CfgNode:
    """
    TODO: add to library to avoid replication.
    """
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation")
    parser.add_argument("--config", type=str, default="config/wilddash_18/wilddash_18_flat.yaml", help="config file")
    parser.add_argument(
        "--file_save", type=str, default="default", help="eval result to save, when lightweight option is on"
    )
    parser.add_argument(
        "opts",
        help="see mseg_semantic/config/test/default_config_360.yaml for all options, model path should be passed in",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


if __name__ == "__main__":
    """
    Example usage:

    python -u mseg_semantic/tool/test_universal_tax.py --config=${config} \
        model_path /path/to/my/model model_name name_of_my_model 
    """
    use_gpu = True
    args = get_parser()

    assert isinstance(args.model_name, str)
    assert isinstance(args.model_path, str)
    assert args.dataset != "default"

    logger.info(args)
    evaluate_universal_tax_model(args, use_gpu)
