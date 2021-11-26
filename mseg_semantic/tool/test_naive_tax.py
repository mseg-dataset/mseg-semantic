"""Test a model w/ that makes predictions in the "naive merge" taxonomy.

Example CLI usage:
python -u ../tool/test_naive_tax.py --config=${config_fpath} dataset ${dataset_name} model_path ${model_fpath} model_name ${model_name}

python mseg_semantic/tool/test_naive_tax.py --config=mseg_semantic/config/test/default_config_360_ms.yaml model_path 
    ../pretrained-semantic-models/mseg-naive-baseline-1m model_name mseg-naive-baseline-1m dataset camvid-11
"""

import argparse
import logging
from pathlib import Path

import mseg.utils.names_utils as names_utils
from mseg.taxonomy.naive_taxonomy_converter import NaiveTaxonomyConverter
from mseg.utils.dataset_config import infos

import mseg_semantic.tool.mseg_dataloaders as dataloader_utils
import mseg_semantic.utils.logger_utils as logger_utils
from mseg_semantic.tool.accuracy_calculator import AccuracyCalculator
from mseg_semantic.tool.inference_task import InferenceTask
from mseg_semantic.utils import config
from mseg_semantic.utils.config import CfgNode


logger = logger_utils.get_logger()


def test_naive_taxonomy_model(args, use_gpu: bool) -> None:
    """

    args.save_folder, str)
    args.dataset, str)
    args.img_name_unique, bool)
    args.print_freq, int)
    args.num_model_classes, int)
    args.model_path, str)
    """

    if "scannet" in args.dataset:
        args.img_name_unique = False
    else:
        args.img_name_unique = True

    args.data_root = infos[args.dataset].dataroot
    dataset_name = args.dataset

    if len(args.scales) > 1:
        scale_type = "ms"  # multi-scale
    else:
        scale_type = "ss"  # single-scale

    model_results_root = f"{Path(args.model_path).parent}/{Path(args.model_path).stem}"
    args.save_folder = f"{model_results_root}/{args.dataset}/{args.base_size}/{scale_type}/"

    ntc = NaiveTaxonomyConverter()
    naive_class_names = ntc.get_naive_taxonomy_classnames()
    args.num_model_classes = len(naive_class_names)

    args.print_freq = 100
    args.test_list = infos[args.dataset].vallist

    eval_taxonomy = "test_dataset"

    itask = InferenceTask(
        args=args,
        base_size=args.base_size,
        crop_h=args.test_h,
        crop_w=args.test_w,
        input_file=None,
        model_taxonomy="naive",
        eval_taxonomy=eval_taxonomy,
        scales=args.scales,
        use_gpu=use_gpu,
    )
    itask.execute()
    
    class_names = names_utils.load_class_names(args.dataset)
    num_eval_classes = len(class_names)

    excluded_ids = []
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
    logger.info(">>>>>>>>> Calculating accuracy from cached results >>>>>>>>>>")
    ac.compute_metrics()


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


if __name__ == "__main__":
    """
    Usage:

    """
    use_gpu = True
    args = get_parser()

    logger.info(args)
    test_naive_taxonomy_model(args, use_gpu)
