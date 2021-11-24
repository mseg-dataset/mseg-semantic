"""Test a model w/ that makes predictions in the "naive merge" taxonomy."""

import argparse
import logging

from mseg.taxonomy.naive_taxonomy_converter import NaiveTaxonomyConverter

from mseg_semantic.tool.accuracy_calculator import AccuracyCalculator
from mseg_semantic.tool.inference_task import InferenceTask
from mseg_semantic.utils import config
from mseg_semantic.utils.config import CfgNode


def get_logger():
    """ """
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


def test_naive_taxonomy_model(args, use_gpu: bool) -> None:
    """
    python -u ../tool/test_naive_tax.py --config=${config_fpath} dataset ${dataset_name} model_path ${model_fpath} model_name ${model_name}
    """

    """
    args.save_folder, str)
    args.dataset, str)
    args.img_name_unique, bool)
    args.print_freq, int)
    args.num_model_classes, int)
    args.model_path, str)
    """


    import pdb; pdb.set_trace()

    ntc = NaiveTaxonomyConverter()
    class_names = ntc.get_naive_taxonomy_classnames()

    eval_taxonomy = "test_dataset"
    args.print_freq = 1
    args.num_model_classes = len(class_names)

    it = InferenceTask(
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

    
    num_eval_classes = 9999
    dataset_name = None
    test_data_list = None
    excluded_ids = None
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
