#!/usr/bin/python3

import torch.utils.data
from typing import List, Tuple

import mseg_semantic.utils.normalization_utils as normalization_utils
from mseg_semantic.utils import dataset, transform, config


def create_test_loader(
    args, use_batched_inference: bool = False
) -> Tuple[torch.utils.data.dataloader.DataLoader, List[Tuple[str, str]]]:
    """Create a Pytorch dataloader from a dataroot and list of relative paths.

    Args:
        args: CfgNode object
        use_batched_inference: whether to process images in batch mode

    Returns:
        test_loader
        data_list: list of 2-tuples (relative rgb path, relative label path)
    """
    preprocess_imgs_in_loader = True if use_batched_inference else False

    if preprocess_imgs_in_loader:
        # resize and normalize images in advance
        mean, std = normalization_utils.get_imagenet_mean_std()
        test_transform = transform.Compose(
            [transform.ResizeShort(args.base_size), transform.ToTensor(), transform.Normalize(mean=mean, std=std)]
        )
    else:
        # no resizing on the fly using OpenCV and also normalize images on the fly
        test_transform = transform.Compose([transform.ToTensor()])
    test_data = dataset.SemData(
        split=args.split, data_root=args.data_root, data_list=args.test_list, transform=test_transform
    )

    index_start = args.index_start
    if args.index_step == 0:
        index_end = len(test_data.data_list)
    else:
        index_end = min(index_start + args.index_step, len(test_data.data_list))
    test_data.data_list = test_data.data_list[index_start:index_end]
    data_list = test_data.data_list

    # limit batch size to 1 if not performing batched inference
    batch_size = args.batch_size_val if use_batched_inference else 1

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )
    return test_loader, data_list
