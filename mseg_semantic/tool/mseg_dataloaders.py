#!/usr/bin/python3

import torch
import torch.utils.data

from mseg_semantic.utils import dataset, transform


def get_test_loader(args, relpath_list: str):
    """
        Args:
        -   args:
        -	split

        Returns:
        -   test_loader
        -   data_list
    """
    test_transform = transform.Compose([transform.ToTensor()])
    test_data = dataset.SemData(
        split=args.split,
        data_root=args.data_root,
        data_list=relpath_list,
        transform=test_transform
    )
    index_start = args.index_start
    if args.index_step == 0:
        index_end = len(test_data.data_list)
    else:
        index_end = min(index_start + args.index_step, len(test_data.data_list))
    test_data.data_list = test_data.data_list[index_start:index_end]
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    return test_loader, test_data.data_list
