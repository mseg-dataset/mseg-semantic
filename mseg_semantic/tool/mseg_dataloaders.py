#!/usr/bin/python3

import torch.utils.data
from typing import List, Tuple

from mseg_semantic.utils import dataset, transform, config


def create_test_loader(args) -> Tuple[torch.utils.data.dataloader.DataLoader, List[Tuple[str,str]]]:
	"""
		Create a Pytorch dataloader from a dataroot and list of 
		relative paths.

		Args:

		Returns:
		-	test_loader
		-	data_list: list of 2-tuples (relative rgb path, relative label path)
	"""
	test_transform = transform.Compose([transform.ToTensor()])
	test_data = dataset.SemData(
		split=args.split,
		data_root=args.data_root,
		data_list=args.test_list,
		transform=test_transform
	)

	index_start = args.index_start
	if args.index_step == 0:
		index_end = len(test_data.data_list)
	else:
		index_end = min(index_start + args.index_step, len(test_data.data_list))
	test_data.data_list = test_data.data_list[index_start:index_end]
	data_list = test_data.data_list
	test_loader = torch.utils.data.DataLoader(
		test_data,
		batch_size=1,
		shuffle=False,
		num_workers=args.workers,
		pin_memory=True
	)
	return test_loader, data_list