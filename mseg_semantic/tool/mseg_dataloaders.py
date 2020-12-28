#!/usr/bin/python3

import torch.utils.data
from typing import List, Tuple

from mseg_semantic.utils import dataset, transform, config
from mseg_semantic.utils.normalization_utils import get_imagenet_mean_std

def create_test_loader(
	args,
	preprocess_imgs_in_loader: bool = False
) -> Tuple[torch.utils.data.dataloader.DataLoader, List[Tuple[str,str]]]:
	"""
		Create a Pytorch dataloader from a dataroot and list of 
		relative paths.

		Args:

		Returns:
		-	test_loader
		-	data_list: list of 2-tuples (relative rgb path, relative label path)
	"""
	if preprocess_imgs_in_loader:
		# resize and normalize images in advance
		mean, std = get_imagenet_mean_std()
		test_transform = transform.Compose([
			transform.ResizeShort(args.base_size),
			transform.ToTensor(),
			transform.Normalize(mean=mean, std=std)
		])
	else:
		# no resizing on the fly using OpenCV and also normalize images on the fly
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
