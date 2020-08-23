#!/usr/bin/python3

import os
import os.path
import cv2
import numpy as np
import pdb
from torch.utils.data import Dataset
import imageio

from typing import Any, List, Mapping, Tuple

from mseg_semantic.utils.dataset import (
	is_image_file,
	make_dataset
)


"""
Pytorch dataloader class to support domain generalization.

Get same size as you expect
But domains inside minibatch will be random
"""


def append_per_tuple(
	dataset_2tuples: List[Tuple[str,str]], 
	new_val: int
	) -> List[Tuple[str,str,int]]:
	"""
	Given a list of 2-tuple elements, append to every 2-tuple another fixed
	item, such that a list of 3-tuples is returned.
	"""
	dataset_3tuples = []
	for (val0, val1) in dataset_2tuples:
		dataset_3tuples += [(val0,val1,new_val)]
	return dataset_3tuples


def pad_to_max_sz(
	tuple_list: List[Tuple[Any,Any,Any]], 
	max_sz: int
	) -> List[Tuple[Any,Any,Any]]:
	"""
	# pad (duplicate) dataset lists of less common datasets.

		Args:
		-	tuple_list:
		-	max_sz:

		Returns:
		-	repeated_data:
	"""
	repeated_data = []
	while len(repeated_data) < max_sz:
		repeated_data.extend(tuple_list)

	# clamp dataset to max dataset length
	repeated_data = repeated_data[:max_sz]
	assert len(repeated_data) == max_sz
	return repeated_data


class CCSA_Data(Dataset):
	""" """
	def __init__(
		self,
		split: str='train',
		data_roots: Mapping[str,str]=None,
		data_lists: Mapping[str,List[Any]]=None,
		transform_dict: Mapping[str, Any]=None
		):
		"""
		Since each dataset requires its own mapping to the universal taxonomy, we
		save each such transform/mapping in a dictionary.

			Args:
			-	split: string representing dataset split
			-	data_roots: Mapping from dataset name to absolute paths to dataset dirs
			-	data_lists: Mapping from dataset name to file paths of datasets images 
					in given split
			-	transform_dict: Mapping from dataset name to data transform object.
		"""
		self.split = split

		# Assign an integer ID to each of the separate "domains".
		self.domain_idx_map = {
			'coco-panoptic-v1-qvga': 0,
			'mapillary_vistas_comm-qvga': 1,
			'ade20k-v1-qvga': 2
		}
		MAX_DATASET_SZ = 118287 # COCO is currently single largest (by #images)

		# data_list contains paths from all domains
		self.data_list = []
		for i, dname in enumerate(self.domain_idx_map.keys()):

			# has (rgb_fpath, label_fpath)
			dataset_2tuples = make_dataset(split, data_roots[dname], data_lists[dname])
			# now has (rgb_fpath, label_fpath, domain_ID)
			dataset_3tuples = append_per_tuple(dataset_2tuples, self.domain_idx_map[dname])
			
			repeated_data = pad_to_max_sz(dataset_3tuples, MAX_DATASET_SZ)
			self.data_list.extend(repeated_data)
			assert len(self.data_list) == MAX_DATASET_SZ * (i+1)

		# should have: num_images = max_dataset_sz * num_domains
		assert len(self.data_list) == len(self.domain_idx_map.keys()) * MAX_DATASET_SZ
		self.transform_dict = transform_dict


	def __len__(self):
		return len(self.data_list)


	def __getitem__(self, index):
		image_path, label_path, domain_idx = self.data_list[index]
		# if 'leftImg8bit' in image_path and ('idd' not in image_path):
		# print(image_path, label_path)
		# logger.info(image_path + ' ' + label_path)
		image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3

		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
		image = np.float32(image)

		label = imageio.imread(label_path) # # GRAY 1 channel ndarray with shape H * W
		label = label.astype(np.int64)

		if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
			raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
		
		# Each dataset requires its own mapping to the universal taxonomy.
		if self.transform_dict is not None:
			if self.split != 'test':
				image, label = self.transform_dict[domain_idx](image, label)
			else:
				image, label = self.transform_dict[domain_idx](image, image[:, :, 0])

		return image, label, domain_idx


