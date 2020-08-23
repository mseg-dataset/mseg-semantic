#!/usr/bin/python3

import argparse
import numpy as np

# from mseg.utils.dataset_config import infos
from mseg.utils.dir_utils import check_mkdir

from mseg_semantic.utils import transform
from mseg_semantic.utils.normalization_utils import get_imagenet_mean_std
from mseg_semantic.domain_generalization.ccsa_data import (
	append_per_tuple,
	pad_to_max_sz,
	CCSA_Data
)


def test_append_per_tuple():
	""" """
	dataset_2tuples = [
		('/path/to/img0', '/path/to/label0'),
		('/path/to/img1', '/path/to/label1'),
		('/path/to/img2', '/path/to/label2')
	]
	new_val = 'ade20k'
	dataset_3tuples = append_per_tuple(dataset_2tuples, new_val)

	gt_dataset_3tuples = [
		('/path/to/img0', '/path/to/label0', 'ade20k'),
		('/path/to/img1', '/path/to/label1', 'ade20k'),
		('/path/to/img2', '/path/to/label2', 'ade20k')
	]
	assert gt_dataset_3tuples == dataset_3tuples


def test_pad_to_max_sz():
	"""
	"""
	tuple_list = [
		('a', 1),
		('b', 2)
	]
	max_sz = 3
	padded_tuple_list = pad_to_max_sz(tuple_list, max_sz)
	assert len(padded_tuple_list) == 3
	gt_tuple_list = [
		('a', 1),
		('b', 2),
		('a', 1)
	]
	assert padded_tuple_list == gt_tuple_list


# def test_ccsa_data():
# 	""" Requires valid file paths.
# 	"""
# 	datasets = [
# 		'ade20k-v1-qvga', 
# 		'coco-panoptic-v1-qvga', 
# 		'mapillary_vistas_comm-qvga', 
# 		'interiornet-37cls-qvga'
# 		]

# 	mean, std = get_imagenet_mean_std()

# 	train_h, train_w = 201, 201
# 	transform_list = [
# 		transform.Crop([train_h, train_w], crop_type='rand', padding=mean, ignore_label=255),
# 		transform.ToTensor()
# 	]
# 	train_transform = transform.Compose(transform_list)

# 	data_roots = {dataset:infos[dataset].dataroot for dataset in datasets}
# 	train_lists = {dataset:infos[dataset].trainlist for dataset in datasets}

# 	COCO_LEN = 118287
# 	train_data = CCSA_Data(
# 		split='train', 
# 		data_roots=data_roots, 
# 		data_lists=train_lists, 
# 		transform_dict={'ade20k-v1-qvga': train_transform}
# 	)
# 	assert len(train_data) == COCO_LEN * 3

# 	check_mkdir('temp_files/ccsa_data')
# 	for i in np.random.randint(low=0,high=COCO_LEN*3,size=(1000,)):
# 		pytorch_img, _, domain = train_data[i]
# 		np_img = pytorch_img.permute(1,2,0).cpu().numpy()
# 		np_img = np_img.astype(np.uint8)
# 		cv2.imwrite(f'temp_files/ccsa_data/domain_{domain}__i_{i}.png', np_img[:,:,::-1])



if __name__ == '__main__':
	"""
	"""
	test_append_per_tuple()
	test_pad_to_max_sz()
	#test_ccsa_data()



