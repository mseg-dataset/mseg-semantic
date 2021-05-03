#!/usr/bin/python3

import numpy as np
import pdb
import torch

from multiobjective_opt.dist_mgda_utils import (
	reduce_to_dict_per_dataset,
	scaled_reduce_dict_to_tensor,
	normalize_tensor_list
)

def test_all_gather_create_tensor_list():
	"""
		NOT EASY TO TEST SINCE MUST BE ON SEPARATE cpus/GPUS FOR IT TO WORK
	"""
	pass


def test_scaled_reduce_dict_to_tensor():
	"""
	"""
	dataset_grad_p_dict = {
		'coco': torch.tensor([1.,2.]),
		'ade20k':  torch.tensor([3.,4.]),
		'mapillary':  torch.tensor([5.,6.])
	}
	dataset_names = ['coco', 'ade20k', 'mapillary']
	scales = {'coco': 1., 'ade20k': 5., 'mapillary': 2.}

	tensor = scaled_reduce_dict_to_tensor(dataset_grad_p_dict, dataset_names, scales=scales)
	gt_tensor = torch.tensor([26., 34.])
	assert torch.allclose(tensor, gt_tensor)


def test_reduce_to_dict_per_dataset():
	"""
	"""
	ngpus_per_node = 8
	tensor_list = [torch.ones(1) * i for i in range(ngpus_per_node) ]
	dataset_gpu_mapping = { 
			'coco':[0,1,2], 
			'mapillary': [3,4,5], 
			'ade20k': [6,7] 
		}

	dataset_loss_dict = reduce_to_dict_per_dataset(tensor_list, dataset_gpu_mapping)
	gt_dataset_loss_dict = {
		'coco': torch.tensor([3./3]),  # (0 + 1 + 2 ) / 3
		'mapillary': torch.tensor([12./3.]), # (3 + 4 + 5) / 3
		'ade20k': torch.tensor([13./2.]) # (6 + 7) / 2
	}
	assert_tensor_dicts_are_equal(dataset_loss_dict, gt_dataset_loss_dict)
	print(dataset_loss_dict)


def assert_tensor_dicts_are_equal(dict1, dict2):
	"""
	"""
	assert set(dict1.keys()) == set(dict2.keys())
	for k, v1 in dict1.items():
		assert torch.allclose(v1, dict2[k])


def test_normalize_tensor_list():
	"""
	"""
	tensor_list = [ 
		torch.arange(5).type(torch.float32), 
		torch.ones(3).type(torch.float32), 
		torch.ones(2).type(torch.float32) * 2
	]
	print('Unnormalized: ', tensor_list)
	normalized_tensor_list, norm = normalize_tensor_list(tensor_list)

	gt_tensor_list = np.array([0,1,2,3,4,1,1,1,2,2.])
	gt_norm = np.linalg.norm(gt_tensor_list)

	assert np.allclose(gt_norm, 6.403, atol=1e-3)
	assert torch.allclose( norm, torch.Tensor([gt_norm]) )

	gt_tensor0 = torch.tensor([0. , 0.156, 0.312, 0.468, 0.624])
	gt_tensor1 = torch.tensor([0.156, 0.156, 0.156])
	gt_tensor2 = torch.tensor([0.312, 0.312])

	assert len(normalized_tensor_list) == 3
	assert torch.allclose(normalized_tensor_list[0], gt_tensor0, atol=1e-2)
	assert torch.allclose(normalized_tensor_list[1], gt_tensor1, atol=1e-2)
	assert torch.allclose(normalized_tensor_list[2], gt_tensor2, atol=1e-2)


if __name__ == '__main__':

	# test_all_gather_create_tensor_list()
	#test_scaled_reduce_dict_to_tensor()
	#test_reduce_to_dict_per_dataset()

	test_normalize_tensor_list()


