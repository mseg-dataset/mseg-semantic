#!/usr/bin/python3

import numpy as np
import pdb
import torch

from mseg_semantic.utils.normalization_utils import normalize_img

def test_normalize_img_test_mean_only():
	""" 
	Take image of shape HWC, i.e. (2 x 2 x 3)
	"""
	image = np.array(
		[ 
			[
				[20,22,24],
				[26,28,30]
			],
			[
				[32,34,36],
				[38,40,42]
			] 
		]
	).astype(np.uint8)
	input = torch.from_numpy(image.transpose((2, 0, 1))).float()
	# tensor is now CHW, i.e. (3,2,2)
	mean = [30,30,30]

	normalize_img(input, mean)

	# subtract 30 from all entries
	gt_input = torch.tensor(
		[
			[
				[-10,-8,-6],
				[  -4,-2, 0]
			],
			[
				[2,4,6],
				[ 8,10,12]
			]
		])
	gt_input = gt_input.permute(2,0,1).float()
	assert torch.allclose(input, gt_input)
	assert isinstance(input, torch.Tensor)

def test_normalize_img_test_mean_std_both():
	""" 
	Take image of shape (2 x 2 x 3)
	"""
	image = np.array(
		[ 
			[
				[20,22,24],
				[26,28,30]
			],
			[
				[32,34,36],
				[38,40,42]
			] 
		]
	).astype(np.uint8)
	input = torch.from_numpy(image.transpose((2, 0, 1))).float()
	# tensor is now CHW, i.e. (3,2,2)
	mean = [30,30,30]
	std = [2,2,2]

	normalize_img(input, mean, std)

	# subtract 30 from all entries
	gt_input = torch.tensor(
		[
			[
				[-10/2, -8/2, -6/2],
				[ -4/2, -2/2,  0/2]
			],
			[
				[ 2/2, 4/2,  6/2],
				[ 8/2, 10/2, 12/2]
			]
		])
	gt_input = gt_input.permute(2,0,1).float()
	assert torch.allclose(input, gt_input)
	assert isinstance(input, torch.Tensor)

if __name__ == '__main__':
	""" """
	test_normalize_img_test_mean_only()
	test_normalize_img_test_mean_std_both()


