#!/usr/bin/python3

import apex
import argparse
from collections import defaultdict
import numpy as np
import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist

from typing import List, Mapping

from mseg_semantic.multiobjective_opt.dist_mgda_utils import (
	scale_loss_and_gradients,
	all_gather_create_tensor_list,
	reduce_to_dict_per_dataset,
	scaled_reduce_dict_to_tensor
)



class LinearModel(nn.Module):

	def __init__(self):
		""" """
		super(LinearModel, self).__init__()



		#self.bn = torch.nn.BatchNorm1d(num_features)

		self.linear = nn.Linear(1, 1, bias=False)


	def forward(self, x):
		""" """

		x = self.bn(x)
		return self.linear(x)


class SyncBatchNormModel(nn.Module):

	def __init__(self):
		""" """
		super(SyncBatchNormModel, self).__init__()
		self.sync_bn = torch.nn.SyncBatchNorm(num_features=1)

	def forward(self, x):
		""" """
		return self.sync_bn(x)



class SpatialBatchNormLayer(nn.Module):

	def __init__(self):
		""" """
		super(SpatialBatchNormLayer, self).__init__()
		num_features = 1
		self.bn = torch.nn.BatchNorm2d(num_features)

	def forward(self, x):
		""" """
		return self.bn(x)


def init_weights(m):
	print(m)
	if type(m) == nn.Linear:
		m.weight.data.fill_(3.0)
		print(m.weight)


def test_single_process():
	""" """
	x = torch.tensor([1.])
	y = torch.tensor([3.])
	net = LinearModel()
	net.apply(init_weights)

	loss = (net(x) - y) ** 2

	loss.backward()
	weight_grad = net.linear.weight.grad
	print('Pytorch grad: ', weight_grad)
	print('Expected grad: ', 2 * (net.linear.weight * x - y) * x)
	



def test_multiple_processes():
	"""

	gloo for cpu, nccl for gpu

		Args:
		-	None

		Returns:
		-	None
	"""
	parser = argparse.ArgumentParser(description='Distributed MGDA Unit Tests')
	parser.add_argument('--use_apex', action='store_true') # default=True
	parser.add_argument('--multiprocessing_distributed', action='store_false') # default=True

	parser.add_argument('--train_gpu', type=List[int], default= [0,1])# [0, 1, 2, 3, 4, 5, 6])
	parser.add_argument('--ngpus_per_node', type=int, default=None)
	parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:6789')
	parser.add_argument('--base_lr', type=float, default=1.)
	parser.add_argument('--world_size', type=int, default=1)
	parser.add_argument('--rank', type=int, default=0)
	parser.add_argument('--dist_backend', type=str, default='nccl') # 'gloo')
	parser.add_argument('--dataset_gpu_mapping', type=Mapping[int,str], 
		default = {
			'coco':[0],
			'mapillary': [1]
		}
		# default = {
		# 	'coco':[0,1,2],
		# 	'mapillary': [3,4],
		# 	'ade20k': [5,6]
		# }
	) 
	parser.add_argument('--opt_level', type=str, default='O0')
	parser.add_argument('--keep_batchnorm_fp32', default=None)
	parser.add_argument('--loss_scale', default=None)
	args = parser.parse_args()

	args.ngpus_per_node = len(args.train_gpu)

	os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
	args.world_size = args.ngpus_per_node * args.world_size

	# Spawns nprocs processes that run fn with args.
	# `main_worker` function is called as fn(i, *args), where i is the process index and 
	# args is the passed through tuple of arguments.
	# nprocs denotes the number of processes to spawn.
	mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
	# main_worker(1, args.ngpus_per_node, args)


def main_worker(gpu: int, ngpus_per_node: int, argss) -> None:
	"""
		Args:
		-	gpu
		-	ngpus_per_node

		Returns:
		-	None
	"""
	global args
	args = argss

	args.rank = args.rank * args.ngpus_per_node + gpu
	# print('Args: ', args)
	# print('Args rank: ', args.rank)
	dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

	# print('rank', dist.get_rank())

	#model = LinearModel()
	#model.apply(init_weights)
	#model = SpatialBatchNormLayer()

	model = SyncBatchNormModel()

	optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr)
	if main_process():
		print('Creating model in main process')

	torch.cuda.set_device(gpu)
	# model = model.cuda()
	model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])
	print('Distributed Model: ', model)

	for name, p in model.named_parameters():
		print(f'name={name}')

	rank_to_dataset_map = {}
	for dataset, gpu_idxs in args.dataset_gpu_mapping.items():
		for gpu_idx in gpu_idxs:
			rank_to_dataset_map[gpu_idx] = dataset

	dataset = rank_to_dataset_map[args.rank]

	num_train_examples = 2
	x = torch.arange(num_train_examples*2).reshape(num_train_examples,2) * args.rank
	x = x.float()
	y = torch.ones(num_train_examples,2) * -1

	print('X shape: ', x.shape)
	print('Y shape: ', y.shape)

	torch.cuda.set_device(gpu)
	train(x, y, model, optimizer, args)



def main_process():
	"""
	"""
	return args.rank % args.ngpus_per_node == 0



def train(inputs, targets, model, optimizer, args) -> None:
	"""
		Note: ddp.no_sync() is only available in Pytorch >1.2.0 (not 1.1.0)

		Everything is working in terms of gathering/setting gradients 
		when we're fully under no_sync() for forward/backward

		SyncBatchNorm works correctly even under ddp.no_sync().
		
		Args:
		-	x
		-	y
		-	model
		-	optimizer
		-	args

		Returns:
		-	
	"""
	rank = dist.get_rank()
	print(f'Before iters: rank={rank}, iter={iter}, Running mean: ', model.module.sync_bn.running_mean)

	num_iters = inputs.shape[0]
	for i in range(num_iters):

		x = inputs[i].reshape(1,1,2,1).cuda(non_blocking=True)
		y = targets[i].reshape(1,1,2,1).cuda(non_blocking=True)
		# print('x and y shape: ', x.shape, y.shape)

		print(f'rank={rank}, iter={i}: x={x}')
		print(f'rank={rank}, iter={i}: y={y}')
		
		with model.no_sync():
			model(x)

		print(f'rank={rank}, iter={i}, Running mean: ', model.module.sync_bn.running_mean)
		continue

		# print(f'rank = {rank}: Loss before detach: ', loss)

		DIST_REGIME = 'all_reduce' # 'mgda' # 'all_gather' #  # 'reduce' #
		
		with model.no_sync():
			optimizer.zero_grad()
			loss = (model(x) - y) ** 2
			loss.backward()

		curr_w = model.module.linear.weight.detach().cpu().numpy()
		print(f'Iter i={i}, rank={rank}, Curr model weight: ', curr_w )

		print(f'Iter i={i}, rank={rank}, Actual grad: ', model.module.linear.weight.grad)
		single_gpu_expected_grads = 2 * (x.cpu().numpy() * curr_w - y.cpu().numpy() ) * x.cpu().numpy()
		print(f'Iter i={i}, rank={rank}, Expected single gpu grad: ',single_gpu_expected_grads)

		all_x = np.arange(2)
		all_y = np.ones(2) * -1
		all_expected_grads = 2 * (all_x * curr_w - all_y ) * all_x
		print(f'Iter i={i}, rank={rank}, Expected averaged grad: ', np.mean(all_expected_grads))
		
		dataset_names = list(args.dataset_gpu_mapping.keys())
		per_dataset_per_param_dict = {}
		# list of all gradients, per each dataset
		dataset_allgrads = defaultdict(list)
		# accumulate the gradients per each task

		# no need to sort these now, names are unique
		for p_name, param in model.named_parameters():
			if param.grad is not None:
				grad_i_tensor_list = all_gather_create_tensor_list(tensor=param.grad, ngpus_per_node=args.ngpus_per_node)
				print(f'grad_i_tensor_list for {p_name}: ', grad_i_tensor_list)
				dataset_grad_p_dict = reduce_to_dict_per_dataset(grad_i_tensor_list, args.dataset_gpu_mapping)
				per_dataset_per_param_dict[p_name] = dataset_grad_p_dict
				print(per_dataset_per_param_dict)

				for dname in dataset_names:
					dataset_allgrads[dname] += [dataset_grad_p_dict[dname].clone().flatten()] # TODO: remove the flatten??
		
		scales = {'coco': 1, 'mapillary': 3}

		# Scaled back-propagation, we must preserve gradients so we will not call optimizer.zero_grad() again
		for p_name, param in model.named_parameters():
			if param.grad is not None:
				# Instead of a second backward pass, just use the results of the original backward pass
				param.grad = scaled_reduce_dict_to_tensor(per_dataset_per_param_dict[p_name], dataset_names, scales)
				print(f'Set {p_name} param.grad to {param.grad}')


		# if DIST_REGIME == 'all_reduce':
		# 	# Reduces the tensor data across all machines in such a way that all get the final result.
		# 	dist.all_reduce(tensor=loss)
		# 	print(f'rank = {rank}: Main loss after all reduce: ', loss)

		# elif DIST_REGIME == 'reduce':
		# 	# Reduces the tensor data across all machines. Only the process with rank dst 
		# 	# is going to receive the final result.
		# 	dist.reduce(tensor=loss, dst=0)
		# 	print(f'rank = {rank}: Main loss after all reduce: ', loss)

		# elif DIST_REGIME == 'all_gather':
		# 	optimizer.zero_grad()
		# 	loss.backward()
		# 	pytorch_grad = model.linear.weight.grad
		# 	expected_grad = 2 * (model.linear.weight * x - y) * x
		# 	print(f'rank = {rank}: Pytorch grad: ', pytorch_grad, ' vs. expected grad: ', expected_grad)
		# 	optimizer.step()
		# 	main_loss = loss.detach() 
		# 	print(f'rank = {rank}: Main loss after detach: ', main_loss)
		# 	tensor_list = all_gather_create_tensor_list(tensor=model.linear.weight.grad, ngpus_per_node=args.ngpus_per_node)
		# 	print(f'rank = {rank}: Tensor list: ', tensor_list)
		# 	print(f'rank = {rank}: model.linear.weight.grad: ', model.linear.weight.grad)
		# 	dataset_grad_dict = { dataset: torch.zeros_like(model.linear.weight.grad) for dataset in args.dataset_gpu_mapping.keys()}
		# 	for dataset, gpu_list in args.dataset_gpu_mapping.items():
		# 		for gpu_idx in gpu_list:
		# 			dataset_grad_dict[dataset] += tensor_list[gpu_idx]
			
		# 	print(dataset_grad_dict)
		# elif DIST_REGIME == 'mgda':
		# 	loss = scale_loss_and_gradients(loss, optimizer, model, args)
			
		# 	# If there was NO MGDA, you would use the following two lines, and nothing would converge!
		# 	# optimizer.zero_grad()
		# 	# dist.all_reduce(tensor=loss)
		# 	# loss.backward()

		print(f'rank={rank}, During Iter {i} ', model.module.linear.weight)
		optimizer.step()
		print(f'rank={rank}, After Iter {i} ', model.module.linear.weight)

if __name__ == '__main__':
	# test_single_process()
	test_multiple_processes()



