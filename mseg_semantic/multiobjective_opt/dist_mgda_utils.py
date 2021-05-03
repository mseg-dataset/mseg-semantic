#!/usr/bin/python3

from collections import defaultdict
import logging
import numpy as np
import os
import pdb
import time
import torch
import torch.distributed as dist

from typing import List, Mapping

from mseg_semantic.multiobjective_opt.min_norm_solvers import MinNormSolver
from mseg_semantic.multiobjective_opt.min_norm_solvers_new import MinNormSolver as MinNormSolverNew



def scale_loss_and_gradients(loss: torch.Tensor, optimizer, model, args) -> torch.Tensor:
	"""
	MGDA --> use Frank-Wolfe iteration to compute scales.

	Find min_norm_element() often takes around 0.51 seconds.

	Args:
	-   loss: Pytorch tensor
	-   optimizer: torch.optim object
	-   model: Network passed by reference
	-   args

	Returns:
	-   loss: Pytorch tensor
	"""
	dataset_names = list(args.dataset_gpu_mapping.keys())
	loss_i_tensor_list = all_gather_create_tensor_list(tensor=loss, ngpus_per_node=args.ngpus_per_node)
	dataset_loss_dict = reduce_to_dict_per_dataset(loss_i_tensor_list, args.dataset_gpu_mapping)

	optimizer.zero_grad()
	# Independent: each process will only have gradients with respect to its own subset of the minibatch

	# Under ddp.no_sync() context, this is doing an independent backward op
	assert not model.require_backward_grad_sync
	loss.backward()

	per_dataset_per_param_dict = {}
	# list of all gradients, per each dataset
	dataset_allgrads = defaultdict(list)
	# accumulate the gradients per each task

######################################## print out unsynced gradients
	# for p_name, param in model.named_parameters():
	# 	if param.grad is not None:
	# 		# grad_i_tensor_list = all_gather_create_tensor_list(tensor=param.grad, ngpus_per_node=args.ngpus_per_node)
	# 		#print(f'grad_i_tensor_list for {p_name}: ', grad_i_tensor_list)
	# 		# dataset_grad_p_dict = reduce_to_dict_per_dataset(grad_i_tensor_list, args.dataset_gpu_mapping)
	# 		# per_dataset_per_param_dict[p_name] = dataset_grad_p_dict
	# 		for dname in dataset_names:
	# 			dataset_allgrads[dname] += [param.grad.clone().flatten()] # TODO: remove the flatten??
	# for dname in dataset_names:
	# 	dataset_allgrads[dname] = torch.cat(dataset_allgrads[dname])

	# for dname in dataset_names:
	# 	norm = torch.norm(dataset_allgrads[dname]).item()
	# 	args.logger.info(f'rank: {args.rank}, {dname}: norm {norm}')
	# no need to sort these now, names are unique
##########################################
	dataset_allgrads = defaultdict(list)
	for p_name, param in model.named_parameters():
		if param.grad is not None:
			grad_i_tensor_list = all_gather_create_tensor_list(tensor=param.grad, ngpus_per_node=args.ngpus_per_node)
			#print(f'grad_i_tensor_list for {p_name}: ', grad_i_tensor_list)
			dataset_grad_p_dict = reduce_to_dict_per_dataset(grad_i_tensor_list, args.dataset_gpu_mapping)
			per_dataset_per_param_dict[p_name] = dataset_grad_p_dict

			for dname in dataset_names:
				dataset_allgrads[dname] += [dataset_grad_p_dict[dname].clone().flatten()] # TODO: remove the flatten??
	
	current_ns_time = lambda: int(round(time.time() * 1e9))

	scales = {}

	# sol, min_norm = MinNormSolver.find_min_norm_element([dataset_allgrads[d] for d in dataset_names])
	# for i, d in enumerate(dataset_names):
	# 	scales[d] = float(sol[i])
		# args.logger.info(f'{d}, {scales[d]}')

	for dname in dataset_names:
		dataset_allgrads[dname] = torch.cat(dataset_allgrads[dname])

	# Optionally, could normalize all gradients here.
	for dname, grad_list in dataset_allgrads.items():
		_, grad_norm = normalize_tensor_list(grad_list) # dataset_allgrads[dname]
		if dist.get_rank() == 0:
			print(f'Gradient norms: {dname}: $ {grad_norm:.2f} $, ns = $ {current_ns_time()} $')

	# args.logger.info(dataset_names)
	# args.logger.info(dataset_allgrads.keys())


	sol, min_norm = MinNormSolverNew.find_min_norm_element([dataset_allgrads[d] for d in dataset_names])
	for i, d in enumerate(dataset_names):
		scales[d] = float(sol[i])

	# args.logger.info(f'{scales}')

	# Scaled back-propagation, we must preserve gradients so we will not call optimizer.zero_grad() again
	for p_name, param in model.named_parameters():
		if param.grad is not None:
			# Instead of a second backward pass, just use the results of the original backward pass
			param.grad = scaled_reduce_dict_to_tensor(per_dataset_per_param_dict[p_name], dataset_names, scales)

	# Multi-task loss -- adding each dataset's scaled loss.
	loss = scaled_reduce_dict_to_tensor(dataset_loss_dict, dataset_names, scales)
	return loss, scales


def reduce_to_dict_per_dataset(tensor_list: List[torch.Tensor], dataset_gpu_mapping: Mapping[str,int]):
	"""
		Reduce a list to a dictionary. Take an average of gradient values, or an average of losses.
		Otherwise loss (and thus gradients) would be larger for whichever dataset gets the most GPUs.

		Args:
		-   tensor_list, where i'th element comes from a specific GPU

		Returns:
		-   dataset_tensor_dict: reduced tensors, reduced from corresponding indices i.
	"""
	assert len(tensor_list) > 0

	item0 = tensor_list[0]
	dataset_tensor_dict = { dataset_name: torch.zeros_like(item0) for dataset_name in dataset_gpu_mapping.keys() }

	for dname, gpu_idxs in dataset_gpu_mapping.items():
		for gpu_idx in gpu_idxs:
			dataset_tensor_dict[dname] += tensor_list[gpu_idx]
		dataset_tensor_dict[dname] /= (1. * len(gpu_idxs))

	return dataset_tensor_dict


def scaled_reduce_dict_to_tensor(dataset_grad_p_dict: Mapping[str,torch.Tensor], dataset_names: List[str], scales=Mapping[str,float]):
    """
        Reduce a dictionary to a single tensor, scaling values in linear combination.

        Args:
        -   dataset_grad_p_dict
        -   dataset_names
        -   scales

        Returns:
        -   sum_tensor
    """
    assert len(dataset_grad_p_dict.values()) > 0

    item0 = list(dataset_grad_p_dict.values())[0]
    sum_tensor = torch.zeros_like(item0)
    for dname in dataset_names:
        sum_tensor += scales[dname] * dataset_grad_p_dict[dname]

    return sum_tensor


def all_gather_create_tensor_list(tensor: torch.Tensor, ngpus_per_node: int) -> List[torch.Tensor]:
    """
		torch.distributed.all_gather() is SYNCHRONOUS, i.e. `async_op=False` by default.
		This ensures a barrier.

        Args:
        -   tensor

        Returns:
        -   tensor_list
    """
    # tensor_list -> Output list. It should contain correctly-sized tensors to be used 
    # for output of the collective.
    tensor_list = [ torch.zeros_like(tensor) for _ in range(ngpus_per_node) ]
    # Gathers tensors from the whole group in a list. 
    # The variable `tensor` will not be affected by this operation.
    dist.all_gather(tensor_list=tensor_list, tensor=tensor)
    return tensor_list


def dump_tensor_list_to_disk(tensor_list):
	"""
	"""
	num_tensors = len(tensor_list)
	print(f'Saving {num_tensors} tensors to disk')


def normalize_tensor_list(tensor):
	"""
		Args:
		-	tensor_list: unnnormalized tensor 

		Returns:
		-	tensor: normalized tensor 
		-	norm: norm of vector representing vstacked list
	"""
	norm = torch.norm(tensor)
	return tensor / norm, norm


def get_tensor_list_norm(tensor_list: List[torch.Tensor]):
	""" Compute the norm of a stacked list of 1d tensors.

		Args:
		-	tensor_list: 

		Returns:
		-	float representing value of norm
	"""
	# return torch.norm(torch.cat(tensor_list, dim=0))
	return torch.norm(tensor_list)

