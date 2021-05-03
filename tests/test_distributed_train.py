
import os
import torch
from util import dataset, transform
import torch.multiprocessing as mp
import torch.distributed as dist


def main_process():
	""" """
	return args['rank'] % 8 == 0


def train(train_loader):
	""" """
	print(args)

	if main_process():
		print('Main process runs in ', args)

	for i, (input, target) in enumerate(train_loader):
		print('hello from training with ', args)



def main_worker(gpu, ngpus_per_node, argss):
	""" """
	global args
	print('Argss: ', argss)
	args = argss
	args['rank'] = gpu
	rank = args['rank'] * ngpus_per_node + gpu
	print(f'Rank: {rank}')
	print(f'Args on {rank}: ', args)
	dist.init_process_group(
		backend=args['dist_backend'], 
		init_method=args['dist_url'], 
		world_size=args['world_size'], 
		rank=args['rank']
	)

	train_transform = transform.Compose([
	transform.RandScale([args.scale_min, args.scale_max])
	])

	train_data = dataset.SemData(
		split='train', 
		data_root=args['data_root'],
		data_list=args['train_list'],
		transform=train_transform
	)
	train_sampler = torch.utils.data.distributed.DistributedSampler(
		train_data, 
		num_replicas=args.num_replica_per_dataset, 
		rank=args.dataset_rank
	)
	train_loader = torch.utils.data.DataLoader(
		train_data,
		batch_size=args.batch_size, 
		shuffle=(train_sampler is None), 
		num_workers=args.workers, 
		pin_memory=True, 
		sampler=train_sampler, 
		drop_last=True
	)


def main():
	""" """
	ngpus_per_node = 8
	world_size = 1
	world_size = ngpus_per_node * world_size
	print(f'World size: {world_size}')
	args = { 
		'world_size' : world_size,
		'dist_url': 'tcp://127.0.0.1:6789',
		'dist_backend': 'nccl',
		'scale_min': 0.5,  # minimum random scale
		'scale_max': 2.0  # maximum random scale
		'data_root':,
		'train_list':
	}
	mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


if __name__ == '__main__':
	main()


