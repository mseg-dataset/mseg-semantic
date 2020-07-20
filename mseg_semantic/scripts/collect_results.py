#!/usr/bin/python3

import argparse
import os
import pdb
from typing import List

import numpy as np
# import scipy.stats.hmean as hmean
from scipy.stats.mstats import gmean

# zero_shot_datasets
zs_datasets = [
	'voc2012',
	'pascal-context-60',
	'camvid-11',
	'wilddash-19',
	'kitti-19',
	'scannet-20'
]

# oracle-trained datasets
o_datasets = [
	'voc2012',
	'pascal-context-60',
	'camvid-11',
	'kitti-19',
	'scannet-20'
]

training_datasets = [
	'coco-panoptic-133_universal',
	'ade20k-150_universal',
	'mapillary-public65_universal',
	'idd-39_universal',
	'bdd_universal',
	'cityscapes-19_universal',
	'sunrgbd-37_universal'
]

# universal taxonomy models
u_models = [
	'coco-panoptic-133-1m',
	'ade20k-150-1m',
	'mapillary-65-1m',
	'idd-39-1m',
	'bdd-1m',
	'cityscapes-19-1m',
	'sunrgbd-37-1m',
	'mseg-1m',
	'mseg-unrelabeled-1m',
	'mseg-mgda-1m',
	'mseg-3m-480p',
	'mseg-3m-720p',
	'mseg-3m',
]

u_names = [
	'COCO',
	'ADE20K',
	'Mapillary',
	'IDD',
	'BDD',
	'Cityscapes',
	'SUN RGBD', 
	'MSeg-1m',
	'MSeg-1m-w/o relabeling',
	'MSeg-MGDA-1m',
	'MSeg-3m-480p',
	'MSeg-3m-720p',
	'MSeg-3m-1080p',
]
# 'naive'

# oracle taxonomy names
o_models = [
	'voc2012-1m',
	'pascal-context-60-1m',
	'camvid-11-1m',
	'kitti-19-1m',
	'scannet-20-1m'
]

o_names = [
	'VOC Oracle',
	'PASCAL Context Oracle',
	'Camvid Oracle',
	'KITTI Oracle',
	'ScanNet Oracle'
]

VERBOSE = False

def parse_result_file(result_file: str) -> float:
	""" """
	if not os.path.isfile(result_file):
		if VERBOSE:
			print(result_file + ' does not exist!')
		return 100000

	with open(result_file, 'r') as f:
		tmp = f.readlines()[0]
		miou = tmp.split('/')[2].split(' ')[-1]
		miou = float(miou) * 100
		# miou = "{:.1f}".format(miou)
	return miou


def parse_folder(folder, resolution: str, scale: str):
	"""
	# folder containing subfolders as 360/720/1080

		Args:
			folder
			resolution
			scale: string representing inference scale option,
				either 'ss' or 'ms' (single-scale or multi-scale)
	"""
	mious = []
	resolutions = ['360', '720', '1080']
	for b in resolutions:
		result_file = os.path.join(folder, b, scale, 'results.txt')
		# parse_file 
		mious.append(parse_result_file(result_file))

	if resolution == 'max':
		max_miou = max(mious)
		return [max_miou]

	else:
		val_idx = resolutions.index(resolution)
		return [mious[val_idx]]

def harmonic_mean(x: np.ndarray) -> float:
	""" 
	1. Take the reciprocal of all numbers in the dataset
	2. Find the arithmetic mean of those reciprocals
	3. Take the reciprocal of that number
	"""
	n = x.shape[0]
	mean = np.sum(1/x) / n
	return 1/mean


def arithmetic_mean(x: np.ndarray) -> float:
	""" """
	return np.mean(x)


def geometric_mean(x: np.ndarray) -> float:
	""" """
	n = x.shape[0]
	prod = np.prod(x)
	return prod ** (1/n)


def collect_results_at_res(
	datasets: List[str],
	resolution: str,
	scale: str,
	output_format: str,
	mean_type: str = 'harmonic'):
	""" """
	print(' '*60, (' '*5).join(datasets), ' '* 10 + 'mean')
	for m, name in zip(u_models, u_names):
		results = []
		for d in datasets:

			# rename 
			if ('mseg' in m) and ('unrelabeled' not in m) and (d in training_datasets):
				d += '_relabeled'

			folder = f'/srv/scratch/jlambert30/MSeg/pretrained-semantic-models/{m}/{m}/{d}'
			mious = parse_folder(folder, resolution, scale)
			results.append(mious)
	
		tmp_results = np.array([r[0] for r in results])
		if mean_type == 'harmonic':
			#results.append([len(tmp_results) / np.sum(1.0/np.array(tmp_results))])
			results.append([harmonic_mean(tmp_results)])
		elif mean_type == 'arithmetic':
			results.append([arithmetic_mean(tmp_results)])
		elif mean_type == 'geometric':
			results.append([geometric_mean(tmp_results)])
		else:
			print('Unknown mean type')
			exit()
		if output_format == 'latex':
			dump_results_latex(name, results)
		elif output_format == 'markdown':
			dump_results_markdown(name, results)


def dump_results_latex(name: str, results) -> None:
	""" """
	results = ['/'.join(["{:.1f}".format(r).rjust(5) for r in x]) for x in results]
	results = ['$ ' + r + ' $' for r in results]
	print(name.rjust(50), ' & ',    ' & '.join(results) + '\\\\')


def dump_results_markdown(name: str, results) -> None:
	""" """
	results = ['|'.join(["{:.1f}".format(r).rjust(5) for r in x]) for x in results]
	results = ['| ' + r + '' for r in results]
	print(name.rjust(50), '  ',    ' '.join(results) + '|')


def collect_oracle_results_at_res(resolution: str, scale: str, output_format: str) -> None:
	""" """
	results = []
	print(' '*60, (' '*5).join(o_datasets), ' '* 10 + 'mean')
	for m, name, d in zip(o_models, o_names, o_datasets):
		folder = f'/srv/scratch/jlambert30/MSeg/pretrained-semantic-models/{m}/{m}/{d}'
		mious = parse_folder(folder, resolution)
		results.append(mious)

	dump_results_latex('Oracle', results)
	dump_results_markdown('Oracle', results)


def collect_zero_shot_results(scale: str, output_format: str) -> None:
	""" """
	# 'ms' vs. 'ss'
	for resolution in ['360','720','1080','max']: #  '480', '2160',
		print(f'At resolution {resolution}')
		collect_results_at_res(zs_datasets, resolution, scale, output_format)


def collect_oracle_results(scale: str, output_format: str) -> None:
	# 'ms' vs. 'ss'
	for resolution in ['360','720','1080','max']: #  '480', '2160',
		print(f'At resolution {resolution}')
		collect_oracle_results_at_res(resolution, scale, output_format)


def collect_training_dataset_results(scale: str, output_format: str):
	""" """
	# 'ss' only
	for resolution in ['360','720','1080','max']: #  '480', '2160',
		print(f'At resolution {resolution}')
		collect_results_at_res(training_datasets, resolution, scale, output_format)


if __name__ == "__main__":
	# Parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--regime",
		required=True,
		type=str,
		help="Testing regime -- either `zero_shot`, `oracle`, or `training_datasets` ",
		choices=['zero_shot','oracle','training_datasets']
	)
	parser.add_argument(
		"--scale",
		required=True,
		type=str,
		help="ss (single-scale) or ms (multi-scale)",
		choices=['ss','ms']
	)
	parser.add_argument(
		"--output_format",
		required=True,
		type=str,
		help="latex or markdown",
		choices=['latex','markdown']
	)
	args = parser.parse_args()
	print(args)

	if args.regime == 'zero_shot':
		collect_zero_shot_results(args.scale, args.output_format)

	elif args.regime == 'oracle':
		collect_oracle_results(args.scale, args.output_format)

	elif args.regime == 'training_datasets':
		collect_training_dataset_results(args.scale, args.output_format)

	else:
		print("Unknown testing regime")


