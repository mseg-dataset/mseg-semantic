

import os
import pdb
import numpy as np
# import scipy.stats.hmean as hmean
from scipy.stats.mstats import gmean

datasets = ['voc2012', 'pascal-context-60', 'camvid-11', 'wilddash-19',  'kitti-19', 'scannet-20']

# datasets = ['coco-panoptic-133_universal','ade20k-150_universal', 'mapillary-public65_universal', 'idd-39_universal', 'bdd_universal',   'cityscapes-19_universal', 'sunrgbd-37_universal']
# datasets = [d + '_relabel' for d in datasets]

models = [
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

names = [
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

def parse_file(result_file):
	""" """
	if not os.path.isfile(result_file):
		print(result_file + ' does not exist!')
		return 100000

	with open(result_file, 'r') as f:
		tmp = f.readlines()[0]
		miou = tmp.split('/')[2].split(' ')[-1]
		miou = float(miou) * 100
		# miou = "{:.1f}".format(miou)
	return miou


def parse_folder(folder, resolution: str):
	"""
	# folder containing subfolders as 360/480/1080
	"""
	mious = []
	resolutions = ['360', '720', '1080']
	for b in resolutions:
		result_file = os.path.join(folder, b, 'ss', 'results.txt')
		# parse_file 
		mious.append(parse_file(result_file))

	if resolution == 'max':
		max_miou = max(mious)
		return [max_miou]

	else:
		val_idx = resolutions.index(resolution)
		return [mious[val_idx]]

def harmonic_mean(x):
	""" 
	1. Take the reciprocal of all numbers in the dataset
	2. Find the arithmetic mean of those reciprocals
	3. Take the reciprocal of that number
	"""
	n = x.shape[0]
	mean = np.sum(1/x) / n
	return 1/mean


def arithmetic_mean(x):
	""" """
	return np.mean(x)


def geometric_mean(x):
	""" """
	n = x.shape[0]
	prod = np.prod(x)
	return prod ** (1/n)


def collect_results(resolution: str, mean_type = 'harmonic'):
	""" """
	print(' '*60, (' '*5).join(datasets), ' '* 10 + 'mean')
	for m, name in zip(models, names):
		results = []
		for f in datasets:
			folder = f'/srv/scratch/jlambert30/MSeg/pretrained-semantic-models/{m}/{m}/{f}'
			mious = parse_folder(folder, resolution)
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
		dump_results_latex(name, results)
		#dump_results_markdown(name, results)


def dump_results_latex(name, results):
	""" """
	results = ['/'.join(["{:.1f}".format(r).rjust(5) for r in x]) for x in results]
	results = ['$ ' + r + ' $' for r in results]
	print(name.rjust(50), ' & ',    ' & '.join(results) + '\\\\')

def dump_results_markdown(name, results):
	""" """
	results = ['/'.join(["{:.1f}".format(r).rjust(5) for r in x]) for x in results]
	results = ['$ ' + r + ' $' for r in results]
	print(name.rjust(50), ' & ',    ' & '.join(results) + '\\\\')




if __name__ == '__main__':
	""" """
	for resolution in ['360', '720', '1080', 'max']:
		print(f'At resolution {resolution}')
		collect_results(resolution)



