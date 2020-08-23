#!/usr/bin/python3

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pdb


def read_txt_lines(fpath):
	"""
	"""
	with open(fpath, 'r') as f:
		return f.readlines()


def parse_norms_and_scales(fpath: str):
	"""
		Args:
		-	fpath: path to log file

		Returns:
		-	None
	"""
	norm_lists = defaultdict(list)
	scales_lists = defaultdict(list)

	txt_lines = read_txt_lines(fpath)
	for line in txt_lines:
		if '$' in line:
			norm, timestamp, dname = parse_norm_line(line)
			norm_lists[dname] += [(timestamp,norm)]
		if 'Scales' in line:
			scales_map = parse_scales_line(line)
			for k,v in scales_map.items():
				scales_lists[k] += [v]

	norm_lists = sort_tuple_lists_by_timestamp(norm_lists)

	for dname, norm_list in norm_lists.items():
		timestamps,norms = list(zip(*norm_list))
		norm_lists[dname] = norms

	plot_lists_single_plot(norm_lists, xlabel="Iteration",ylabel="Gradient Norm")
	plot_lists_multiple_subplots(norm_lists, xlabel="Iteration",ylabel="Gradient Norm")

	plot_lists_single_plot(scales_lists, xlabel="Iteration",ylabel="MGDA Scale")
	plot_lists_multiple_subplots(scales_lists, xlabel="Iteration",ylabel="MGDA Scale")


def plot_lists_single_plot(val_lists, xlabel, ylabel):
	"""
		Args:
		-	val_lists
		-	xlabel: 
		-	ylabel: 

		Returns:
		-	None
	"""
	# Use Shared Plots
	fig= plt.figure(dpi=200, facecolor='white')
	for dname, val_list in val_lists.items():
		plt.plot(range(len(val_list)), val_list, label=dname)
		# plt.plot(range(len(val_list)), val_list, 0.1, marker='.', label=dname)

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend(loc='upper left')
	fig.tight_layout(pad=4)
	plt.show() #savefig('fig.pdf')


def plot_lists_multiple_subplots(val_lists, xlabel, ylabel):
	"""
		Args:
		-	val_lists
		-	xlabel
		-	ylabel

		Returns:
		-	None
	"""
	# Use Individual Plots
	fig= plt.figure(dpi=200, facecolor='white')
	subplot_counter = 1
	axes =[]
	for dname, val_list in val_lists.items():
		if subplot_counter == 1:
			axes += [ plt.subplot(4,1,subplot_counter) ]
		else:
			axes += [ plt.subplot(4,1,subplot_counter, sharex=axes[0], sharey=axes[0]) ]
		plt.plot(range(len(val_list)), val_list, label=dname)
		plt.xlabel(xlabel )
		plt.ylabel(ylabel)
		plt.title(dname)
		subplot_counter += 1

	plt.show()


def parse_norm_line(line):
	"""
		Args:
		-	line

		Returns:
		-	norm
		-	timestamp
		-	dname
	"""
	def find_next(str, token='$'):
		return str.find(token)

	dname = line[find_next(line, ':')+1:]
	dname = dname[:find_next(dname, ':')]

	k = find_next(line)
	line = line[k+1:]
	norm_str = line[1:find_next(line)]
	line = line[find_next(line)+1:]
	line = line[find_next(line)+1:]
	time_str = line[1:find_next(line)]

	norm = float(norm_str)
	timestamp = float(time_str)

	return norm, timestamp, dname.strip()

def parse_scales_line(line):
	"""
		Args:
		-	line:

		Returns:
		-	scales_dict
	"""
	def advance_past_token(str, token):
		return str[str.find(token) + len(token):]

	scales_dict = {}
	line = advance_past_token(line, 'Scales:')
	pair_str = line.split(',')
	for pair_str in pair_str:
		dname, scale = pair_str.split(':')
		scales_dict[dname.strip()] = float(scale)
	return scales_dict


def test_parse_norm_line_1():
	"""
	"""
	line = 'Gradient norms: ade20k-v1-qvga: $ 11.55 $, ns = $ 1569682972195191808 $'
	norm, timestamp, dname = parse_norm_line(line)
	assert dname == 'ade20k-v1-qvga'
	assert timestamp == 1569682972195191808
	assert norm == 11.55


def test_parse_norm_line_2():
	"""
	"""
	line = 'Gradient norms: coco-panoptic-v1-qvga: $ 13.65 $, ns = $ 1569682976771436288 $[2019-09-28 08:02:56,933 INFO train.py line 543 91056] Scales: coco-panoptic-v1-qvga: 0.26 , mapillary_vistas_comm-qvga: 0.21 , ade20k-v1-qvga: 0.23 , interiornet-37cls-qvga: 0.29'
	norm, timestamp, dname = parse_norm_line(line)
	assert dname == 'coco-panoptic-v1-qvga'
	assert timestamp == 1569682976771436288
	assert norm == 13.65


def test_parse_scales_line_1():
	"""
	"""
	line = '[2019-09-28 08:02:58,476 INFO train.py line 543 91056] Scales: coco-panoptic-v1-qvga: 0.28 , mapillary_vistas_comm-qvga: 0.20 , ade20k-v1-qvga: 0.24 , interiornet-37cls-qvga: 0.28'
	scales_dict = parse_scales_line(line)
	gt_scales_dict = {
		'coco-panoptic-v1-qvga': 0.28 , 
		'mapillary_vistas_comm-qvga': 0.20 , 
		'ade20k-v1-qvga': 0.24 , 
		'interiornet-37cls-qvga': 0.28
	}
	assert_dict_equal(scales_dict, gt_scales_dict)

def assert_dict_equal(dict1, dict2):
	"""
	"""
	assert set(dict1.keys()) == set(dict2.keys())
	for k, v in dict1.items():
		assert v == dict2[k]



def test_parse_scales_line_2():
	"""
	"""
	line = 'Gradient norms: coco-panoptic-v1-qvga: $ 13.65 $, ns = $ 1569682976771436288 $[2019-09-28 08:02:56,933 INFO train.py line 543 91056] Scales: coco-panoptic-v1-qvga: 0.26 , mapillary_vistas_comm-qvga: 0.21 , ade20k-v1-qvga: 0.23 , interiornet-37cls-qvga: 0.29'
	scales_dict = parse_scales_line(line)
	gt_scales_dict = {
		'coco-panoptic-v1-qvga': 0.26, 
		'mapillary_vistas_comm-qvga': 0.21, 
		'ade20k-v1-qvga': 0.23, 
		'interiornet-37cls-qvga': 0.29
	}
	assert_dict_equal(scales_dict, gt_scales_dict)



def sort_tuple_lists_by_timestamp(norm_lists):
	"""
	"""
	get_timestamp = lambda pair: pair[0]
	for k, norm_list in norm_lists.items():
		norm_lists[k] = sorted(norm_list, key=get_timestamp)


	return norm_lists



def test_sort_tuple_lists_by_timestamp():
	""" """
	norm_lists = {
		# tuple has order (timestamp, norm)
		'a': [(1, 3.5), (3, 1.5), (2, 0.5)],
		'b': [(4,0.6), (0, 1.6), (5, 2.6)]
	}
	
	sorted_lists = sort_tuple_lists_by_timestamp(norm_lists)
	gt_sorted_lists = {
		'a': [(1, 3.5), (2, 0.5), (3, 1.5)], 
		'b': [(0, 1.6), (4, 0.6), (5, 2.6)]
	}
	assert_dict_equal(sorted_lists, gt_sorted_lists)




def visualize_losses():
	"""
		Get the train loss values from each training run (saved in SLURM output
		scripts) and plot them.
	"""
	expname_to_fname_dict = {
		'camvid-qvga-50epochs-bs16-nomgda' : 'slurm-130924.out',
		'nyudepthv2-36-qvga-50epochs-nomgda-bs16' : 'slurm-138433.out',
		'A-C-M-mgda-10-epochs-6-gpus' : 'slurm-139445.out',
		'A-C-M-I-mgda-3epochs-bs128' : 'slurm-139759.out', # scales uniform after 10%
		'C-no-mgda-bs-32-10epochs' : 'slurm-140714.out',
		'A-C-M-I-3Iepochs-normalize_before_FW-mgda-bs128' : 'slurm-140886.out',
		'A-C-M-I-mgda-3epochs-bs128-nomgda' : 'slurm-140963.out',
		'A-C-M-I-3epochs_2gpus_each_bs128-normalizeunitbeforeFW-mgda-lr1' : 'slurm-141004.out',
		'A-C-M-I-NOMGDA-12epochs_2gpus_each_bs128' : 'slurm-141015.out',
		'A-C-M-I-6epochs_2gpus_each_bs128_crop201_no_mgda-crashed' : 'slurm-141016.out',
		'completed-A-C-M-I-NOMGDA-12epochs_2gpus_each_bs128' : 'slurm-141134.out',
		'A-C-M-I-6epochs_2gpus_each_bs128-no_mgda' : 'slurm-141135.out',
		'A-C-M-I-24epochs_2gpus_each_bs128-no_mgda' : 'slurm-141142.out',
		'A-C-M-I-3epochs_2gpus_each_bs256_no_mgda_lrpoint01' : 'slurm-141362.out',
		'A-C-M-I-3epochs_2gpus_each_bs256-no_mgda_lr1' : 'slurm-141363.out',
		'A-C-M-I-3epochs-2gpus_each_bs256_no_mgda_lrpoint1' : 'slurm-141364.out',
		'A-C-M-I-3epochs_2gpus_each_bs128_no_mgda_lrpoint1' : 'slurm-141365.out',
		'A-C-M-I-3epochs_1gpu_each_bs64_crop201_no_mgda_lrpoint01' : 'slurm-141375.out',
		'A-C-M-I-3epochs_1gpu_each_bs64_crop201_no_mgda_lrpoint001' : 'slurm-141376.out',
		'A-C-M-I-3epochs_1gpu_each_bs64_no_mgda_lrpoint001' : 'slurm-141377.out',
		'A-C-M-I-3epochs_1gpu_each_bs32_no_mgda_lrpoint01' : 'slurm-141378.out',
		'A-C-M-I-3epochs_1gpu_each_bs32_no_mgda_lrpoint001' : 'slurm-141379.out',
	}

	SLURM_FILE_DIR = '/Users/johnlamb/Documents/SLURM_FILES'

	for expname, fname in expname_to_fname_dict.items():
		metrics_dict = defaultdict(list)
		fpath = f'{SLURM_FILE_DIR}/{fname}'
		txt_lines = read_txt_lines(fpath)
		for line in txt_lines:
			if 'MainLoss' not in line:
				continue
			MainLoss, AuxLoss, Loss, Accuracy = parse_iter_info_line(line)
			metrics_dict['MainLoss'] += [MainLoss]
			metrics_dict['AuxLoss'] += [AuxLoss]
			metrics_dict['Loss'] += [Loss]
			metrics_dict['Accuracy'] += [Accuracy]

		plot_sublots_with_metrics(expname, metrics_dict)


def plot_sublots_with_metrics(expname: str, metrics_dict: Mapping[str, List[float]] ):
	"""
		Render or save a plot of training metrics (e.g. training loss,
		training accuracy). Share the x-axis, representing training iterations,
		but use different y-axes for different quantities.

		Args:
		-	metrics_dict: Dictionary mapping the name of a metric to a list
				of values.

		Returns:
		-	None
	"""
	subplot_counter = 1
	fig = plt.figure(dpi=200, facecolor='white')
	
	axes = []
	for metric, val_list in metrics_dict.items():
		if subplot_counter == 1:
			axes += [ plt.subplot(4,1,subplot_counter) ]
			plt.title(expname)
		else:
			axes += [ plt.subplot(4,1,subplot_counter, sharex=axes[0]) ]
		plt.plot(range(len(val_list)), val_list, label=metric)
		xlabel = 'iter'
		plt.xlabel(xlabel)
		ylabel = metric
		plt.ylabel(ylabel)
		subplot_counter += 1

	#plt.show()
	plt.savefig(f'loss_plots/{expname}.png')



def parse_iter_info_line(line: str) -> Tuple[float,float,float,float]:
	"""
		Args:
		-	line: string representing output file line

		Returns:
		-	MainLoss: float representing PSPNet CE primary loss value
		-	AuxLoss: float representing PSPNet CE auxiliary loss value
		-	Loss: float representing combined loss
		-	Accuracy: float representing pixel accuracy
	"""
	MainLoss = get_substr(line, start_token='MainLoss', end_token='AuxLoss')
	AuxLoss = get_substr(line, start_token='AuxLoss', end_token='Loss')
	Loss = get_substr(line, start_token=' Loss', end_token='Accuracy')
	Accuracy = get_substr(line, start_token='Accuracy', end_token='.current_iter', alt_end_token='.\n')
	return MainLoss, AuxLoss, Loss, Accuracy


def get_substr(line: str, start_token: str, end_token: str, alt_end_token: str = None) -> float:
	""" 
		Search a string for a substring that will be contained between two specified tokens.
		If the end token may not be always found in the string, an alternate end token can be 
		provided as well.

		Args:
		-	line: string representing line of text
		-	start_token: string 
		-	end_token: string
		-	alt_end_token: string

		Returns:
		-	val: floating point number retrieved
	"""
	i = line.find(start_token)
	j = i + len(start_token)

	# `rel_line` is relevant portion of line
	rel_line = line[j:]

	if end_token not in rel_line:
		rel_line += '\n'
		end_token = alt_end_token
	k = rel_line.find(end_token)
	val = rel_line[:k]

	return float(val)



def test_parse_iter_info_line():
	"""
		3 Simple test cases to make sure that we can parse file lines appropriately.
	"""
	line = '[2019-10-05 07:09:13,411 INFO train.py line 538 112397] Epoch: [101/101][280/281] Data 0.000 (0.072) Batch 0.812 (0.909) Remain 00:01:30 MainLoss 3.3073 AuxLoss 3.3141 Loss 4.6329 Accuracy 0.1890.current_iter: 28380'
	MainLoss, AuxLoss, Loss, Accuracy = parse_iter_info_line(line)
	assert MainLoss == 3.3073
	assert AuxLoss == 3.3141
	assert Loss == 4.6329
	assert Accuracy == 0.1890

	line = '[2019-10-02 15:55:39,707 INFO train.py line 538 27363] Epoch: [2/124][2380/3696] Data 0.000 (0.010) Batch 0.775 (0.763) Remain 95:14:16 MainLoss 0.7233 AuxLoss 1.0095 Loss 1.1271 Accuracy 0.7905.current_iter: 6076'
	MainLoss, AuxLoss, Loss, Accuracy = parse_iter_info_line(line)
	assert MainLoss == 0.7233
	assert AuxLoss == 1.0095
	assert Loss == 1.1271
	assert Accuracy == 0.7905
	
	line = '[2019-09-06 03:12:48,857 INFO train.py line 480 43364] Epoch: [49/50][10/23] Data 0.000 (2.382) Batch 0.220 (2.599) Remain 00:01:33 MainLoss 0.2480 AuxLoss 0.2670 Loss 0.3548 Accuracy 0.9102.'
	MainLoss, AuxLoss, Loss, Accuracy = parse_iter_info_line(line)
	assert MainLoss == 0.2480
	assert AuxLoss == 0.2670
	assert Loss == 0.3548
	assert Accuracy == 0.9102

	print('All tests passed.')


if __name__ == '__main__':

	# FILE BELOW WAS WHEN I NORMALIZED TO TO UNIT LENGTH
	# fpath = '/Users/johnlamb/Documents/train-20190928_080102.log'
	#fpath = '/Users/johnlamb/Documents/train-20190928_095930.log' # training A/C/M/I 3 I epochs w/ unit normalization

	# normalize to unit length, but increase learning rate
	fpath = '/Users/johnlamb/Documents/slurm-141004.out'

	# FILE BELOW WAS WHEN I DO NOT NORMALIZE TO UNIT LENGTH
	#fpath = '/Users/johnlamb/Documents/train-20190928_093558.log'

	#parse_norms_and_scales(fpath)

	# test_parse_norm_line_1()
	# test_parse_norm_line_2()
	# test_parse_scales_line_1()
	# test_parse_scales_line_2()
	# test_sort_tuple_lists_by_timestamp()

	# visualize_losses()
	#test_parse_iter_info_line()


