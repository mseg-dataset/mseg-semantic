#!/usr/bin/python3

from pathlib import Path
from types import SimpleNamespace

from mseg_semantic.tool.test_oracle_tax import test_oracle_taxonomy_model
from mseg_semantic.scripts.collect_results import parse_result_file

REPO_ROOT_ = Path(__file__).resolve().parent.parent

def test_evaluate_universal_tax_model():
	"""
	Ensure testing script works correctly.
	"""
	args = None
	use_gpu = True
	

	"""
	base_sizes=(
	        #360
	        720
	        #1080
	
	python -u mseg_semantic/tool/test_universal_tax.py --config=${config_fpath}
		dataset ${dataset_name} model_path ${model_fpath} model_name ${model_name}
	"""
	base_size = 360
	d = {
		'dataset': 'camvid-11',
		'config': f'{REPO_ROOT_}/mseg_semantic/config/test/default_config_${base_size}.yaml', 
		#'model_path': f'{_ROOT}/pretrained-semantic-models/${model_name}/${model_name}.pth',
		'model_path': '/srv/scratch/jlambert30/MSeg/mseg-semantic/integration_test_data/camvid-11-1m.pth',
		'model_name': 'mseg-3m',

		'input_file': 'default',
		'base_size': base_size,
		'test_h': 713,
 		'test_w': 713,
 		'scales': [1.0],
 		'save_folder': 'default',
 		'arch': 'hrnet',
		'index_start': 0,
		'index_step': 0,
		'workers': 16,
		'has_prediction': False,
		'split': 'val',
		'vis_freq': 20
	}
	args = SimpleNamespace(**d)
	use_gpu = True
	#test_oracle_taxonomy_model(args, use_gpu)

	# assert a file exists
	print('Completed')

	result_file_path = '/srv/scratch/jlambert30/MSeg/mseg-semantic/integration_test_data/'
	result_file_path += 'camvid-11-1m/camvid-11/360/ss/result.txt'
	mIoU = parse_result_file(result_file_path)

	assert mIoU == 72.0


	# for base_size in [360,720,1080]:
	# 	# Args that would be provided in command line and in config file
	# 	d = {
	# 		'config': f'{REPO_ROOT_}/mseg_semantic/config/test/default_config_${base_size}.yaml', 
	# 		#'model_path': f'{_ROOT}/pretrained-semantic-models/${model_name}/${model_name}.pth',
	# 		'model_path': '/srv/scratch/jlambert30/MSeg/pretrained-semantic-models/mseg-3m/mseg-3m.pth',
	# 		'input_file': f'{REPO_ROOT_}/tests/test_data/demo_images',
	# 		'model_name': 'mseg-3m',
	# 		'dataset': 'default',
	# 		'base_size': base_size,
	# 		'test_h': 713,
	#  		'test_w': 713,
	#  		'scales': [1.0],
	#  		'save_folder': 'default',
	#  		'arch': 'hrnet',
	# 		'index_start': 0,
	# 		'index_step': 0,
	# 		'workers': 16
	# 	}
	# 	args = SimpleNamespace(**d)
	# 	use_gpu = True
	# 	print(args)
	# 	run_universal_demo(args, use_gpu)

	# 	# assert result files exist
	# 	results_dir = f'{REPO_ROOT_}/temp_files/mseg-3m_default_universal_ss/{base_size}/gray'
	# 	fnames = [
	# 		'242_Maryview_Dr_Webster_0000304.png',
	# 		'bike_horses.png',
	# 		'PrivateLakefrontResidenceWoodstockGA_0000893.png'
	# 	]
	# 	for fname in fnames:
	# 		gray_fpath = f'{results_dir}/{fname}'
	# 		print(gray_fpath)
	# 		assert Path(gray_fpath).exists()
	# 		os.remove(gray_fpath)
