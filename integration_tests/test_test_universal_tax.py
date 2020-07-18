#!/usr/bin/python3

from pathlib import Path
from types import SimpleNamespace

from mseg_semantic.scripts.collect_results import parse_result_file
from mseg_semantic.tool.test_universal_tax import evaluate_universal_tax_model

REPO_ROOT_ = Path(__file__).resolve().parent.parent

def test_evaluate_universal_tax_model():
	"""
	Ensure testing script works correctly.

	base_sizes=(
	        #360
	        720
	        #1080
	
	python -u mseg_semantic/tool/test_universal_tax.py --config=${config_fpath}
		dataset ${dataset_name} model_path ${model_fpath} model_name ${model_name}
	"""
	base_size = 360
	# Args that would be provided in command line and in config file
	d = {
		'dataset': 'camvid-11'
		'config': f'{REPO_ROOT_}/mseg_semantic/config/test/default_config_${base_size}.yaml', 
		#'model_path': f'{_ROOT}/pretrained-semantic-models/${model_name}/${model_name}.pth',
		'model_path': '/srv/scratch/jlambert30/MSeg/mseg-semantic/integration_test_data/mseg-3m.pth',
		'model_name': 'mseg-3m',
		
		'input_file': 'default'
		'base_size': base_size,
		'test_h': 713,
 		'test_w': 713,
 		'scales': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
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
	print(args)
	evaluate_universal_tax_model(args, use_gpu)

	# assert a file exists
	print('Completed')

	result_file_path = '/srv/scratch/jlambert30/MSeg/mseg-semantic/integration_test_data/'
	result_file_path += 'camvid-11-1m/camvid-11/360/ss/result.txt'
	mIoU = parse_result_file(result_file_path)
	print(f"mIoU: {mIoU}")
	assert mIoU = 83.3


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







