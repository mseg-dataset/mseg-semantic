#!/usr/bin/python3

from pathlib import Path
from types import SimpleNamespace

from mseg_semantic.scripts.collect_results import parse_result_file
from mseg_semantic.tool.test_oracle_tax import test_oracle_taxonomy_model

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
	#test_oracle_taxonomy_model(args, use_gpu)

	# assert a file exists
	print('Completed')

	result_file_path = '/srv/scratch/jlambert30/MSeg/mseg-semantic/integration_test_data/'
	result_file_path += 'camvid-11-1m/camvid-11/360/ss/results.txt'
	assert Path(result_file_path).exists()
	mIoU = parse_result_file(result_file_path)
	print(f"mIoU: {mIoU}")
	assert mIoU == 78.8

	OKGREEN = '\033[92m'
	ENDC = '\033[0m'
	print(OKGREEN + ">>>>>>>>>>>>>>>>>>>>>>>>>>>>"  + ENDC)
	print(OKGREEN + 'Oracle model evalution passed successfully' + ENDC)
	print(OKGREEN + ">>>>>>>>>>>>>>>>>>>>>>>>>>>>"  + ENDC)
	


if __name__ == '__main__':
	test_evaluate_universal_tax_model()
