#!/usr/bin/python3

from pathlib import Path
from types import SimpleNamespace

from mseg_semantic.tool.universal_demo import run_universal_demo

ROOT_ = Path(__file__).resolve().parent


def test_run_universal_demo():
	"""
	Ensure demo script works correctly
	base_sizes=(
	        #360
	        720
	        #1080
	
	python -u mseg_semantic/tool/test_universal_tax.py --config=${config_fpath}
		dataset ${dataset_name} model_path ${model_fpath} model_name ${model_name}
	"""
	base_size = 360
	d = {
		'config': f'{ROOT_}/mseg_semantic/config/test/default_config_${base_size}.yaml', 
		#'model_path': f'{_ROOT}/pretrained-semantic-models/${model_name}/${model_name}.pth',
		'model_path': '/srv/scratch/jlambert30/MSeg/pretrained-semantic-models/mseg-3m/mseg-3m.pth',
		'input_file': 'test_data/temp_files/demo_images',
		'model_name': 'mseg-3m',
		'dataset': 'default',
		'base_size': base_size
	}
	args = SimpleNamespace(**d)
	use_gpu = True
	run_universal_demo(args, use_gpu)

	# assert a file exists




