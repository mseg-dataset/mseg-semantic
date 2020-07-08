#!/usr/bin/python3

from types import SimpleNamespace

from mseg_semantic.tool.test_universal_tax import evaluate_universal_tax_model

ROOT_ = Path(__file__).resolve().parent.parent

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
		'config': f'{ROOT_}/mseg_semantic/config/test/default_config_${base_size}_ss.yaml', 
		#'model_path': f'{_ROOT}/pretrained-semantic-models/${model_name}/${model_name}.pth',
		'model_path': '',
		'dataset': '',
		'model_name': 'mseg-3m'
	}
	args = SimpleNamespace(**d)
	use_gpu = True
	evaluate_universal_tax_model(args, use_gpu)

	# assert a file exists





