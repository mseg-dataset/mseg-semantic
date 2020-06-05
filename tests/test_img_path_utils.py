#!/usr/bin/python3

from pathlib import Path
from mseg_semantic.utils.img_path_utils import dump_relpath_txt

_ROOT = Path(__file__).resolve().parent

def test_dump_relpath_txt():
	""" """
	jpg_dir = f'{_ROOT}/test_data/test_imgs_relpaths'
	txt_output_dir = f'{_ROOT}/test_data/temp_files'
	txt_save_fpath = dump_relpath_txt(jpg_dir, txt_output_dir)
	lines = open(txt_save_fpath).readlines()
	lines = [line.strip() for line in lines]
	
	gt_lines = [
		'0016E5_08159.png',
		'ADE_train_00000001.jpg'
	]
	assert gt_lines == lines