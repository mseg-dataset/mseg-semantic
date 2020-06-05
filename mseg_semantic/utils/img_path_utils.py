#!/usr/bin/python3

import glob
from pathlib import Path
import pdb

from mseg.utils.txt_utils import (
	get_last_n_path_elements_as_str,
	write_txt_lines
)

from mseg.utils.dir_utils import check_mkdir


def dump_relpath_txt(jpg_dir: str, txt_output_dir: str) -> str:
	"""
	Dump relative paths.

		Args:
		-	jpg_dir:
		-	txt_output_dir:

		Returns:
		-	txt_save_fpath:
	"""
	fpaths = []
	dirname = Path(jpg_dir).stem
	for suffix in ['jpg','JPG','jpeg','JPEG','png','PNG']:
		suffix_fpaths = glob.glob(f'{jpg_dir}/*.{suffix}')
		fpaths.extend(suffix_fpaths)
	
	txt_lines = [get_last_n_path_elements_as_str(fpath, n=1) for fpath in fpaths]
	txt_lines.sort()
	check_mkdir(txt_output_dir)
	txt_save_fpath = f'{txt_output_dir}/{dirname}_relative_paths.txt'
	write_txt_lines(txt_save_fpath, txt_lines)
	return txt_save_fpath

