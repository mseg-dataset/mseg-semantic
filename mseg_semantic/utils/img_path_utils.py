#!/usr/bin/python3

import glob
from pathlib import Path

import mseg.utils.dir_utils as dir_utils
from mseg.utils.txt_utils import get_last_n_path_elements_as_str, write_txt_lines


def dump_relpath_txt(jpg_dir: str, txt_output_dir: str) -> str:
    """Dump relative file paths for all images found within a directory.

    Args:
        jpg_dir: image directory.
        txt_output_dir:

    Returns:
        txt_save_fpath: file path where .txt file will be saved.
    """
    fpaths = []
    dirname = Path(jpg_dir).stem
    for suffix in ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]:
        suffix_fpaths = glob.glob(f"{jpg_dir}/*.{suffix}")
        fpaths.extend(suffix_fpaths)

    txt_lines = [get_last_n_path_elements_as_str(fpath, n=1) for fpath in fpaths]
    txt_lines.sort()
    dir_utils.check_mkdir(txt_output_dir)
    txt_save_fpath = f"{txt_output_dir}/{dirname}_relative_paths.txt"
    write_txt_lines(txt_save_fpath, txt_lines)
    return txt_save_fpath


def get_unique_stem_from_last_k_strs(fpath: str, k: int = 4) -> str:
    """Generate a unique stem for a file from last k parts of its absolute file path.

    For datasets like ScanNet where image filename stem is not unique.
    Will not return the suffix, e.g.
    'aiport_terminal/ADE_train_00000001_seg.png'
    would be returned as 'aiport_terminal_ADE_train_00000001_seg'

    Args:
        fpath: absolute file path
        k: integer representing number of subdirs in filepath to use
            in new filename, starting at leaf in filesystem tree

    Returns:
        unique_stem: string
    """
    parts = Path(fpath).parts
    concat_kparent_dirs = "_".join(parts[-k:-1])
    unique_stem = concat_kparent_dirs + "_" + Path(fpath).stem
    return unique_stem
