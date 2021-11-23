#!/usr/bin/python3

from pathlib import Path

import mseg_semantic.utils.img_path_utils as img_path_utils

_ROOT = Path(__file__).resolve().parent


def test_dump_relpath_txt() -> None:
    """ """
    jpg_dir = f"{_ROOT}/test_data/test_imgs_relpaths"
    txt_output_dir = f"{_ROOT}/test_data/temp_files"
    txt_save_fpath = img_path_utils.dump_relpath_txt(jpg_dir, txt_output_dir)
    lines = open(txt_save_fpath).readlines()
    lines = [line.strip() for line in lines]

    gt_lines = ["0016E5_08159.png", "ADE_train_00000001.jpg"]
    assert gt_lines == lines


def test_get_unique_stem_from_last_k_strs_k1() -> None:
    """ """
    fpath = "ADE20K_2016_07_26/images/training/a/aiport_terminal/ADE_train_00000001_seg.png"
    k = 1
    new_fname = img_path_utils.get_unique_stem_from_last_k_strs(fpath, k=1)
    assert new_fname == "_ADE_train_00000001_seg"


def test_get_unique_stem_from_last_k_strs_k2() -> None:
    """ """
    fpath = "ADE20K_2016_07_26/images/training/a/aiport_terminal/ADE_train_00000001_seg.png"
    k = 1
    new_fname = img_path_utils.get_unique_stem_from_last_k_strs(fpath, k=2)
    assert new_fname == "aiport_terminal_ADE_train_00000001_seg"


def test_get_unique_stem_from_last_k_strs_k3() -> None:
    """ """
    fpath = "ADE20K_2016_07_26/images/training/a/aiport_terminal/ADE_train_00000001_seg.png"
    k = 1
    new_fname = img_path_utils.get_unique_stem_from_last_k_strs(fpath, k=3)
    assert new_fname == "a_aiport_terminal_ADE_train_00000001_seg"


def test_get_unique_stem_from_last_k_strs_k4() -> None:
    """ """
    fpath = "ADE20K_2016_07_26/images/training/a/aiport_terminal/ADE_train_00000001_seg.png"
    k = 1
    new_fname =img_path_utils. get_unique_stem_from_last_k_strs(fpath, k=4)
    assert new_fname == "training_a_aiport_terminal_ADE_train_00000001_seg"


def test_get_unique_stem_from_last_k_strs_k5() -> None:
    """ """
    fpath = "ADE20K_2016_07_26/images/training/a/aiport_terminal/ADE_train_00000001_seg.png"
    k = 1
    new_fname = img_path_utils.get_unique_stem_from_last_k_strs(fpath, k=5)
    assert new_fname == "images_training_a_aiport_terminal_ADE_train_00000001_seg"


if __name__ == "__main__":
    """ """
    test_dump_relpath_txt()
    test_get_unique_stem_from_last_k_strs_k1()
    test_get_unique_stem_from_last_k_strs_k2()
    test_get_unique_stem_from_last_k_strs_k3()
    test_get_unique_stem_from_last_k_strs_k4()
    test_get_unique_stem_from_last_k_strs_k5()
