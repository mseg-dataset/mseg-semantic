from pathlib import Path

from mseg_semantic.utils.dataset import SemData, make_dataset

TEST_DATA_ROOT = Path(__file__).resolve().parent / "test_data"


def test_make_dataset() -> None:
    """Ensure make_dataset() returns the proper outputs"""
    split = "train"
    data_root = "/home/dummy_data_root"
    data_list_fpath = str(TEST_DATA_ROOT / "dummy_camvid_train.txt")
    image_label_list = make_dataset(split, data_root, data_list_fpath)

    expected_image_label_list = [
        (f"{data_root}/701_StillsRaw_full/0001TP_006690.png", f"{data_root}/semseg11/0001TP_006690_L.png"),
        (f"{data_root}/701_StillsRaw_full/0001TP_006720.png", f"{data_root}/semseg11/0001TP_006720_L.png"),
        (f"{data_root}/701_StillsRaw_full/0001TP_006750.png", f"{data_root}/semseg11/0001TP_006750_L.png"),
    ]
    assert image_label_list == expected_image_label_list
