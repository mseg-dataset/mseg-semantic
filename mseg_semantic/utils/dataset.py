#!/usr/bin/python3

import os
import os.path
from typing import Callable, List, Optional, Tuple

import cv2
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset

"""
Could duplicate samples here to reduce overhead between epochs.
Modified from https://github.com/hszhao/semseg/blob/master/util/dataset.py
"""

IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm"]


def is_image_file(filename: str) -> bool:
    """Check if file represents an image, by comparing against several known image file extensions."""
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split: str, data_root: str, data_list_fpath: str) -> List[Tuple[str, str]]:
    """Create list of (image file path, label file path) pairs.

    Args:
        split: string representing split of data set to use, must be either 'train','val','test'
        data_root: path to where data lives, and where relative image paths are relative to
        data_list_fpath: path to .txt file with relative image paths

    Returns:
        image_label_list: list of 2-tuples, each 2-tuple is comprised of an absolute image path
            and an absolute label path
    """
    assert split in ["train", "val", "test"]
    if not os.path.isfile(data_list_fpath):
        raise (RuntimeError("Image list file do not exist: " + data_list_fpath + "\n"))
    image_label_list = []
    list_read = open(data_list_fpath).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))

    for line in list_read:
        line = line.strip()
        line_split = line.split(" ")
        if split == "test":
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = image_name  # just set place holder for label_name, not for use
        else:
            if len(line_split) != 2:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = os.path.join(data_root, line_split[1])
        """
        following check costs some time
        if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
            item = (image_name, label_name)
            image_label_list.append(item)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
        """
        item = (image_name, label_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))

    return image_label_list


class SemData(Dataset):
    def __init__(
        self, split: str = "train", data_root: str = None, data_list: str = None, transform: Optional[Callable] = None
    ) -> None:
        """Dataloader class for semantic segmentation datasets.

        Args:
            split: string representing split of data set to use, must be either 'train','val','test'
            data_root: path to where data lives, and where relative image paths are relative to
            data_list: path to .txt file containing relative image paths
            transform: Pytorch torchvision transform
        """
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ """
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)

        label = imageio.imread(label_path)  # # GRAY 1 channel ndarray with shape H * W
        label = label.astype(np.int64)

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            if self.split != "test":
                image, label = self.transform(image, label)
            else:
                # use dummy label in transform, since label unknown for test
                image, label = self.transform(image, image[:, :, 0])

        return image, label
