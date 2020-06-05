#!/usr/bin/python3

import cv2
import imageio
import numpy as np
import os
import os.path
import pdb
from torch.utils.data import Dataset
from typing import List, Tuple

"""
Could duplicate samples here to reduce overhead between epochs.
"""

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(
    split: str='train',
    data_root: str=None,
    data_list=None
    ) -> List[Tuple[str,str]]:
    """
        Args:
        -   split: string representing split of data set to use, must be either 'train','val','test'
        -   data_root: path to where data lives, and where relative image paths are relative to
        -   data_list: path to .txt file with relative image paths
        
        Returns:
        -   image_label_list: list of 2-tuples, each 2-tuple is comprised of a relative image path
                and a relative label path
    """
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))

    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if split == 'test':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = image_name  # just set place holder for label_name, not for use
        else:
            if len(line_split) != 2:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = os.path.join(data_root, line_split[1])
        '''
        following check costs some time
        if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
            item = (image_name, label_name)
            image_label_list.append(item)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
        '''
        item = (image_name, label_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))

    return image_label_list


class SemData(Dataset):
    def __init__(self, split: str='train', data_root: str=None, data_list: str=None, transform=None):
        """
            Args:
            -   split: string representing split of data set to use, must be either 'train','val','test'
            -   data_root: path to where data lives, and where relative image paths are relative to
            -   data_list: path to .txt file with relative image paths
            -   transform: Pytorch torchvision transform

            Returns:
            -   None
        """
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """ """
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)

        label = imageio.imread(label_path) # # GRAY 1 channel ndarray with shape H * W
        label = label.astype(np.int64)


        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            if self.split != 'test':
                image, label = self.transform(image, label)
            else:
                # use dummy label in transform, since label unknown for test
                image, label = self.transform(image, image[:, :, 0])

        return image, label

