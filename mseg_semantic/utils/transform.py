#!/usr/bin/python3

import collections
import math
import random
from typing import Tuple, Union

import cv2
import numpy as np
import numbers
import torch

from mseg.taxonomy.naive_taxonomy_converter import NaiveTaxonomyConverter
from mseg.taxonomy.taxonomy_converter import TaxonomyConverter

"""
Provides a set of Pytorch transforms that use OpenCV instead of PIL (Pytorch default)
for image manipulation.

Most transformations are applied identically to both the image and ground truth label map.

Modified from https://github.com/hszhao/semseg/blob/master/util/transform.py
"""


class Compose(object):
    # Composes segtransforms: segtransform.Compose([segtransform.RandScale([0.5, 2.0]), segtransform.ToTensor()])
    def __init__(self, segtransform):
        self.segtransform = segtransform

    def __call__(self, image, label):
        for t in self.segtransform:
            image, label = t(image, label)
        return image, label


class ToTensor(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __call__(self, image: np.ndarray, label: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(image, np.ndarray) or not isinstance(label, np.ndarray):
            raise (
                RuntimeError("segtransform.ToTensor() only handle np.ndarray" "[eg: data readed by cv2.imread()].\n")
            )
        if len(image.shape) > 3 or len(image.shape) < 2:
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n"))
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        if not len(label.shape) == 2:
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray labellabel with 2 dims.\n"))

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        label = torch.from_numpy(label)
        if not isinstance(label, torch.LongTensor):
            label = label.long()

        return image, label


class ToUniversalLabel(object):
    def __init__(self, dataset: str, use_naive_taxonomy: bool = False) -> None:
        self.dataset = dataset
        if use_naive_taxonomy:
            self.tax_converter = NaiveTaxonomyConverter()
        else:
            self.tax_converter = TaxonomyConverter()

    def __call__(self, image, label):
        return image, self.tax_converter.transform_label(label, self.dataset)


class Normalize(object):
    # Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.std is None:
            for t, m in zip(image, self.mean):
                t.sub_(m)
        else:
            for t, m, s in zip(image, self.mean, self.std):
                t.sub_(m).div_(s)
        return image, label


class Resize(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, size):
        assert isinstance(size, collections.Iterable) and len(size) == 2
        self.size = size

    def __call__(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        image = cv2.resize(image, self.size[::-1], interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, self.size[::-1], interpolation=cv2.INTER_NEAREST)
        return image, label


class ResizeShort(object):
    """Resize the input such that its shorter size meets the prescribed size.
    Note that 'size' is a float or int.
    """

    def __init__(self, size: Union[int, float]) -> None:
        self.size = size

    def __call__(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Resize image such that the shorter side equals predefined size."""
        old_image_shape, old_label_shape = image.shape, label.shape
        h, w = image.shape[0], image.shape[1]
        shorter_size = min(h, w)
        if shorter_size == h:
            new_h = self.size
            scale = new_h / float(h)
            new_w = int(w * scale)
        else:
            new_w = self.size
            scale = new_w / float(w)
            new_h = int(h * scale)

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        return image, label


class RandScale(object):
    # Randomly resize image & label with scale factor in [scale_min, scale_max]
    def __init__(self, scale, aspect_ratio=None):
        assert isinstance(scale, collections.Iterable) and len(scale) == 2
        if (
            isinstance(scale, collections.Iterable)
            and len(scale) == 2
            and isinstance(scale[0], numbers.Number)
            and isinstance(scale[1], numbers.Number)
            and 0 < scale[0] < scale[1]
        ):
            self.scale = scale
        else:
            raise (RuntimeError("segtransform.RandScale() scale param error.\n"))
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif (
            isinstance(aspect_ratio, collections.Iterable)
            and len(aspect_ratio) == 2
            and isinstance(aspect_ratio[0], numbers.Number)
            and isinstance(aspect_ratio[1], numbers.Number)
            and 0 < aspect_ratio[0] < aspect_ratio[1]
        ):
            self.aspect_ratio = aspect_ratio
        else:
            raise (RuntimeError("segtransform.RandScale() aspect_ratio param error.\n"))

    def __call__(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Randomly scale an RGB image and label map identically.

        Args:
            image: array of shape (H,W,C) representing RGB image
            label: array of shape (H,W) representing ground truth label map
        Returns:
            image: array of shape (H,W,C) representing *randomly scaled* RGB image
            label: array of shape (H,W) representing *randomly scaled* ground truth label map
        """
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale / temp_aspect_ratio
        image = cv2.resize(image, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)
        return image, label


class Crop(object):
    """Crops the given ndarray image (H*W*C or H*W).
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """

    def __init__(self, size, crop_type: str = "center", padding=None, ignore_label: int = 255) -> None:
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif (
            isinstance(size, collections.Iterable)
            and len(size) == 2
            and isinstance(size[0], int)
            and isinstance(size[1], int)
            and size[0] > 0
            and size[1] > 0
        ):
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if crop_type == "center" or crop_type == "rand":
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center\n"))
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))

    def __call__(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return a random crop or center crop of the specified size, from both an RGB image and label map"""
        h, w = label.shape
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError("segtransform.Crop() need padding while padding argument is None\n"))
            image = cv2.copyMakeBorder(
                image,
                pad_h_half,
                pad_h - pad_h_half,
                pad_w_half,
                pad_w - pad_w_half,
                cv2.BORDER_CONSTANT,
                value=self.padding,
            )
            label = cv2.copyMakeBorder(
                label,
                pad_h_half,
                pad_h - pad_h_half,
                pad_w_half,
                pad_w - pad_w_half,
                cv2.BORDER_CONSTANT,
                value=self.ignore_label,
            )
        h, w = label.shape
        if self.crop_type == "rand":
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)
        image = image[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w]
        label = label[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w]
        return image, label


class RandRotate(object):
    # Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]
    def __init__(
        self, rotate: Tuple[float, float], padding: Tuple[int, int, int], ignore_label: int = 255, p: float = 0.5
    ) -> None:
        assert isinstance(rotate, collections.Iterable) and len(rotate) == 2
        if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) and rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise (RuntimeError("segtransform.RandRotate() scale param error.\n"))
        assert padding is not None
        assert isinstance(padding, list) and len(padding) == 3
        if all(isinstance(i, numbers.Number) for i in padding):
            self.padding = padding
        else:
            raise (RuntimeError("padding in RandRotate() should be a number list\n"))
        assert isinstance(ignore_label, int)
        self.ignore_label = ignore_label
        self.p = p

    def __call__(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """With probability p, apply a random rotation to both an RGB image and label map.

        Args:
            image: array of shape (H,W,C) representing RGB image
            label: array of shape (H,W) representing ground truth label map
        Returns:
            image: array of shape (H,W,C) representing *randomly rotated* RGB image
            label: array of shape (H,W) representing *randomly rotated* ground truth label map
        """
        if random.random() < self.p:
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            h, w = label.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image = cv2.warpAffine(
                image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding
            )
            label = cv2.warpAffine(
                label,
                matrix,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=self.ignore_label,
            )
        return image, label


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """With probability p, horizontally flip both an RGB image and label map.

        Args:
            image: array of shape (H,W,C) representing RGB image
            label: array of shape (H,W) representing ground truth label map
        Returns:
            image: array of shape (H,W,C) representing *randomly hflipped* RGB image
            label: array of shape (H,W) representing *randomly hflipped* ground truth label map
        """
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        return image, label


class RandomVerticalFlip(object):
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """With probability p, vertically flip both an RGB image and label map.

        Args:
            image: array of shape (H,W,C) representing RGB image
            label: array of shape (H,W) representing ground truth label map
        Returns:
            image: array of shape (H,W,C) representing *randomly vflipped* RGB image
            label: array of shape (H,W) representing *randomly vflipped* ground truth label map
        """
        if random.random() < self.p:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)
        return image, label


class RandomGaussianBlur(object):
    def __init__(self, radius: float = 5) -> None:
        self.radius = radius

    def __call__(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """With probability p, blur an RGB image with a (radius x radius) Gaussian kernel.

        Args:
            image: array of shape (H,W,C) representing RGB image
            label: array of shape (H,W) representing ground truth label map
        Returns:
            image: array of shape (H,W,C) representing *randomly blurred* RGB image
            label: array of shape (H,W) representing the unmodified ground truth label map
        """
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        return image, label


class RGB2BGR(object):
    # Converts image from RGB order to BGR order, for model initialized from Caffe
    def __call__(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, label


class BGR2RGB(object):
    # Converts image from BGR order to RGB order, for model initialized from Pytorch
    def __call__(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, label
