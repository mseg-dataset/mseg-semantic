#!/usr/bin/python3

"""Unit test on image padding and rescaling for inference."""

import numpy as np

from mseg_semantic.tool.inference_task import resize_by_scaled_short_side, pad_to_crop_sz


def test_resize_by_scaled_short_side1() -> None:
    """
    Resize simple (4x6) RGB image to 8p, with 1.0 scale.
    """
    img = np.zeros((4, 6, 3), dtype=np.uint8)
    img[:, 3:] = 255
    base_size = 8
    scale = 1.0
    scaled_img = resize_by_scaled_short_side(img, base_size, scale)
    assert scaled_img.shape == (8, 12, 3)


def test_resize_by_scaled_short_side2() -> None:
    """
    Resize simple (4x6) RGB image to 8p, with 1.5 scale.
    """
    img = np.zeros((4, 6, 3), dtype=np.uint8)
    img[:, 3:] = 255
    base_size = 8
    scale = 1.5
    scaled_img = resize_by_scaled_short_side(img, base_size, scale)
    assert scaled_img.shape == (12, 18, 3)


def test_pad_to_crop_sz() -> None:
    """
    Want to feed image into network, but it is smaller than
    default crop size. Thus, we pad it at its edges to a mean value.
    Here the mean value is 50.
    """
    img = np.zeros((4, 6, 3), dtype=np.uint8)
    img[1:3, 2:4] = 255
    crop_h = 8
    crop_w = 8
    mean = [50, 100, 150]
    padded_img, pad_h_half, pad_w_half = pad_to_crop_sz(img, crop_h, crop_w, mean)
    assert padded_img.shape == (crop_h, crop_w, 3)
    # fmt: off
    gt_slice = np.array(
        [
            [50, 50, 50,  50,  50, 50, 50, 50],
            [50, 50, 50,  50,  50, 50, 50, 50],
            [50,  0,  0,   0,   0,  0,  0, 50],
            [50,  0,  0, 255, 255,  0,  0, 50],
            [50,  0,  0, 255, 255,  0,  0, 50],
            [50,  0,  0,   0,   0,  0,  0, 50],
            [50, 50, 50,  50,  50, 50, 50, 50],
            [50, 50, 50,  50,  50, 50, 50, 50],
        ],
        dtype=np.uint8,
    )
    # fmt: on
    assert np.allclose(padded_img[:, :, 0], gt_slice)
    assert pad_h_half == 2
    assert pad_w_half == 1
