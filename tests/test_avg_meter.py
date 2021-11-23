#!/usr/bin/python3

import time

import numpy as np

from mseg_semantic.utils.avg_meter import exclusion


def test_exclusion() -> None:
    iou_class = np.array([0.3, 0.8, 0.9, 0.1])
    excluded_ids = [1, 3]
    relevant_ious = exclusion(iou_class, excluded_ids)
    gt_relevant_ious = np.array([0.3, 0.9])
    assert np.allclose(gt_relevant_ious, relevant_ious)


def exclusion_slow(array: np.ndarray, excluded_ids: np.ndarray) -> np.ndarray:
    """
    id is excluded id
    take in array of iou/acc., return non-excluded values
    """
    total = len(array)

    count = []
    for i in range(total):
        if i not in excluded_ids:
            count += [i]

    return array[count]


def test_exclusion_generic() -> None:
    """Compare w/ speed of set approach"""

    for _ in range(100):
        iou_class = np.random.rand(194)
        excluded_ids = np.random.randint(low=0, high=194, size=70)
        # start = time.time()
        gt_relevant_ious = exclusion_slow(iou_class, excluded_ids)
        relevant_ious = exclusion(iou_class, excluded_ids)

        assert np.allclose(gt_relevant_ious, relevant_ious)
        # end = time.time()
        # duration = end - start
        # print(duration)


if __name__ == "__main__":
    test_exclusion()
    test_exclusion_generic()
