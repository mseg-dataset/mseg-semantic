#!/usr/bin/python3

import numpy as np

import mseg_semantic.utils.confusion_matrix_utils as confusion_matrix_utils


def test_plot_confusion_matrix() -> None:
    """Ensure that confusion matrix is generated properly."""
    y_pred = np.array([0, 1, 2, 0]).astype(np.uint32)
    y_true = np.array([0, 1, 1, 1]).astype(np.uint32)
    confusion_mat = confusion_matrix_utils.plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        classes=np.array(["car", "pedestrian", "road"]),  # dummy labels for 3 dummy classes
        normalize=False,
    )
    # fmt: off
    expected_confusion_mat = np.array(
        [
            [1, 0, 0],
            [1, 1, 1],
            [0, 0, 0]
        ]
    )
    # fmt: on
    assert np.allclose(confusion_mat, expected_confusion_mat)
