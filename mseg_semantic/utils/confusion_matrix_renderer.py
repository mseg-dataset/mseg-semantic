#!/usr/bin/python3

from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import mseg_semantic.utils.confusion_matrix_utils as confusion_matrix_utils


class ConfusionMatrixRenderer:
    def __init__(self, save_folder: str, class_names: List[str], dataset_name: str) -> None:
        """ """
        self.save_folder = save_folder
        self.class_names = np.array(class_names)
        self.dataset_name = dataset_name
        self.y_pred = np.zeros((0, 1), dtype=np.int64)
        self.y_true = np.zeros((0, 1), dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        """
        Args:
            pred
            target
        """
        self.y_pred = np.vstack([self.y_pred, pred.reshape(-1, 1)])
        self.y_true = np.vstack([self.y_true, target.reshape(-1, 1)])

    def render(self) -> None:
        """ """
        self.y_true, self.y_pred = remove_ignored_pixels(self.y_true, self.y_pred)
        # Only unlabeled pixels were found (test split), or zero images were processed
        if self.y_true.size == 0:
            return

        title_str = f"{self.dataset_name}_confusion_matrix_unnormalized"
        _ = confusion_matrix_utils.plot_confusion_matrix(
            self.y_true, self.y_pred, self.class_names, normalize=True, title=title_str
        )
        figure_save_fpath = f"{self.save_folder}/{title_str}.png"
        plt.savefig(figure_save_fpath, dpi=400)


def remove_ignored_pixels(
    y_true: np.ndarray, y_pred: np.ndarray, ignore_index: int = 255
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        y_true: ground truth labels.
        y_pred: predictions.
        ignore_index

    Returns:
        y_true: after ignored classes are set to the ignore_index.
        y_pred: after ignored classes are set to the ignore_index.
    """
    valid_idx = y_true != ignore_index
    y_pred = y_pred[valid_idx]
    y_true = y_true[valid_idx]
    assert y_true.shape == y_pred.shape, "Target vector and predicted label vector are not aligned."
    return y_true, y_pred
