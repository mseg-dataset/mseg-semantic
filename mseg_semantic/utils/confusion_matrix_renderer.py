#!/usr/bin/python3


import numpy as np
import matplotlib.pyplot as plt
from mseg_semantic.utils.confusion_matrix_utils import plot_confusion_matrix


class ConfusionMatrixRenderer():
    def __init__(self, save_folder, class_names, dataset_name):
        """
        """
        self.save_folder = save_folder
        self.class_names = np.array(class_names)
        self.dataset_name = dataset_name
        self.y_pred = np.zeros((0,1), dtype=np.int64)
        self.y_true = np.zeros((0,1), dtype=np.int64)

    def update(self, pred, target):
        """ 
            Args:
            -   pred
            -   target

            Returns:
            -   None
        """
        self.y_pred = np.vstack( [self.y_pred, pred.reshape(-1,1)] )
        self.y_true = np.vstack( [self.y_true, target.reshape(-1,1)] )

    def render(self):
        """
            Args:
            -   

            Returns:
            -   None
        """
        self.y_true, self.y_pred = remove_ignored_pixels(self.y_true, self.y_pred)
        # Only unlabeled pixels were found (test split), or zero images were processed
        if self.y_true.size == 0:
            return

        title_str = f'{self.dataset_name}_confusion_matrix_unnormalized'
        _ = plot_confusion_matrix(self.y_true, self.y_pred, self.class_names, normalize=True, title=title_str)
        figure_save_fpath = f'{self.save_folder}/{title_str}.png'
        plt.savefig(figure_save_fpath, dpi=400)


def remove_ignored_pixels(y_true, y_pred, ignore_index=255):
    """ 
        Args:

        Returns:
        -   y_true
        -   y_pred
    """
    valid_idx = y_true != ignore_index
    y_pred = y_pred[valid_idx]
    y_true = y_true[valid_idx]
    assert y_true.shape == y_pred.shape, 'Target vector and predicted label vector are not aligned.'
    return y_true, y_pred

