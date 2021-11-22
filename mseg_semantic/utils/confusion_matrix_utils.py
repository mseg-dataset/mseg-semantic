#!/usr/bin/python3

"""Utilities for creating a confusion matrix from predictions and GT labels."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import sklearn.utils.multiclass


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
    normalize: bool = False,
    title: Optional[str] = None,
    cmap=plt.cm.Blues,
) -> np.ndarray:
    """
    Ref:
        https://scikit-learn.org/stable/auto_examples/model_selection/
        plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Args:
        y_true: (N,) array representing ground truth labels
        y_pred: (N,) array representing predicted labels
        classes: (K,) numpy array of strings, representing ordered class names
        normalize: whether to normalize confusion matrix entries
        title: string representing figure title
        cmap: Matplotlib colormap

    Returns:
        cm: Array representing 2d confusion matrix. Rows are GT, and cols are preds.
            In other words, i-th row and j-th column entry indicates the number of
            samples with true label being i-th class and predicted label being j-th class.
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[sklearn.utils.multiclass.unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black"
            )
    fig.tight_layout()
    return cm
