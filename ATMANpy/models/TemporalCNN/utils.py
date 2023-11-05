# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 13:49:37 2023

@author: gianl
"""

import numpy as np
import matplotlib.pyplot as plt

from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Streams import DecapStream

from itertools import product

def clip_to_stream(clip, zoom = 1):
    clipArr = clip.numpy()
    if not clipArr.dtype == 'uint8':
        clipArr = (clipArr * 255).astype('uint8')
    clipArr = np.moveaxis(clipArr, 0, -1) ### right now, shape is C, T, H, W --> we want T, H, W, C
    clipArr = np.flip(clipArr, -1) ### additionally, flip RGB to BGR
    clipFrms = BGRFrames(clipArr).zoom(zoom)
    clipStm = DecapStream(clipFrms, start_frame = 0, stop_frame = len(clipArr))
    return clipStm

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure
