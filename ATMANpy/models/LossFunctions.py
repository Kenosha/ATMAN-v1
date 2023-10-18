# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 19:05:31 2023

@author: gianl
"""

import numpy as np

def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / len(y_true)
