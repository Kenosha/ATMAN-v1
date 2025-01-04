# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 12:20:06 2023

@author: gianl
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.nn as nn

from ATMANpy.funcs.util_funcs import split_into_contiguous
from ATMANpy.types.Frames import BGRFrame

from itertools import product

def tensor_to_frame(tensor, zoom = 1):
    tensArr = tensor.numpy()
    if not tensArr.dtype == 'uint8':
        tensArr = (tensArr * 255).astype('uint8')
    tensArr = np.moveaxis(tensArr, 0, -1) ### right now, shape is C, H, W --> we want H, W, C
    # tensArr = np.flip(tensArr, -1) ### additionally, flip RGB to BGR
    tensFrm = BGRFrame(tensArr).zoom(zoom)
    return tensFrm