# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:30:35 2022

@author: gianl
"""

import os
import cv2 as cv
import numpy as np
from dataclasses import dataclass

from ATMANpy.type_funcs import typecheck, arraycheck
from ATMANpy.cv_funcs import show, get_frame    
from ATMANpy.Points import Point
from ATMANpy.Lines import Line, Lines, Region

from multipledispatch import dispatch


@dataclass(frozen = True)
class Contour:
    points: np.ndarray
    
    def moments(self):
        return cv.moments(self)
    
    def area(self):
        return cv.contourArea(self)
    
    def perimeter(self, closedBool = True):
        return cv.arcLength(self, closedBool)
    










