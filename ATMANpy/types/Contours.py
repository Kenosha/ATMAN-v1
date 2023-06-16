# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:30:35 2022

@author: gianl
"""

import os
import cv2 as cv
import numpy as np
from dataclasses import dataclass

from ATMANpy.funcs.type_funcs import typecheck, arraycheck
from ATMANpy.funcs.cv_funcs import show, get_frame    
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import Line, Lines, Region

from multipledispatch import dispatch


@dataclass(frozen = True)
class Contour:
    points: np.ndarray
    
    def moments(self):
        return cv.moments(self.points)
    
    def area(self):
        return cv.contourArea(self.points)
    
    def perimeter(self, closedBool = True):
        return cv.arcLength(self, closedBool)
    
    def center(self):
        M = self.moments()
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return Point2D((cX, cY))

    def rescale(self, scale_factor):
        return Contour((self.points * scale_factor).astype(int))
    
    def enclosing_region(self, margin = 0):
        x, y, w, h = cv.boundingRect(self.points)
        p1, p2 = Point2D((x - margin, y - margin)), Point2D((x + w + margin, y + h + margin))
        return Region.from_points(p1, p2)
        
    def fit_ellipse(self):
        ## returns: centerCoords, majminAxis, rotAngle 
        ## -> parameters of the rotated rectangle in which the ellipse is inscribed
        # centerCoords: center or rectangle and ellipse
        # majminAxis: major and minor axis of the rectangle, i.e: width and heigth!
        # rotAngle: theta, the rotation angle of the rectangle.
        return cv.fitEllipse(self.points)

    def compress(self, target_number_of_points):
                
        current_number_of_points = self.points.shape[0]
        
        if current_number_of_points <= target_number_of_points:
            return self
        else:
            sampleIndices = np.linspace(0, current_number_of_points, num = target_number_of_points, 
                                        dtype = int, endpoint = False)
            samplePoints = self.points[sampleIndices]
            
            return Contour(samplePoints)




