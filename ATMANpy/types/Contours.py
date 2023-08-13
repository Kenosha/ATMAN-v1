# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:30:35 2022

@author: gianl
"""

import os
import cv2 as cv
import numpy as np
from dataclasses import dataclass
from typing import List

import ATMANpy.auxil.constants as const
from ATMANpy.funcs.type_funcs import typecheck, arraycheck
from ATMANpy.funcs.cv_funcs import show, get_frame    
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import Line, Lines, Region, Regions

from multipledispatch import dispatch


@dataclass(frozen = True)
class Contour:
    points: np.ndarray
    
    @staticmethod
    def from_series(mySer):
        singlefeatDict = mySer.to_dict()
                      
        x_vals = [v for k, v in singlefeatDict.items() if k.startswith('x') if not v == const.MISSING_SENTINEL_VALUE]
        y_vals = [v for k, v in singlefeatDict.items() if k.startswith('y') if not v == const.MISSING_SENTINEL_VALUE]
        
        assert len(x_vals) == len(y_vals)
        num_points = len(x_vals)
        
        myPoints = np.dstack([x_vals, y_vals]).reshape((num_points, 1, 2)) ### bring them into shape
        return Contour(myPoints)
        
    def __eq__(self, othCnt):
        return np.array(self.points == othCnt.points).all()
        
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

    def shift(self, x_offset = 0, y_offset = 0):
        return Contour((self.points + [x_offset, y_offset]))

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

    def contains(self, somePt):
        return cv.pointPolygonTest(self.points, somePt.coords, False) == 1

    def compress(self, target_number_of_points):
                
        current_number_of_points = self.points.shape[0]
        
        if current_number_of_points <= target_number_of_points:
            return self
        else:
            sampleIndices = np.linspace(0, current_number_of_points, num = target_number_of_points, 
                                        dtype = int, endpoint = False)
            samplePoints = self.points[sampleIndices]
            
            return Contour(samplePoints)


@dataclass(frozen = True)
class Contours:
    contourList: List[Contour]
    
    @staticmethod
    def from_frame(myFrm):
        outList = []
        for idx, rowSer in myFrm.iterrows():
            outList.append(Contour.from_series(rowSer))
        return Contours(outList)
    
    def __repr__(self):
        return f'Contours(length = {self.length()})'
    
    def __iter__(self):
        return self.contourList.__iter__()
        
    def __getitem__(self, index):
        return self.contourList[index]
    
    def drop(self, index):
        return Contours([cnt for idx, cnt in enumerate(self) if idx != index])
    
    def length(self):
        return len(self.contourList)
    
    def is_empty(self):
        return self.length() == 0
    
    def moments(self):
        return [cnt.moments() for cnt in self]
    
    def area(self):
        return [cnt.area() for cnt in self]
    
    def perimeter(self, closedBool = True):
        return [cnt.perimeter() for cnt in self]
    
    def center(self):
        return [cnt.center() for cnt in self] ### this should be a list of Point2D --> Points2D class!!!

    def rescale(self, scale_factor):
        return Contours([cnt.rescale(scale_factor) for cnt in self])
    
    def enclosing_regions(self, margin = 0):
        return Regions([cnt.enclosing_region() for cnt in self])
        
    def smallest(self):
        return self[np.argmin(self.area())]
        
    def largest(self):
        return self[np.argmax(self.area())]
    
    def fit_ellipse(self):
        ## returns: centerCoords, majminAxis, rotAngle 
        ## -> parameters of the rotated rectangle in which the ellipse is inscribed
        # centerCoords: center or rectangle and ellipse
        # majminAxis: major and minor axis of the rectangle, i.e: width and heigth!
        # rotAngle: theta, the rotation angle of the rectangle.
        return [cnt.fit_ellipse() for cnt in self]

    def contains(self, somePt):
        return any([myCnt.contains(somePt) for myCnt in self])

    def compress(self, target_number_of_points):
        return Contours([cnt.compress(target_number_of_points) for cnt in self])


