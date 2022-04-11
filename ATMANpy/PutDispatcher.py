# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 15:11:50 2022

@author: gianl
"""

import cv2 as cv
from ATMANpy.Points import Point
from ATMANpy.Lines import Line, Lines, Region
from ATMANpy.Contours import Contour

def put(someFrm, someObj, 
        color = None, radius = None, thickness = None, where = None, size = None):
    
    InstClass = someFrm.__class__
    self_copy = someFrm.array.copy() # to preverve immutability of frame
    
    if isinstance(someObj, Point):
        # conf_point = point.conform_to(self)
        return InstClass(cv.circle(self_copy, someObj.round_coords(), radius, color, thickness))
    
    if isinstance(someObj, Line):
        # conf_point = point.conform_to(self)
        c1, c2 = someObj.leftPt.round_coords(), someObj.rightPt.round_coords()
        return InstClass(cv.line(self_copy, c1, c2, color, thickness))
    
    if isinstance(someObj, Lines):
        for line in someObj:
            c1, c2 = line.leftPt.round_coords(), line.rightPt.round_coords()
            self_copy = cv.line(self_copy, c1, c2, color, thickness)
        return InstClass(self_copy)
        
    if isinstance(someObj, Region):
        return put(someFrm, Lines(someObj.lineList), 
                   color = color, radius = radius, thickness = thickness, where = where, size = size)
    
    if isinstance(someObj, Contour):
        return InstClass(cv.drawContours(self_copy, [someObj.points], -1, color, thickness = thickness))
    
    if isinstance(someObj, str):
        font = cv.FONT_HERSHEY_SIMPLEX
        return InstClass(cv.putText(self_copy, someObj, where, font, size, color, thickness))
    
    if isinstance(someObj, list):
        if len(someObj) == 0:
            return someFrm
        else:
            p, ps = someObj[0], someObj[1:]
            return put(put(someFrm, p, color = color, radius = radius, thickness = thickness, where = where, size = size), 
                       ps, color = color, radius = radius, thickness = thickness, where = where, size = size)
    ####################
