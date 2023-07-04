# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 15:11:50 2022

@author: gianl
"""

import cv2 as cv
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import Line, Lines, Region
from ATMANpy.types.Contours import Contour, Contours

def put(someFrm, someObj, 
        color = None, radius = None, thickness = None, where = None, size = None,
        annotation = None, verbose = False):
    
    offset_margin = 20
    
    if verbose:
        print('Calling put dispatcher with keywords')
        print(f'    color = {color}, radius = {radius}, thickness = {thickness}, where = {where}, size = {size}')
        
    InstClass = someFrm.__class__
    self_copy = someFrm.array.copy() # to preverve immutability of frame
    
    if someObj is None:
        return someFrm
    
    if isinstance(someObj, Point2D):
        if verbose: print('Point2D!')
        myFrm = InstClass(cv.circle(self_copy, someObj.round_coords(), radius, color, thickness))
        if annotation is not None:
            myFrm = myFrm.put(str(annotation), where = (someObj - offset_margin).coords)
        return myFrm
    
    if isinstance(someObj, Line):
        if verbose: print('Line!')
        c1, c2 = someObj.leftPt.round_coords(), someObj.rightPt.round_coords()
        myFrm = InstClass(cv.line(self_copy, c1, c2, color, thickness))
        if annotation is not None:
            myFrm = myFrm.put(str(annotation), where = (someObj.leftPt - offset_margin).coords)
        return myFrm
    
    if someObj.__class__.__name__ in ['Lines', 'CourtLines', 'NetLines']: ### when the putting does not work, it's likely due to autoreload, which makes the above evaluate to false....
        if verbose: print('Lines!')
        for line in someObj:
            c1, c2 = line.leftPt.round_coords(), line.rightPt.round_coords()
            self_copy = cv.line(self_copy, c1, c2, color, thickness)
        return InstClass(self_copy)
        
    if isinstance(someObj, Region):
        if verbose: print('Region!')
        return put(someFrm, Lines(someObj.lineList), 
                   color = color, radius = radius, thickness = thickness, where = where, size = size)
    
    if isinstance(someObj, Contour):
        if verbose: print('Contour!')
        myFrm = InstClass(cv.drawContours(self_copy, [someObj.points], -1, color, thickness = thickness))
        if annotation is not None:
            try:
                cntPnt = someObj.center()
            except ZeroDivisionError:
                pass
            else:
                myFrm = myFrm.put(str(annotation), where = (cntPnt - offset_margin).coords)
        return myFrm
          
    if isinstance(someObj, Contours):
        if verbose: print('Contours!')
        myFrm = InstClass(cv.drawContours(self_copy, [cnt.points for cnt in someObj], -1, color, thickness = thickness))
        if annotation is not None:
            for cnt in someObj:
                try:
                    cntPnt = cnt.center()
                except ZeroDivisionError:
                    pass
                else:
                    myFrm = myFrm.put(str(annotation), where = (cntPnt - offset_margin).coords)
        return myFrm
      
    if isinstance(someObj, str):
        if verbose: print('String!')
        font = cv.FONT_HERSHEY_SIMPLEX
        return InstClass(cv.putText(self_copy, someObj, where, font, size, color, thickness))
    
    if isinstance(someObj, list):
        if verbose: print('List!')
        if len(someObj) == 0:
            return someFrm
        else:
            p, ps = someObj[0], someObj[1:]
            currFrm = put(someFrm, p, color = color, radius = radius, thickness = thickness, where = where, size = size,
                          annotation = annotation)
            new_annotation = annotation + 1 if annotation is not None else None
            return put(currFrm, ps, color = color, radius = radius, thickness = thickness, 
                       where = where, size = size, annotation = new_annotation)
            
    if verbose: print('Not Any Type!')
    return someFrm ### if any other entry type --> we can't display it, so we just return original frame, instead of None
    ####################



