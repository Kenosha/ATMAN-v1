# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:30:35 2022

@author: gianl
"""

import cv2 as cv
import numpy as np
from dataclasses import dataclass

from type_funcs import typecheck, arraycheck, pointcheck
from cv_funcs import show, get_frame    

# from typing import Union

@dataclass
class Point:
    """
    A point: tuple of integers, signifying x and y coordinates.
    """   
    coords: tuple
    
    def __post_init__(self):
        typecheck(self)
        pointcheck(self)
        self.x, self.y = self.coords
        ### prohibit negative points.
        if not (self.x >= 0 and self.y >= 0):
            self = None ## or maybe make none point?
        
    # def conform_to(self, image: Union[BGRFrame, GreyFrame]): ### circular annotation not allowed...
    def conform_to(self, image):
        # force the point to be in bounds
        corr_x = min(max(0, self.x), image.cols)
        corr_y = min(max(0, self.y), image.rows)
        return Point((corr_x, corr_y))

    def up(self, points):
        return Point((self.x, self.y - points))

    def down(self, points):
        return Point((self.x, self.y + points))

    def left(self, points):
        return Point((self.x - points, self.y))

    def right(self, points):
        return Point((self.x + points, self.y))

    

@dataclass
class GreyFrame:
    array: np.ndarray
    name: str = 'frame'
    
    def __post_init__(self):
        typecheck(self)
        
        ### we can actually deal with a three dim array, by turning it back to Grey
        arraycheck(self, channel_number = 3, color_channel_length = None)
            
        ### convert a 1-dim array to a line, with 1 row.
        if len(self.array.shape) == 1:
            self.array = np.array([self.array])
        
        ### convert grayscale to a grey BGR
        if len(self.array.shape) == 3:
            self.array = cv.cvtColor(self.array,  cv.COLOR_BGR2GRAY)
        
        self.rows, self.cols = self.array.shape[:2]
        self.shape = self.array.shape
    
    ### CONSTRUCTORS ###
    @staticmethod
    def from_video(cap, frame_num, name = 'frame'):
        return GreyFrame(get_frame(cap, frame_num))
    ####################
    
    ### TRANSFORMS ###
    def to_BGR(self):
        self_copy = self.array.copy() # to preverve immutability of frame
        return BGRFrame(cv.cvtColor(self_copy, cv.COLOR_GRAY2BGR), name = self.name)
    ##################     
    
    ### METHODS ###
    def show(self):
        show(self.array, name = self.name)

    def overlay(self, point: Point, color = 255, radius = 0, thickness = 1):
        self_copy = self.array.copy() # to preverve immutability of frame
        conf_point = point.conform_to(self)
        return GreyFrame(cv.circle(self_copy, conf_point.coords, radius, color, thickness))
    
    def eval_at(self, point: Point):
        ### first, check whether point is in bounds
        # (we have to check since numpy will wrap around in )
        return self.array[point.y, point.x]
    ###############


@dataclass
class BGRFrame:
    array: np.ndarray
    name: str = 'frame'
    
    def __post_init__(self):
        typecheck(self)
        
        arraycheck(self, channel_number = 3, color_channel_length = 3)
            
        ### convert a 1-dim array to a line, with 1 row.
        if len(self.array.shape) == 1:
            self.array = np.array([self.array])
        
        ### convert grayscale to a grey BGR
        if len(self.array.shape) == 2:
            self.array = cv.cvtColor(self.array,  cv.COLOR_GRAY2BGR)
        
        self.rows, self.cols = self.array.shape[:2]
        self.shape = self.array.shape
       
    ### CONSTRUCTORS ###
    @staticmethod
    def from_video(cap, frame_num, name = 'frame'):
        return BGRFrame(get_frame(cap, frame_num))
    ####################
    
    ### TRANSFORMS ###
    def to_grey(self):
        self_copy = self.array.copy() # to preverve immutability of frame
        return GreyFrame(cv.cvtColor(self_copy, cv.COLOR_BGR2GRAY), name = self.name)
    ##################     
    
    ### METHODS ###
    def show(self):
        show(self.array, name = self.name)
    
    def overlay(self, point: Point, color = (0, 255, 0), radius = 0, thickness = 1):
        self_copy = self.array.copy() # to preverve immutability of frame
        conf_point = point.conform_to(self)
        return BGRFrame(cv.circle(self_copy, conf_point.coords, radius, color, thickness))
    
    def eval_at(self, point: Point):
        return self.array[point.y, point.x]
    ###############
    

if __name__ == '__main__':
    
    coords = (0, 0)
    a = np.zeros((500, 350), dtype = 'uint8')
    
    aFrm = BGRFrame(a)
    aPt = Point(coords)
    
    oFrm = aFrm.overlay(aPt, radius = 5)
    oFrm.show()
    
    gFrm = GreyFrame(a)
    oFrm2 = gFrm.overlay(aPt, radius = 5)
    oFrm2.show()
    
    cFrm = gFrm.to_BGR()
    cFrm = cFrm.overlay(aPt, radius = 5)
    cFrm.show()

