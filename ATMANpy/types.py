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

from typing import Optional

@dataclass(frozen = True)
class Point:
    """
    A point: tuple of integers, signifying x and y coordinates.
    """   
    coords: tuple
    value: Optional[None] = None
    
    def __post_init__(self):
        #### IMMUTABILITY BREAKS ####
        object.__setattr__(self, 'x', self.coords[0])
        object.__setattr__(self, 'y', self.coords[1])
        #############################  
        
    ### METHODS ###
    def up(self, points):
        return Point((self.x, max(self.y - points, 0)))

    def down(self, points):
        return Point((self.x, self.y + points))

    def left(self, points):
        return Point((max(self.x - points, 0), self.y))

    def right(self, points):
        return Point((self.x + points, self.y))
    
    def conform_to(self, image):
        # force the point to be in bounds
        corr_x = min(max(0, self.x), image.cols - 1)
        corr_y = min(max(0, self.y), image.rows - 1)
        return Point((corr_x, corr_y))
    ###########
    
    
@dataclass(frozen = True)
class SafePoint(Point):
    """
    A typechecked point
    """
    def __post_init__(self):
        #### TYPE CHECKS ####
        typecheck(self)
        pointcheck(self)
        
        super(SafePoint, self).__post_init__()
        #####################  

class FrameIterator:
    def __init__(self, frame):
        self._frame = frame
        self._rowIdx = 0
        self._colIdx = 0
        
    def __next__(self):
        rows, cols = self._frame.rows, self._frame.cols
        if self._colIdx >= cols:
            self._colIdx = 0
            self._rowIdx += 1
        
        if self._rowIdx < rows:
            value = self._frame.array[self._rowIdx, self._colIdx]
            out = Point((self._colIdx, self._rowIdx), value = value)
            self._colIdx += 1
            return out
        else:
            raise StopIteration
        
    
@dataclass(frozen = True)
class Frame:
    array: np.ndarray
    name: str = 'frame'
    
    def __post_init__(self):
        #### TYPE CHECKS ####
        typecheck(self)
        #####################
        
        #### IMMUTABILITY BREAKS ####
        object.__setattr__(self, 'rows', self.array.shape[0])
        object.__setattr__(self, 'cols', self.array.shape[1])
        object.__setattr__(self, 'shape', self.array.shape)
        
        if len(self.array.shape) == 1:
            object.__setattr__(self, 'array', np.array([self.array]))
        #############################  
    
    def __iter__(self):
        return FrameIterator(self)
    
    ### METHODS ###
    def show(self):
        show(self.array, name = self.name)

    def eval_at(self, point: Point):
        point = point.conform_to(self)
        return self.array[point.y, point.x]
    
    def fast_eval_at(self, point: Point):
        return self.array[point.y, point.x]
    
    def maybe_eval_at(self, point: Point):
        try:
            return self.array[point.y, point.x]
        except IndexError:
            return None
        
    def mask(self, boolFrm):
        maskArr = boolFrm.array.astype('uint8') # cv mask does not accept bools
        return GreyFrame(cv.bitwise_and(self.array, self.array, mask = maskArr))
        
    ###############

@dataclass(frozen = True)
class BoolFrame(Frame):
    array: np.ndarray
    name: str = 'frame'
    
    def __post_init__(self):
        ### call post init of super class
        super(BoolFrame, self).__post_init__()

        #### TYPE CHECKS ####
        arraycheck(self, member_type = 'bool', 
                   channel_number = 2, color_channel_length = None)
        #####################

    def __or__(self, other):
        return BoolFrame(np.bitwise_or(self.array, other.array), 
                         name = f"{self.name} or {other.name}")

    def __and__(self, other):
        return BoolFrame(np.bitwise_or(self.array, other.array), 
                         name = f"{self.name} and {other.name}")

    def to_grey(self):
        return GreyFrame(self.array.astype('uint8') * 255)

    


@dataclass(frozen = True)
class GreyFrame(Frame):
    array: np.ndarray
    name: str = 'frame'
    
    def __post_init__(self):
        ### call post init of super class
        super(GreyFrame, self).__post_init__()
        
        #### TYPE CHECKS ####
        # we can actually deal with a three dim array, by turning it back to Grey
        arraycheck(self, channel_number = 3, color_channel_length = None)
        ########################  
        
        #### IMMUTABILITY BREAKS ####        
        ### convert BGR to Grey
        if len(self.array.shape) == 3:
            object.__setattr__(self, 'array', cv.cvtColor(self.array,  cv.COLOR_BGR2GRAY))
        #############################  
    
    ### INEQUALITIES ###
    #for a grey frame, these are defined with respect to integers, and return a boolean frame.
    def __lt__(self, other: int):
        return BoolFrame(self.array < other)
    def __le__(self, other: int):
        return BoolFrame(self.array <= other)
    def __gt__(self, other: int):
        return BoolFrame(self.array > other)
    def __ge__(self, other: int):
        return BoolFrame(self.array >= other)
    #####################
    
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
    def overlay(self, point: Point, color = 255, radius = 0, thickness = 1):
        self_copy = self.array.copy() # to preverve immutability of frame
        # conf_point = point.conform_to(self)
        return GreyFrame(cv.circle(self_copy, point.coords, radius, color, thickness))
    
    def zero(self):
        return GreyFrame(np.zeros(self.shape, dtype = 'uint8'))
    ###############

@dataclass(frozen = True)
class BGRFrame(Frame):
    array: np.ndarray
    name: str = 'frame'
    
    def __post_init__(self):
        ### call post init of super class
        super(BGRFrame, self).__post_init__()
        
        #### TYPE CHECKS ####
        arraycheck(self, channel_number = 3, color_channel_length = 3)
        #####################   
        
        #### IMMUTABILITY BREAKS ####
        # convert grayscale to a grey BGR
        if len(self.array.shape) == 2:
            object.__setattr__(self, 'array', cv.cvtColor(self.array,  cv.COLOR_GRAY2BGR))
        #############################  
        
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
    def overlay(self, point: Point, color = (0, 255, 0), radius = 0, thickness = 1):
        self_copy = self.array.copy() # to preverve immutability of frame
        # conf_point = point.conform_to(self)
        return BGRFrame(cv.circle(self_copy, point.coords, radius, color, thickness))
    ###############
    

if __name__ == '__main__':
    
    ### Point and ArbPoint: Does typechecking work?
    coords = (-1, 2)
    coords = (1.5, 2)
    Point(coords)
    # SafePoint(coords)
    
    coords = (0, 0)
    a = np.zeros((10, 10), dtype = 'uint8')
    
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


    Point((4, 4)).up(10)

    ### typecheck works? ###
    # aFrm = BGRFrame(1)

    ### immutability preserved?
    gFrm = GreyFrame(a)
    print(gFrm.shape)
    cv.cvtColor(gFrm.array, cv.COLOR_GRAY2BGR)
    print(gFrm.shape)
    # gFrm.array = cv.cvtColor(gFrm.array, cv.COLOR_GRAY2BGR)
    # print(gFrm.shape)


    cFrm.rows

    ##### Check Iteration
    for cPt in cFrm:
        print(cPt)




